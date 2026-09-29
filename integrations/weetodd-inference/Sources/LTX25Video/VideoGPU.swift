import Foundation
import Metal
import MetalPerformanceShaders

public enum VideoDecodeError: Error, LocalizedError {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

/// THWC storage. Every operation completes before returning, so replacing a stage
/// releases its buffers instead of retaining the complete decoder execution graph.
struct VideoTensor {
  let buffer: MTLBuffer
  let shape: [Int]
  var count: Int { shape.reduce(1, *) }
}

final class VideoGPU {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let pipelines: [String: MTLComputePipelineState]
  private(set) var lastNormalizationWorkspaceBytes = 0
  private(set) var lastConvolutionCommandCount = 0

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), MPSSupportsMTLDevice(device),
      let queue = device.makeCommandQueue() else {
      throw VideoDecodeError.invalid("A Metal device with MPS matrix support is required.")
    }
    self.device = device; self.queue = queue
    let library = try device.makeLibrary(source: Self.shader + Self.upscaleShader, options: nil)
    var states: [String: MTLComputePipelineState] = [:]
    for name in ["patches", "bias", "norm_silu", "norm_scales", "add", "shuffle", "group_norm_silu"] {
      guard let function = library.makeFunction(name: name) else {
        throw VideoDecodeError.invalid("Missing video kernel: \(name)")
      }
      states[name] = try device.makeComputePipelineState(function: function)
    }
    pipelines = states
  }

  func buffer(count: Int) throws -> MTLBuffer {
    guard count > 0, count <= device.maxBufferLength / 4,
      let result = device.makeBuffer(length: count * 4, options: .storageModeShared) else {
      throw VideoDecodeError.invalid("Video buffer exceeds the Metal allocation limit.")
    }
    return result
  }

  func tensor(_ values: [Float], shape: [Int]) throws -> VideoTensor {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }), shape.reduce(1, *) == values.count else {
      throw VideoDecodeError.invalid("Video tensor shape does not match its values.")
    }
    let storage = try buffer(count: values.count)
    values.withUnsafeBufferPointer { storage.contents().copyMemory(from: $0.baseAddress!, byteCount: values.count * 4) }
    return VideoTensor(buffer: storage, shape: shape)
  }

  func values(_ input: VideoTensor) -> [Float] {
    Array(UnsafeBufferPointer(start: input.buffer.contents().assumingMemoryBound(to: Float.self), count: input.count))
  }

  func makeCommand() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else { throw VideoDecodeError.invalid("Cannot allocate video command buffer.") }
    return command
  }

  func finish(_ command: MTLCommandBuffer) throws {
    command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else {
      throw VideoDecodeError.invalid("Video GPU command failed: \(command.error?.localizedDescription ?? "unknown error")")
    }
  }

  func encode(_ name: String, buffers: [MTLBuffer], parameters: [UInt32], count: Int,
    command: MTLCommandBuffer) throws {
    guard let encoder = command.makeComputeCommandEncoder(), let state = pipelines[name] else {
      throw VideoDecodeError.invalid("Cannot encode video kernel \(name).")
    }
    encoder.setComputePipelineState(state)
    for (index, storage) in buffers.enumerated() { encoder.setBuffer(storage, offset: 0, index: index) }
    parameters.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(state.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    encoder.endEncoding()
  }

  func normalize(_ x: VideoTensor) throws -> VideoTensor {
    return try autoreleasepool {
      let output = try buffer(count: x.count)
      let command = try makeCommand()
      try encode("norm_silu", buffers: [x.buffer,output], parameters: [UInt32(x.shape[3])],
        count: x.count / x.shape[3], command: command)
      try finish(command)
      return VideoTensor(buffer: output, shape: x.shape)
    }
  }

  func add(_ x: VideoTensor, residual: VideoTensor) throws -> VideoTensor {
    return try autoreleasepool {
      guard x.shape == residual.shape else { throw VideoDecodeError.invalid("Residual shapes differ.") }
      let output = try buffer(count: x.count)
      let command = try makeCommand()
      try encode("add", buffers: [x.buffer,residual.buffer,output], parameters: [0], count: x.count, command: command)
      try finish(command)
      return VideoTensor(buffer: output, shape: x.shape)
    }
  }

  func shuffle(_ x: VideoTensor, spatial: Int, temporal: Int, unpatch: Bool = false) throws -> VideoTensor {
    return try autoreleasepool {
      let factor = spatial * spatial * temporal
      guard factor > 0, x.shape[3] % factor == 0 else { throw VideoDecodeError.invalid("Invalid channel shuffle factors.") }
      let crop = temporal > 1 ? 1 : 0
      let shape = [x.shape[0] * temporal - crop, x.shape[1] * spatial, x.shape[2] * spatial, x.shape[3] / factor]
      let count = shape.reduce(1, *)
      let output = try buffer(count: count)
      let command = try makeCommand()
      let parameters = (x.shape + shape + [spatial,temporal,crop,unpatch ? 1 : 0]).map(UInt32.init)
      try encode("shuffle", buffers: [x.buffer,output], parameters: parameters, count: count, command: command)
      try finish(command)
      return VideoTensor(buffer: output, shape: shape)
    }
  }

  /// Weights are O,I,T,H,W; im2col preserves that reduction order. Only
  /// `windowSites` output sites are expanded, irrespective of video dimensions.
  func convolution(_ x: VideoTensor, weights: MTLBuffer, bias: MTLBuffer,
    outputChannels: Int, causal: Bool, windowSites: Int,
    normalizeInput: Bool = false, windowsPerCommand: Int = 16,
    kernelDepth: Int = 3, zeroTemporalPadding: Bool = false,
    checkCancelled: () throws -> Void) throws -> VideoTensor {
    lastNormalizationWorkspaceBytes = 0; lastConvolutionCommandCount = 0
    guard (1...4096).contains(windowSites), (1...64).contains(windowsPerCommand),
      [1,3].contains(kernelDepth), !(kernelDepth == 1 && causal),
      outputChannels > 0, weights.length >= outputChannels*x.shape[3]*kernelDepth*9*4,
      bias.length >= outputChannels*4 else { throw VideoDecodeError.invalid("Invalid convolution weights or window configuration.") }
    try checkCancelled()
    let sites = x.shape[0] * x.shape[1] * x.shape[2], reduction = x.shape[3] * kernelDepth * 9
    let output = try buffer(count: sites * outputChannels)
    let workspace = try buffer(count: min(windowSites,sites) * reduction)
    let scales = normalizeInput ? try buffer(count: sites) : x.buffer
    if normalizeInput {
      let command = try makeCommand()
      try encode("norm_scales",buffers: [x.buffer,scales],parameters: [UInt32(x.shape[3])],count: sites,command: command)
      try finish(command); lastConvolutionCommandCount += 1
      lastNormalizationWorkspaceBytes = sites*4
    }
    let right = MPSMatrix(buffer: weights, descriptor: MPSMatrixDescriptor(rows: outputChannels,
      columns: reduction, rowBytes: reduction * 4, dataType: .float32))
    for batchStart in stride(from: 0, to: sites, by: windowSites*windowsPerCommand) {
      try checkCancelled()
      try autoreleasepool {
        let command = try makeCommand()
        for start in stride(from: batchStart,to: min(sites,batchStart+windowSites*windowsPerCommand),by: windowSites) {
        try checkCancelled()
        let rows = min(windowSites, sites-start)
        try encode("patches", buffers: [x.buffer,workspace,scales],
          parameters: (x.shape + [start,causal ? 1 : 0,normalizeInput ? 1 : 0,kernelDepth,zeroTemporalPadding ? 1 : 0]).map(UInt32.init),
          count: rows * reduction, command: command)
        let left = MPSMatrix(buffer: workspace, descriptor: MPSMatrixDescriptor(rows: rows,
          columns: reduction, rowBytes: reduction * 4, dataType: .float32))
        let destination = MPSMatrix(buffer: output, offset: start * outputChannels * 4,
          descriptor: MPSMatrixDescriptor(rows: rows, columns: outputChannels,
            rowBytes: outputChannels * 4, dataType: .float32))
        let multiply = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
          resultRows: rows, resultColumns: outputChannels, interiorColumns: reduction, alpha: 1, beta: 0)
        multiply.encode(commandBuffer: command, leftMatrix: left, rightMatrix: right, resultMatrix: destination)
        }
        try finish(command); lastConvolutionCommandCount += 1
      }
    }
    try checkCancelled()
    let command = try makeCommand()
    try encode("bias", buffers: [output,bias], parameters: [UInt32(outputChannels)],
      count: sites * outputChannels, command: command)
    try finish(command); lastConvolutionCommandCount += 1
    return VideoTensor(buffer: output, shape: [x.shape[0],x.shape[1],x.shape[2],outputChannels])
  }

  private static let shader = """
  #include <metal_stdlib>
  using namespace metal;
  kernel void patches(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    device const float* scales [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint F=p[0], H=p[1], W=p[2], C=p[3], area=p[7]*9, K=C*area;
    uint site=i/K+p[4], k=i%K, c=k/area, q=k%area;
    int t=int(site/(H*W))+int(q/9)-(p[5] ? 2 : int(p[7]/2));
    int y=int((site/W)%H)+int((q/3)%3)-1;
    int z=int(site%W)+int(q%3)-1;
    if(p[8] && (t<0 || t>=int(F))) { out[i]=0.0f; return; }
    t=clamp(t,0,int(F)-1);
    if(y<0 || y>=int(H) || z<0 || z>=int(W)) { out[i]=0.0f; return; }
    uint inputSite=(t*H+y)*W+z; float value=x[inputSite*C+c];
    if(p[6]) { value*=scales[inputSite]; value=value/(1.0f+exp(-value)); }
    out[i]=value;
  }
  kernel void bias(device float* x [[buffer(0)]], device const float* b [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) { x[i]+=b[i%p[0]]; }
  kernel void norm_silu(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint site [[thread_position_in_grid]]) {
    uint C=p[0], start=site*C; float sum=0;
    for(uint c=0;c<C;c++) sum+=x[start+c]*x[start+c];
    float scale=rsqrt(sum/float(C)+1e-8f);
    for(uint c=0;c<C;c++) { float v=x[start+c]*scale; out[start+c]=v/(1.0f+exp(-v)); }
  }
  kernel void norm_scales(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint site [[thread_position_in_grid]]) {
    uint C=p[0],start=site*C; float sum=0;
    for(uint c=0;c<C;c++) sum+=x[start+c]*x[start+c];
    out[site]=rsqrt(sum/float(C)+1e-8f);
  }
  kernel void add(device const float* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    out[i]=x[i]+r[i];
  }
  kernel void shuffle(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint H=p[1], W=p[2], C=p[3], OH=p[5], OW=p[6], OC=p[7], sf=p[8], tf=p[9];
    uint c=i%OC, site=i/OC, z=site%OW, y=(site/OW)%OH, t=site/(OW*OH)+p[10];
    uint ci=p[11] ? ((c*sf+z%sf)*sf+y%sf) : (((c*tf+t%tf)*sf+y%sf)*sf+z%sf);
    out[i]=x[(((t/tf)*H+y/sf)*W+z/sf)*C+ci];
  }
  """
}
