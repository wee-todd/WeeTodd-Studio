import Foundation
import Metal
import MetalPerformanceShaders

extension TextMatrixGPU {
  /// Upload Q/K/V once, perform all heads and softmax on the GPU, and download
  /// only the final interleaved output. One score matrix is reused across heads.
  func attention(q: [Float], k: [Float], v: [Float], tokens: Int, heads: Int,
    kvHeads: Int, width: Int, scale: Float, window: Int?, causal: Bool,
    maximumWorkingBytes: Int = 256 * 1024 * 1024,
    checkCancelled: () throws -> Void = { try Task.checkCancellation() }) throws -> [Float] {
    lastAttentionReadbacks = 0; lastAttentionCommands = 0
    try checkCancelled()
    guard (1...1024).contains(tokens), (1...64).contains(heads),
      (1...64).contains(kvHeads), (1...512).contains(width), heads % kvHeads == 0,
      q.count == tokens*heads*width, k.count == tokens*kvHeads*width, v.count == k.count,
      scale.isFinite, window == nil || window! > 0,
      q.allSatisfy(\.isFinite), k.allSatisfy(\.isFinite), v.allSatisfy(\.isFinite) else {
      throw TextEncodingError.invalid("Invalid GPU text attention shape, values or mask.")
    }
    // Four Q-sized buffers (input, packed input/output, interleaved output),
    // four KV-sized buffers, and one reused score matrix. Caller arrays are separate.
    let workingBytes = (4*q.count + 4*k.count + tokens*tokens)*4
    guard maximumWorkingBytes > 0, workingBytes <= maximumWorkingBytes else {
      throw TextEncodingError.invalid("Text attention exceeds its working-buffer budget.")
    }
    return try autoreleasepool {
      if attentionPipelines == nil {
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: Self.attentionShader, options: options)
        var states: [String: MTLComputePipelineState] = [:]
        for name in ["pack_heads", "attention_softmax"] {
          guard let function = library.makeFunction(name: name) else {
            throw TextEncodingError.invalid("Missing text attention kernel.")
          }
          states[name] = try device.makeComputePipelineState(function: function)
        }
        attentionPipelines = states
      }
      func buffer(_ count: Int, values: [Float]? = nil) throws -> MTLBuffer {
        guard count > 0, count <= device.maxBufferLength/4,
          let storage = device.makeBuffer(length: count*4, options: .storageModeShared) else {
          throw TextEncodingError.invalid("Cannot allocate text attention buffer.")
        }
        if let values {
          values.withUnsafeBytes { storage.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        return storage
      }
      let qi = try buffer(q.count, values: q), ki = try buffer(k.count, values: k), vi = try buffer(v.count, values: v)
      let qp = try buffer(q.count), kp = try buffer(k.count), vp = try buffer(v.count)
      let scores = try buffer(tokens*tokens), packedOutput = try buffer(q.count), output = try buffer(q.count)
      guard let command = queue.makeCommandBuffer() else { throw TextEncodingError.invalid("Cannot allocate attention command.") }
      func encode(_ name: String, _ buffers: [MTLBuffer], _ params: [UInt32], _ count: Int) throws {
        guard let encoder = command.makeComputeCommandEncoder(), let state = attentionPipelines?[name] else {
          throw TextEncodingError.invalid("Cannot encode text attention.")
        }
        encoder.setComputePipelineState(state)
        for (i,b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        params.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
        encoder.dispatchThreads(MTLSize(width: count,height: 1,depth: 1),
          threadsPerThreadgroup: MTLSize(width: min(256,state.maxTotalThreadsPerThreadgroup),height: 1,depth: 1))
        encoder.endEncoding()
      }
      for (input,packed,count) in [(qi,qp,heads),(ki,kp,kvHeads),(vi,vp,kvHeads)] {
        try encode("pack_heads", [input,packed], [tokens,count,width,0].map(UInt32.init), tokens*count*width)
      }
      let headStride = tokens*width*4
      let headDescriptor = MPSMatrixDescriptor(rows: tokens, columns: width, rowBytes: width*4, dataType: .float32)
      let scoreDescriptor = MPSMatrixDescriptor(rows: tokens, columns: tokens, rowBytes: tokens*4, dataType: .float32)
      let scoreMatrix = MPSMatrix(buffer: scores, descriptor: scoreDescriptor)
      let qk = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
        resultRows: tokens,resultColumns: tokens,interiorColumns: width,alpha: Double(scale),beta: 0)
      let av = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
        resultRows: tokens,resultColumns: width,interiorColumns: tokens,alpha: 1,beta: 0)
      for head in 0..<heads {
        try checkCancelled()
        let keyHead = head/(heads/kvHeads)
        qk.encode(commandBuffer: command,
          leftMatrix: MPSMatrix(buffer: qp,offset: head*headStride,descriptor: headDescriptor),
          rightMatrix: MPSMatrix(buffer: kp,offset: keyHead*headStride,descriptor: headDescriptor), resultMatrix: scoreMatrix)
        try encode("attention_softmax", [scores], [UInt32(tokens),UInt32(min(window ?? tokens,tokens)),causal ? 1 : 0], tokens)
        av.encode(commandBuffer: command, leftMatrix: scoreMatrix,
          rightMatrix: MPSMatrix(buffer: vp,offset: keyHead*headStride,descriptor: headDescriptor),
          resultMatrix: MPSMatrix(buffer: packedOutput,offset: head*headStride,descriptor: headDescriptor))
      }
      try encode("pack_heads", [packedOutput,output], [tokens,heads,width,1].map(UInt32.init), q.count)
      command.commit(); lastAttentionCommands = 1; command.waitUntilCompleted()
      guard command.status == .completed else {
        throw TextEncodingError.invalid("Text attention GPU command failed: \(command.error?.localizedDescription ?? "unknown")")
      }
      try checkCancelled()
      let result = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self),count: q.count))
      lastAttentionReadbacks = 1
      guard result.allSatisfy(\.isFinite) else { throw TextEncodingError.invalid("Nonfinite GPU text attention result.") }
      return result
    }
  }

  private static let attentionShader = """
  #include <metal_stdlib>
  using namespace metal;
  kernel void pack_heads(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint t=p[0],h=p[1],w=p[2];
    if(p[3]==0) { uint head=i/(t*w),row=(i/w)%t; y[i]=x[(row*h+head)*w+i%w]; }
    else { uint row=i/(h*w),head=(i/w)%h; y[i]=x[(head*t+row)*w+i%w]; }
  }
  kernel void attention_softmax(device float* scores [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint row [[thread_position_in_grid]]) {
    uint n=p[0],lo=p[2] ? (row+1>p[1] ? row+1-p[1] : 0) : 0,hi=p[2] ? row+1 : n;
    float maximum=-INFINITY;
    for(uint col=lo;col<hi;col++) maximum=max(maximum,scores[row*n+col]);
    float sum=0;
    for(uint col=0;col<n;col++) {
      float value=(col>=lo && col<hi) ? exp(scores[row*n+col]-maximum) : 0;
      scores[row*n+col]=value; sum+=value;
    }
    for(uint col=0;col<n;col++) scores[row*n+col]/=sum;
  }
  """
}
