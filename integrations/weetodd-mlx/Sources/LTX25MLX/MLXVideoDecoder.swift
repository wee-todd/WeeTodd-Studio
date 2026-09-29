import Foundation
import MLX
import TensorIO
import LTX25Engine
import LTX25Video

/// One weighted layer at a time, with bounded depth slices evaluated as MLX
/// Conv2D. Temporal replication and spatial zero padding retain the trained 3D
/// operator; windows partition output frames and never average overlaps.
public final class MLXVideoDecoder {
  private let file:SafeTensorFile
  private let causal:Bool
  private let precision:MLXVideoPrecision
  public let cacheLimitBytes:Int
  private let gate=NSLock()
  private static let workspaceGate=NSLock()
  public private(set) var residentWeightBytes=0
  public private(set) var stageSeconds:[String:Double]=[:]
  static let windowElements=8*1024*1024
  // The reference BF16 decoder uses compiled SiLU: keep its intermediate
  // sigmoid in the fused kernel instead of rounding it to BF16 separately.
  private static let silu: @Sendable (MLXArray) -> MLXArray = compile(shapeless:true) { x in x*sigmoid(x) }
  public init(checkpoint:URL,precision:MLXVideoPrecision = .bfloat16,cacheLimitBytes:Int?=nil) throws {
    let cache=cacheLimitBytes ?? (precision == .bfloat16 ? 2048 : 128)*1024*1024
    guard (0...2*1024*1024*1024).contains(cache) else { throw LTXError.invalid("Video allocator cache must be within 0–2 GiB.") }
    self.cacheLimitBytes=cache
    // Reuse the complete checkpoint/header contract, without loading weights.
    _ = try VideoDecoder(checkpoint:checkpoint)
    file=try SafeTensorFile(url:checkpoint)
    self.precision=precision
    let config=try JSONSerialization.jsonObject(with:Data(file.metadata["config"]!.utf8)) as! [String:Any]
    causal=(config["vae"] as! [String:Any])["causal_decoder"] as! Bool
  }
  public func decode(latent:MLXArray,configuration c:VideoDecodeConfiguration = .init(),
    progress:(Int,Int) throws -> Void = { _,_ in },receive:(VideoFrameChunk) throws -> Void) throws {
    try decodeFrames(latent:latent,configuration:c,progress:progress) { index,image in
      try receive(VideoFrameChunk(startFrame:index,frameCount:1,width:image.shape[1],height:image.shape[0],frameRate:c.frameRate,rgb:image.asType(.float32).asArray(Float.self)))
    }
  }
  public func decodeRGB8(latent:MLXArray,configuration c:VideoDecodeConfiguration = .init(),
    progress:(Int,Int) throws -> Void = { _,_ in },receive:(Int,Data) throws -> Void) throws {
    try decodeFrames(latent:latent,configuration:c,progress:progress) { index,image in try receive(index,Self.rgb8(image)) }
  }
  static func rgb8(_ image:MLXArray) -> Data {
    Data(((clip(image.asType(.float32),min:-1,max:1)+1)*Float(127.5)).asType(.uint8).asArray(UInt8.self))
  }
  private func decodeFrames(latent:MLXArray,configuration c:VideoDecodeConfiguration,
    progress:(Int,Int) throws -> Void,receive:(Int,MLXArray) throws -> Void) throws {
    let plan=try MLXVideoDecodePlan(shape:latent.shape,configuration:c,precision:precision)
    let largest=file.tensors.filter { $0.key.hasPrefix("decoder.") && $0.key.hasSuffix(".weight") }
      .map { Int($0.value.shape.reduce(1,*))*precision.bytes }.max()!
    guard largest<=c.maximumWeightBytes else { throw LTXError.invalid("Video convolution exceeds its layer weight allowance.") }
    guard latent.dtype == .float32 else {
      throw LTXError.invalid("Video decode requires finite Float32 BCFHW latents.")
    }
    try Self.checkFinite(latent)
    guard gate.try() else { throw LTXError.invalid("Video decoder is already running.") }
    defer { gate.unlock() }
    // MLX's workspace control is process-global. Serialize decoder instances
    // and restore it on every exit, including a throwing progress callback.
    guard Self.workspaceGate.try() else { throw LTXError.invalid("Another MLX video decoder is running.") }
    let workspaceKey="MLX_CONV_WINOGRAD_WORKING_SET"
    let oldWorkspace=getenv(workspaceKey).map { String(cString:$0) }
    setenv(workspaceKey,"1073741824",1)
    defer {
      if let oldWorkspace { setenv(workspaceKey,oldWorkspace,1) } else { unsetenv(workspaceKey) }
      Self.workspaceGate.unlock()
    }
    // A 128 MiB allocator cache repeatedly destroys convolution work buffers
    // larger than the cache. Reserve bounded reuse during this stage only.
    let cacheBytes=cacheLimitBytes
    let oldCache=Memory.cacheLimit;Memory.cacheLimit=cacheBytes
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=oldCache }
    try Task.checkCancellation()
    stageSeconds=[:]
    var completed=0
    func finish(_ value:MLXArray) throws -> MLXArray {
      eval(value);try Task.checkCancellation()
      let start=Date();defer { stageSeconds["finite_checks",default:0] += Date().timeIntervalSince(start) }
      try Self.checkFinite(value)
      return value
    }
    func conv(_ input:MLXArray,_ name:String,normalized:Bool=false) throws -> MLXArray {
      try Task.checkCancellation()
      let bytes=Int(file.tensors[name+".weight"]!.shape.reduce(1,*))*precision.bytes
      guard bytes <= c.maximumWeightBytes else { throw LTXError.invalid("Video convolution exceeds its layer weight allowance.") }
      let out=try autoreleasepool {
        let readStart=Date()
        let weight=try MLXWeight.read(file,name+".weight").asType(precision.dtype)
        let bias=try MLXWeight.read(file,name+".bias").asType(precision.dtype)
        eval(weight,bias);residentWeightBytes=weight.nbytes+bias.nbytes
        stageSeconds["weight_read_evaluate",default:0] += Date().timeIntervalSince(readStart)
        defer { residentWeightBytes=0 }
        try Self.checkFinite(weight);try Self.checkFinite(bias)
        let computeStart=Date()
        let result=try Self.convolve(input,weight:weight,bias:bias,causal:causal,normalized:normalized,maximumWindowElements:precision.windowElements)
        eval(result)
        stageSeconds["convolution_evaluate",default:0] += Date().timeIntervalSince(computeStart)
        return try finish(result)
      }
      if Memory.cacheMemory>cacheBytes { Memory.clearCache() }
      completed += 1;try progress(completed,42);try Task.checkCancellation()
      return out
    }
    let mean=try MLXWeight.read(file,"per_channel_statistics.mean-of-means").asType(precision.dtype)
    let std=try MLXWeight.read(file,"per_channel_statistics.std-of-means").asType(precision.dtype)
    try Self.checkFinite(mean);try Self.checkFinite(std)
    guard std.min().item(Float.self)>0 else { throw LTXError.invalid("Invalid decoder normalization statistics.") }
    var x=try finish(latent[0].asType(precision.dtype).transposed(1,2,3,0)*std+mean)
    x=try conv(x,"decoder.conv_in.conv")
    let counts=[2,2,4,6,4],scales=[(2,2),(2,2),(1,2),(2,1)]
    for stage in 0..<5 {
      for block in 0..<counts[stage] {
        let residual=x.reshaped(x.shape),prefix="decoder.up_blocks.\(stage*2).res_blocks.\(block)"
        x=try conv(x,prefix+".conv1.conv",normalized:true)
        x=try conv(x,prefix+".conv2.conv",normalized:true)
        x=try finish(x+residual)
      }
      if stage<4 {
        x=try conv(x,"decoder.up_blocks.\(stage*2+1).conv.conv")
        x=try finish(Self.shuffle(x,spatial:scales[stage].0,temporal:scales[stage].1))
      }
    }
    x=try conv(x,"decoder.conv_out.conv",normalized:true)
    x=try finish(Self.shuffle(x,spatial:4,temporal:1,unpatch:true))
    guard x.shape == plan.outputShape else { throw LTXError.invalid("Decoded video shape disagrees with admission.") }
    for frame in 0..<x.shape[0] {
      try Task.checkCancellation()
      try receive(frame,x[frame])
    }
  }
  /// Slice before reducing: flattening a valid multidimensional activation can
  /// itself exceed MLX's Int32 shape-dimension limit. Bounded axis slices also
  /// keep isFinite's boolean intermediates small, including on strided views.
  static func checkFinite(_ value:MLXArray,maximumWindowElements:Int=windowElements) throws {
    guard maximumWindowElements>0 else { throw LTXError.invalid("Finite-check window must be positive.") }
    let limit=min(maximumWindowElements,Int(Int32.max))
    func scan(_ part:MLXArray) throws {
      try Task.checkCancellation()
      guard part.size>0 else { return }
      if part.size<=limit {
        guard MLX.isFinite(part).all().item(Bool.self) else {
          throw LTXError.invalid("Nonfinite video decoder tensor.")
        }
        return
      }
      let plane=part.size/part.shape[0]
      if plane>limit {
        // A single frame/plane may still be too large: drop this axis and
        // partition the remaining dimensions, without copying the full tensor.
        for index in 0..<part.shape[0] { try autoreleasepool { try scan(part[index]) } }
      } else {
        let depth=max(1,limit/plane)
        for start in stride(from:0,to:part.shape[0],by:depth) {
          try autoreleasepool { try scan(part[start..<min(part.shape[0],start+depth)]) }
        }
      }
    }
    try scan(value)
  }
  static func convolve(_ x:MLXArray,weight:MLXArray,bias:MLXArray,causal:Bool,normalized:Bool,
    maximumWindowElements:Int) throws -> MLXArray {
    guard maximumWindowElements>0 else { throw LTXError.invalid("Convolution depth window must be positive.") }
    let t=x.shape[0],h=x.shape[1],w=x.shape[2],ic=x.shape[3],oc=weight.shape[0]
    let depth=max(1,maximumWindowElements/(h*w*max(ic,oc)))
    let output=try x.dtype == .float32 ? MLXVideoWindowBuffer(shape:[t,h,w,oc]) : nil
    var windows:[MLXArray]=[]
    for start in stride(from:0,to:t,by:depth) {
      try Task.checkCancellation()
      let end=min(t,start+depth)
      try autoreleasepool {
        let haloStart=start-(causal ? 2 : 1),haloEnd=end+(causal ? 0 : 1)
        var prepared:MLXArray?
        if normalized {
          let halo=take(x,MLXArray((haloStart..<haloEnd).map { Int32(max(0,min(t-1,$0))) }),axis:0)
          let norm=MLXFast.rmsNorm(halo,weight:.ones([ic],dtype:x.dtype),eps:1e-8)
          prepared=x.dtype == .bfloat16 ? Self.silu(norm) : norm*sigmoid(norm)
          eval(prepared!)
        }
        var accumulated:MLXArray?
        for d in 0..<3 {
          try Task.checkCancellation()
          let lo=start+d-(causal ? 2 : 1),hi=end+d-(causal ? 2 : 1)
          var input:MLXArray
          if let prepared { input=prepared[d..<(d+end-start)] }
          else if lo>=0 && hi<=t { input=x[lo..<hi] }
          else { input=take(x,MLXArray((lo..<hi).map { Int32(max(0,min(t-1,$0))) }),axis:0) }
          let kernel=weight[0...,0...,d,0...,0...].transposed(0,2,3,1)
          let contribution=conv2d(input,kernel,padding:1)
          eval(contribution)
          accumulated=accumulated.map { $0+contribution } ?? contribution
          eval(accumulated!)
        }
        let window=accumulated!+bias
        if let output { try output.append(window,start:start) }
        else { eval(window);windows.append(window) }
      }
    }
    if let output { return try output.finish() }
    return windows.count == 1 ? windows[0] : concatenated(windows,axis:0)
  }
  static func shuffle(_ x:MLXArray,spatial s:Int,temporal t:Int,unpatch:Bool=false) -> MLXArray {
    let f=x.shape[0],h=x.shape[1],w=x.shape[2],c=x.shape[3]/(s*s*t)
    if unpatch { return x.reshaped([f,h,w,c,s,s]).transposed(0,1,5,2,4,3).reshaped([f,h*s,w*s,c]) }
    let result=x.reshaped([f,h,w,c,t,s,s]).transposed(0,4,1,5,2,6,3).reshaped([f*t,h*s,w*s,c])
    return t>1 ? result[1...] : result
  }
}
