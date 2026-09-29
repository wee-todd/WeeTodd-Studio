import Foundation
import MLX
import TensorIO
import LTX25Video
import LTX25Engine

/// The released spatial x2 network, evaluated without downloading the latent.
/// One convolution's weights at a time; GroupNorm covers the complete volume.
public final class MLXLatentUpscaler {
  private let file:SafeTensorFile
  private let statistics:SafeTensorFile
  private let gate=NSLock()
  public private(set) var residentWeightBytes=0
  public init(checkpoint:URL,statisticsCheckpoint:URL) throws {
    _ = try LatentUpscaler(checkpoint:checkpoint,statisticsCheckpoint:statisticsCheckpoint)
    file=try SafeTensorFile(url:checkpoint);statistics=try SafeTensorFile(url:statisticsCheckpoint)
  }
  static func admit(shape:[Int],maximumActivationBytes:Int) throws -> LatentUpscalePlan {
    var c=LatentUpscaleConfiguration();c.maximumActivationBytes=maximumActivationBytes
    let plan=try LatentUpscalePlan(shape:shape,configuration:c)
    let workspace=plan.outputShape.dropLast().reduce(1,*)*1024*27*4
    guard plan.activationBytes <= maximumActivationBytes-workspace else {
      throw LTXError.invalid("MLX upscaler exceeds its activation/workspace allowance.")
    }
    return plan
  }
  public func upscale(_ input:MLXArray,maximumActivationBytes:Int=2*1024*1024*1024,
    progress:(String) throws -> Void = { _ in }) throws -> MLXArray {
    let plan=try Self.admit(shape:input.shape,maximumActivationBytes:maximumActivationBytes)
    // Include a conservative full im2col workspace, even when the MLX backend
    // selects an implicit convolution that requires less temporary storage.
    guard input.dtype == .float32 else { throw LTXError.invalid("MLX upscaler requires Float32 input.") }
    guard gate.try() else { throw LTXError.invalid("MLX upscaler is already running.") }
    defer { gate.unlock() }
    let oldCache=Memory.cacheLimit;Memory.cacheLimit=128*1024*1024
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=oldCache }
    func finish(_ x:MLXArray) throws -> MLXArray {
      eval(x);try Task.checkCancellation()
      guard MLX.isFinite(x).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite MLX upscaler tensor.") }
      return x
    }
    func report(_ stage:String) throws { try Task.checkCancellation();try progress(stage);try Task.checkCancellation() }
    func read(_ name:String) throws -> MLXArray { try MLXWeight.read(file,name).asType(.float32) }
    func conv(_ x:MLXArray,_ name:String) throws -> MLXArray {
      try Task.checkCancellation()
      return try autoreleasepool {
        let w=try read(name+".weight"),b=try read(name+".bias")
        residentWeightBytes=w.nbytes+b.nbytes;defer { residentWeightBytes=0 }
        return try finish(Self.convolution(x,weight:w,bias:b))
      }
    }
    func norm(_ x:MLXArray,_ name:String,_ residual:MLXArray?=nil) throws -> MLXArray {
      try finish(Self.normalized(x,weight:read(name+".weight"),bias:read(name+".bias"),residual:residual))
    }
    let mean=try finish(MLXWeight.read(statistics,"per_channel_statistics.mean-of-means").asType(.float32))
    let std=try finish(MLXWeight.read(statistics,"per_channel_statistics.std-of-means").asType(.float32))
    guard std.min().item(Float.self)>0 else { throw LTXError.invalid("Invalid latent normalization statistics.") }
    var x=try finish(input*std+mean)
    x=try norm(conv(x,"initial_conv"),"initial_norm");try report("initial")
    for stage in ["res_blocks","post_upsample_res_blocks"] {
      if stage == "post_upsample_res_blocks" {
        x=try finish(Self.shuffle(conv(x,"upsampler.0")));try report("spatial_x2")
      }
      for block in 0..<4 {
        x=try autoreleasepool {
          let residual=x,stem="\(stage).\(block)"
          let first=try norm(conv(x,stem+".conv1"),stem+".norm1")
          return try norm(conv(first,stem+".conv2"),stem+".norm2",residual)
        }
        try report("\(stage).\(block)")
      }
    }
    x=try finish((conv(x,"final_conv")-mean)/std)
    guard x.shape==plan.outputShape else { throw LTXError.invalid("MLX upscaler output geometry differs.") }
    try report("complete")
    return x
  }
  static func convolution(_ x:MLXArray,weight:MLXArray,bias:MLXArray) -> MLXArray {
    if weight.ndim==4 { return conv2d(x,weight.transposed(0,2,3,1),padding:1)+bias }
    return conv3d(x.expandedDimensions(axis:0),weight.transposed(0,2,3,4,1),padding:1)[0]+bias
  }
  static func normalized(_ x:MLXArray,weight:MLXArray,bias:MLXArray,residual:MLXArray?=nil) -> MLXArray {
    let channels=x.shape[3],sites=x.size/channels,groupSize=channels/32
    let grouped=x.reshaped([sites,32,groupSize]).transposed(1,0,2).reshaped([32,sites*groupSize])
    let normalized=MLXFast.layerNorm(grouped,eps:1e-5)
      .reshaped([32,sites,groupSize]).transposed(1,0,2).reshaped(x.shape)
    var result=normalized*weight+bias
    if let residual { result=result+residual }
    return result*sigmoid(result)
  }
  static func shuffle(_ x:MLXArray) -> MLXArray {
    let t=x.shape[0],h=x.shape[1],w=x.shape[2],c=x.shape[3]/4
    return x.reshaped([t,h,w,c,2,2]).transposed(0,1,4,2,5,3).reshaped([t,h*2,w*2,c])
  }
}
