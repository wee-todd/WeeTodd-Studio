import Foundation
import MLX
import TensorIO
import LTX25Engine

/// Bounds complete stage tensors plus overlapping residual/packing/convolution
/// buffers. Convolutions use bounded windows; this is not tiled VAE encoding.
public struct MLXVideoEncodePlan:Sendable {
  public let latentShape:[Int]
  public let ownedBufferBytes:Int
  public init(frames:Int,width:Int,height:Int,maximumOwnedBufferBytes:Int=4*1024*1024*1024) throws {
    guard (1...1501).contains(frames),(frames-1)%8==0,
      (32...4096).contains(width),(32...4096).contains(height),width%32==0,height%32==0,
      maximumOwnedBufferBytes>0 else { throw LTXError.invalid("Video encoding requires 8n+1 frames and bounded dimensions divisible by32.") }
    var t=frames,h=height/4,w=width/4
    var peak=0
    let channels=[128,256,512,1024,1024],spatial=[2,1,2,2],temporal=[1,2,2,2]
    for stage in 0..<5 {
      let padded=t+(stage<4 && temporal[stage]==2 ? 1 : 0)
      peak=max(peak,padded*h*w*channels[stage]*4*8)
      if stage<4 { t=(t-1)/temporal[stage]+1;h/=spatial[stage];w/=spatial[stage] }
    }
    ownedBufferBytes=peak+frames*height*width*3*4+384*1024*1024
    guard ownedBufferBytes<=maximumOwnedBufferBytes else {
      throw LTXError.invalid("Video encoder exceeds its owned-buffer budget: needs \(ownedBufferBytes) bytes.")
    }
    latentShape=[t,h,w,128]
  }
}

/// Released causal convolutional VAE encoder. Inputs are FHWC RGB [-1,1];
/// outputs are normalized FHWC latents. Shares the decoder's bounded depth
/// convolution and the image encoder's checkpoint validation; no second model
/// copy or Python process. Only the active layer's weights are resident.
public final class MLXVideoEncoder {
  private let file:SafeTensorFile
  private let gate=NSLock()
  public private(set) var residentWeightBytes=0
  public init(checkpoint:URL) throws {
    _ = try MLXImageEncoder(checkpoint:checkpoint)
    file=try SafeTensorFile(url:checkpoint)
  }
  public func encode(_ pixels:MLXArray,maximumOwnedBufferBytes:Int=4*1024*1024*1024,
    progress:(Int) throws -> Void = { _ in }) throws -> MLXArray {
    guard gate.try() else { throw LTXError.invalid("Video encoder is already running.") }
    defer { gate.unlock() }
    guard pixels.ndim==4,pixels.shape[3]==3,pixels.dtype == .float32 else {
      throw LTXError.invalid("Video encoder requires Float32 FHWC RGB pixels.")
    }
    let plan=try MLXVideoEncodePlan(frames:pixels.shape[0],width:pixels.shape[2],height:pixels.shape[1],
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    try MLXVideoDecoder.checkFinite(pixels)
    guard pixels.min().item(Float.self)>=(-1),pixels.max().item(Float.self)<=1 else {
      throw LTXError.invalid("Video encoder pixels must be in[-1,1].")
    }
    let oldCache=Memory.cacheLimit;Memory.cacheLimit=128*1024*1024
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=oldCache }
    var completed=0
    func finish(_ value:MLXArray) throws -> MLXArray {
      eval(value);try Task.checkCancellation();try MLXVideoDecoder.checkFinite(value);return value
    }
    func convolution(_ input:MLXArray,_ name:String,normalized:Bool=false) throws -> MLXArray {
      try Task.checkCancellation()
      let result=try autoreleasepool {
        let weight=try MLXWeight.read(file,name+".weight").asType(.float32)
        let bias=try MLXWeight.read(file,name+".bias").asType(.float32)
        eval(weight,bias);residentWeightBytes=weight.nbytes+bias.nbytes
        defer { residentWeightBytes=0 }
        try MLXVideoDecoder.checkFinite(weight);try MLXVideoDecoder.checkFinite(bias)
        return try finish(MLXVideoDecoder.convolve(input,weight:weight,bias:bias,causal:true,
          normalized:normalized,maximumWindowElements:MLXVideoDecoder.windowElements))
      }
      if Memory.cacheMemory>128*1024*1024 { Memory.clearCache() }
      completed+=1;try progress(completed);try Task.checkCancellation();return result
    }
    var x=try finish(Self.patch(pixels))
    x=try convolution(x,"encoder.conv_in.conv")
    let blocks=[4,6,4,2,2],channels=[256,512,1024,1024],spatial=[2,1,2,2],temporal=[1,2,2,2]
    for stage in 0..<5 {
      for block in 0..<blocks[stage] {
        let residual=x.reshaped(x.shape),prefix="encoder.down_blocks.\(stage*2).res_blocks.\(block)"
        x=try convolution(x,prefix+".conv1.conv",normalized:true)
        x=try convolution(x,prefix+".conv2.conv",normalized:true)
        x=try finish(x+residual)
      }
      if stage<4 {
        // The trained downsampler prepends before BOTH branches, including
        // convolution. Padding after convolution misaligns moving frames.
        x=try finish(Self.temporalPad(x,stride:temporal[stage]))
        let skip=Self.downpack(x,spatial:spatial[stage],temporal:temporal[stage])
        let group=skip.shape[3]/channels[stage]
        let residual=try finish(skip.reshaped([skip.shape[0],skip.shape[1],skip.shape[2],channels[stage],group]).mean(axis:-1))
        let projected=try convolution(x,"encoder.down_blocks.\(stage*2+1).conv.conv")
        x=try finish(Self.downpack(projected,spatial:spatial[stage],temporal:temporal[stage])+residual)
      }
    }
    x=try convolution(x,"encoder.conv_out.conv",normalized:true)
    let mean=try MLXWeight.read(file,"per_channel_statistics.mean-of-means").asType(.float32)
    let std=try MLXWeight.read(file,"per_channel_statistics.std-of-means").asType(.float32)
    try MLXVideoDecoder.checkFinite(mean);try MLXVideoDecoder.checkFinite(std)
    guard std.min().item(Float.self)>0 else { throw LTXError.invalid("Invalid video encoder statistics.") }
    return try finish(((x[0...,0...,0...,0..<128]-mean)/std).reshaped(plan.latentShape))
  }
  static func patch(_ x:MLXArray) -> MLXArray {
    let f=x.shape[0],h=x.shape[1],w=x.shape[2]
    return x.reshaped([f,h/4,4,w/4,4,3]).transposed(0,1,3,5,4,2).reshaped([f,h/4,w/4,48])
  }
  static func temporalPad(_ x:MLXArray,stride:Int) -> MLXArray {
    stride==2 ? concatenated([x[0..<1],x],axis:0) : x
  }
  /// Input already includes the causal leading frame for temporal stride two.
  static func downpack(_ x:MLXArray,spatial s:Int,temporal t:Int) -> MLXArray {
    let f=x.shape[0],h=x.shape[1],w=x.shape[2],c=x.shape[3]
    return x.reshaped([f/t,t,h/s,s,w/s,s,c]).transposed(0,2,4,6,1,3,5).reshaped([f/t,h/s,w/s,c*t*s*s])
  }
}
