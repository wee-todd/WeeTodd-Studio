import Foundation
import MLX
import TensorIO

/// The released 784-channel temporal-consistent decoder. Memory belongs to an
/// explicit clip carry; each callback receives at most four normalized RGB frames.
public final class FlashVSRTinyDecoder {
  public struct Carry {
    public private(set) var sourceID:String
    public fileprivate(set) var emittedFrames=0
    fileprivate var memories:[Int:MLXArray]=[:]
    fileprivate var canvas:[Int]?,model:String?
    fileprivate var closed=false
    public init(sourceID:String) { self.sourceID=sourceID }
    public mutating func reset(sourceID:String) { self=Self(sourceID:sourceID) }
    public var cachedBytes:Int { memories.values.reduce(0) { $0+$1.nbytes } }
  }
  private var weights:[String:MLXArray]
  private let identity:String
  public var isLoaded:Bool { !weights.isEmpty }
  public init(checkpoint:URL,dtype:DType = .bfloat16)throws {
    weights=try FlashVSRDecoderOps.load(checkpoint:checkpoint,profile:.tinyDecoder,dtype:dtype,include:{ _ in true })
    guard weights.count==66,weights["decoder.1.weight"]?.shape==[512,784,3,3],weights["decoder.26.weight"]?.shape==[3,128,3,3] else {
      throw FlashVSRError.invalid("Selected Tiny checkpoint is not the released 784-channel FlashVSR decoder.")
    }
    identity=TorchTensorImportProfile.tinyDecoder.sourceSHA256+"/"+String(describing:dtype)
  }
  public func unload() { weights.removeAll() }
  /// [1,16,L,h,w] and matched source RGB [1,3,4L−3 (first) or 4L,8h,8w]
  /// become [L,h,w,768+16]. Only initial causal history repeats the first frame.
  public static func pack(latents:MLXArray,sourceRGB:MLXArray,first:Bool)throws->MLXArray {
    guard latents.ndim==5,sourceRGB.ndim==5,latents.shape[0]==1,latents.shape[1]==16,
      latents.shape[2]>0,latents.shape[3]>0,latents.shape[4]>0,
      sourceRGB.shape[0]==1,sourceRGB.shape[1]==3,
      sourceRGB.shape[2]==4*latents.shape[2]-(first ? 3:0),
      sourceRGB.shape[3]==8*latents.shape[3],sourceRGB.shape[4]==8*latents.shape[4],
      latents.dtype.isFloatingPoint,sourceRGB.dtype.isFloatingPoint else { throw FlashVSRError.invalid("Tiny requires 16-channel latents and its exactly aligned RGB source groups at 8× canvas scale.") }
    let source=first ? concatenated([repeated(sourceRGB[0...,0...,0..<1],count:3,axis:2),sourceRGB],axis:2):sourceRGB
    let frames=latents.shape[2],h=latents.shape[3],w=latents.shape[4]
    let packed=source.reshaped([1,3,frames,4,h,8,w,8]).transposed(0,2,4,6,1,3,5,7).reshaped([frames,h,w,768])
    let z=latents.transposed(0,2,3,4,1).reshaped([frames,h,w,16])
    return concatenated([packed.asType(latents.dtype),z],axis:3)
  }
  public func decode(latents:MLXArray,sourceRGB:MLXArray,sourceID:String,carry:inout Carry,
    receive:(Int,MLXArray)throws->Void)throws {
    try Task.checkCancellation()
    guard isLoaded else { throw FlashVSRError.invalid("Tiny decoder is unloaded; load the selected released TCDecoder checkpoint before decoding.") }
    guard !sourceID.isEmpty,sourceID==carry.sourceID,!carry.closed,carry.model==nil || carry.model==identity,
      latents.ndim==5,latents.shape[0]==1,latents.shape[1]==16,(1...4).contains(latents.shape[2]),
      latents.shape[3]>0,latents.shape[4]>0,sourceRGB.ndim==5,sourceRGB.shape[0]==1,sourceRGB.shape[1]==3,
      sourceRGB.shape[3]==latents.shape[3]*8,sourceRGB.shape[4]==latents.shape[4]*8,
      latents.dtype.isFloatingPoint,sourceRGB.dtype.isFloatingPoint else {
      throw FlashVSRError.invalid("Tiny stream source, decoder identity or bounded latent/RGB geometry changed; reset carry for a new clip.")
    }
    let first=carry.emittedFrames==0,natural=latents.shape[2]*4-(first ? 3:0),visible=sourceRGB.shape[2]
    guard visible>0,visible<=natural,visible>natural-4,
      carry.canvas==nil || carry.canvas==Array(latents.shape[3...]) else { throw FlashVSRError.invalid("Tiny source frame count/canvas differs from its frozen latent interval.") }
    let dtype=weights["decoder.1.weight"]!.dtype
    var source=sourceRGB.asType(dtype)
    if visible<natural { source=concatenated([source,repeated(source[0...,0...,(visible-1)..<visible],count:natural-visible,axis:2)],axis:2) }
    let packed=try Self.pack(latents:latents.asType(dtype),sourceRGB:source,first:first)
    var completed=false
    defer { if !completed { carry.memories.removeAll();carry.closed=true } }
    carry.canvas=Array(latents.shape[3...]);carry.model=identity
    var remaining=visible
    for frame in 0..<latents.shape[2] {
      try Task.checkCancellation()
      var value=packed[frame..<frame+1]
      value=tanh(value/3)*3
      value=try FlashVSRDecoderOps.conv2D(value,key:"decoder.1",weights:weights)
      value=maximum(value,0)
      value=maximum(try FlashVSRDecoderOps.conv2D(value,key:"decoder.3",weights:weights),0)
      for index in [5,6,7] { value=try memory(value,index:index,carry:&carry) }
      value=FlashVSRDecoderOps.nearest2x(value)
      value=try FlashVSRDecoderOps.temporalGrow(value,weight:FlashVSRDecoderOps.weight("decoder.9.conv.weight",weights))
      value=try FlashVSRDecoderOps.conv2D(value,key:"decoder.10",weights:weights)
      for index in [11,12,13] { value=try memory(value,index:index,carry:&carry) }
      value=FlashVSRDecoderOps.nearest2x(value)
      value=try FlashVSRDecoderOps.temporalGrow(value,weight:FlashVSRDecoderOps.weight("decoder.15.conv.weight",weights))
      value=try FlashVSRDecoderOps.conv2D(value,key:"decoder.16",weights:weights)
      for index in [17,18,19] { value=try memory(value,index:index,carry:&carry) }
      value=FlashVSRDecoderOps.nearest2x(value)
      value=try FlashVSRDecoderOps.temporalGrow(value,weight:FlashVSRDecoderOps.weight("decoder.21.conv.weight",weights))
      value=try FlashVSRDecoderOps.conv2D(value,key:"decoder.22",weights:weights)
      value=maximum(value,0)
      value=maximum(try FlashVSRDecoderOps.conv2D(value,key:"decoder.24",weights:weights),0)
      value=try FlashVSRDecoderOps.conv2D(value,key:"decoder.26",weights:weights)
      // The initial temporal history produces three hidden RGB frames. Trim
      // once per clip; terminal source padding is never published.
      if first && frame==0 { value=value[3..<4] }
      let count=min(remaining,value.shape[0])
      if count>0 {
        let rgb=(value[0..<count]*2-1).asType(latents.dtype)
        eval(rgb);try Task.checkCancellation();try receive(carry.emittedFrames,rgb)
        carry.emittedFrames += count;remaining -= count
      }
    }
    guard remaining==0 else { throw FlashVSRError.invalid("Tiny decoder did not restore the requested visible interval.") }
    carry.closed=visible<natural;completed=true
  }
  private func memory(_ input:MLXArray,index:Int,carry:inout Carry)throws->MLXArray {
    var previous=carry.memories[index]
    let output=try FlashVSRDecoderOps.memoryBlock(input,previous:&previous,weights:weights,prefix:"decoder.\(index)")
    carry.memories[index]=previous;return output
  }
}
