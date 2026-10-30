import Foundation
import MLX
import TensorIO

/// Wan 2.1 decoding with explicit per-tile temporal input carry. This uses only
/// the exact FlashVSR-pinned VAE, independent of any other installed Wan codec.
public final class FlashVSRWanDecoder {
  public struct Tiling:Sendable,Equatable {
    public let height:Int,width:Int,strideHeight:Int,strideWidth:Int
    private init() { height=30;width=52;strideHeight=15;strideWidth=26 }
    public static let released=Self()
    public init(height:Int,width:Int,strideHeight:Int,strideWidth:Int)throws {
      guard height>0,width>0,strideHeight>0,strideWidth>0,strideHeight<=height,strideWidth<=width,
        height<=128,width<=128 else { throw FlashVSRError.invalid("Wan decoder tiles require positive sizes/strides with stride no greater than size (maximum 128 latent pixels).") }
      self.height=height;self.width=width;self.strideHeight=strideHeight;self.strideWidth=strideWidth
    }
  }
  fileprivate struct TileCarry {
    var caches:[String:MLXArray]=[:]
    var upsampleStarted=Set<String>()
    var cachedBytes:Int { caches.values.reduce(0) { $0+$1.nbytes } }
  }
  public struct Carry {
    public private(set) var sourceID:String
    public fileprivate(set) var emittedFrames=0
    fileprivate var tiles:[String:TileCarry]=[:]
    fileprivate var canvas:[Int]?,model:String?
    fileprivate var closed=false
    fileprivate var spool:FlashVSRDecoderCarryStore?
    fileprivate let cacheDirectory:URL?
    public init(sourceID:String,cacheDirectory:URL?=nil) { self.sourceID=sourceID;self.cacheDirectory=cacheDirectory }
    public mutating func reset(sourceID:String) { self=Self(sourceID:sourceID,cacheDirectory:cacheDirectory) }
    public var spilledBytes:UInt64 { spool?.storedBytes ?? 0 }
    public var cachedBytes:Int { tiles.values.reduce(0) { $0+$1.cachedBytes } }
  }
  private var weights:[String:MLXArray]
  private let identity:String
  public let tiling:Tiling?
  public var isLoaded:Bool { !weights.isEmpty }
  public init(checkpoint:URL,dtype:DType = .bfloat16,tiling:Tiling? = .released)throws {
    if let tiling {
      let bytes=UInt64(tiling.height)*UInt64(tiling.width)*151328*UInt64(dtype == .float32 ? 4:2)
      guard bytes<=1024*1024*1024 else { throw FlashVSRError.invalid("Selected Wan tile exceeds the 1 GiB activation-carry bound; reduce tile size.") }
    }
    weights=try FlashVSRDecoderOps.load(checkpoint:checkpoint,profile:.wanVAE,dtype:dtype,
      include:{ $0.hasPrefix("decoder.") || $0.hasPrefix("conv2.") })
    guard weights["conv2.weight"]?.shape==[16,16,1,1,1],weights["decoder.conv1.weight"]?.shape==[384,16,3,3,3],
      weights["decoder.head.2.weight"]?.shape==[3,96,3,3,3] else {
      throw FlashVSRError.invalid("Selected Full decoder requires the released FlashVSR Wan2.1 VAE checkpoint.")
    }
    self.tiling=tiling
    identity=TorchTensorImportProfile.wanVAE.sourceSHA256+"/"+String(describing:dtype)+"/"+String(describing:tiling)
  }
  public func unload() { weights.removeAll() }
  private static let latentMean:[Float]=[-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]
  private static let latentStd:[Float]=[2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]
  public static func denormalize(_ latents:MLXArray)throws->MLXArray {
    guard latents.ndim==5,latents.shape[0]>0,latents.shape[1]==16,latents.shape[2]>0,latents.shape[3]>0,
      latents.shape[4]>0,latents.dtype.isFloatingPoint else { throw FlashVSRError.invalid("Full decoder expects normalized floating Wan latents [B,16,F,h,w].") }
    let mean=MLXArray(latentMean).asType(latents.dtype).reshaped([1,16,1,1,1])
    // Released buffers store reciprocal std in FP32, then cast. Preserve that
    // rounding order for BF16 instead of multiplying by a separately cast std.
    let inv=(1/MLXArray(latentStd)).asType(latents.dtype).reshaped([1,16,1,1,1])
    return latents/inv+mean
  }
  public static func temporalShuffle(_ x:MLXArray)throws->MLXArray {
    guard x.ndim==5,x.shape[4]>0,x.shape[4]%2==0 else { throw FlashVSRError.invalid("Wan temporal expansion requires paired channel groups.") }
    let b=x.shape[0],t=x.shape[1],h=x.shape[2],w=x.shape[3],c=x.shape[4]/2
    return x.reshaped([b,t,h,w,2,c]).transposed(0,1,4,2,3,5).reshaped([b,t*2,h,w,c])
  }
  public static func tileStarts(length:Int,size:Int,stride:Int)throws->[Int] {
    guard length>0,size>0,stride>0,stride<=size else { throw FlashVSRError.invalid("Wan tile stride must cover its positive canvas.") }
    let end=length-min(size,length)
    var starts=Array(Swift.stride(from:0,through:end,by:stride))
    if starts.last != end { starts.append(end) };return starts
  }
  static func blendMask(height:Int,width:Int,top:Bool,bottom:Bool,left:Bool,right:Bool,borderHeight:Int,borderWidth:Int)->MLXArray {
    var rows=[Float](repeating:1,count:height),columns=[Float](repeating:1,count:width)
    let bh=min(borderHeight,height),bw=min(borderWidth,width)
    if !top && bh>0 { for i in 0..<bh { rows[i]=Float(i+1)/Float(bh) } }
    if !bottom && bh>0 { for i in 0..<bh { rows[height-bh+i]=Float(bh-i)/Float(bh) } }
    if !left && bw>0 { for i in 0..<bw { columns[i]=Float(i+1)/Float(bw) } }
    if !right && bw>0 { for i in 0..<bw { columns[width-bw+i]=Float(bw-i)/Float(bw) } }
    return minimum(MLXArray(rows).reshaped([1,height,1,1]),MLXArray(columns).reshaped([1,1,width,1]))
  }
  /// Each callback emits [1 or <=4,H,W,3] RGB on the caller's unchanged frame
  /// interval. visibleFrames trims only a terminal padded latent group.
  public func decode(latents:MLXArray,sourceID:String,carry:inout Carry,visibleFrames:Int?=nil,
    receive:(Int,MLXArray)throws->Void)throws {
    try Task.checkCancellation()
    guard isLoaded else { throw FlashVSRError.invalid("Full decoder is unloaded; load the selected released Wan2.1_VAE checkpoint before decoding.") }
    guard !sourceID.isEmpty,sourceID==carry.sourceID,!carry.closed,carry.model==nil || carry.model==identity,
      latents.ndim==5,latents.shape[0]==1,latents.shape[1]==16,(1...4).contains(latents.shape[2]),
      latents.shape[3]>0,latents.shape[4]>0,latents.dtype.isFloatingPoint,
      carry.canvas==nil || carry.canvas==Array(latents.shape[3...]) else {
      throw FlashVSRError.invalid("Full decoder source, identity or bounded latent canvas changed; reset carry for a new clip.")
    }
    let first=carry.emittedFrames==0,natural=4*latents.shape[2]-(first ? 3:0),visible=visibleFrames ?? natural
    guard visible>0,visible<=natural,visible>natural-4 else { throw FlashVSRError.invalid("Full decoder visible interval differs from its latent clock.") }
    let dtype=weights["conv2.weight"]!.dtype
    let unscaled=try Self.denormalize(latents.asType(dtype)).transposed(0,2,3,4,1)
    var completed=false
    defer { if !completed { carry.tiles.removeAll();carry.spool=nil;carry.closed=true } }
    carry.canvas=Array(latents.shape[3...]);carry.model=identity
    var remaining=visible
    for index in 0..<latents.shape[2] {
      try Task.checkCancellation()
      var unused:MLXArray?
      let projected=try FlashVSRDecoderOps.causal(unscaled[0...,index..<index+1],
        weight:FlashVSRDecoderOps.weight("conv2.weight",weights),bias:weights["conv2.bias"],cache:&unused,spatialPadding:0)
      let decoded=try decodeOneTiled(projected,carry:&carry)
      let count=min(remaining,decoded.shape[0])
      if count>0 {
        let rgb=clip(decoded[0..<count],min:-1,max:1).asType(latents.dtype)
        eval(rgb);try Task.checkCancellation();try receive(carry.emittedFrames,rgb)
        carry.emittedFrames += count;remaining -= count
      }
    }
    guard remaining==0 else { throw FlashVSRError.invalid("Full decoder did not restore its requested visible interval.") }
    carry.closed=visible<natural;completed=true
  }
  private func decodeOneTiled(_ input:MLXArray,carry:inout Carry)throws->MLXArray {
    let h=input.shape[2],w=input.shape[3]
    guard let tiling else {
      var state=carry.tiles["full"] ?? TileCarry()
      let output=try decodeOne(input,carry:&state)
      carry.tiles["full"]=state
      return output.reshaped([output.shape[1],output.shape[2],output.shape[3],3])
    }
    let th=min(tiling.height,h),tw=min(tiling.width,w)
    let ys=try Self.tileStarts(length:h,size:th,stride:min(tiling.strideHeight,th)),xs=try Self.tileStarts(length:w,size:tw,stride:min(tiling.strideWidth,tw))
    var values:MLXArray?,weights:MLXArray?
    for y in ys { for x in xs {
      try Task.checkCancellation()
      let key="\(y),\(x)",bottom=y+th,right=x+tw
      let stateFromDisk:TileCarry?
      if let spool=carry.spool,spool.contains(key) {
        let saved=try spool.load(key:key);stateFromDisk=TileCarry(caches:saved.caches,upsampleStarted:saved.upsampleStarted)
      } else { stateFromDisk=nil }
      var state=carry.tiles[key] ?? stateFromDisk ?? TileCarry()
      let tile=try decodeOne(input[0...,0...,y..<bottom,x..<right,0...],carry:&state)
      // Multiple spatial tiles keep only one tile's causal features resident.
      // Private disk state preserves exact segmented history without retaining
      // a full canvas of intermediate decoder activations in unified memory.
      if ys.count*xs.count>1 {
        if carry.spool==nil { carry.spool=try FlashVSRDecoderCarryStore(parent:carry.cacheDirectory,sourceID:carry.sourceID,modelIdentity:identity) }
        try carry.spool!.save(key:key,caches:state.caches,upsampleStarted:state.upsampleStarted)
      } else { carry.tiles[key]=state }
      let rgb=tile.reshaped([tile.shape[1],tile.shape[2],tile.shape[3],3]).asType(.float32)
      guard rgb.shape[1]==th*8,rgb.shape[2]==tw*8 else { throw FlashVSRError.invalid("Wan decoder tile scale differs from the released 8× canvas.") }
      if values==nil {
        values=MLXArray.zeros([rgb.shape[0],h*8,w*8,3]);weights=MLXArray.zeros([1,h*8,w*8,1])
      }
      guard values!.shape[0]==rgb.shape[0] else { throw FlashVSRError.invalid("Wan decoder tiles disagree on their temporal clock.") }
      let mask=Self.blendMask(height:th*8,width:tw*8,top:y==0,bottom:bottom==h,left:x==0,right:right==w,
        borderHeight:max(0,th-tiling.strideHeight)*8,borderWidth:max(0,tw-tiling.strideWidth)*8)
      let yr=(y*8)..<(bottom*8),xr=(x*8)..<(right*8)
      values![0...,yr,xr,0...]=values![0...,yr,xr,0...]+rgb*mask
      weights![0...,yr,xr,0...]=weights![0...,yr,xr,0...]+mask
      eval(values!,weights!)
    } }
    guard let values,let weights else { throw FlashVSRError.invalid("Wan spatial decode produced no tiles.") }
    let result=(values/maximum(weights,Float(1e-8))).asType(input.dtype);eval(result);return result
  }
  private func causal(_ x:MLXArray,key:String,carry:inout TileCarry)throws->MLXArray {
    let weight=try FlashVSRDecoderOps.weight(key+".weight",weights)
    var cache=carry.caches[key]
    let value=try FlashVSRDecoderOps.causal(x,weight:weight,bias:weights[key+".bias"],cache:&cache,spatialPadding:weight.shape[3]/2)
    carry.caches[key]=cache;return value
  }
  private func residual(_ x:MLXArray,key:String,carry:inout TileCarry)throws->MLXArray {
    let shortcut:MLXArray
    if let weight=weights[key+".shortcut.weight"] {
      var unused:MLXArray?
      shortcut=try FlashVSRDecoderOps.causal(x,weight:weight,bias:weights[key+".shortcut.bias"],cache:&unused,spatialPadding:0)
    } else { shortcut=x }
    var value=try FlashVSRDecoderOps.rms(x,gamma:FlashVSRDecoderOps.weight(key+".residual.0.gamma",weights))
    value=try causal(FlashVSRDecoderOps.silu(value),key:key+".residual.2",carry:&carry)
    value=try FlashVSRDecoderOps.rms(value,gamma:FlashVSRDecoderOps.weight(key+".residual.3.gamma",weights))
    value=try causal(FlashVSRDecoderOps.silu(value),key:key+".residual.6",carry:&carry)
    let result=value+shortcut;eval(result);return result
  }
  private func upsample(_ input:MLXArray,key:String,temporal:Bool,carry:inout TileCarry)throws->MLXArray {
    var x=input
    if temporal {
      if carry.upsampleStarted.contains(key) {
        x=try Self.temporalShuffle(causal(x,key:key+".time_conv",carry:&carry))
      } else {
        // First frame bypasses the temporal convolution. Its feature is not
        // retained: the next call begins with the release's zero-history sentinel.
        carry.upsampleStarted.insert(key)
      }
    }
    let b=x.shape[0],t=x.shape[1],h=x.shape[2],w=x.shape[3],c=x.shape[4]
    let pixels=FlashVSRDecoderOps.nearest2x(x.reshaped([b*t,h,w,c]))
    let output=try FlashVSRDecoderOps.conv2D(pixels,key:key+".resample.1",weights:weights)
    return output.reshaped([b,t,output.shape[1],output.shape[2],output.shape[3]])
  }
  private func decodeOne(_ input:MLXArray,carry:inout TileCarry)throws->MLXArray {
    var value=try causal(input,key:"decoder.conv1",carry:&carry)
    value=try residual(value,key:"decoder.middle.0",carry:&carry)
    value=try FlashVSRDecoderOps.attention(value,gamma:FlashVSRDecoderOps.weight("decoder.middle.1.norm.gamma",weights),
      qkv:FlashVSRDecoderOps.weight("decoder.middle.1.to_qkv.weight",weights),qkvBias:FlashVSRDecoderOps.weight("decoder.middle.1.to_qkv.bias",weights),
      projection:FlashVSRDecoderOps.weight("decoder.middle.1.proj.weight",weights),projectionBias:FlashVSRDecoderOps.weight("decoder.middle.1.proj.bias",weights))
    value=try residual(value,key:"decoder.middle.2",carry:&carry)
    for layer in 0..<15 {
      try Task.checkCancellation()
      let key="decoder.upsamples.\(layer)"
      if [3,7,11].contains(layer) { value=try upsample(value,key:key,temporal:layer != 11,carry:&carry) }
      else { value=try residual(value,key:key,carry:&carry) }
    }
    value=try FlashVSRDecoderOps.rms(value,gamma:FlashVSRDecoderOps.weight("decoder.head.0.gamma",weights))
    return try causal(FlashVSRDecoderOps.silu(value),key:"decoder.head.2",carry:&carry)
  }
}
