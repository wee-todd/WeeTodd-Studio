import Foundation
import MLX

/// Explicit causal projection session. Input is normalized RGB [B,C,F,H,W].
/// Warm-up consumes one first frame; each later call consumes four source frames.
public final class LQProjectionSession {
  private let weights:[String:MLXArray]
  public private(set) var sourceID:String
  public private(set) var chunks=0
  private var cache1:MLXArray?, cache2:MLXArray?
  private var canvas:[Int]?
  public init(checkpoint:FlashVSRCheckpoint, sourceID:String) throws {
    self.sourceID=sourceID
    var loaded=[String:MLXArray]()
    for key in FlashVSRCheckpoint.expectedShapes(component:.projection).keys { loaded[key]=try checkpoint.tensor(key) }
    weights=loaded
  }
  // Small deterministic fixtures can exercise the state machine without full weights.
  init(weights:[String:MLXArray], sourceID:String) { self.weights=weights; self.sourceID=sourceID }
  public func reset(sourceID:String) { self.sourceID=sourceID; chunks=0; cache1=nil; cache2=nil; canvas=nil }
  public static func normalizeRGB8(_ value:MLXArray) -> MLXArray { value.asType(.float32)/127.5-1 }
  public static func pixelUnshuffle(_ x:MLXArray, factor:Int=16) throws -> MLXArray {
    guard x.ndim == 5, factor > 0, x.shape[3]%factor == 0, x.shape[4]%factor == 0 else { throw FlashVSRError.invalid("LQ projection canvas must divide by its spatial factor.") }
    let b=x.shape[0],c=x.shape[1],t=x.shape[2],h=x.shape[3]/factor,w=x.shape[4]/factor
    return x.reshaped([b,c,t,h,factor,w,factor]).transposed(0,1,4,6,2,3,5).reshaped([b,c*factor*factor,t,h,w])
  }
  public static func causalConvolution(_ x:MLXArray, weight:MLXArray, bias:MLXArray, cache:MLXArray?, spatialPadding:Int=1) throws -> MLXArray {
    guard x.ndim == 5, weight.ndim == 5, weight.shape[1] == x.shape[1], weight.shape[2] == 4, x.shape[2] > 0,
      bias.shape == [weight.shape[0]], spatialPadding >= 0 else { throw FlashVSRError.invalid("Invalid LQ causal convolution.") }
    var padded=x
    if let cache {
      guard cache.shape[0..<2] == x.shape[0..<2], cache.shape[3...] == x.shape[3...], cache.shape[2] <= 2 else { throw FlashVSRError.invalid("LQ carry dimensions changed.") }
      padded=concatenated([cache.asType(x.dtype),x],axis:2)
    }
    let missing=2-(cache?.shape[2] ?? 0)
    if missing > 0 { padded=concatenated([repeated(padded[0...,0...,0..<1],count:missing,axis:2),padded],axis:2) }
    if spatialPadding > 0 {
      let rows=MLXArray(((-spatialPadding)..<(x.shape[3]+spatialPadding)).map { Int32(Swift.max(0,Swift.min(x.shape[3]-1,$0))) })
      let cols=MLXArray(((-spatialPadding)..<(x.shape[4]+spatialPadding)).map { Int32(Swift.max(0,Swift.min(x.shape[4]-1,$0))) })
      padded=take(take(padded,rows,axis:3),cols,axis:4)
    }
    let result=conv3d(padded.transposed(0,2,3,4,1),weight.transposed(0,2,3,4,1),stride:[2,1,1])+bias
    return result.transposed(0,4,1,2,3)
  }
  private func normalizeActivate(_ x:MLXArray, key:String) throws -> MLXArray {
    guard let weight=weights[key] else { throw FlashVSRError.invalid("Missing LQ normalization.") }
    // Release ChannelRMSNorm is F.normalize * sqrt(C), with norm clamp 1e-12.
    let f=x.asType(.float32), norm=maximum(sqrt(sum(f*f,axis:1,keepDims:true)),MLXArray(Float(1e-12)))
    let normalized=(f/norm*sqrt(Float(x.shape[1]))).asType(x.dtype)*weight.expandedDimensions(axis:0)
    return normalized*sigmoid(normalized)
  }
  /// Source indices expose the unique first frame, one-frame lookahead, and EOF-only
  /// repeat padding explicitly. The pipeline crops decoded output to its visible interval.
  public static func groupSourceIndices(frameCount:Int,firstGroup:Bool,isFinal:Bool) throws -> [[Int]] {
    let needed=firstGroup ? 17 : 16
    guard frameCount > 0, frameCount <= needed, isFinal || frameCount == needed else {
      throw FlashVSRError.invalid("Projection needs 17 initial / 16 later source frames; only a final group may repeat its tail.")
    }
    var slices=[[Int]]()
    if firstGroup { slices.append([0]) }
    let offset=firstGroup ? 1 : 0
    for chunk in 0..<4 { slices.append((0..<4).map { Swift.min(offset+4*chunk+$0,frameCount-1) }) }
    return slices
  }
  public func processGroup(_ video:MLXArray,sourceID:String,isFinal:Bool=false) throws -> [MLXArray] {
    guard video.ndim == 5, chunks == 0 || chunks%4 == 1 else { throw FlashVSRError.invalid("Projection group begins at an incomplete stream boundary.") }
    let slices=try Self.groupSourceIndices(frameCount:video.shape[2],firstGroup:chunks == 0,isFinal:isFinal)
    var parts=[MLXArray]()
    for slice in slices {
      if let value=try process(take(video,MLXArray(slice.map(Int32.init)),axis:2),sourceID:sourceID) { parts.append(value[0]) }
    }
    return [concatenated(parts,axis:1)]
  }
  public func process(_ video:MLXArray, sourceID:String) throws -> [MLXArray]? {
    guard sourceID == self.sourceID, !sourceID.isEmpty, video.ndim == 5,
      video.shape[0] == 1, video.shape[1] == 3, video.shape[2] == (chunks == 0 ? 1 : 4) else {
      throw FlashVSRError.invalid("LQ stream requires the same source, one warm-up frame, then four-frame chunks.")
    }
    let dimensions=Array(video.shape[3...])
    guard canvas == nil || canvas == dimensions else { throw FlashVSRError.invalid("LQ source canvas changed.") }
    var input=video
    if chunks == 0 { input=repeated(video,count:4,axis:2) }
    let packed=try Self.pixelUnshuffle(input)
    guard let w1=weights["conv1.weight"],let b1=weights["conv1.bias"],let w2=weights["conv2.weight"],let b2=weights["conv2.bias"],let linear=weights["linear_layers.0.weight"],let bias=weights["linear_layers.0.bias"] else { throw FlashVSRError.invalid("Incomplete LQ projection weights.") }
    let first=try normalizeActivate(Self.causalConvolution(packed.asType(w1.dtype),weight:w1,bias:b1,cache:cache1),key:"norm1.weight")
    // Compact copies avoid retaining previous source tensors in carry views.
    let next1=take(packed,MLXArray([Int32(packed.shape[2]-2),Int32(packed.shape[2]-1)]),axis:2).asType(w1.dtype), next2=take(first,MLXArray([Int32(first.shape[2]-2),Int32(first.shape[2]-1)]),axis:2)
    var output:[MLXArray]?
    if chunks > 0 {
      let second=try normalizeActivate(Self.causalConvolution(first,weight:w2,bias:b2,cache:cache2),key:"norm2.weight")
      let tokens=second.transposed(0,2,3,4,1).reshaped([1,-1,second.shape[1]])
      let projected=matmul(tokens,linear.transposed())+bias; eval(projected); output=[projected]
    }
    eval(next1,next2); cache1=next1; cache2=next2; canvas=dimensions; chunks += 1
    return output
  }
}
