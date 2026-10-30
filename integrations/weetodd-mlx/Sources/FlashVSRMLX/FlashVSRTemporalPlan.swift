import Foundation
import InferenceContracts
import MLX

/// Expansion is an internal model index map. The source's rational cadence and
/// visible interval remain unchanged; EOF padding is never merged into a real tail.
public struct FlashVSRTemporalPlan:Codable,Equatable,Sendable {
  public static let version="flashvsr-native-temporal-v1"
  public let sourceFrames:Int,effectiveFrames:Int,expandedFrames:Int,repeatFactor:Int,groupCount:Int
  public let gaussianWeights:[Float]
  public init(recipe:NativeFlashVSRRecipe)throws {
    try recipe.validate()
    sourceFrames=recipe.frames;effectiveFrames=recipe.effectiveFrames;expandedFrames=recipe.expandedFrames
    repeatFactor=recipe.repeatFactor;groupCount=recipe.groupCount
    let center=Float(repeatFactor-1)/2,sigma=Float(recipe.temporalSigma)
    let weights=(0..<repeatFactor).map { index -> Float in
      let position=(Float(index)-center)/sigma;return exp(-0.5*position*position)
    }
    let sum=weights.reduce(0,+);gaussianWeights=weights.map { $0/sum }
    guard gaussianWeights.allSatisfy({ $0.isFinite }),sum>0 else { throw FlashVSRError.invalid("Temporal Gaussian weights are not finite at the selected sigma.") }
  }
  public func sourceIndex(expandedIndex:Int)->Int {
    let index=max(0,min(expandedFrames-1,expandedIndex))
    return min(sourceFrames-1,index==0 ? 0:1+(index-1)/repeatFactor)
  }
  public func projectionSourceIndices(group:Int)throws->[Int] {
    guard (0..<groupCount).contains(group) else { throw FlashVSRError.invalid("Projection group lies outside the frozen temporal plan.") }
    let start=group==0 ? 0:1+group*16,count=group==0 ? 17:16
    return (start..<start+count).map { sourceIndex(expandedIndex:$0) }
  }
  public func decoderExpandedRange(group:Int)throws->Range<Int> {
    guard (0..<groupCount).contains(group) else { throw FlashVSRError.invalid("Decoder group lies outside the frozen temporal plan.") }
    let start=group==0 ? 0:13+(group-1)*16
    return start..<start+(group==0 ? 13:16)
  }
  func coordinate(_ expanded:Int)->(frame:Int,weight:Float,complete:Bool)? {
    guard expanded>=0,expanded<expandedFrames else { return nil }
    if expanded==0 { return (0,1,true) }
    let frame=1+(expanded-1)/repeatFactor,position=(expanded-1)%repeatFactor
    guard frame<sourceFrames else { return nil }
    return (frame,gaussianWeights[position],position==repeatFactor-1)
  }
}

/// Holds at most one FP32 accumulated visible frame, including when a model
/// callback splits a duplicate group. Input offsets must be strictly contiguous.
final class FlashVSRTemporalMerger {
  private let plan:FlashVSRTemporalPlan
  private var expectedExpanded=0,nextFrame=0,pending:MLXArray?,canvas:[Int]?
  private var closed=false
  init(plan:FlashVSRTemporalPlan) { self.plan=plan }
  func consume(offset:Int,frames:MLXArray,receive:(Int,MLXArray)throws->Void)throws {
    guard !closed,offset==expectedExpanded,frames.ndim==4,(1...4).contains(frames.shape[0]),
      frames.shape[1]>0,frames.shape[2]>0,frames.shape[3]==3,frames.dtype.isFloatingPoint,
      canvas==nil || canvas==Array(frames.shape[1...]) else { throw FlashVSRError.invalid("Decoded frame clock/canvas is discontinuous before temporal merge.") }
    canvas=Array(frames.shape[1...])
    do {
      for index in 0..<frames.shape[0] {
        try Task.checkCancellation()
        if let coordinate=plan.coordinate(expectedExpanded) {
          guard coordinate.frame==nextFrame else { throw FlashVSRError.invalid("Temporal merge received an unexpected visible frame index.") }
          let value=frames[index].asType(.float32)*coordinate.weight
          pending=pending.map { $0+value } ?? value
          eval(pending!)
          if coordinate.complete {
            let output=pending!;pending=nil
            try receive(nextFrame,output);nextFrame += 1
          }
        }
        expectedExpanded += 1
      }
    } catch { closed=true;pending=nil;throw error }
  }
  func finish()throws {
    guard !closed,nextFrame==plan.sourceFrames,pending==nil,expectedExpanded>=plan.expandedFrames else { throw FlashVSRError.invalid("Temporal merge did not receive the complete visible source interval.") }
    closed=true
  }
}
