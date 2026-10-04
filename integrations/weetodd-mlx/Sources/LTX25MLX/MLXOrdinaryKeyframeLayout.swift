import Foundation
import MLX
import LTX25Engine

/// Ordinary timed-image conditioning and optional generated slots. Plain primary
/// keyframes use global attention: no reference-isolation mask is requested.
/// Order is the main canvas, explicit nonzero-frame anchors in submitted order,
/// then generated slots. Only the trailing slotTokens rows receive the learned
/// keyframe marker. The existing ordinary Python path uses these slots in stage
/// one only; stage two must construct this layout with generatedCount zero.
public struct MLXOrdinaryKeyframeLayout:Sendable {
  public struct Anchor:Sendable,Equatable {
    public let frame:Int
    public let strength:Float
    public init(frame:Int,strength:Float) { self.frame=frame;self.strength=strength }
  }
  public let geometry:AVGeometry
  public let anchors:[Anchor]
  public let generatedFrames:[Int]
  public let positions:[Float]
  public var frameTokens:Int { geometry.latentHeight*geometry.latentWidth }
  public var anchorTokens:Int { anchors.filter { $0.frame != 0 }.count*frameTokens }
  public var slotTokens:Int { generatedFrames.count*frameTokens }
  public var videoTokens:Int { geometry.videoTokens+anchorTokens+slotTokens }
  public var videoAttentionGroups:[Int] { [] }

  public init(geometry:AVGeometry,anchors:[Anchor],generatedCount:Int,maximumAnchors:Int=8) throws {
    try Task.checkCancellation()
    guard (1...32).contains(maximumAnchors),anchors.count<=maximumAnchors,Set(anchors.map(\.frame)).count==anchors.count,
      anchors.allSatisfy({ $0.frame>=0 && $0.frame<geometry.frames && $0.strength.isFinite && (0...1).contains($0.strength) }) else {
      throw LTXError.invalid("Keyframes require at most \(maximumAnchors) unique bounded frames and strengths.")
    }
    self.geometry=geometry;self.anchors=anchors
    generatedFrames=try Self.interiorFrames(count:generatedCount,frames:geometry.frames)
    let total=geometry.videoTokens+(anchors.filter({ $0.frame != 0 }).count+generatedFrames.count)*geometry.latentHeight*geometry.latentWidth
    guard total<=131072 else { throw LTXError.invalid("Ordinary keyframe tokens exceed video admission.") }
    var result=geometry.videoPositions
    result.reserveCapacity(total*3)
    for frame in anchors.filter({ $0.frame != 0 }).map(\.frame)+generatedFrames {
      try Task.checkCancellation()
      let time=Float(Double(frame)+0.5)/Float(geometry.fps)
      for h in 0..<geometry.latentHeight { for w in 0..<geometry.latentWidth {
        result += [time,Float(h*32+16),Float(w*32+16)]
      } }
    }
    positions=result
  }

  /// Integer arithmetic implements nearest-even rounding without float drift.
  /// Factored separately so exact half ties are covered even though the public
  /// 8n+1 canvas and count<=8 rarely produce them.
  static func interiorFrames(count:Int,frames:Int) throws -> [Int] {
    guard (0...8).contains(count),frames>=1,frames<=4097,
      count==0 || frames>=count+2 else { throw LTXError.invalid("Generated keyframes require 0–8 bounded interior slots.") }
    guard count>0 else { return [] }
    let denominator=count+1
    let result=(1...count).map { index -> Int in
      let numerator=index*(frames-1),whole=numerator/denominator,remainder=numerator%denominator
      return whole+((remainder*2>denominator || (remainder*2==denominator && whole%2==1)) ? 1 : 0)
    }
    guard Set(result).count==count,result.allSatisfy({ $0>0 && $0<frames-1 }) else {
      throw LTXError.invalid("Generated keyframe positions must be unique interior frames.")
    }
    return result
  }

  /// The runner owns RNG and stage-noise blending. initialSlots, when present,
  /// are the already initialized stage state; clean slots remain zero and their
  /// denoise factors remain one. No source array wrapper is returned directly.
  public func prepare(generated:MLXArray,anchors values:[MLXArray],initialSlots:MLXArray?=nil) throws
    -> (latent:MLXArray,condition:MLXVideoDenoiseCondition) {
    try Task.checkCancellation()
    func valid(_ value:MLXArray,rows:Int) -> Bool {
      value.dtype == .float32 && value.shape == [rows,128] && MLX.isFinite(value).all().item(Bool.self)
    }
    guard valid(generated,rows:geometry.videoTokens),values.count==anchors.count,
      initialSlots == nil || slotTokens>0,
      initialSlots.map({ valid($0,rows:slotTokens) }) ?? true else {
      throw LTXError.invalid("Ordinary canvas, anchor count or generated slots differ from admitted geometry.")
    }
    for value in values {
      try Task.checkCancellation()
      guard valid(value,rows:frameTokens) else { throw LTXError.invalid("Ordinary keyframe latent must be finite normalized Float32 at this stage's image geometry.") }
    }
    var parts:[MLXArray]=[],clean:[MLXArray]=[],mask=[Float](repeating:1,count:geometry.videoTokens)
    if let index=anchors.firstIndex(where:{ $0.frame==0 }) {
      parts.append(values[index].reshaped(values[index].shape));clean.append(values[index].reshaped(values[index].shape))
      if geometry.videoTokens>frameTokens {
        parts.append(generated[frameTokens..<geometry.videoTokens]);clean.append(.zeros([geometry.videoTokens-frameTokens,128]))
      }
      for row in 0..<frameTokens { mask[row]=1-anchors[index].strength }
    } else {
      parts.append(generated.reshaped(generated.shape));clean.append(.zeros([geometry.videoTokens,128]))
    }
    for (index,anchor) in anchors.enumerated() where anchor.frame != 0 {
      try Task.checkCancellation()
      parts.append(values[index].reshaped(values[index].shape));clean.append(values[index].reshaped(values[index].shape))
      mask += [Float](repeating:1-anchor.strength,count:frameTokens)
    }
    if slotTokens>0 {
      parts.append(initialSlots?.reshaped([slotTokens,128]) ?? .zeros([slotTokens,128]))
      clean.append(.zeros([slotTokens,128]));mask += [Float](repeating:1,count:slotTokens)
    }
    let latent=concatenated(parts,axis:0),reference=concatenated(clean,axis:0)
    eval(latent,reference)
    try Task.checkCancellation()
    return (latent,try MLXVideoDenoiseCondition(clean:reference,mask:mask))
  }
}
