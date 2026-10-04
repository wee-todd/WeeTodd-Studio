import Foundation
import MLX
import LTX25Engine

/// Plain single-stage conditioning shares global attention. The existing
/// primary pipeline appends ordinary images, then ordered IC/Union guides,
/// then generated slots. Only frame zero replaces the main canvas. Only the
/// trailing slotTokens rows receive the learned keyframe marker. RNG belongs
/// to the runner: this helper never creates noise or changes its draw order.
public struct MLXSingleStageControlLayout:Sendable {
  public let geometry:AVGeometry
  public let ordinaryLayout:MLXOrdinaryKeyframeLayout
  private let anchorLayout:MLXOrdinaryKeyframeLayout
  private let guideRows:[Int]
  private let guideStrengths:[Float]
  public let guideTokens:Int
  public let videoTokens:Int
  public let positions:[Float]
  public var slotTokens:Int { ordinaryLayout.slotTokens }
  public var generatedFrames:[Int] { ordinaryLayout.generatedFrames }
  public var videoAttentionGroups:[Int] { [] }

  public init(geometry:AVGeometry,anchors:[MLXOrdinaryKeyframeLayout.Anchor],
    generatedCount:Int,icControl:MLXICControl?=nil,unionStrength:Float?=nil) throws {
    try Task.checkCancellation()
    guard icControl == nil || unionStrength == nil else {
      throw LTXError.invalid("Single-stage IC and Union controls are mutually exclusive.")
    }
    self.geometry=geometry
    ordinaryLayout=try MLXOrdinaryKeyframeLayout(geometry:geometry,anchors:anchors,generatedCount:generatedCount)
    anchorLayout=try MLXOrdinaryKeyframeLayout(geometry:geometry,anchors:anchors,generatedCount:0)
    let guidePositions:[Float]
    if let icControl {
      let layout=try MLXICControlLayout(geometry:geometry,control:icControl)
      guideRows=Array(repeating:layout.referenceTokens,count:layout.strengths.count)
      guideStrengths=layout.strengths
      guidePositions=Array(layout.positions.dropFirst(geometry.videoTokens*3))
    } else if let unionStrength {
      let layout=try MLXUnionControlLayout(geometry:geometry,strength:unionStrength)
      guideRows=[layout.referenceTokens];guideStrengths=[layout.strength]
      guidePositions=Array(layout.positions.dropFirst(geometry.videoTokens*3))
    } else {
      guideRows=[];guideStrengths=[];guidePositions=[]
    }
    guideTokens=guideRows.reduce(0,+)
    videoTokens=ordinaryLayout.videoTokens+guideTokens
    guard videoTokens<=131072 else {
      throw LTXError.invalid("Combined ordinary images, control guides and generated slots exceed video admission.")
    }
    positions=anchorLayout.positions + guidePositions +
      Array(ordinaryLayout.positions.suffix(ordinaryLayout.slotTokens*3))
    guard positions.count==videoTokens*3 else {
      throw LTXError.invalid("Single-stage conditioning positions differ from the admitted rows.")
    }
    try Task.checkCancellation()
  }

  /// Per-token timestep modulation is required by any explicit anchor or guide.
  public var requiresPerTokenVideo:Bool { !ordinaryLayout.anchors.isEmpty || guideTokens>0 }
  /// The CFG++ runner and all admission paths reserve the same combined rows.
  public func cfgppReserveBytes(audioTokens:Int) throws -> Int {
    guard (1...1501).contains(audioTokens) else { throw LTXError.invalid("Invalid single-stage audio token count.") }
    return (videoTokens*2+audioTokens)*128*4*8+1024*6144*4
  }

  /// Inputs are already normalized Float32 at the selected guide geometry.
  /// initialSlots is the runner's initialized state, not a new RNG request.
  /// Clean slots remain zero and their denoise masks remain one.
  public func prepare(generated:MLXArray,anchors:[MLXArray],guides:[MLXArray],
    initialSlots:MLXArray?=nil) throws -> (latent:MLXArray,condition:MLXVideoDenoiseCondition) {
    try Task.checkCancellation()
    func valid(_ value:MLXArray,rows:Int) -> Bool {
      value.dtype == .float32 && value.shape == [rows,128] && MLX.isFinite(value).all().item(Bool.self)
    }
    guard guides.count==guideRows.count,initialSlots == nil || slotTokens>0,
      initialSlots.map({ valid($0,rows:slotTokens) }) ?? true else {
      throw LTXError.invalid("Single-stage guide count or initialized slots differ from the admitted layout.")
    }
    for (index,guide) in guides.enumerated() {
      try Task.checkCancellation()
      guard valid(guide,rows:guideRows[index]) else {
        throw LTXError.invalid("Single-stage guide must be finite normalized Float32 at its trained reference grid.")
      }
    }
    let base=try anchorLayout.prepare(generated:generated,anchors:anchors)
    var parts=[base.latent],clean=[base.condition.clean],mask=base.condition.mask
    for (index,guide) in guides.enumerated() {
      try Task.checkCancellation()
      parts.append(guide.reshaped(guide.shape));clean.append(guide.reshaped(guide.shape))
      mask += Array(repeating:1-guideStrengths[index],count:guideRows[index])
    }
    if slotTokens>0 {
      parts.append(initialSlots?.reshaped([slotTokens,128]) ?? .zeros([slotTokens,128]))
      clean.append(.zeros([slotTokens,128]));mask += Array(repeating:1,count:slotTokens)
    }
    let latent=concatenated(parts,axis:0),reference=concatenated(clean,axis:0)
    eval(latent,reference)
    try Task.checkCancellation()
    return (latent,try MLXVideoDenoiseCondition(clean:reference,mask:mask))
  }
}
