import Foundation
import LTX25Engine

/// Scene v2 images address the delivered global timeline, excluding its extra
/// causal frame. Balanced routing chooses the first covering window; strict
/// routing repeats an image in every overlap. Neither policy snaps timestamps.
public struct MLXSceneImageRouting:Sendable {
  public enum Policy:String,Sendable { case balanced,strict }
  public struct Anchor:Sendable,Equatable {
    public let id:String,path:String
    public let frame:Int,strength:Float,crf:Int
    public init(id:String,path:String,frame:Int,strength:Float,crf:Int=33) throws {
      guard !id.isEmpty,id.utf8.count<=256,!id.utf8.contains(0),
        path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0),frame>=0,
        strength.isFinite,(0...1).contains(strength),(0...51).contains(crf) else {
        throw LTXError.invalid("Invalid scene image identity, path, timestamp or strength.")
      }
      self.id=id;self.path=path;self.frame=frame;self.strength=strength;self.crf=crf
    }
  }
  public struct WindowAnchor:Sendable,Equatable {
    public let anchor:Anchor
    public let frame:Int
    public var layoutAnchor:MLXOrdinaryKeyframeLayout.Anchor { .init(frame:frame,strength:anchor.strength) }
  }
  public let anchors:[Anchor]
  public let windows:[[WindowAnchor]]
  public let retainedConditioningBytes:Int
  public init(plan:LTX25ScenePlan,anchors:[Anchor],policy:Policy,width:Int,height:Int) throws {
    try Task.checkCancellation()
    guard anchors.count<=32,Set(anchors.map(\.id)).count==anchors.count,
      Set(anchors.map(\.frame)).count==anchors.count,
      anchors.allSatisfy({ $0.frame<plan.totalFrames-1 }),
      (64...4096).contains(width),(64...4096).contains(height),width%64==0,height%64==0 else {
      throw LTXError.invalid("Scenes require up to 32 unique images inside the delivered timeline and two-stage geometry.")
    }
    self.anchors=anchors.sorted { $0.frame<$1.frame }
    var routed=[[WindowAnchor]](repeating:[],count:plan.windowFrames.count)
    for anchor in self.anchors {
      try Task.checkCancellation()
      let owners=plan.windowFrames.indices.filter {
        plan.windowStarts[$0]<=anchor.frame && anchor.frame-plan.windowStarts[$0]<plan.windowFrames[$0]
      }
      guard !owners.isEmpty else { throw LTXError.invalid("A scene image has no covering window.") }
      for index in policy == .balanced ? [owners[0]] : owners {
        routed[index].append(WindowAnchor(anchor:anchor,frame:anchor.frame-plan.windowStarts[index]))
      }
    }
    let count=routed.reduce(0) { $0+$1.count }
    // Two retained normalized latent scales plus their three position channels.
    let bytes=count*((width/32)*(height/32)+(width/64)*(height/64))*131*4
    guard bytes<=256*1024*1024 else { throw LTXError.invalid("Scene image conditioning exceeds 256 MiB.") }
    windows=routed;retainedConditioningBytes=bytes
  }
}
