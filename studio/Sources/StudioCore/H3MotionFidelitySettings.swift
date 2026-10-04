import Foundation

/// Independent movie repair, distinct from a saved-tail continuityMode motion.
public struct H3MotionFidelitySettings:Codable,Equatable {
  public enum Mode:String,Codable,CaseIterable { case adaptive,uniform }
  public var sourceVideo:String
  public var sourceIn:Double
  public var mode:Mode
  public var strength:Double
  public var maxHold:Int
  public var sensitivity:Double
  public var maxFrames:Int
  public var evaluations:Int?
  public var experimentalEnabled:Bool
  public init(sourceVideo:String,sourceIn:Double=0,mode:Mode = .adaptive,strength:Double=0.5,maxHold:Int=2,sensitivity:Double=0.5,maxFrames:Int=345,evaluations:Int?=nil,experimentalEnabled:Bool=false) {
    self.sourceVideo=sourceVideo;self.sourceIn=sourceIn;self.mode=mode;self.strength=strength;self.maxHold=maxHold
    self.sensitivity=sensitivity;self.maxFrames=maxFrames;self.evaluations=evaluations;self.experimentalEnabled=experimentalEnabled
  }
  public func validate(duration:Double,seed:Int,width:Int,height:Int) throws {
    _=try H3JointLatentArtifact.localPath(sourceVideo)
    guard experimentalEnabled,sourceIn.isFinite,(0...86400).contains(sourceIn),duration.isFinite,
      (2.5...Double(345)/24).contains(duration),abs(sourceIn*24-(sourceIn*24).rounded(.toNearestOrEven))<=0.001,
      abs(duration*24-(duration*24).rounded(.toNearestOrEven))<=0.001,strength.isFinite,strength>0,strength<=1,
      (2...4).contains(maxHold),sensitivity.isFinite,(0...1).contains(sensitivity),(73...345).contains(maxFrames),
      (0...Int(UInt32.max)).contains(seed),evaluations==nil || (1...64).contains(evaluations!),
      (32...2048).contains(width),(32...2048).contains(height),width%32==0,height%32==0,width*height<=1376*768 else {
      throw StudioError.invalid("H3 Motion Fidelity requires explicit experimental opt-in, native 24 fps trim, admitted canvas and bounded repair controls.")
    }
    let frames=Int((duration*24).rounded(.toNearestOrEven))
    let expanded=mode == .uniform ? frames*maxHold : maxFrames
    let padded=mode == .uniform ? expanded+((5-expanded)%17+17)%17 : maxFrames
    guard padded<=maxFrames,padded*width*height<=160_000_000,padded*width*height*3<=1024*1024*1024 else {
      throw StudioError.invalid("H3 Motion Fidelity exceeds its expanded temporal or RGB memory budget.")
    }
  }
}
