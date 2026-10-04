import Foundation

/// Nil on existing projects preserves the historical reference noise and schedule.
public struct H3ReferenceSettings: Codable, Equatable {
  public var visualConditionStrength:Double?
  public var audioConditionStrength:Double?
  public init(visualConditionStrength:Double?=nil,audioConditionStrength:Double?=nil) {
    self.visualConditionStrength=visualConditionStrength;self.audioConditionStrength=audioConditionStrength
  }
  public func validate(task:String) throws {
    for value in [visualConditionStrength,audioConditionStrength].compactMap({$0}) {
      guard value.isFinite,(0...1).contains(value) else { throw StudioError.invalid("H3 reference strengths must be finite numbers from 0 to 1.") }
    }
    guard ["ref2va","a2v","i2v","fflf","extension"].contains(task) else {
      throw StudioError.invalid("Reference noise strengths require an H3 image, reference, audio-driven or extension task.")
    }
  }
}
public enum H3ReferenceFrame:Codable,Equatable {
  case index(Int),last
  public init(from decoder:Decoder) throws {
    let c=try decoder.singleValueContainer()
    if let text=try? c.decode(String.self),text == "last" { self = .last;return }
    let value=try c.decode(Int.self)
    guard value>=0 else { throw DecodingError.dataCorruptedError(in:c,debugDescription:"H3 reference frame must be nonnegative or last.") }
    self = .index(value)
  }
  public func encode(to encoder:Encoder) throws {
    var c=encoder.singleValueContainer();switch self { case .index(let i):try c.encode(i);case .last:try c.encode("last") }
  }
  public func wire(visibleFrames:Int) throws -> Any {
    guard visibleFrames>0 else { throw StudioError.invalid("H3 reference placement needs a visible frame interval.") }
    switch self {
    case .last:return "last"
    case .index(let i):guard (0..<visibleFrames).contains(i) else { throw StudioError.invalid("H3 reference frame lies outside the visible clip.") };return i
    }
  }
}
public enum H3VideoReferenceSizePolicy:String,Codable,CaseIterable { case matchOutput="match_output",nativeH3="native_h3" }
public enum H3VideoReferenceTemporalDensity:String,Codable,CaseIterable { case full,half,quarter,automatic }
public struct H3ReferencePlacement:Codable,Equatable {
  public var frame:H3ReferenceFrame?
  /// A movie sidecar replaces its embedded sound; it is one AV reference.
  public var soundtrackPath:String?
  public var imagePixelBudgetPercent:Int?
  public var videoSizePolicy:H3VideoReferenceSizePolicy?
  public var videoTemporalDensity:H3VideoReferenceTemporalDensity?
  public init(frame:H3ReferenceFrame?=nil,soundtrackPath:String?=nil,imagePixelBudgetPercent:Int?=nil,
    videoSizePolicy:H3VideoReferenceSizePolicy?=nil,videoTemporalDensity:H3VideoReferenceTemporalDensity?=nil) {
    self.frame=frame;self.soundtrackPath=soundtrackPath;self.imagePixelBudgetPercent=imagePixelBudgetPercent
    self.videoSizePolicy=videoSizePolicy;self.videoTemporalDensity=videoTemporalDensity
  }
}
public enum H3LoRAProfile:String,Codable,CaseIterable { case auto,standard,turbo }
public enum H3LoRAQKVLayout:String,Codable,CaseIterable {
  case auto,nativeInterleaved="native_interleaved",contiguousQKV="contiguous_qkv"
}
public struct H3LoRASettings:Codable,Equatable {
  public var profile:H3LoRAProfile
  public var qkvLayout:H3LoRAQKVLayout
  public var startAfterEvaluations:Int
  public init(profile:H3LoRAProfile = .auto,qkvLayout:H3LoRAQKVLayout = .auto,startAfterEvaluations:Int=0) {
    self.profile=profile;self.qkvLayout=qkvLayout;self.startAfterEvaluations=startAfterEvaluations
  }
  public func validate(strength:Double,evaluations:Int,samplingMethod:String) throws {
    guard strength.isFinite,(-10...10).contains(strength),(0...99).contains(startAfterEvaluations),
      startAfterEvaluations<evaluations else { throw StudioError.invalid("H3 LoRA needs signed strength −10 to 10 and an activation inside the actual evaluation schedule.") }
    guard profile != .turbo || (evaluations==4 && samplingMethod=="euler" && startAfterEvaluations==0) else {
      throw StudioError.invalid("H3 Turbo requires Euler, four evaluations and immediate activation.")
    }
  }
  public func wire(path:String,strength:Double) -> [String:Any] {
    ["path":path,"strength":strength,"profile":profile.rawValue,"qkv_layout":qkvLayout.rawValue,
      "start_after_evaluations":startAfterEvaluations]
  }
}
public enum H3JointRefinementMode:String,Codable { case initialized,spatial }
public enum H3JointResizeMethod:String,Codable,CaseIterable {
  case nearestExact="nearest exact",bilinear,bicubic,lanczos3="lanczos-3"
}
public struct H3JointSettings:Codable,Equatable {
  public var saveFullLatents:Bool
  public var refinement:H3JointRefinementSettings?
  public init(saveFullLatents:Bool=false,refinement:H3JointRefinementSettings?=nil) { self.saveFullLatents=saveFullLatents;self.refinement=refinement }
}
public struct H3JointRefinementSettings:Codable,Equatable {
  public var mode:H3JointRefinementMode
  public var source:H3JointLatentArtifact
  public var strength:Double
  public var preserveAudio:Bool
  public var startVideoSigma:Double?
  public var evaluations:Int?
  public var resizeMethod:H3JointResizeMethod?
  /// Explicit v2 spatial admission. Nil preserves the historical v1 canvas.
  public var expandedSpatialTarget:Bool?
  public var learnedUpscalerPath:String?
  public var learnedUpscalerHeaderSHA256:String?
  public init(mode:H3JointRefinementMode,source:H3JointLatentArtifact,strength:Double=0.5,preserveAudio:Bool=true) {
    self.mode=mode;self.source=source;self.strength=strength;self.preserveAudio=preserveAudio
    self.resizeMethod=mode == .spatial ? .bilinear : nil
  }
}
