import Foundation

/// Research/import metadata is intentionally Encodable only and separate from
/// NativeCapabilityDescriptor. These records cannot enable a runtime profile.
public enum NativeAccelerationAvailability:String,Encodable,Sendable { case importCandidate,researchOnly }
public enum NativeAccelerationVelocity:String,Encodable,Sendable {
  case dataWard,sigma
  public var signToSigmaVelocity:Double { self == .dataWard ? -1:1 }
}
public enum NativeAccelerationUpdate:String,Encodable,Sendable {
  case dataWardEuler,sigmaEuler,denoiseThenRenoise,publisherMultistepUnqualified,parallelHeadIntegration
}
public enum NativeAccelerationNoise:String,Encodable,Sendable {
  case initialVideoThenAudio,initialAndEveryStepVideoThenAudio,unqualified
}
public enum NativeAccelerationHead:String,Encodable,Sendable {
  case ordinaryH3,h3Parallel32,ltxUnqualified
  public func validate(videoWeight:[Int],audioWeight:[Int])throws {
    guard self != .ltxUnqualified else { throw ContractError.invalid("LTX candidate heads need their own qualified layout.") }
    let heads=self == .h3Parallel32 ? 32:1
    guard videoWeight == [96*heads,5376],audioWeight == [32*heads,5376] else {
      throw ContractError.invalid("Candidate H3 output heads differ from the ordinary or 32-head PDD layout.")
    }
  }
  /// Pinned PDD channel-major layout inserts heads before (channel, patch).
  /// This indexes metadata only; it does not widen or repeat teacher weights.
  public func outputRow(head:Int,baseRow:Int,video:Bool)throws->Int {
    let heads=self == .h3Parallel32 ? 32:1,width=video ? 96:32
    guard self != .ltxUnqualified,(0..<heads).contains(head),(0..<width).contains(baseRow) else {
      throw ContractError.invalid("Invalid candidate output-head coordinate.")
    }
    return head*width+baseRow
  }
}
public struct NativeAccelerationCandidate:Encodable,Sendable {
  public let id:String,family:NativeModelFamily,availability:NativeAccelerationAvailability
  public let repository:String,modelRevision:String?,sourceRevision:String,asset:String
  public let tasks:[String],evaluations:[Int],videoSigmas:[Double]?,audioSigmas:[Double]?
  public let modelTimeBasis:String,velocity:NativeAccelerationVelocity,update:NativeAccelerationUpdate
  public let noise:NativeAccelerationNoise,heads:NativeAccelerationHead,precisionIdentity:String
  public let requiredQualification:[String]
  public var runnable:Bool { false }
  public var directRegenerationLatentsQualified:Bool { false }
}
public struct NativeVDNStageIdentity:Encodable,Sendable {
  public enum DownstreamRoute:String,Encodable,Sendable { case decodedMovie }
  public let stage:String,evaluations:Int,modelRevision:String
  public let requiredRoles:[NativeComponentRole]
  public var downstreamRoute:DownstreamRoute { .decodedMovie }
  public var directRegenerationLatentsQualified:Bool { false }
}

public enum NativeAccelerationRegistry {
  public static let vdnStages:[NativeVDNStageIdentity] = [
    .init(stage:"stage-dmd-step-250",evaluations:8,modelRevision:"2ce18deb1d1c6b115d6bb3732f3750fdbac95173",requiredRoles:[.transformer,.linearBranch,.standardAdapter,.turboAdapter]),
    .init(stage:"stage-b-step-2000",evaluations:50,modelRevision:"2ce18deb1d1c6b115d6bb3732f3750fdbac95173",requiredRoles:[.transformer,.linearBranch,.standardAdapter])]
  private static let missing=["exact source/base and all adapter targets", "numerical projection and joint latent fixtures", "native memory/time and audiovisual review", "packaged Swift-only setup/runtime"]
  private static func f32(_ values:[Double])->[Double] { values.map{Double(Float($0))} }
  private static let video4=f32([1,36.0/37,12.0/13,0.8,0])
  private static func pdmd(_ nfe:Int,precision:String="bf16",paper:Bool=false)->NativeAccelerationCandidate {
    let suffix=nfe == 2 ? (paper ? "-paper":"-recommended"):""
    return .init(id:"pdmd-h3-\(nfe)nfe-\(precision)"+suffix,family:.h3,availability:.importCandidate,
      repository:"pdmd2026/pdmd_\(nfe)NFE_lora",
      modelRevision:nfe == 4 ? "9913253b4cee1702671c38637bf3941e1de489f5":"3bc1bf125da77483a5d7819dae6b745b0404dadf",
      sourceRevision:"6b6e10635495a2d9b989fee393235d406d3840c4",asset:precision == "f32" ? "lora_model_0_fp32.safetensors":"lora_model_0.safetensors",
      tasks:["t2va"],evaluations:[nfe],videoSigmas:nfe == 4 ? video4:f32([1,12.0/13,0]),
      audioSigmas:nfe == 4 ? f32([1,0.9,0.75,0.5,0]):f32([1,paper ? 0.75:6.0/7,0]),
      modelTimeBasis:"oneMinusSigma",velocity:.dataWard,update:.dataWardEuler,noise:.initialVideoThenAudio,
      heads:.ordinaryH3,precisionIdentity:precision,requiredQualification:missing+["PDMD is projected-distribution training, not FastGen PDD", "312 rank-128 pairs, alpha128, one MLP half swap", "publisher NFE+1 counts terminal point"])
  }
  private static func dmad(fullCritic:Bool)->NativeAccelerationCandidate {
    .init(id:fullCritic ? "dmad-h3-live-1600-full-critic":"dmad-h3-ema-800-paper",family:.h3,availability:.importCandidate,
      repository:"ZhengmingYu/DMAD",modelRevision:"439136a1dd4f435b8137a084f73d558d2cecda2a",
      sourceRevision:"32ac967324eb0ebba654faeffda99f5def1e86cb",
      asset:fullCritic ? "dmad_minimax_h3_4step_full_critic.safetensors":"dmad_minimax_h3_4step_lora_critic.safetensors",
      tasks:["t2va"],evaluations:[4],videoSigmas:video4,audioSigmas:f32([1,6.0/7,2.0/3,0.4,0]),
      modelTimeBasis:"oneMinusSigma",velocity:.dataWard,update:.denoiseThenRenoise,
      noise:.initialAndEveryStepVideoThenAudio,heads:.ordinaryH3,precisionIdentity:"bf16",
      requiredQualification:missing+["mixed attention down/up and FFN A/B suffixes", "initial and every inter-step video/audio noise tensor identity", "Euler is a separate unqualified profile"])
  }
  private static func fastgen(_ id:String,family:NativeModelFamily,pdd:Bool)->NativeAccelerationCandidate {
    .init(id:id,family:family,availability:.researchOnly,repository:"NVlabs/FastGen",modelRevision:nil,
      sourceRevision:"deb93e6879fce9d262780232d5d04bd054bab0dd",asset:"no matching released student verified",
      tasks:["t2va"],evaluations:pdd ? [4,8]:[4],videoSigmas:nil,audioSigmas:nil,
      modelTimeBasis:family == .h3 ? "oneMinusSigma":"family-specific-unqualified",
      velocity:.sigma,update:pdd ? .parallelHeadIntegration:.sigmaEuler,noise:.unqualified,
      heads:family == .h3 ? (pdd ? .h3Parallel32:.ordinaryH3):.ltxUnqualified,precisionIdentity:"training-config-only",
      requiredQualification:missing+["training configuration is not a student checkpoint", "freeze actual modality grid and head integration with exact student"])
  }
  public static let candidates:[NativeAccelerationCandidate] = [
    pdmd(4),pdmd(2),pdmd(2,paper:true),pdmd(2,precision:"f32"),pdmd(2,precision:"f32",paper:true),
    dmad(fullCritic:false),dmad(fullCritic:true),
    .init(id:"matlow-h3-fused-convrot",family:.h3,availability:.importCandidate,
      repository:"MATLOWAI/minimax-h3-fused-turbo-int8-convrot",modelRevision:"8a8dffaa0cd99c6184833ae0a3b4e9b0089c17b3",
      sourceRevision:"8a8dffaa0cd99c6184833ae0a3b4e9b0089c17b3",asset:"minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
      tasks:["publisher-fused-recipe-unqualified"],evaluations:[4],videoSigmas:nil,audioSigmas:nil,
      modelTimeBasis:"oneMinusSigma",velocity:.dataWard,update:.publisherMultistepUnqualified,noise:.unqualified,
      heads:.ordinaryH3,precisionIdentity:"int8-convrot-original-markers",
      requiredQualification:missing+["rootless1025x8 basis, 934 tensor schema and all ConvRot marker payloads", "baked Ref2VA/Turbo/Mystic ancestry", "res_multistep, shifts12/3 and SLA are not native admission", "freeze actual grid before implementation"]),
    fastgen("fastgen-h3-dmd2",family:.h3,pdd:false),fastgen("fastgen-h3-pdd",family:.h3,pdd:true),
    fastgen("fastgen-ltx25-dmd2",family:.ltx25,pdd:false)]
}
