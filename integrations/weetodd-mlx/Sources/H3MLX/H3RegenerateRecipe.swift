import CryptoKit
import Foundation
import InferenceContracts

public enum H3SourceEncoderProfile:String,Codable,Sendable {
  case nativeNormalizedLatents,nativePosteriorMeanF16,publisherPosteriorFP32
}

/// Immutable settings admission only: never loads a checkpoint or changes FPS.
public enum H3RegenerateRecipe {
  public struct Component:Codable,Sendable { public let path:String;public let identity:NativeComponentIdentity }
  public struct Source:Codable,Sendable {
    public let contentSHA256:String,audioSHA256:String,normalization:String
    public let width:Int,height:Int,visibleFrames:Int,paddedFrames:Int
    public let fps:Double
    public let encoderProfile:H3SourceEncoderProfile
    public let posteriorNoiseSHA256:String?
  }
  public struct Target:Codable,Sendable { public let width:Int,height:Int }
  public struct Decoder:Codable,Sendable {
    public enum Kind:String,Codable,Sendable { case releasedH3,fastH3 }
    public let kind:Kind
    public let identity:NativeComponentIdentity
  }
  public struct Budget:Codable,Sendable {
    public let maximumStageBytes:UInt64,availableTemporaryBytes:UInt64
    public let textRows:Int
  }
  private struct Document:Codable {
    let version:Int,task:String,profile:String
    let schedule:H3RegenerateSchedule
    let components:[Component]
    let source:Source,target:Target
    let promptSHA256:String,embeddingsSHA256:String,noiseSHA256:String,silenceSHA256:String
    let decoder:Decoder,budget:Budget
  }
  public struct Prepared:Sendable {
    public let profile:H3RegenerateProfile,schedule:H3RegenerateSchedule
    public let components:[Component],source:Source,target:Target,decoder:Decoder
    public let promptSHA256:String,embeddingsSHA256:String,noiseSHA256:String,silenceSHA256:String
    public let admission:H3RegenerateAdmission
    public let fingerprint:String,encodedRecipe:Data
    public func component(_ role:NativeComponentRole) -> Component { components.first { $0.identity.role == role }! }
  }
  static func digest(_ text:String) -> Bool {
    text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
  }
  private static func keys(_ value:Any?,_ allowed:Set<String>) throws -> [String:Any] {
    guard let object=value as? [String:Any],Set(object.keys).isSubset(of:allowed) else {
      throw H3CheckpointError.invalid("Unknown or malformed regeneration recipe fields.")
    }
    return object
  }
  public static func prepare(data:Data) throws -> Prepared {
    guard data.count <= 1024*1024 else { throw H3CheckpointError.invalid("Regenerate recipe exceeds its size allowance.") }
    let root=try keys(JSONSerialization.jsonObject(with:data),["version","task","profile","schedule","components","source","target",
      "promptSHA256","embeddingsSHA256","noiseSHA256","silenceSHA256","decoder","budget"])
    _ = try keys(root["source"],["contentSHA256","audioSHA256","normalization","width","height","visibleFrames","paddedFrames","fps","encoderProfile","posteriorNoiseSHA256"])
    _ = try keys(root["target"],["width","height"])
    _ = try keys(root["decoder"],["kind","identity"])
    _ = try keys(root["budget"],["maximumStageBytes","availableTemporaryBytes","textRows"])
    guard let components=root["components"] as? [Any] else { throw H3CheckpointError.invalid("Missing regenerate components.") }
    for component in components { _ = try keys(component,["path","identity"]) }
    let document=try JSONDecoder().decode(Document.self,from:data)
    guard document.version == 1,document.task == "h3_regenerate",
      let profile=H3RegenerateProfile(rawValue:document.profile),profile == .publisherThreeStep,
      document.source.normalization == "h3_sampler_normalized_v1",
      [document.source.contentSHA256,document.source.audioSHA256,document.promptSHA256,
       document.embeddingsSHA256,document.noiseSHA256,document.silenceSHA256].allSatisfy(digest),
      [document.source.width,document.source.height].allSatisfy({ (32...4096).contains($0) && $0.isMultiple(of:32) }),
      document.components.allSatisfy({ $0.path.hasPrefix("/") && !$0.path.contains("://") && !$0.path.utf8.contains(0) }) else {
      throw H3CheckpointError.invalid("Unsupported regenerate profile, source normalization, component path or content identity.")
    }
    if document.source.encoderProfile == .publisherPosteriorFP32 {
      guard let noise=document.source.posteriorNoiseSHA256,digest(noise) else {
        throw H3CheckpointError.invalid("Publisher source encoding requires an explicit posterior-noise identity.")
      }
    } else if document.source.posteriorNoiseSHA256 != nil {
      throw H3CheckpointError.invalid("This source preparation does not consume posterior noise.")
    }
    let descriptor=NativeCapabilityDescriptor.h3Regenerate
    try descriptor.validate(components:document.components.map(\.identity),sampling:descriptor.sampling)
    try document.schedule.contract.validate()
    try document.decoder.identity.validate()
    guard document.decoder.identity.family == .h3,document.decoder.identity.role == .decoder,
      document.components.first(where: { $0.identity.role == .sharpInput })?.identity.sourceSHA256 == H3SharpCurve.tableSHA256 else {
      throw H3CheckpointError.invalid("Regenerate decoder role or pinned Sharp table identity differs.")
    }
    let admission=try H3RegenerateAdmission(width:document.target.width,height:document.target.height,
      visibleFrames:document.source.visibleFrames,paddedFrames:document.source.paddedFrames,fps:document.source.fps,
      textRows:document.budget.textRows,maximumStageBytes:document.budget.maximumStageBytes,
      availableTemporaryBytes:document.budget.availableTemporaryBytes)
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    let canonical=try encoder.encode(document)
    return Prepared(profile:profile,schedule:document.schedule,components:document.components,source:document.source,
      target:document.target,decoder:document.decoder,promptSHA256:document.promptSHA256,
      embeddingsSHA256:document.embeddingsSHA256,noiseSHA256:document.noiseSHA256,silenceSHA256:document.silenceSHA256,
      admission:admission,fingerprint:SHA256.hash(data:canonical).map { String(format:"%02x",$0) }.joined(),encodedRecipe:canonical)
  }
}
