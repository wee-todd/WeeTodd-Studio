import Foundation
import MLX
import LTX25Engine

/// Explicit IC-LoRA families preserve their trained guide count and spatial grid.
/// Source audio is publication-only; it is never used as an A2V driver.
public struct MLXICControl: Codable, Sendable {
  private struct AnyKey:CodingKey {
    let stringValue:String
    var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { return nil }
  }
  private static func exact(_ decoder:Decoder,_ fields:Set<String>) throws {
    let all=try decoder.container(keyedBy:AnyKey.self)
    guard Set(all.allKeys.map(\.stringValue)) == fields else {
      throw LTXError.invalid("IC control has missing or unsupported fields.")
    }
  }
  public struct Adapter: Codable, Sendable {
    public let path: String, family: String
    public let strength: Float
    enum CodingKeys:String,CodingKey { case path,family,strength }
    public init(from decoder:Decoder) throws {
      try MLXICControl.exact(decoder,["path","family","strength"])
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);family=try c.decode(String.self,forKey:.family)
      strength=try c.decode(Float.self,forKey:.strength)
    }
  }
  public struct Guide: Codable, Sendable {
    public let path: String, sourceSHA256: String, role: String
    public let strength: Float
    enum CodingKeys: String, CodingKey { case path, role, strength, sourceSHA256="source_sha256" }
    public init(from decoder:Decoder) throws {
      try MLXICControl.exact(decoder,["path","source_sha256","role","strength"])
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
      role=try c.decode(String.self,forKey:.role);strength=try c.decode(Float.self,forKey:.strength)
    }
  }
  public struct PublicationAudio: Codable, Sendable {
    public let path: String, sourceSHA256: String
    public let sourceStartSeconds: Double, sourceDurationSeconds: Double
    enum CodingKeys: String, CodingKey {
      case path, sourceSHA256="source_sha256", sourceStartSeconds="source_start_seconds",
        sourceDurationSeconds="source_duration_seconds"
    }
    public init(from decoder:Decoder) throws {
      try MLXICControl.exact(decoder,["path","source_sha256","source_start_seconds","source_duration_seconds"])
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
      sourceStartSeconds=try c.decode(Double.self,forKey:.sourceStartSeconds)
      sourceDurationSeconds=try c.decode(Double.self,forKey:.sourceDurationSeconds)
    }
  }
  public let family: String
  public let adapters: [Adapter]
  public let guides: [Guide]
  public let publicationAudio: PublicationAudio?
  public var referenceDownscale: Int { family == "motion_track" ? 2 : 1 }
  enum CodingKeys: String, CodingKey, CaseIterable { case family, adapters, guides, publicationAudio="publication_audio" }

  public init(from decoder: Decoder) throws {
    try Self.exact(decoder,Set(CodingKeys.allCases.map(\.rawValue)))
    let c=try decoder.container(keyedBy:CodingKeys.self)
    family=try c.decode(String.self,forKey:.family)
    adapters=try c.decode([Adapter].self,forKey:.adapters)
    guides=try c.decode([Guide].self,forKey:.guides)
    publicationAudio=try c.decodeIfPresent(PublicationAudio.self,forKey:.publicationAudio)
    let expectedAdapters: [String], expectedGuides: [String]
    switch family {
    case "motion_track": expectedAdapters=["motion_track"]; expectedGuides=["control"]
    case "crossview_warp": expectedAdapters=["crossview_warp"]; expectedGuides=["warp","source"]
    case "crossview_ingredients":
      expectedAdapters=["crossview_warp","ingredients_reference_sheet"]
      expectedGuides=["warp","source","ingredients"]
    default: throw LTXError.invalid("Unsupported IC control family.")
    }
    func absolute(_ path: String) -> Bool {
      path.hasPrefix("/") && path.utf8.count<=4096 && !path.utf8.contains(0)
    }
    func digest(_ value: String) -> Bool {
      value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    guard adapters.map(\.family) == expectedAdapters, guides.map(\.role) == expectedGuides,
      Set(adapters.map(\.path)).count == adapters.count, Set(guides.map(\.path)).count == guides.count,
      adapters.allSatisfy({ absolute($0.path) && $0.strength.isFinite && $0.strength>0 && $0.strength<=3 }),
      guides.allSatisfy({ absolute($0.path) && digest($0.sourceSHA256) && $0.strength.isFinite && (0...1).contains($0.strength) }),
      (publicationAudio != nil) == family.hasPrefix("crossview") else {
      throw LTXError.invalid("IC control adapter families, ordered guides, strengths or source audio do not match the selected task.")
    }
    if let audio=publicationAudio {
      guard absolute(audio.path),digest(audio.sourceSHA256),audio.sourceStartSeconds.isFinite,
        (0...86400).contains(audio.sourceStartSeconds),audio.sourceDurationSeconds.isFinite,
        (0.001...30).contains(audio.sourceDurationSeconds) else {
        throw LTXError.invalid("CrossView source audio requires a frozen, finite publication interval.")
      }
    }
  }

  public func guideGeometry(target: AVGeometry) throws -> AVGeometry {
    guard target.width % (32*referenceDownscale) == 0,
      target.height % (32*referenceDownscale) == 0,
      family != "crossview_ingredients" || target.frames>=121 else {
      throw LTXError.invalid("IC control guide dimensions or Ingredients duration do not match the trained layout.")
    }
    return try AVGeometry(width:target.width/referenceDownscale,height:target.height/referenceDownscale,
      frames:target.frames,fps:target.fps)
  }
}

/// References share causal time positions while their spatial centers are
/// mapped onto the stage-one canvas. Multiple groups retain their supplied order.
public struct MLXICControlLayout: Sendable {
  public let geometry: AVGeometry, referenceGeometry: AVGeometry
  public let strengths: [Float], videoTokens: Int, positions: [Float]
  public var referenceTokens: Int { referenceGeometry.videoTokens }
  public init(geometry: AVGeometry, control: MLXICControl) throws {
    self.geometry=geometry;referenceGeometry=try control.guideGeometry(target:geometry)
    strengths=control.guides.map(\.strength)
    videoTokens=geometry.videoTokens+referenceGeometry.videoTokens*strengths.count
    guard videoTokens<=131072 else { throw LTXError.invalid("IC control exceeds the admitted video token budget.") }
    var guide=referenceGeometry.videoPositions
    for index in stride(from:0,to:guide.count,by:3) {
      guide[index+1] *= Float(control.referenceDownscale)
      guide[index+2] *= Float(control.referenceDownscale)
    }
    positions=geometry.videoPositions+strengths.flatMap { _ in guide }
  }
  public func prepare(generated: MLXArray, references: [MLXArray]) throws
    -> (latent: MLXArray, condition: MLXVideoDenoiseCondition) {
    guard generated.dtype == .float32,generated.shape == [geometry.videoTokens,128],
      references.count == strengths.count,
      references.allSatisfy({ $0.dtype == .float32 && $0.shape == [referenceTokens,128] && MLX.isFinite($0).all().item(Bool.self) }),
      MLX.isFinite(generated).all().item(Bool.self) else {
      throw LTXError.invalid("IC control references differ from the admitted ordered guide layout.")
    }
    let latent=concatenated([generated]+references,axis:0)
    let clean=concatenated([MLXArray.zeros([geometry.videoTokens,128])]+references,axis:0)
    let mask=[Float](repeating:1,count:geometry.videoTokens)
      + strengths.flatMap { [Float](repeating:1-$0,count:referenceTokens) }
    eval(latent,clean)
    return (latent,try MLXVideoDenoiseCondition(clean:clean,mask:mask))
  }
}
