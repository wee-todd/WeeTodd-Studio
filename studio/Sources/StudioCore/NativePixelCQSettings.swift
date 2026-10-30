import Foundation

/// A combined native 2× pass. Timing and audio always belong to its source movie.
public struct NativePixelCQSettings: Codable, Equatable, Sendable {
  public var pixelStrength: Double = 1
  public var cqStrength: Double = 0.5
  public var refinementStrength: Double = 0.75
  public var profileID = "auto"
  public var pixelAdapterPath: String?
  public var cqAdapterPath: String?
  public init() {}
  private enum CodingKeys: String, CodingKey, CaseIterable {
    case pixelStrength, cqStrength, refinementStrength, profileID, pixelAdapterPath, cqAdapterPath
  }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    pixelStrength = try c.decodeIfPresent(Double.self, forKey: .pixelStrength) ?? 1
    cqStrength = try c.decodeIfPresent(Double.self, forKey: .cqStrength) ?? 0.5
    refinementStrength = try c.decodeIfPresent(Double.self, forKey: .refinementStrength) ?? 0.75
    profileID = try c.decodeIfPresent(String.self, forKey: .profileID) ?? "auto"
    pixelAdapterPath = try c.decodeIfPresent(String.self, forKey: .pixelAdapterPath)
    cqAdapterPath = try c.decodeIfPresent(String.self, forKey: .cqAdapterPath)
    try validate()
  }
  public func validate() throws {
    guard pixelStrength.isFinite, (0.05...2).contains(pixelStrength),
      cqStrength.isFinite, (0...2).contains(cqStrength),
      refinementStrength.isFinite, (0.05...0.909375).contains(refinementStrength),
      profileID == "auto" || Self.validPath(profileID),
      [pixelAdapterPath, cqAdapterPath].compactMap({ $0 }).allSatisfy(Self.validPath) else {
      throw StudioError.invalid("Native Pixel Spatial + CQ needs Pixel strength 0.05–2, CQ strength 0–2 and refinement 0.05–0.909375. Choose local adapters and a native LTX profile.")
    }
  }
  static func validPath(_ value: String) -> Bool {
    value.hasPrefix("/") && value.utf8.count <= 4096 && !value.utf8.contains(0)
  }
  public var movieSettings: LTX25MovieUpscaleSettings {
    var s = LTX25MovieUpscaleSettings()
    s.experimentalEnabled = true; s.mode = .pixelSpatial; s.sizePolicy = .strict
    s.pixelStrength = pixelStrength; s.refinementStrength = refinementStrength
    s.pixelSpatialAdapterPath = pixelAdapterPath; s.audioPolicy = .source
    return s
  }
}
