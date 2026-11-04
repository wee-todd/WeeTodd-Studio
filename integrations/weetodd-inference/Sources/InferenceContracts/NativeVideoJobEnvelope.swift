import CryptoKit
import Foundation

/// Small, engine-neutral request identity. A worker validates this before it
/// opens a checkpoint; the engine then validates the complete recipe itself.
public struct NativeVideoJobEnvelope: Codable, Sendable {
  public enum Engine: String, Codable, Sendable { case h3, ltx25; case flashVSR = "flashvsr" }

  public let version: Int
  public let jobID: UUID
  public let engine: Engine
  public let recipePath: String
  public let recipeSHA256: String
  public let outputDirectory: String
  public let ffmpegPath: String?
  /// Display-only. Never participates in the recipe or final decoder selection.
  public let preview: NativeVideoPreviewSettings?

  public init(jobID: UUID, engine: Engine, recipePath: String, recipeData: Data,
    outputDirectory: String, ffmpegPath: String?, preview: NativeVideoPreviewSettings? = nil) {
    version = 1
    self.jobID = jobID
    self.engine = engine
    self.recipePath = recipePath
    recipeSHA256 = Self.digest(recipeData)
    self.outputDirectory = outputDirectory
    self.ffmpegPath = ffmpegPath
    self.preview = preview
  }

  public static func decode(_ data: Data, expectedEngine: Engine,
    expectedOutputDirectory: String) throws -> Self {
    guard data.count <= 65536,
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys).isSubset(of: ["version", "jobID", "engine", "recipePath",
        "recipeSHA256", "outputDirectory", "ffmpegPath", "preview"]) else {
      throw ContractError.invalid("Invalid native video job envelope.")
    }
    let value = try JSONDecoder().decode(Self.self, from: data)
    guard value.version == 1, value.engine == expectedEngine,
      value.jobID != UUID(uuidString: "00000000-0000-0000-0000-000000000000"),
      validAbsolutePath(value.recipePath), validAbsolutePath(value.outputDirectory),
      standardized(value.outputDirectory) == standardized(expectedOutputDirectory),
      value.recipeSHA256.count == 64,
      value.recipeSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      value.ffmpegPath == nil || validAbsolutePath(value.ffmpegPath!),
      value.preview == nil || value.engine == .ltx25 else {
      throw ContractError.invalid("Native video job identity or path is invalid.")
    }
    return value
  }

  public func validateRecipe(_ data: Data) throws {
    guard data.count <= 1024 * 1024, Self.digest(data) == recipeSHA256 else {
      throw ContractError.invalid("The native video recipe changed after job submission.")
    }
  }

  private static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func validAbsolutePath(_ path: String) -> Bool {
    path.hasPrefix("/") && path.utf8.count <= 4096 && !path.utf8.contains(0)
      && !path.unicodeScalars.contains(where: { $0.value < 32 })
  }

  private static func standardized(_ path: String) -> String {
    URL(fileURLWithPath: path).standardizedFileURL.path
  }
}

public struct NativeVideoPreviewSettings: Codable, Sendable {
  public enum Decoder: String, Codable, Sendable {
    case standard = "taeltx2_3", wide = "taeltx2_3_wide"
  }
  public let decoder: Decoder
  public let checkpointPath: String
  private enum CodingKeys: String, CodingKey { case decoder, checkpointPath }
  private struct Key: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  public init(from decoder: any Swift.Decoder) throws {
    let all = try decoder.container(keyedBy: Key.self)
    guard Set(all.allKeys.map(\.stringValue)) == ["decoder", "checkpointPath"] else {
      throw ContractError.invalid("Preview settings require a TAE decoder and checkpoint path only.")
    }
    let fields = try decoder.container(keyedBy: CodingKeys.self)
    self.decoder = try fields.decode(Decoder.self, forKey: .decoder)
    checkpointPath = try fields.decode(String.self, forKey: .checkpointPath)
    guard checkpointPath.hasPrefix("/"), checkpointPath.utf8.count <= 4096,
      !checkpointPath.unicodeScalars.contains(where: { $0.value < 32 }) else {
      throw ContractError.invalid("Preview weights require an absolute local checkpoint path.")
    }
  }
}
