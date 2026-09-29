import CryptoKit
import Foundation

/// Small, engine-neutral request identity. A worker validates this before it
/// opens a checkpoint; the engine then validates the complete recipe itself.
public struct NativeVideoJobEnvelope: Codable, Sendable {
  public enum Engine: String, Codable, Sendable { case h3, ltx25 }

  public let version: Int
  public let jobID: UUID
  public let engine: Engine
  public let recipePath: String
  public let recipeSHA256: String
  public let outputDirectory: String
  public let ffmpegPath: String?

  public init(jobID: UUID, engine: Engine, recipePath: String, recipeData: Data,
    outputDirectory: String, ffmpegPath: String?) {
    version = 1
    self.jobID = jobID
    self.engine = engine
    self.recipePath = recipePath
    recipeSHA256 = Self.digest(recipeData)
    self.outputDirectory = outputDirectory
    self.ffmpegPath = ffmpegPath
  }

  public static func decode(_ data: Data, expectedEngine: Engine,
    expectedOutputDirectory: String) throws -> Self {
    guard data.count <= 65536,
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys).isSubset(of: ["version", "jobID", "engine", "recipePath",
        "recipeSHA256", "outputDirectory", "ffmpegPath"]) else {
      throw ContractError.invalid("Invalid native video job envelope.")
    }
    let value = try JSONDecoder().decode(Self.self, from: data)
    guard value.version == 1, value.engine == expectedEngine,
      value.jobID != UUID(uuidString: "00000000-0000-0000-0000-000000000000"),
      validAbsolutePath(value.recipePath), validAbsolutePath(value.outputDirectory),
      standardized(value.outputDirectory) == standardized(expectedOutputDirectory),
      value.recipeSHA256.count == 64,
      value.recipeSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      value.ffmpegPath == nil || validAbsolutePath(value.ffmpegPath!) else {
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
