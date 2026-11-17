import CryptoKit
import Foundation
import ImageLayout

public struct MOSSLayoutAnalysisRequest: Codable, Sendable {
  public var schema = "weetodd-moss-layout-request-v1"
  public var requestID, sourcePath, sourceSHA256, modelDirectory: String
  public var targetPixels = 2_000_000, maxTokens = 8192
  public var seed: UInt64 = 42
  public var outputDirectory: String?
  public var mossTokenizerDirectory: String?
  public init(requestID: String, sourcePath: String, sourceSHA256: String, modelDirectory: String) {
    self.requestID = requestID; self.sourcePath = sourcePath; self.sourceSHA256 = sourceSHA256; self.modelDirectory = modelDirectory
  }
  public func validate() throws {
    guard schema == "weetodd-moss-layout-request-v1", UUID(uuidString: requestID) != nil,
      [sourcePath, modelDirectory].allSatisfy({ $0.hasPrefix("/") && !$0.utf8.contains(0) }),
      sourceSHA256.count == 64, sourceSHA256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }),
      (262_144...4_194_304).contains(targetPixels), (256...8192).contains(maxTokens) else {
      throw MOSSError("Invalid local image analysis request, identity or token budget.")
    }
    if let outputDirectory, !outputDirectory.hasPrefix("/") || outputDirectory.utf8.contains(0) {
      throw MOSSError("Image analysis requires a local output directory.")
    }
    if let mossTokenizerDirectory, !mossTokenizerDirectory.hasPrefix("/") || mossTokenizerDirectory.utf8.contains(0) {
      throw MOSSError("The MOSS prompt tokenizer must be a local folder.")
    }
  }
}

final class MOSSLayoutModelIdentity {
  let directory: URL, index: MOSSCheckpointIndex
  let identity: String
  let family: ImageLayoutVisionModel
  private let configurationIdentity: String
  private static let assets = ["config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "generation_config.json", "model.safetensors.index.json"]
  init(directory: URL) throws {
    self.directory = directory.standardizedFileURL.resolvingSymlinksInPath()
    let configURL = self.directory.appendingPathComponent("config.json")
    let bytes = try Data(contentsOf: configURL)
    guard bytes.count <= 1024 * 1024, let root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw MOSSError("Invalid analyzer model configuration.")
    }
    family = try ImageLayoutVisionModel.identify(root)
    index = try MOSSCheckpointIndex(directory: self.directory)
    guard index.descriptors.count < 4000, index.files.values.allSatisfy({ $0.metadata["format"] == "mlx" }) else {
      throw MOSSError("The analyzer requires an admitted MLX-format vision checkpoint.")
    }
    switch family {
    case .qwen35_4b:
      try index.require("vision_tower.patch_embed.proj.weight", shape: [1024, 2, 16, 16, 3])
      try index.require("language_model.model.norm.weight", shape: [2560])
    case .gemma4_12b:
      try index.require("vision_embedder.patch_dense.weight", shape: [3840, 1728], dtypes: ["U32"])
      try index.require("vision_embedder.patch_dense.scales", shape: [3840, 108])
      try index.require("vision_embedder.patch_ln1.weight", shape: [6912])
      try index.require("language_model.model.norm.weight", shape: [3840])
    }
    configurationIdentity = try Self.assetsIdentity(self.directory, family: family)
    identity = MOSSImagePreflight.digest(Data((try index.identity() + configurationIdentity).utf8))
  }
  func checkUnchanged() throws {
    guard try Self.assetsIdentity(directory, family: family) == configurationIdentity,
      MOSSImagePreflight.digest(Data((try index.identity() + configurationIdentity).utf8)) == identity else {
      throw MOSSError("The vision model changed during analysis.")
    }
  }
  private static func assetsIdentity(_ directory: URL, family: ImageLayoutVisionModel) throws -> String {
    try assets.map { name in
      let file = directory.appendingPathComponent(name)
      if !FileManager.default.fileExists(atPath: file.path), !family.requiredAssets.contains(name) { return name + ":absent" }
      let bytes = try Data(contentsOf: file)
      guard bytes.count <= 64 * 1024 * 1024 else { throw MOSSError("Analyzer configuration exceeds its size limit.") }
      return name + ":" + MOSSImagePreflight.digest(bytes)
    }.joined(separator: "\n")
  }
}
