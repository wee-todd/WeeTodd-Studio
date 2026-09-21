import Foundation

public enum CharacterPipelineStageState: String, Codable { case pending, submitted, uncertain, completed, failed, cancelled }
public struct CharacterPipelineStage: Codable, Equatable, Identifiable {
  public var id: String { key }
  public var key: String
  public var inputDigest: String
  public var state: CharacterPipelineStageState
  public var outputPath: String?
  public var outputHash: String?
  public var recoveryDirectory: String?
  public var inputHashes: [String]?
  public var message: String?
  public var draft: DrawThingsImageDraft?
}
public struct CharacterSheetPipelineManifest: Codable, Equatable {
  public var version = 1
  public var stages: [CharacterPipelineStage] = []
  public init() {}
  public mutating func record(key: String, inputDigest: String, state: CharacterPipelineStageState,
    outputPath: String? = nil, outputHash: String? = nil, recoveryDirectory: String? = nil, inputHashes: [String]? = nil, message: String? = nil, draft: DrawThingsImageDraft? = nil) {
    let value = CharacterPipelineStage(key: key, inputDigest: inputDigest, state: state,
      outputPath: outputPath, outputHash: outputHash, recoveryDirectory: recoveryDirectory, inputHashes: inputHashes, message: message, draft: draft)
    if let index = stages.firstIndex(where: { $0.key == key }) { stages[index] = value }
    else { stages.append(value) }
  }
  public func canReuse(key: String, inputDigest: String) -> Bool {
    guard let stage = stages.first(where: { $0.key == key && $0.inputDigest == inputDigest }),
      stage.state == .completed, let path = stage.outputPath, let hash = stage.outputHash,
      let current = try? CharacterArtifactHash.file(path) else { return false }
    return hash == current
  }
  public var hasUncertainSubmission: Bool { stages.contains { [.submitted, .uncertain].contains($0.state) } }
}

public extension DrawThingsImageDraft {
  /// Hash only executable request semantics, including ordered input content hashes.
  /// UI input UUIDs and per-submission request IDs must not invalidate completed work.
  func characterExecutionDigest() throws -> String {
    let canonical = try request(id: "character-pipeline")
    let data = try JSONSerialization.data(withJSONObject: canonical, options: [.sortedKeys])
    return try CharacterArtifactHash.value(JSONDecoder().decode(JSONValue.self, from: data))
  }
}

/// Portable execution evidence for an assembled sheet; contains hashes rather than local media paths.
public struct CharacterAssemblyProvenance: Codable, Equatable {
  public var version = 1
  public var detection: CharacterPanelDetection
  public var headSourceSHA256: String?
  public var headArtifactSHA256: [String: String]?
  public var headPreprocessingVersion: String?
  public var panelTakes: [CharacterPanelTakeProvenance]
  public init(detection: CharacterPanelDetection, head: CharacterHeadReference?, stages: [CharacterPipelineStage]) {
    self.detection = detection; headSourceSHA256 = head?.sourceSHA256
    headArtifactSHA256 = head?.artifactSHA256; headPreprocessingVersion = head?.preprocessingVersion
    panelTakes = stages.compactMap { stage in
      guard let draft = stage.draft, let context = draft.characterPanel else { return nil }
      return CharacterPanelTakeProvenance(inputDigest: stage.inputDigest, inputHashes: stage.inputHashes ?? [],
        outputSHA256: stage.outputHash ?? "", modelID: draft.modelID, prompt: draft.prompt,
        width: draft.width, height: draft.height, seed: draft.seed, steps: draft.steps,
        guidance: draft.guidance, loras: draft.loras, context: context)
    }
  }
}
public struct CharacterPanelTakeProvenance: Codable, Equatable {
  public var inputDigest: String
  public var inputHashes: [String]
  public var outputSHA256: String
  public var modelID: String
  public var prompt: String
  public var width: Int
  public var height: Int
  public var seed: Int
  public var steps: Int
  public var guidance: Double
  public var loras: [DrawThingsLoRA]
  public var context: CharacterPanelPromptContext
}
