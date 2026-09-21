import CryptoKit
import Foundation

public enum CharacterSourceRole: String, Codable, CaseIterable, Identifiable {
  case character, style, face
  public var id: String { rawValue }
  public var label: String {
    switch self { case .character: return "Character image"; case .style: return "Style image"; case .face: return "Reference face" }
  }
}

public struct CharacterFieldProposal: Codable, Equatable, Identifiable {
  public var id: String
  public var field: String
  public var value: String
  public var state: String
  public var evidence: String
  public var uncertainty: String
  public var selected = false
  public init(id: String = UUID().uuidString, field: String, value: String, state: String = "value", evidence: String = "", uncertainty: String = "") {
    self.id = id; self.field = field; self.value = value; self.state = state
    self.evidence = evidence; self.uncertainty = uncertainty
  }
}

public struct CharacterExtractionMetadata: Codable, Equatable {
  public var schemaVersion: Int
  public var extractionVersion: Int
  public var promptVersion: Int
  public var modelFingerprint: String
  public init(schemaVersion: Int, extractionVersion: Int, promptVersion: Int,
              modelFingerprint: String) {
    self.schemaVersion = schemaVersion; self.extractionVersion = extractionVersion
    self.promptVersion = promptVersion; self.modelFingerprint = modelFingerprint
  }
}

public struct CharacterProposalDiagnostic: Codable, Equatable {
  public var code: String
  public var proposalID: String?
  public var field: String?
  public var message: String
  public init(code: String, proposalID: String? = nil, field: String? = nil,
              message: String) {
    self.code = code; self.proposalID = proposalID; self.field = field; self.message = message
  }
}

public struct CharacterProposalBatch: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var documentID: UUID
  public var revision: Int
  public var sourcePath: String
  public var sourceHash: String
  public var role: String
  public var proposals: [CharacterFieldProposal]
  public var metadata: CharacterExtractionMetadata?
  public var diagnostics: [CharacterProposalDiagnostic]?
  public var stale = false
  public init(documentID: UUID, revision: Int, sourcePath: String, sourceHash: String,
              role: String, proposals: [CharacterFieldProposal],
              metadata: CharacterExtractionMetadata? = nil,
              diagnostics: [CharacterProposalDiagnostic]? = nil) {
    self.documentID = documentID; self.revision = revision; self.sourcePath = sourcePath
    self.sourceHash = sourceHash; self.role = role; self.proposals = proposals
    self.metadata = metadata; self.diagnostics = diagnostics
  }
}

public struct CharacterRefinementSettings: Codable, Equatable {
  public var modelID = ""
  public var detailLoRAID = ""
  public var detailStrength = 1.0
  public var headLoRAID = ""
  public var headStrength = 1.0
  public var replaceFaces = false
  public var twoPass = false
  public var steps = 4
  public var guidance = 1.0
  public var seed = 0
  public init() {}
}

public struct CharacterSheetDocument: Codable, Equatable, Identifiable {
  public var format = "weetodd-character-director-v1"
  public var version = 1
  public var id: UUID
  public var title: String
  public var revision = 0
  public var definition = CharacterSheetDefinition.newDraft()
  public var originalDescription = ""
  public var subjectKey: String?
  public var sources: [String: String] = [:]
  public var styleUsesCharacterImage = false
  public var draft: DrawThingsImageDraft
  public var refinement = CharacterRefinementSettings()
  public var proposals: [CharacterProposalBatch] = []
  public var candidates: [MediaAsset] = []
  public var initialSheetPath: String?
  public var panels: CharacterPanelDetection?
  public var cropsApproved = false
  public var headReference: CharacterHeadReference?
  public var pipeline = CharacterSheetPipelineManifest()
  public init(id: UUID = UUID(), title: String = "New character") {
    self.id = id; self.title = title
    draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: id))
    draft.width = 1920; draft.height = 1088; draft.steps = 8; draft.guidance = 1
  }
  public mutating func changed() { revision += 1 }
}

public struct CharacterSheetDocumentStore {
  public let root: URL
  public init(root: URL) { self.root = root }
  public func url(id: UUID) -> URL { root.appendingPathComponent(id.uuidString).appendingPathComponent("character.json") }
  public func directory(id: UUID) -> URL { url(id: id).deletingLastPathComponent() }
  public func save(_ document: CharacterSheetDocument) throws {
    guard document.version == 1, document.format == "weetodd-character-director-v1" else {
      throw StudioError.invalid("This Character Director document requires a newer Studio version.")
    }
    let destination = url(id: document.id)
    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(document).write(to: destination, options: .atomic)
  }
  public func load(id: UUID) throws -> CharacterSheetDocument { try load(from: url(id: id)) }
  public func load(from url: URL) throws -> CharacterSheetDocument {
    let document = try JSONDecoder().decode(CharacterSheetDocument.self, from: Data(contentsOf: url))
    guard document.version == 1, document.format == "weetodd-character-director-v1" else {
      throw StudioError.invalid("This Character Director document requires a newer Studio version.")
    }
    return document
  }
  public func documents() -> [CharacterSheetDocument] {
    ((try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [])
      .compactMap { try? load(from: $0.appendingPathComponent("character.json")) }
      .sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
  }
}

public enum CharacterArtifactHash {
  public static func file(_ path: String) throws -> String {
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path)); defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  public static func value<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
  }
}
