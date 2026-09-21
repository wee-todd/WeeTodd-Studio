import XCTest
@testable import StudioCore

final class CharacterSheetDocumentTests: XCTestCase {
  func testAtomicStandaloneRoundTripDoesNotCreateMovie() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = CharacterSheetDocumentStore(root: root)
    let document = CharacterSheetDocument(title: "Ada")
    try store.save(document)
    XCTAssertEqual(try store.load(id: document.id), document)
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Autosave.weetodd").path))
  }
  func testResumeRejectsChangedArtifactAndPreservesCompletedSiblings() throws {
    var manifest = CharacterSheetPipelineManifest()
    manifest.record(key: "front", inputDigest: "input-a", state: .completed, outputPath: "/missing.png", outputHash: "hash")
    manifest.record(key: "side", inputDigest: "input-b", state: .failed)
    XCTAssertFalse(manifest.canReuse(key: "front", inputDigest: "input-a"))
    XCTAssertEqual(manifest.stages.first { $0.key == "front" }?.state, .completed)
    XCTAssertEqual(manifest.stages.first { $0.key == "side" }?.state, .failed)
  }
  func testUnknownVersionDoesNotOverwriteSavedFile() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = CharacterSheetDocumentStore(root: root)
    var document = CharacterSheetDocument(title: "Future")
    document.version = 999
    XCTAssertThrowsError(try store.save(document))
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
  }
  func testProposalBatchPersistsExtractionMetadataAndDiagnostics() throws {
    let metadata = CharacterExtractionMetadata(schemaVersion: 1, extractionVersion: 1,
      promptVersion: 2, modelFingerprint: "qwen-hash")
    let diagnostic = CharacterProposalDiagnostic(code: "proposal.state", proposalID: "p1",
      field: "eyes.color", message: "Only value proposals can be applied.")
    let batch = CharacterProposalBatch(documentID: UUID(), revision: 4, sourcePath: "/image.png",
      sourceHash: "source", role: "character",
      proposals: [.init(id: "p1", field: "eyes.color", value: "Blue")],
      metadata: metadata, diagnostics: [diagnostic])
    let decoded = try JSONDecoder().decode(CharacterProposalBatch.self,
      from: JSONEncoder().encode(batch))
    XCTAssertEqual(decoded.metadata, metadata)
    XCTAssertEqual(decoded.diagnostics, [diagnostic])
  }
  func testLegacyProposalBatchWithoutMetadataStillDecodes() throws {
    let id = UUID(), documentID = UUID()
    let json = """
      {"id":"\(id.uuidString)","documentID":"\(documentID.uuidString)","revision":0,
       "sourcePath":"","sourceHash":"","role":"character","proposals":[],"stale":false}
      """.data(using: .utf8)!
    let batch = try JSONDecoder().decode(CharacterProposalBatch.self, from: json)
    XCTAssertNil(batch.metadata)
    XCTAssertNil(batch.diagnostics)
  }
}
