import Foundation
import XCTest
@testable import StudioCore

final class CharacterPortabilityTests: XCTestCase {
  func testExportImportCollectsExplicitMediaAndRemapsDocumentIdentity() throws {
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: workspace) }
    let media = workspace.appendingPathComponent("source media")
    try FileManager.default.createDirectory(at: media, withIntermediateDirectories: true)
    func file(_ name: String, _ bytes: [UInt8]) throws -> String {
      let url = media.appendingPathComponent(name); try Data(bytes).write(to: url); return url.path
    }
    let shared = try file("shared.png", [1, 2, 3])
    let initial = try file("initial.png", [4])
    let crop = try file("head.png", [5])
    let mask = try file("mask.png", [6])
    let cutout = try file("cutout.png", [7])
    let matte = try file("matte.png", [8])
    let pipeline = try file("panel.png", [9])
    let storeRoot = workspace.appendingPathComponent("library")
    let exportRoot = workspace.appendingPathComponent("Ada.character")
    var document = CharacterSheetDocument(title: "Ada")
    let oldID = document.id
    document.sources = ["character": shared, "style": shared]
    document.draft.canvas = ImageWorkspaceInput(path: shared)
    document.draft.moodboard = [ImageWorkspaceInput(path: shared)]
    document.proposals = [.init(documentID: oldID, revision: 2, sourcePath: shared,
      sourceHash: "source-hash", role: "character", proposals: [],
      context: try CharacterProposalContext.capture(document, role: "character"))]
    var candidate = MediaAsset(name: "Candidate", kind: .image, path: shared, scope: .global)
    candidate.thumbnail = shared
    candidate.generation = ImageGeneration(provider: "drawThings", requestFingerprint: "request-hash",
      modelID: "model-id-is-not-a-file", prompt: "prompt")
    document.candidates = [candidate]
    document.initialSheetPath = initial
    document.headReference = CharacterHeadReference(originalAssetID: UUID(), headCropAssetID: UUID(),
      maskAssetID: UUID(), rgbaCutoutAssetID: UUID(), whiteMatteAssetID: UUID(), sourcePath: initial,
      headCropPath: crop, maskPath: mask, rgbaCutoutPath: cutout, whiteMattePath: matte,
      sourceSHA256: "original-hash", preprocessingVersion: "head-v1")
    document.pipeline.record(key: "front", inputDigest: "pipeline-input", state: .completed,
      outputPath: pipeline, outputHash: "pipeline-hash", draft: document.draft)
    let detection = CharacterPanelDetection(sourceSHA256: "sheet-hash", sourceOrientation: 1,
      detectorVersion: "vision-v1", candidates: [], status: .detected, diagnostics: [])
    document.candidates[0].generation?.characterAssembly = CharacterAssemblyProvenance(
      detection: detection, head: document.headReference, stages: document.pipeline.stages)

    let store = CharacterSheetDocumentStore(root: storeRoot)
    try store.export(document, to: exportRoot)
    let exported = try store.load(from: exportRoot.appendingPathComponent("character.json"))
    let manifest = try JSONDecoder().decode(CharacterDocumentPortableManifest.self,
      from: Data(contentsOf: exportRoot.appendingPathComponent("portability.json")))
    XCTAssertEqual(manifest.originalRoot, storeRoot.path)
    XCTAssertEqual(exported.sources["character"], exported.sources["style"])
    XCTAssertTrue(exported.sources["character"]?.hasPrefix("assets/") == true)
    XCTAssertFalse(exportRoot.appendingPathComponent("model-id-is-not-a-file").exists)

    let imported = try store.importDocument(from: exportRoot)
    XCTAssertNotEqual(imported.id, oldID)
    XCTAssertEqual(imported.draft.destination, ImageAssetDestination(scope: .global, projectID: imported.id))
    XCTAssertEqual(imported.proposals[0].documentID, imported.id)
    XCTAssertEqual(imported.proposals[0].sourceHash, "source-hash")
    XCTAssertEqual(imported.proposals[0].context, try CharacterProposalContext.capture(imported, role: "character"))
    XCTAssertEqual(imported.headReference?.sourceSHA256, "original-hash")
    XCTAssertEqual(imported.pipeline.stages[0].inputDigest, "pipeline-input")
    XCTAssertEqual(imported.pipeline.stages[0].outputHash, "pipeline-hash")
    XCTAssertEqual(imported.candidates[0].generation?.requestFingerprint, "request-hash")
    XCTAssertEqual(imported.candidates[0].generation?.characterAssembly?.detection, detection)
    XCTAssertEqual(imported.candidates[0].generation?.characterAssembly?.headSourceSHA256, "original-hash")
    for path in [imported.sources["character"], imported.initialSheetPath,
                 imported.headReference?.headCropPath, imported.pipeline.stages[0].outputPath].compactMap({ $0 }) {
      XCTAssertTrue(path.hasPrefix(store.directory(id: imported.id).path + "/"))
      XCTAssertTrue(FileManager.default.fileExists(atPath: path))
    }
  }

  func testExportRejectsMissingExplicitMedia() throws {
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: workspace) }
    var document = CharacterSheetDocument(title: "Missing")
    document.sources["character"] = workspace.appendingPathComponent("missing.png").path
    XCTAssertThrowsError(try CharacterSheetDocumentStore(root: workspace.appendingPathComponent("store"))
      .export(document, to: workspace.appendingPathComponent("export")))
  }

  func testExportImportPreservesCapturedSubjectContextIdentity() throws {
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: workspace) }
    let store = CharacterSheetDocumentStore(root: workspace.appendingPathComponent("library"))
    let document = CharacterSheetDocument(title: "Edited Ada")
    try store.save(document)
    let signature = store.directory(id: document.id).appendingPathComponent("subject-context.sha256")
    let captured = String(repeating: "a", count: 64)
    try Data((captured + "\n").utf8).write(to: signature)
    let package = workspace.appendingPathComponent("Ada.character")

    try store.export(document, to: package)
    let imported = try store.importDocument(from: package)

    XCTAssertEqual(try String(contentsOf: store.directory(id: imported.id)
      .appendingPathComponent("subject-context.sha256"), encoding: .utf8), captured + "\n")
  }

  func testImportRejectsTraversalWithoutReadingOutsideExport() throws {
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: workspace) }
    let package = workspace.appendingPathComponent("Bad.character")
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    let outside = workspace.appendingPathComponent("outside.png"); try Data([1]).write(to: outside)
    var document = CharacterSheetDocument(title: "Bad")
    document.sources["character"] = "assets/../outside.png"
    try JSONEncoder().encode(document).write(to: package.appendingPathComponent("character.json"))
    XCTAssertThrowsError(try CharacterSheetDocumentStore(root: workspace.appendingPathComponent("store"))
      .importDocument(from: package))
  }

  func testImportRejectsMissingCollectedMedia() throws {
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: workspace) }
    let package = workspace.appendingPathComponent("Missing.character")
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    var document = CharacterSheetDocument(title: "Missing")
    document.initialSheetPath = "assets/gone.png"
    try JSONEncoder().encode(document).write(to: package.appendingPathComponent("character.json"))
    XCTAssertThrowsError(try CharacterSheetDocumentStore(root: workspace.appendingPathComponent("store"))
      .importDocument(from: package))
  }
}

final class PlanningSubjectAppearanceTests: XCTestCase {
  func testLegacyRoundTripPreservesRevisionWhenAppearanceAbsent() throws {
    let legacy = PlanningSubject(name: "Ada", kind: .character)
    let before = legacy.revision
    let restored = try JSONDecoder().decode(PlanningSubject.self, from: JSONEncoder().encode(legacy))
    XCTAssertNil(restored.characterAppearance)
    XCTAssertEqual(restored.revision, before)
  }

  func testAppearanceChangesRevisionAndProjectionPreservesOriginalDescription() {
    var appearance = CharacterAppearance()
    appearance.setChoice("identity.species", id: "human", displayValue: "Human")
    appearance.setText("hair.style", "long braids")
    var subject = PlanningSubject(name: "Ada", kind: .character)
    subject.details = "Original prose about Ada."
    let legacyRevision = subject.revision
    subject.applyAcceptedAppearance(appearance, sourceDescription: subject.details)
    XCTAssertEqual(subject.originalAppearanceDescription, "Original prose about Ada.")
    XCTAssertTrue(subject.details.contains("identity species: Human"))
    XCTAssertTrue(subject.details.contains("hair style: long braids"))
    XCTAssertNotEqual(subject.revision, legacyRevision)
    XCTAssertEqual(subject.revision, PlanningSubject(name: "Ada", kind: .character).with {
      $0.details = subject.details
      $0.characterAppearance = appearance
      $0.originalAppearanceDescription = "Different provenance that does not alter identity."
    }.revision)
  }
}

private extension URL {
  var exists: Bool { FileManager.default.fileExists(atPath: path) }
}

private extension PlanningSubject {
  func with(_ edit: (inout PlanningSubject) -> Void) -> PlanningSubject {
    var copy = self; edit(&copy); return copy
  }
}
