import XCTest
@testable import WeeToddStudio
import StudioCore

@MainActor final class CharacterDirectorSessionTests: XCTestCase {
  func testSeparateStyleChangesDoNotDiscardReviewedCharacterValues() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var document = CharacterSheetDocument()
    document.originalDescription = "A slim character with blue eyes"
    document.definition.appearance.setText("body.build", "old test character build")
    let controller = CharacterSheetSessionController(document: document, store: store,
      storage: .init(root: root.appendingPathComponent("Characters")),
      bridge: Bridge(invocation: { _, _, _, _ in
        ["proposals": [["id": "build", "field": "body.build", "value": "slender torso and narrow shoulders"],
                       ["id": "eyes", "field": "eyes.color", "value": "Blue"]]]
      }))
    try await controller.analyze(role: "text", modelPath: "test")
    let batchID = try XCTUnwrap(controller.document.proposals.last?.id)
    controller.edit { $0.sources["style"] = "separate-style.jpg" }
    controller.selectStyle("clay")
    controller.setRequired("eyes.color", true)
    await controller.applyProposals(batchID: batchID, selectedIDs: ["build"])
    XCTAssertEqual(controller.document.definition.entry(at: "body.build")?.displayString,
      "slender torso and narrow shoulders")
    XCTAssertNil(controller.error)
    XCTAssertEqual(controller.document.proposals.first?.proposals.map(\.id), ["eyes"])
    await controller.applyProposals(batchID: batchID, selectedIDs: ["eyes"])
    XCTAssertEqual(controller.document.definition.entry(at: "eyes.color")?.displayString, "Blue")
    XCTAssertTrue(controller.document.proposals.isEmpty)
    XCTAssertEqual(controller.document.definition.settings.stylePresetID, "clay")
    let saved = try controller.storage.load(id: document.id)
    XCTAssertEqual(saved.definition, controller.document.definition)
    controller.undo()
    XCTAssertNil(controller.document.definition.entry(at: "eyes.color"))
    XCTAssertEqual(controller.document.definition.entry(at: "body.build")?.displayString,
      "slender torso and narrow shoulders")
  }

  func testEmptySelectionDoesNotRemoveProposalsOrAdvanceRevision() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var document = CharacterSheetDocument()
    let batch = CharacterProposalBatch(documentID: document.id, revision: document.revision,
      sourcePath: "", sourceHash: "", role: "text", proposals: [.init(field: "eyes.color", value: "Blue")])
    document.proposals = [batch]
    let controller = CharacterSheetSessionController(document: document, store: store, storage: .init(root: root))
    await controller.applyProposals(batchID: batch.id, selectedIDs: [])
    XCTAssertEqual(controller.document.revision, document.revision)
    XCTAssertEqual(controller.document.proposals, [batch])
  }

  func testCharacterImageProposalsSurviveStyleImportAndReloadButRejectRelevantChanges() async throws {
    // A real stored batch must apply after unrelated edits, while source/field edits must block it.
    for change in ["style", "field", "record", "sourceBinding", "sourceBytes"] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      let source = root.appendingPathComponent("source.jpg")
      try Data("original source".utf8).write(to: source)
      var document = CharacterSheetDocument()
      document.sources["character"] = source.path
      document.definition.appearance.setText("body.build", "previous test value")
      document.definition.appearance.garments = [.init(order: 0)]
      let batch = CharacterProposalBatch(documentID: document.id, revision: document.revision,
        sourcePath: source.path, sourceHash: try CharacterArtifactHash.file(source.path), role: "character",
        proposals: [.init(id: "build", field: "body.build", value: "new narrow torso")],
        context: try CharacterProposalContext.capture(document, role: "character"))
      document.proposals = [batch]
      let store = StudioStore(dataDirectory: root, restoreSession: false)
      let storage = CharacterSheetDocumentStore(root: root.appendingPathComponent("Characters"))
      let controller = CharacterSheetSessionController(document: document, store: store, storage: storage)
      switch change {
      case "style": await controller.importSource(source, role: .style)
      case "field": controller.setText("body.build", "user edited value")
      case "record": controller.removeRecord("garments", id: document.definition.appearance.garments[0].id)
      case "sourceBinding": await controller.importSource(source, role: .character) // Same bytes, different binding.
      default: try Data("replaced source".utf8).write(to: source); controller.save()
      }
      let reloaded = CharacterSheetSessionController(document: try storage.load(id: document.id),
        store: store, storage: storage)
      await reloaded.applyProposals(batchID: batch.id, selectedIDs: ["build"])
      if change == "style" {
        XCTAssertEqual(reloaded.document.definition.entry(at: "body.build")?.displayString, "new narrow torso")
        XCTAssertNil(reloaded.error)
        XCTAssertTrue(reloaded.status.contains("Applied 1 values"))
      } else {
        XCTAssertEqual(reloaded.document.definition, controller.document.definition, change)
        XCTAssertTrue(try XCTUnwrap(reloaded.document.proposals.first).stale, change)
        XCTAssertNotNil(reloaded.error, change)
      }
    }
  }

  func testTextProposalsProtectAuthoredDescriptionAndSettingsOwnedValues() async throws {
    for change in ["description", "body.pose", "face.expression"] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let store = StudioStore(dataDirectory: root, restoreSession: false)
      var document = CharacterSheetDocument()
      document.originalDescription = "Standing character"
      let controller = CharacterSheetSessionController(document: document, store: store, storage: .init(root: root),
        bridge: Bridge(invocation: { _, _, _, _ in
          ["proposals": [["id": "pose", "field": "body.pose", "value": "standing"]]]
        }))
      try await controller.analyze(role: "text", modelPath: "test")
      let batchID = try XCTUnwrap(controller.document.proposals.last?.id)
      if change == "description" { controller.edit { $0.originalDescription = "Seated character" } }
      else { controller.setText(change, "user choice") }
      let before = controller.document.definition
      await controller.applyProposals(batchID: batchID, selectedIDs: ["pose"])
      XCTAssertEqual(controller.document.definition, before, change)
      XCTAssertTrue(try XCTUnwrap(controller.document.proposals.first).stale, change)
    }
  }

  func testStyleProposalsAllowAppearanceChangesButProtectStyleFieldsAndSourceRole() async throws {
    for change in ["appearance", "lighting", "sourceRole"] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      let source = root.appendingPathComponent("style.jpg")
      try Data("style image".utf8).write(to: source)
      let store = StudioStore(dataDirectory: root, restoreSession: false)
      var document = CharacterSheetDocument()
      document.sources = ["style": source.path, "character": source.path]
      let controller = CharacterSheetSessionController(document: document, store: store, storage: .init(root: root),
        bridge: Bridge(invocation: { _, _, payload, _ in
          ["sourceHash": (payload["sourceImage"] as? [String: String])?["sha256"] ?? "",
           "proposals": [["id": "preset", "field": "style.presetID", "value": "clay"]]]
        }))
      try await controller.analyze(role: "style", modelPath: "test")
      let batchID = try XCTUnwrap(controller.document.proposals.last?.id)
      switch change {
      case "appearance": controller.setText("body.build", "narrow torso")
      case "lighting": controller.setText("lighting.key", "user choice")
      default: controller.edit { $0.styleUsesCharacterImage = true }
      }
      await controller.applyProposals(batchID: batchID, selectedIDs: ["preset"])
      XCTAssertEqual(controller.document.definition.settings.stylePresetID, change == "appearance" ? "clay" : "photograph", change)
      XCTAssertEqual(controller.error == nil, change == "appearance", change)
    }
  }

  func testPartialApplyCanContinueAfterReloadAndUndoRestoresReviewedState() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let storage = CharacterSheetDocumentStore(root: root.appendingPathComponent("Characters"))
    let controller = CharacterSheetSessionController(document: .init(), store: store, storage: storage,
      bridge: Bridge(invocation: { _, _, _, _ in
        ["proposals": [["id": "build", "field": "body.build", "value": "slender"],
                       ["id": "eyes", "field": "eyes.color", "value": "Blue"]]]
      }))
    try await controller.analyze(role: "text", modelPath: "test")
    let batchID = try XCTUnwrap(controller.document.proposals.last?.id)
    await controller.applyProposals(batchID: batchID, selectedIDs: ["build"])
    let reloaded = CharacterSheetSessionController(document: try storage.load(id: controller.id), store: store, storage: storage)
    await reloaded.applyProposals(batchID: batchID, selectedIDs: ["eyes"])
    XCTAssertEqual(reloaded.document.definition.entry(at: "eyes.color")?.displayString, "Blue")
    reloaded.undo()
    await reloaded.applyProposals(batchID: batchID, selectedIDs: ["eyes"])
    XCTAssertEqual(reloaded.document.definition.entry(at: "eyes.color")?.displayString, "Blue")
    XCTAssertNil(reloaded.error)
  }

  func testStyleAnalysisCapturesTheActualSharedOrSeparateSourceRole() {
    XCTAssertEqual(CharacterSheetSessionController.analysisSourceRole(
      role: "style", styleUsesCharacterImage: true), "character")
    XCTAssertEqual(CharacterSheetSessionController.analysisSourceRole(
      role: "style", styleUsesCharacterImage: false), "style")
    XCTAssertEqual(CharacterSheetSessionController.analysisSourceRole(
      role: "face", styleUsesCharacterImage: true), "face")
  }

  func testExtractionOmissionsRemainVisibleForReview() {
    let diagnostics = CharacterSheetSessionController.extractionDiagnostics([
      "diagnostics": [["code": "proposal.invalid", "field": "body.height",
        "message": "Omitted an unsupported height measurement.", "index": 2]]
    ])
    XCTAssertEqual(diagnostics.count, 1)
    XCTAssertEqual(diagnostics.first?.field, "body.height")
    XCTAssertEqual(diagnostics.first?.message, "body.height: Omitted an unsupported height measurement.")
    XCTAssertNil(diagnostics.first?.proposalID)
  }
  func testStandaloneEditsDoNotChangeMovieDraftAndStaleProposalsDoNotApply() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let project = store.project
    let controller = CharacterSheetSessionController(document: .init(title: "Ada"), store: store,
      storage: .init(root: root.appendingPathComponent("Characters")))
    let captured = controller.document.revision
    controller.setText("eyes.color", "Brown")
    controller.document.proposals = [CharacterProposalBatch(documentID: controller.document.id,
      revision: captured, sourcePath: "", sourceHash: "", role: "character",
      proposals: [.init(field: "eyes.color", value: "Blue")])]
    await controller.applyProposals(batchID: controller.document.proposals[0].id,
      selectedIDs: Set(controller.document.proposals[0].proposals.map(\.id)))
    XCTAssertEqual(controller.document.definition.entry(at: "eyes.color")?.displayString, "Brown")
    XCTAssertEqual(store.project, project)
    XCTAssertNil(store.imageDraft)
    XCTAssertTrue(controller.document.proposals[0].stale)
  }
  func testRequiredAndStyleChangesPreserveAppearance() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let controller = CharacterSheetSessionController(document: .init(), store: store, storage: .init(root: root))
    controller.setText("identity.species", "human")
    let appearance = controller.document.definition.appearance
    controller.selectStyle("clay")
    controller.setRequired("eyes.color", true)
    XCTAssertEqual(controller.document.definition.appearance, appearance)
    XCTAssertTrue(controller.document.definition.settings.requiredFieldPaths.contains("eyes.color"))
    controller.undo()
    XCTAssertFalse(controller.document.definition.settings.requiredFieldPaths.contains("eyes.color"))
  }
  func testUndoRedoAlwaysAdvanceRevisionSoCapturedAnalysisNeverBecomesCurrentAgain() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let controller = CharacterSheetSessionController(document: .init(), store: store,
      storage: .init(root: root.appendingPathComponent("Characters")))
    let captured = controller.document.revision
    controller.setText("eyes.color", "Brown")
    let edited = controller.document.revision
    controller.undo()
    let undone = controller.document.revision
    controller.redo()
    let redone = controller.document.revision
    XCTAssertGreaterThan(edited, captured)
    XCTAssertGreaterThan(undone, edited)
    XCTAssertGreaterThan(redone, undone)
    XCTAssertNotEqual(undone, captured)
  }
  func testOriginalSourceMutationDuringBridgeIsRejectedBeforePublishingProposals() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let source = root.appendingPathComponent("character.png")
    try Data("original character bytes".utf8).write(to: source)
    let originalHash = try CharacterArtifactHash.file(source.path)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let controller = CharacterSheetSessionController(document: .init(), store: store,
      storage: .init(root: root.appendingPathComponent("Characters")))

    // Models a bridge invocation completing after the source was replaced at the same path.
    try Data("replacement character bytes".utf8).write(to: source, options: .atomic)

    do {
      try await controller.verifyOriginalSourceUnchanged(path: source.path, expectedSHA256: originalHash)
      XCTFail("A same-path source replacement must invalidate the analysis result.")
    } catch {
      XCTAssertTrue(error.localizedDescription.contains("original character source changed"))
    }
    XCTAssertTrue(controller.document.proposals.isEmpty)
  }
  func testMaliciousProposalStateAndInvalidValueRemainUnapplied() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let controller = CharacterSheetSessionController(document: .init(), store: store,
      storage: .init(root: root.appendingPathComponent("Characters")))
    let revision = controller.document.revision
    let malicious = CharacterFieldProposal(field: "eyes.color", value: "Blue", state: "apply")
    let invalid = CharacterFieldProposal(field: "body.height", value: "very tall")
    controller.document.proposals = [.init(documentID: controller.document.id,
      revision: revision, sourcePath: "", sourceHash: "", role: "text",
      proposals: [malicious, invalid])]
    let batch = controller.document.proposals[0]
    await controller.applyProposals(batchID: batch.id, selectedIDs: Set(batch.proposals.map(\.id)))
    XCTAssertNil(controller.document.definition.entry(at: "eyes.color"))
    XCTAssertNil(controller.document.definition.entry(at: "body.height"))
    XCTAssertFalse(controller.document.proposals.isEmpty)
    XCTAssertGreaterThanOrEqual(controller.document.proposals[0].diagnostics?.count ?? 0, 2)
  }
}
