import XCTest
@testable import WeeToddStudio
import StudioCore

@MainActor final class CharacterDirectorSessionTests: XCTestCase {
  func testStandaloneEditsDoNotChangeMovieDraftAndStaleProposalsDoNotApply() throws {
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
    controller.applyProposals(batchID: controller.document.proposals[0].id,
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
  func testMaliciousProposalStateAndInvalidValueRemainUnapplied() throws {
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
    controller.applyProposals(batchID: batch.id, selectedIDs: Set(batch.proposals.map(\.id)))
    XCTAssertNil(controller.document.definition.entry(at: "eyes.color"))
    XCTAssertNil(controller.document.definition.entry(at: "body.height"))
    XCTAssertFalse(controller.document.proposals.isEmpty)
    XCTAssertGreaterThanOrEqual(controller.document.proposals[0].diagnostics?.count ?? 0, 2)
  }
}
