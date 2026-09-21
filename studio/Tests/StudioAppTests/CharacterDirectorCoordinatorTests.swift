import StudioCore
import XCTest
@testable import WeeToddStudio

final class CharacterDirectorCoordinatorTests: XCTestCase {
  private func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }

  private func context(_ key: String = "hero") -> ReferenceSheetContext {
    var value = ReferenceSheetContext(subjectKey: key, name: "Ada", kind: .character,
      description: "Short black hair and a red jacket.")
    value.template = .characterSheet
    return value
  }

  @MainActor func testStandaloneSessionUsesSeparateStorageWithoutChangingMovieOrImageDraft() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project.name = "Existing movie"
    var ordinary = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    ordinary.prompt = "Existing image draft"
    store.imageDraft = ordinary
    let beforeProject = store.project

    let coordinator = CharacterDirectorCoordinator(store: store)
    let session = coordinator.session()

    XCTAssertEqual(store.project, beforeProject)
    XCTAssertEqual(store.imageDraft, ordinary)
    XCTAssertEqual(session.storage.root, root.appendingPathComponent("Character Director"))
    XCTAssertEqual(session.document.draft.destination.projectID, session.document.id)
  }

  @MainActor func testSubjectSessionRestoresBySubjectKeyAndKeepsOneControllerOwner() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    let first = coordinator.session(context: context())
    first.setText("identity.type", "human")

    let same = coordinator.session(context: context())
    XCTAssertTrue(first === same)

    let reopened = CharacterDirectorCoordinator(store: store).session(context: context())
    XCTAssertEqual(reopened.document.id, first.document.id)
    XCTAssertEqual(reopened.document.definition.entry(at: "identity.type")?.displayString, "human")
    XCTAssertEqual(reopened.document.subjectKey, "hero")
  }

  @MainActor func testDocumentIDRestoresExactDocumentWithoutChangingAnotherDraft() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    let first = coordinator.session(context: context("first"))
    let second = coordinator.session(context: context("second"))

    XCTAssertTrue(coordinator.session(documentID: first.id) === first)
    XCTAssertTrue(coordinator.session(documentID: second.id) === second)
    XCTAssertNotEqual(first.id, second.id)
  }

  @MainActor func testRecentDocumentsExposeSavedStandaloneCharacters() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    let zed = coordinator.session(context: context("zed")); zed.edit { $0.title = "Zed" }
    let ada = coordinator.session(context: context("ada")); ada.edit { $0.title = "Ada" }

    XCTAssertEqual(coordinator.recentDocuments.map(\.title), ["Ada", "Zed"])
    XCTAssertEqual(coordinator.recentDocuments.map(\.id), [ada.id, zed.id])
  }

  @MainActor func testCorruptSavedDocumentDoesNotBecomeBlankDocumentWithSameID() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    let id = UUID()
    let url = root.appendingPathComponent("Character Director/\(id.uuidString)/character.json")
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("not-json".utf8).write(to: url)

    XCTAssertNil(coordinator.session(documentID: id))
    XCTAssertTrue(store.error?.contains("could not be opened") == true)
    XCTAssertEqual(String(data: try Data(contentsOf: url), encoding: .utf8), "not-json")
  }

  @MainActor func testChangedSubjectContextCreatesNewDocumentWithoutOverwritingLocalEdits() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    var original = context()
    original.characterDefinition = .newDraft()
    let first = coordinator.session(context: original)
    first.setText("hair.color", "Black")

    XCTAssertTrue(coordinator.session(context: original) === first,
      "Local document edits must not prevent reuse when the captured subject input is unchanged")
    var changed = original
    changed.description = "Long silver hair and a blue jacket."
    var replacement = CharacterSheetDefinition.newDraft()
    replacement.appearance.setText("hair.color", "Silver")
    changed.characterDefinition = replacement
    let second = coordinator.session(context: changed)

    XCTAssertNotEqual(second.id, first.id)
    XCTAssertEqual(first.document.definition.entry(at: "hair.color")?.displayString, "Black")
    XCTAssertEqual(second.document.definition.entry(at: "hair.color")?.displayString, "Silver")
  }

  @MainActor func testEmbeddedCandidateMustMatchSubjectAndCurrentDefinition() throws {
    var active = context()
    active.characterDefinition = .newDraft()
    var matching = MediaAsset(name: "Current", kind: .image)
    var generation = ImageGeneration(provider: "drawThings", requestFingerprint: "hash",
      modelID: "model", prompt: "prompt")
    generation.referenceSheet = active
    matching.generation = generation
    XCTAssertNil(CharacterSheetEmbeddedHost.candidateIssue(matching, context: active,
      definition: active.characterDefinition!))

    var stale = matching
    stale.generation?.referenceSheet?.characterDefinition?.appearance.setText("hair.color", "Silver")
    XCTAssertNotNil(CharacterSheetEmbeddedHost.candidateIssue(stale, context: active,
      definition: active.characterDefinition!))
    var wrongSubject = matching
    wrongSubject.generation?.referenceSheet?.subjectKey = "other"
    XCTAssertNotNil(CharacterSheetEmbeddedHost.candidateIssue(wrongSubject, context: active,
      definition: active.characterDefinition!))
  }

  func testManualCropRecoveryAddsUnusedRoleAndNeverFabricatesQuarterLayout() {
    var detection = CharacterPanelDetection(sourceSHA256: "source", sourceOrientation: 1,
      detectorVersion: "vision", candidates: [], status: .needsReview, diagnostics: [])
    XCTAssertTrue(CharacterDirectorWindow.addManualCrop(to: &detection, revision: 3))
    XCTAssertEqual(detection.candidates[0].role, .front)
    XCTAssertEqual(detection.candidates[0].sourcePixelRect, PanelPixelRect(x: 0, y: 0, width: 240, height: 400))
    XCTAssertEqual(detection.status, .needsReview)
    XCTAssertTrue(CharacterDirectorWindow.addManualCrop(to: &detection, revision: 4))
    XCTAssertEqual(detection.candidates[1].role, .side)
    XCTAssertNotEqual(detection.candidates[0].sourcePixelRect.width, 480)
  }

  @MainActor func testCloseChoiceKeepsOrCancelsOnlySelectedSession() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let coordinator = CharacterDirectorCoordinator(store: store)
    let keep = coordinator.session(context: context("keep"))
    let cancel = coordinator.session(context: context("cancel"))

    coordinator.applyCloseChoice(.keepRunning, to: keep)
    XCTAssertFalse(keep.cancelled)
    coordinator.applyCloseChoice(.cancelJob, to: cancel)
    XCTAssertTrue(cancel.cancelled)
    XCTAssertFalse(keep.cancelled)
  }
}
