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
}
