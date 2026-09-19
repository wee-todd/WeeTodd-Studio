import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class ReferenceImageWorkspaceTests: XCTestCase {
  private func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }
  private func context(_ key: String = "actor") -> ReferenceSheetContext {
    ReferenceSheetContext(subjectKey: key, name: key, kind: .character, description: "Reviewed identity")
  }
  @MainActor func testExplicitReferenceConnectionSurvivesOrdinaryWorkspaceRestoreAndPersistence() throws {
    let dir = try directory(), store = StudioStore(dataDirectory: dir, restoreSession: false)
    store.restoringImageWorkspace = false
    store.drawThingsConnections = [DrawThingsConnection(id: "cloud"), DrawThingsConnection(id: "local")]
    var ordinary = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    ordinary.profileID = "cloud"; ordinary.prompt = "Ordinary work"
    store.imageDraft = ordinary
    let lease = ReferenceWorkspaceLease(store: store, subjectKey: "actor")
    store.imageDraft = store.makeReferenceImageDraft(context(), previousDraft: ordinary)
    store.selectImageConnection("local")
    XCTAssertTrue(lease.restore(store: store))
    XCTAssertEqual(store.imageDraft, ordinary)
    let restored = try ImageWorkspaceLibrary.read(from: dir.appendingPathComponent("image-workspaces.json"))
    store.imageWorkspaceLibrary = restored
    XCTAssertEqual(store.makeReferenceImageDraft(context("next"), previousDraft: ordinary).profileID, "local")
    store.selectImageConnection("cloud")
    XCTAssertEqual(store.imageWorkspaceLibrary.referenceConnectionID, "local")
  }
  @MainActor func testSavedSubjectSettingsTakePrecedenceOverLastConnection() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    store.drawThingsConnections = [DrawThingsConnection(id: "cloud"), DrawThingsConnection(id: "local")]
    store.imageWorkspaceLibrary.referenceConnectionID = "local"
    var saved = store.makeReferenceImageDraft(context(), previousDraft: nil)
    saved.profileID = "cloud"; saved.modelID = "explicit-model"; saved.steps = 13; saved.prompt = "Custom saved prompt"
    store.imageWorkspaceLibrary.record(saved, preview: "/candidate.png")
    XCTAssertEqual(store.makeReferenceImageDraft(context(), previousDraft: nil), saved)
  }
  @MainActor func testPlacementOnlyChangeDoesNotRecoverStaleReferencePrompt() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var subject = PlanningSubject(name: "Wiglaf", kind: .character)
    var ring = PlanningSubject(name: "Ring", kind: .prop); ring.details = "Gold arm-ring."
    subject.relationships = [ObjectRelationship(targetID: ring.id, role: .holds, placement: "Held in his hand.")]
    var original = context()
    original.linkedDefinitions = ReferenceSheetLinks.definitions(subject: subject, inventory: [subject, ring])
    var saved = store.makeReferenceImageDraft(original, previousDraft: nil)
    saved.prompt = "Old prompt with premature handover"
    store.imageWorkspaceLibrary.record(saved, preview: nil)
    subject.relationships?[0].placement = "Receives only after the dragon falls."
    var updated = original
    updated.linkedDefinitions = ReferenceSheetLinks.definitions(subject: subject, inventory: [subject, ring])
    let restored = store.makeReferenceImageDraft(updated, previousDraft: nil)
    XCTAssertNotEqual(restored.prompt, saved.prompt)
    XCTAssertTrue(restored.prompt.contains("Receives only after the dragon falls."))
    XCTAssertEqual(restored.referenceSheet?.description, original.description)
  }
  @MainActor func testRemovedReferencePreferenceDoesNotSilentlyChooseCloud() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    store.drawThingsConnections = [DrawThingsConnection(id: "cloud")]
    store.imageWorkspaceLibrary.referenceConnectionID = "removed-local"
    var ordinary = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    ordinary.profileID = "cloud"
    XCTAssertEqual(store.makeReferenceImageDraft(context(), previousDraft: ordinary).profileID, "")
    store.imageWorkspaceLibrary.referenceConnectionID = nil
    XCTAssertEqual(store.makeReferenceImageDraft(context(), previousDraft: nil).profileID, "")
    XCTAssertEqual(store.makeReferenceImageDraft(context(), previousDraft: ordinary).profileID, "cloud")
  }
  func testOlderImageWorkspaceLibraryNeedsNoMigration() throws {
    let library = try JSONDecoder().decode(ImageWorkspaceLibrary.self, from: Data(#"{"version":1,"sessions":{}}"#.utf8))
    XCTAssertNil(library.referenceConnectionID)
  }
  @MainActor func testReferenceAttemptClearsOnlyItsOwnErrorAndKeepsGlobalError() async throws {
    var fail = true
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { _, _, _, _ in
      if fail { throw StudioError.invalid("connection_failed") }
      return ["eligibility": "allowed"]
    })
    store.drawThingsConnections = [DrawThingsConnection(id: "local")]
    var draft = store.makeReferenceImageDraft(context(), previousDraft: nil)
    draft.profileID = "local"; draft.modelID = "test-model"; store.imageDraft = draft
    store.error = "Unrelated project error"
    await store.prepareImageGeneration()
    XCTAssertTrue(store.referenceImageError?.contains("connection_failed") == true)
    XCTAssertEqual(store.error, "Unrelated project error")
    fail = false
    await store.prepareImageGeneration()
    XCTAssertNil(store.referenceImageError)
    XCTAssertEqual(store.error, "Unrelated project error")
    XCTAssertEqual(store.imageEstimate?["eligibility"] as? String, "allowed")
  }
  @MainActor func testLateReferenceFailureCannotLandInAnotherSubjectOrDocument() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let first = store.makeReferenceImageDraft(context(), previousDraft: nil)
    store.imageDraft = first
    let key = store.beginImageAttempt(first)
    store.imageDraft = store.makeReferenceImageDraft(context("other"), previousDraft: nil)
    store.recordImageFailure("old failure", referenceKey: key)
    XCTAssertNil(store.referenceImageError)
    store.imageDraft = first
    try store.replaceDocument(store.project, url: nil, isDirty: false)
    store.imageDraft = first
    store.recordImageFailure("old document failure", referenceKey: key)
    XCTAssertNil(store.referenceImageError)
  }
  @MainActor func testReopenedReferenceSupersedesOldLeaseWithoutLosingOrdinaryWorkspace() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var ordinary = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    ordinary.prompt = "Ordinary work"
    store.imageDraft = ordinary; store.imagePreviewPath = "/ordinary.png"
    let first = ReferenceWorkspaceLease(store: store, subjectKey: "actor")
    store.imageDraft = store.makeReferenceImageDraft(context(), previousDraft: first.previousDraft)
    store.imageDraft?.prompt = "First editor work"
    let oldAttemptKey = store.beginImageAttempt(try XCTUnwrap(store.imageDraft))
    let second = ReferenceWorkspaceLease(store: store, subjectKey: "actor")
    var reopened = store.makeReferenceImageDraft(context(), previousDraft: second.previousDraft)
    reopened.prompt = "Second editor work"; store.imageDraft = reopened
    store.recordImageFailure("Late old editor failure", referenceKey: oldAttemptKey)
    XCTAssertNil(store.referenceImageError)
    XCTAssertThrowsError(try first.validate(store: store))
    XCTAssertFalse(first.restore(store: store))
    XCTAssertEqual(store.imageDraft, reopened)
    XCTAssertEqual(second.previousDraft, ordinary)
    XCTAssertTrue(second.restore(store: store))
    XCTAssertEqual(store.imageDraft, ordinary)
    XCTAssertEqual(store.imagePreviewPath, "/ordinary.png")
  }
  @MainActor func testOverlappingReferenceLeasesCannotNilNewEditorDraft() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let first = ReferenceWorkspaceLease(store: store, subjectKey: "actor")
    store.imageDraft = store.makeReferenceImageDraft(context(), previousDraft: nil)
    let second = ReferenceWorkspaceLease(store: store, subjectKey: "actor")
    store.imageDraft = store.makeReferenceImageDraft(context(), previousDraft: second.previousDraft)
    XCTAssertFalse(first.restore(store: store))
    XCTAssertNotNil(store.imageDraft)
    store.selectImageConnection("local")
    XCTAssertEqual(store.imageDraft?.profileID, "local")
    XCTAssertTrue(second.restore(store: store))
    XCTAssertNil(store.imageDraft)
  }

  @MainActor func testRecoveredReferenceDraftExposesScopedErrorWithoutReferenceToolsOrLease() async throws {
    let dir = try directory()
    var fail = true
    let store = StudioStore(dataDirectory: dir, restoreSession: false, invocation: { _, _, _, _ in
      if fail { throw StudioError.invalid("Recovered connection_failed") }
      return ["eligibility": "allowed"]
    })
    store.drawThingsConnections = [DrawThingsConnection(id: "local")]
    var draft = store.makeReferenceImageDraft(context(), previousDraft: nil)
    draft.profileID = "local"; draft.modelID = "test-model"
    var library = ImageWorkspaceLibrary()
    library.record(draft, preview: "/saved-candidate.png")
    try library.write(to: dir.appendingPathComponent("image-workspaces.json"))
    store.restoreImageWorkspaces()
    XCTAssertEqual(store.imageDraft, draft)
    XCTAssertFalse(store.referenceSheetOpen)
    XCTAssertNil(store.activeReferenceLease)
    store.error = "Unrelated error"
    await store.prepareImageGeneration()
    XCTAssertTrue(store.referenceImageError?.contains("Recovered connection_failed") == true)
    XCTAssertEqual(store.error, "Unrelated error")
    fail = false
    await store.prepareImageGeneration()
    XCTAssertNil(store.referenceImageError)
    XCTAssertEqual(store.error, "Unrelated error")
  }

}
