import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class DirectorSessionTests: XCTestCase {
  private func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }
  private func summary(_ revision: String) throws -> WorkflowRunSummary {
    try JSONDecoder().decode(WorkflowRunSummary.self, from: Data("{\"status\":\"awaiting_approval\",\"totalSeconds\":1,\"revision\":\"\(revision)\",\"steps\":{},\"outputs\":{}}".utf8))
  }
  @MainActor func testDraftsAndNavigationSurviveNewControllerWithoutChangingApprovedResult() async throws {
    let url = try directory().appendingPathComponent("session.json")
    let first = DirectorSessionController()
    try await first.open(url)
    first.state.fields["brief"] = "Unfinished movie brief"
    first.state.selectedStep = "classify"
    first.state.reviews["classify"] = DirectorReviewDraft()
    first.state.reviews["classify"]?.selectedSubject = "actor"
    first.state.reviews["classify"]?.referencePaths["actor"] = ["/second.png", "/first.png"]
    first.state.result = try summary("approved-r1")
    try await first.flush()
    let reopened = DirectorSessionController()
    try await reopened.open(url)
    XCTAssertEqual(reopened.state.fields["brief"], "Unfinished movie brief")
    XCTAssertEqual(reopened.state.selectedStep, "classify")
    XCTAssertEqual(reopened.state.reviews["classify"]?.selectedSubject, "actor")
    XCTAssertEqual(reopened.state.reviews["classify"]?.referencePaths["actor"], ["/second.png", "/first.png"])
    XCTAssertEqual(reopened.state.result?.revision, "approved-r1")
  }
  @MainActor func testInvalidImportedJobKeepsCurrentSession() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    controller.state.fields["brief"] = "Keep me"
    controller.state.result = try summary("old")
    let job = WorkflowJob(definition: ["format": .string("weetodd-workflow-v1"), "id": .string("test"), "steps": .array([.object(["id": .string("one")])]), "inputs": .object(["images": .object(["type": .string("image_list")])])], inputs: ["images": .array([.string("missing")])], models: [:], assets: [:], runDirectory: dir.path)
    let url = dir.appendingPathComponent("job.json")
    try JSONEncoder().encode(job).write(to: url)
    do { try await controller.importJob(url, validateDefinition: { _ in }); XCTFail("Missing bindings must fail") } catch {}
    XCTAssertEqual(controller.state.fields["brief"], "Keep me")
    XCTAssertEqual(controller.state.result?.revision, "old")
  }
  @MainActor func testCancelledRunRefreshesCheckpointAndKeepsPreviousResultIfUnreadable() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    controller.state.runDirectory = dir.path
    controller.state.result = try summary("old")
    try JSONEncoder().encode(summary("saved-before-cancel")).write(to: dir.appendingPathComponent("run.json"))
    await controller.refreshCheckpoint()
    XCTAssertEqual(controller.state.result?.revision, "saved-before-cancel")
    try Data("broken".utf8).write(to: dir.appendingPathComponent("run.json"))
    await controller.refreshCheckpoint()
    XCTAssertEqual(controller.state.result?.revision, "saved-before-cancel")
  }
  @MainActor func testOversizedCheckpointRejectsWholeCandidate() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    controller.state.fields["brief"] = "Keep me"
    let job = WorkflowJob(definition: ["format": .string("weetodd-workflow-v1"), "id": .string("test"), "steps": .array([.object(["id": .string("one")])])], inputs: [:], models: [:], assets: [:], runDirectory: dir.path)
    let url = dir.appendingPathComponent("job.json")
    try JSONEncoder().encode(job).write(to: url)
    try Data(repeating: 32, count: 16 * 1024 * 1024 + 1).write(to: dir.appendingPathComponent("run.json"))
    do { try await controller.importJob(url, validateDefinition: { _ in }); XCTFail("Oversized checkpoint accepted") } catch {}
    XCTAssertEqual(controller.state.fields["brief"], "Keep me")
  }
  @MainActor func testDocumentTargetRejectsSameUUIDReopenedAndChangedPlanning() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let target = DirectorDocumentTarget(store: store)
    let project = store.project
    try store.replaceDocument(project, url: nil, isDirty: false)
    XCTAssertEqual(store.project.id, project.id)
    XCTAssertThrowsError(try target.validate(store: store))
    let current = DirectorDocumentTarget(store: store)
    store.changePlanning { $0.sourceText = "New brief" }
    XCTAssertThrowsError(try current.validate(store: store))
  }
}

extension DirectorSessionTests {
  @MainActor func testPromptContextRejectsChangedGenerationInputsAndSameUUIDReopen() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let clip = Clip()
    store.project.clips = [clip]
    let context = PromptAssistantContext(project: store.project, clip: clip, documentSessionID: store.documentSessionID)
    store.project.clips[0].duration += 1
    XCTAssertThrowsError(try context.validate(project: store.project, image: nil, documentSessionID: store.documentSessionID))
    store.project.clips = [clip]
    let project = store.project
    try store.replaceDocument(project, url: nil, isDirty: false)
    XCTAssertThrowsError(try context.validate(project: store.project, image: nil, documentSessionID: store.documentSessionID))
  }
  @MainActor func testReferenceWorkspaceDoesNotRestoreIntoReopenedDocument() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let lease = ReferenceWorkspaceLease(store: store)
    store.referenceSheetOpen = true
    let project = store.project
    try store.replaceDocument(project, url: nil, isDirty: false)
    var replacement = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: project.id))
    replacement.prompt = "New session workspace"
    store.imageDraft = replacement
    XCTAssertFalse(store.referenceSheetOpen)
    XCTAssertFalse(lease.restore(store: store))
    XCTAssertEqual(store.imageDraft?.prompt, "New session workspace")
    XCTAssertThrowsError(try lease.validate(store: store))
  }
}

@MainActor private final class DirectorDeferredResponse {
  var continuation: CheckedContinuation<[String: Any], Error>?
  var entered: (() -> Void)?
  func wait() async throws -> [String: Any] {
    try await withCheckedThrowingContinuation { continuation in self.continuation = continuation; entered?() }
  }
}
extension DirectorSessionTests {
  @MainActor func testFailedSuspendedExecutionRetainsCheckpointAndNeverClearsReview() async throws {
    let dir = try directory(), controller = DirectorSessionController(), deferred = DirectorDeferredResponse()
    controller.state.runDirectory = dir.path; controller.state.result = try summary("reviewed")
    let entered = expectation(description: "bridge suspended"); deferred.entered = { entered.fulfill() }
    let task = Task { try await controller.execute { try await deferred.wait() } }
    await fulfillment(of: [entered], timeout: 2)
    XCTAssertEqual(controller.state.result?.revision, "reviewed")
    try JSONEncoder().encode(summary("checkpoint-r2")).write(to: dir.appendingPathComponent("run.json"))
    deferred.continuation?.resume(throwing: CancellationError())
    do { try await task.value; XCTFail("Cancellation must be reported") } catch {}
    XCTAssertEqual(controller.state.result?.revision, "checkpoint-r2")
  }
  @MainActor func testLateExecutionDoesNotReplaceNewSessionResult() async throws {
    let controller = DirectorSessionController(), deferred = DirectorDeferredResponse()
    let entered = expectation(description: "bridge suspended"); deferred.entered = { entered.fulfill() }
    let task = Task { try await controller.execute { try await deferred.wait() } }
    await fulfillment(of: [entered], timeout: 2)
    controller.state = DirectorSession(); controller.state.result = try summary("new")
    deferred.continuation?.resume(returning: ["status": "completed", "totalSeconds": 1, "revision": "old", "steps": [:], "outputs": [:]])
    do { try await task.value; XCTFail("Stale result must not be applied") } catch {}
    XCTAssertEqual(controller.state.result?.revision, "new")
  }
  func testReferenceBindingPreservesAddRemoveOrderAndUsesExistingKeys() throws {
    var bindings = ["asset:a": "/a.png", "asset:b": "/b.png"]
    let added = try DirectorReferenceAssets.bind(["/b.png", "/new.png", "/a.png"], in: &bindings)
    XCTAssertEqual(added.first, "asset:b"); XCTAssertEqual(added.last, "asset:a")
    XCTAssertEqual(added.map { bindings[$0] }, ["/b.png", "/new.png", "/a.png"])
    let removed = try DirectorReferenceAssets.bind(["/new.png"], in: &bindings)
    XCTAssertEqual(removed, [added[1]])
    XCTAssertEqual(try DirectorReferenceAssets.bind([], in: &bindings), [])
    XCTAssertThrowsError(try DirectorReferenceAssets.bind(["/a.png", "/a.png"], in: &bindings))
  }
}

extension DirectorSessionTests {
  @MainActor func testRelinkedPromptReferenceInvalidatesProposal() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let asset = MediaAsset(name: "Face", kind: .image, path: "/old.png")
    var clip = Clip(); clip.attachments = [Attachment(assetID: asset.id, role: .keyframe)]
    store.project.assets = [asset]; store.project.clips = [clip]
    let context = PromptAssistantContext(project: store.project, clip: clip)
    store.project.assets[0].path = "/new.png"
    XCTAssertThrowsError(try context.validate(project: store.project, image: nil))
  }
  @MainActor func testExplicitImageWorkspacePersistenceUsesInjectedDirectory() throws {
    let dir = try directory(), store = StudioStore(dataDirectory: dir, restoreSession: false)
    store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    draft.prompt = "Temporary test draft"; store.imageDraft = draft
    let loaded = try ImageWorkspaceLibrary.read(from: dir.appendingPathComponent("image-workspaces.json"))
    XCTAssertEqual(loaded.sessions[draft.storageKey]?.draft.prompt, "Temporary test draft")
  }
  @MainActor func testRejectedSharedDefinitionValidationKeepsCurrentSession() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    controller.state.fields["brief"] = "Keep draft"
    let job = WorkflowJob(definition: ["format": .string("weetodd-workflow-v1"), "id": .string("test"), "steps": .array([.object(["id": .string("one")])])], inputs: [:], models: [:], assets: [:], runDirectory: dir.path)
    let url = dir.appendingPathComponent("job.json"); try JSONEncoder().encode(job).write(to: url)
    do {
      try await controller.importJob(url, validateDefinition: { _ in throw StudioError.invalid("Invalid operation contract") })
      XCTFail("Validation rejection must keep old session")
    } catch {}
    XCTAssertEqual(controller.state.fields["brief"], "Keep draft")
  }
}

extension DirectorSessionTests {
  @MainActor func testInvalidBriefFieldSurvivesRelaunchSeparatelyFromSavedBrief() async throws {
    let brief = try JSONDecoder().decode(CreativeBrief.self, from: Data(#"{"sourceText":"A pilot enters.","facts":[],"questions":[],"preferences":{"durationSeconds":30,"targetClipSeconds":5,"frameRate":24,"visualStyle":"Cinematic","presentation":"Landscape","cameraStyle":"Gentle","audioStyle":"Ambience","designPolicy":"Review","constraints":""},"referenceObservations":[]}"#.utf8))
    let url = try directory().appendingPathComponent("session.json"), first = DirectorSessionController()
    try await first.open(url)
    var draft = DirectorReviewDraft(); draft.brief = DirectorBriefDraft(brief); draft.brief?.duration = "unfinished number"
    first.state.reviews["brief"] = draft; try await first.flush()
    let reopened = DirectorSessionController(); try await reopened.open(url)
    XCTAssertEqual(reopened.state.reviews["brief"]?.brief?.duration, "unfinished number")
    XCTAssertEqual(reopened.state.reviews["brief"]?.brief?.baseline, brief)
  }
  @MainActor func testUnreadableDraftIsNeverOverwritten() async throws {
    let url = try directory().appendingPathComponent("session.json"), original = Data("unreadable preserved draft".utf8)
    try original.write(to: url)
    let controller = DirectorSessionController()
    do { try await controller.open(url); XCTFail("Invalid draft accepted") } catch {}
    controller.state.fields["brief"] = "New text"; try await controller.flush()
    XCTAssertEqual(try Data(contentsOf: url), original)
  }
  @MainActor func testImportedCheckpointFromAnotherWorkflowPreservesDraft() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    controller.state.fields["brief"] = "Keep me"
    let job = WorkflowJob(definition: ["format": .string("weetodd-workflow-v1"), "id": .string("expected"), "steps": .array([.object(["id": .string("one")])])], inputs: [:], models: [:], assets: [:], runDirectory: dir.path)
    let url = dir.appendingPathComponent("job.json"); try JSONEncoder().encode(job).write(to: url)
    try Data(#"{"format":"weetodd-workflow-run-v1","workflowID":"other","status":"completed","totalSeconds":1,"steps":{},"outputs":{}}"#.utf8).write(to: dir.appendingPathComponent("run.json"))
    do { try await controller.importJob(url, validateDefinition: { _ in }); XCTFail("Mismatched checkpoint accepted") } catch {}
    XCTAssertEqual(controller.state.fields["brief"], "Keep me")
  }
}

extension DirectorSessionTests {
  @MainActor func testOpeningDirectorAnchorsFreshMovieBeforeFirstProjectEdit() async throws {
    let dir = try directory(), first = StudioStore(dataDirectory: dir, restoreSession: false)
    let originalID = first.project.id
    let draftURL = try first.directorSessionURL(key: "intake")
    let draft = DirectorSessionController(); try await draft.open(draftURL)
    draft.state.fields["brief"] = "A first-launch movie idea"; try await draft.flush()
    XCTAssertFalse(first.dirty)
    XCTAssertNil(first.projectURL)
    let relaunched = StudioStore(dataDirectory: dir, restoreSession: false)
    relaunched.restoreAutosavedProject()
    XCTAssertEqual(relaunched.project.id, originalID)
    let restoredURL = try relaunched.directorSessionURL(key: "intake")
    XCTAssertEqual(restoredURL, draftURL)
    let restoredDraft = DirectorSessionController(); try await restoredDraft.open(restoredURL)
    XCTAssertEqual(restoredDraft.state.fields["brief"], "A first-launch movie idea")
  }
  @MainActor func testDirectorAnchorPreservesSavedProjectAndDirtyRecovery() throws {
    let dir = try directory(), store = StudioStore(dataDirectory: dir, restoreSession: false)
    let savedURL = dir.appendingPathComponent("saved.weetodd")
    try ProjectStorage.write(store.project, to: savedURL)
    let savedBytes = try Data(contentsOf: savedURL)
    store.projectURL = savedURL
    store.project.name = "Unsaved change"; store.dirty = true
    let session = store.documentSessionID
    _ = try store.directorSessionURL(key: "intake")
    XCTAssertEqual(try Data(contentsOf: savedURL), savedBytes)
    XCTAssertEqual(store.projectURL, savedURL)
    XCTAssertTrue(store.dirty)
    XCTAssertEqual(store.documentSessionID, session)
    XCTAssertEqual(try ProjectStorage.read(dir.appendingPathComponent("Autosave.weetodd")).name, "Unsaved change")
    XCTAssertEqual(try ProjectStorage.read(dir.appendingPathComponent("Recovery/\(session.uuidString).weetodd")).name, "Unsaved change")
  }
}

extension DirectorSessionTests {
  @MainActor func testSameWorkflowCheckpointWithDifferentInputsOrDefinitionCannotExposeApprovals() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    let definition: [String: JSONValue] = ["format": .string("weetodd-workflow-v1"), "id": .string("same"), "steps": .array([.object(["id": .string("one")])]), "inputs": .object(["brief": .object(["type": .string("text")])])]
    let inputs: [String: JSONValue] = ["brief": .string("Current movie")]
    let job = WorkflowJob(definition: definition, inputs: inputs, models: [:], assets: [:], runDirectory: dir.path)
    let url = dir.appendingPathComponent("job.json"); try JSONEncoder().encode(job).write(to: url)
    for mismatch in ["inputs", "definition", "missing"] {
      controller.state = DirectorSession(); controller.state.fields["brief"] = "Keep current draft"
      controller.state.result = try summary("current-review")
      var envelope: [String: JSONValue] = ["format": .string("weetodd-workflow-run-v1"), "workflowID": .string("same"), "status": .string("completed"), "totalSeconds": .integer(1), "revision": .string("foreign-approved"), "steps": .object(["one": .object(["name": .string("One"), "status": .string("completed"), "approved": .boolean(true)])]), "outputs": .object([:]), "definition": .object(definition), "inputs": .object(inputs)]
      if mismatch == "inputs" { envelope["inputs"] = .object(["brief": .string("Another movie")]) }
      if mismatch == "definition" { var changed = definition; changed["name"] = .string("Changed recipe"); envelope["definition"] = .object(changed) }
      if mismatch == "missing" { envelope.removeValue(forKey: "inputs") }
      try JSONEncoder().encode(envelope).write(to: dir.appendingPathComponent("run.json"))
      do { try await controller.importJob(url, validateDefinition: { _ in }); XCTFail("Accepted mismatched \(mismatch)") } catch {}
      XCTAssertEqual(controller.state.fields["brief"], "Keep current draft", mismatch)
      XCTAssertEqual(controller.state.result?.revision, "current-review", mismatch)
    }
  }
  @MainActor func testMatchingCheckpointAcceptsDefaultedInputsAndChangedRuntimeBindings() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    let definition: [String: JSONValue] = ["format": .string("weetodd-workflow-v1"), "id": .string("same"), "steps": .array([.object(["id": .string("one")])]), "inputs": .object(["brief": .object(["type": .string("text"), "default": .string("Default brief")])])]
    let job = WorkflowJob(definition: definition, inputs: [:], models: ["assistant": "/new-model.ckpt"], assets: [:], runDirectory: dir.path, maxTokens: 512)
    let url = dir.appendingPathComponent("job.json"); try JSONEncoder().encode(job).write(to: url)
    let envelope: [String: JSONValue] = ["format": .string("weetodd-workflow-run-v1"), "workflowID": .string("same"), "status": .string("completed"), "totalSeconds": .integer(1), "revision": .string("approved"), "steps": .object([:]), "outputs": .object([:]), "definition": .object(definition), "inputs": .object(["brief": .string("Default brief")]), "runtimeFingerprints": .object(["assistant": .string("old-runtime")]), "generationSettings": .object(["maxTokens": .integer(1024)])]
    try JSONEncoder().encode(envelope).write(to: dir.appendingPathComponent("run.json"))
    try await controller.importJob(url, validateDefinition: { _ in })
    XCTAssertEqual(controller.state.result?.revision, "approved")
    XCTAssertEqual(controller.state.modelPaths["assistant"], "/new-model.ckpt")
    XCTAssertEqual(controller.state.tokens, 512)
    XCTAssertEqual(controller.state.fields["brief"], "Default brief")
  }
}

extension DirectorSessionTests {
  func testBatchGuardIncludesUnselectedSubjectReferencesAndClosedShotEditorDrafts() throws {
    let subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: Data(#"{"id":"ada","name":"Ada","kind":"character","description":"Red coat","aliases":[],"evidence":[],"suggestions":[],"referenceAssets":["asset:a"]}"#.utf8))
    let clip = try JSONDecoder().decode(WorkflowClipDraft.self, from: Data(#"{"id":"c1","startFrame":0,"frameCount":120,"action":"Walk","startState":"Door","endState":"Desk","location":"Office","characters":["ada"],"continuity":"cut"}"#.utf8))
    let outputs: [String: JSONValue] = ["subjects": try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode([subject])), "clips": .object(["fps": .integer(24), "totalFrames": .integer(120), "characters": .array([]), "clips": .array([try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(clip))])])]
    var draft = DirectorReviewDraft()
    XCTAssertFalse(draft.hasUnsavedChanges(outputs: outputs, referenceBindings: ["asset:a": "/old.png"]))
    draft.selectedSubject = "someone-else"
    draft.referencePaths["ada"] = ["/new.png"]
    XCTAssertTrue(draft.hasUnsavedChanges(outputs: outputs, referenceBindings: ["asset:a": "/old.png"]))
    draft.referencePaths = [:]; var changed = clip; changed.action = "Run"; draft.clips["c1"] = changed
    XCTAssertTrue(draft.hasUnsavedChanges(outputs: outputs, referenceBindings: [:]))
    draft.clips = [:]; draft.subjects["ada"] = subject
    XCTAssertFalse(draft.hasUnsavedChanges(outputs: outputs, referenceBindings: [:]))
    draft.subjects["ada"]?.description = "Blue coat"
    XCTAssertTrue(draft.hasUnsavedChanges(outputs: outputs, referenceBindings: [:]))
  }
  func testRepairScopeAndBeforeSnapshotSurviveSessionAndOlderDraftsStillDecode() throws {
    var draft = try JSONDecoder().decode(DirectorReviewDraft.self, from: Data(#"{"subjects":{},"referencePaths":{},"clips":{},"repairs":{}}"#.utf8))
    XCTAssertNil(draft.repairScopes)
    draft.repairScopes = ["c1": .states]
    let reopened = try JSONDecoder().decode(DirectorReviewDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertEqual(reopened.repairScopes?["c1"], .states)
  }
  func testStartedAndImportedJobsCannotChangeReviewModeEvenWithoutAResult() throws {
    var session = DirectorSession()
    session.definition = ["id": .string("weetodd.guided-movie-planning"), "version": .string("1.1.0"), "steps": .array([])]
    XCTAssertFalse(session.reviewModeLocked)
    session.executionStarted = true
    XCTAssertTrue(session.reviewModeLocked)
    XCTAssertThrowsError(try session.setReviewMode(.detailed))
  }
}


extension DirectorSessionTests {
  @MainActor func testLargeCheckpointRefreshAndImportKeepDefinitionImportBounded() async throws {
    let dir = try directory(), controller = DirectorSessionController()
    let definition: [String: JSONValue] = ["format": .string("weetodd-workflow-v1"), "id": .string("same"), "steps": .array([.object(["id": .string("one")])])]
    let job = WorkflowJob(definition: definition, inputs: [:], models: [:], assets: [:], runDirectory: dir.path)
    let jobURL = dir.appendingPathComponent("job.json")
    try JSONEncoder().encode(job).write(to: jobURL)
    let evidence = String(repeating: "Verified word and beat evidence. ", count: 70_000)
    let envelope: [String: JSONValue] = ["format": .string("weetodd-workflow-run-v1"), "workflowID": .string("same"), "status": .string("completed"), "totalSeconds": .integer(1), "revision": .string("large-reviewed"), "steps": .object([:]), "outputs": .object([:]), "definition": .object(definition), "inputs": .object([:]), "evidence": .string(evidence)]
    try JSONEncoder().encode(envelope).write(to: dir.appendingPathComponent("run.json"))
    controller.state.runDirectory = dir.path
    await controller.refreshCheckpoint()
    XCTAssertEqual(controller.state.result?.revision, "large-reviewed")
    try await controller.importJob(jobURL, validateDefinition: { _ in })
    XCTAssertEqual(controller.state.result?.revision, "large-reviewed")
    var oversizedJob = job
    oversizedJob.inputs["brief"] = .string(evidence)
    try JSONEncoder().encode(oversizedJob).write(to: jobURL)
    do { try await controller.importJob(jobURL, validateDefinition: { _ in }); XCTFail("Oversized job accepted") } catch {}
    XCTAssertEqual(controller.state.result?.revision, "large-reviewed")
  }
}

extension DirectorSessionTests {
  private func inputChangeSession() throws -> DirectorSession {
    var session = DirectorSession()
    let subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: Data(#"{"id":"ada","name":"Ada","kind":"character","description":"Red coat","aliases":[],"evidence":[],"suggestions":[],"referenceAssets":["asset:a"]}"#.utf8))
    let clip = try JSONDecoder().decode(WorkflowClipDraft.self, from: Data(#"{"id":"c1","startFrame":0,"frameCount":120,"action":"Walk","startState":"Door","endState":"Desk","location":"Office","characters":["ada"],"continuity":"cut"}"#.utf8))
    let outputs: [String: JSONValue] = ["subjects": try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode([subject])), "clips": .object(["fps": .integer(24), "totalFrames": .integer(120), "characters": .array([]), "clips": .array([try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(clip))])])]
    let envelope: [String: JSONValue] = ["status": .string("paused"), "totalSeconds": .integer(1), "revision": .string("saved"), "steps": .object(["review": .object(["name": .string("Review objects and clips"), "status": .string("completed"), "outputs": .object(outputs)])]), "outputs": .object([:])]
    session.result = try JSONDecoder().decode(WorkflowRunSummary.self, from: JSONEncoder().encode(envelope))
    session.fields = ["engine": "ltx25", "minimum_seconds": "1"]
    session.reviewAssetBindings = ["asset:a": "/original.png"]
    var draft = DirectorReviewDraft()
    draft.subjects = ["ada": subject]; draft.referencePaths = ["ada": ["/original.png"]]; draft.clips = ["c1": clip]
    draft.selectedSubject = "ada"; draft.repairs = ["c1": "Keep the original camera direction"]
    draft.repairScopes = ["c1": .states]; draft.repairBaselines = ["c1": clip]
    session.reviews = ["review": draft]
    return session
  }
  func testInputChangesClearOnlyCleanReviewCachesAndAllowFollowingEdits() throws {
    var session = try inputChangeSession()
    XCTAssertFalse(session.hasUnsavedReviews)
    try session.changeInputs { $0.fields["engine"] = "drawThings" }
    XCTAssertNil(session.result)
    XCTAssertFalse(session.hasUnsavedReviews)
    XCTAssertTrue(session.reviews["review"]?.subjects.isEmpty == true)
    XCTAssertTrue(session.reviews["review"]?.referencePaths.isEmpty == true)
    XCTAssertTrue(session.reviews["review"]?.clips.isEmpty == true)
    XCTAssertEqual(session.reviews["review"]?.selectedSubject, "ada")
    XCTAssertEqual(session.reviews["review"]?.repairs["c1"], "Keep the original camera direction")
    XCTAssertEqual(session.reviews["review"]?.repairScopes?["c1"], .states)
    XCTAssertEqual(session.reviews["review"]?.repairBaselines?["c1"]?.action, "Walk")
    try session.changeInputs { $0.fields["minimum_seconds"] = "4" }
    XCTAssertEqual(session.fields["engine"], "drawThings")
    XCTAssertEqual(session.fields["minimum_seconds"], "4")
    XCTAssertFalse(session.hasUnsavedReviews)
  }
  func testGenuineSubjectReferenceAndClipEditsBlockInputChangesWithoutLoss() throws {
    for change in ["subject", "references", "clip"] {
      var session = try inputChangeSession()
      if change == "subject" { session.reviews["review"]?.subjects["ada"]?.description = "Blue coat" }
      if change == "references" { session.reviews["review"]?.referencePaths["ada"] = ["/new.png"] }
      if change == "clip" { session.reviews["review"]?.clips["c1"]?.action = "Run" }
      let before = try JSONEncoder().encode(session)
      XCTAssertTrue(session.hasUnsavedReviews, change)
      XCTAssertThrowsError(try session.changeInputs { $0.fields["engine"] = "drawThings" }, change)
      let after = try JSONEncoder().encode(session)
      XCTAssertEqual(try JSONDecoder().decode(JSONValue.self, from: before), try JSONDecoder().decode(JSONValue.self, from: after), change)
    }
  }
}
