import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

@MainActor private final class SuspendedBridge {
  var command: String
  var suspendLimit = Int.max
  var inspectRuntimeRoot: String?
  private var suspensionCount = 0
  var calls: [(String, [String: Any])] = []
  var continuation: CheckedContinuation<[String: Any], Error>?
  var entered: (() -> Void)?
  init(_ command: String) { self.command = command }
  func invoke(_ name: String, _ runtime: RuntimeSettings, _ payload: [String: Any], _ output: URL?) async throws -> [String: Any] {
    calls.append((name, payload))
    if name == command && suspensionCount < suspendLimit {
      suspensionCount += 1
      return try await withCheckedThrowingContinuation { continuation in
        self.continuation = continuation
        entered?()
      }
    }
    if name == "inspect" {
      if let inspectRuntimeRoot, runtime.root != inspectRuntimeRoot {
        throw StudioError.invalid("Wrong runtime for completed output")
      }
      return ["duration": payload["path"] as? String == "/tmp/context.mov" ? 8.0 : 12.0]
    }
    if name == "prepare" { return ["recipePath": "/tmp/prepared/recipe.json", "prompt": "prompt", "report": [:]] }
    return [:]
  }
}

final class StudioReliabilityTests: XCTestCase {
  func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    return directory
  }

  @MainActor func testOpenSeparatesUndoAndPreservesDepartingDirtyDocument() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Unsaved A" }
    let a = store.project
    var b = StudioProject(); b.name = "Saved B"
    let url = directory.appendingPathComponent("B.weetodd")
    try ProjectStorage.write(b, to: url)
    store.load(url)
    XCTAssertFalse(store.canUndo)
    store.undo(); store.save()
    XCTAssertEqual(try ProjectStorage.read(url).name, "Saved B")
    let recovered = (FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? [])
      .filter { $0.pathExtension == "weetodd" }.compactMap { try? ProjectStorage.read($0) }
    XCTAssertTrue(recovered.contains(a), "Departed dirty content must be recoverable even before debounce fires")
  }

  @MainActor func testNewProjectClearsUndoAndSelections() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.selectedAssetID = UUID(); store.selectedAudioID = UUID()
    store.newProject()
    XCTAssertFalse(store.canUndo)
    XCTAssertNil(store.selectedAssetID); XCTAssertNil(store.selectedAudioID)
    store.undo()
    XCTAssertTrue(store.project.clips.isEmpty)
  }

  @MainActor func testSuspendedPreflightNeverPreparesAnotherSelection() async throws {
    let fake = SuspendedBridge("describe-generation")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.addClip()
    let first = store.project.clips[0].id
    let entered = expectation(description: "description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    store.select(first)
    fake.continuation?.resume(returning: ["fingerprint": "resolved"])
    await task.value
    XCTAssertFalse(fake.calls.contains { $0.0 == "prepare" })
    XCTAssertNil(store.preparedRecipe)
    XCTAssertTrue(store.validationErrors.isEmpty)
  }

  @MainActor func testLateNativeRenderDoesNotPromoteOverEdits() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.prompt = "Original"; $0.sourcePath = "/tmp/original.mov" }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.prompt = "Revised"; $0.sourceIn = 2; $0.duration = 3 }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/original.mov")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2)
    XCTAssertEqual(store.selectedClip?.duration, 3)
    XCTAssertEqual(store.selectedClip?.versions.last?.path, "/tmp/completed.mov")
  }

  @MainActor func testLateNativeRenderCannotAttachToReopenedCopy() async throws {
    let directory = try temporaryDirectory()
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let url = directory.appendingPathComponent("copy.weetodd")
    try ProjectStorage.write(store.project, to: url)
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.load(url)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertTrue(store.selectedClip?.versions.isEmpty == true)
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
    XCTAssertTrue(store.error?.contains("/tmp/completed.mov") == true)
  }
  @MainActor func testMovieTransportEndUsesFullMovieDuration() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.duration = 3 }
    store.addClip(); store.editClip { $0.duration = 7 }
    store.select(store.project.clips[0].id)
    store.previewMode = "Movie"
    store.seekToEnd()
    XCTAssertEqual(store.effectivePreviewDuration, 10)
    XCTAssertEqual(store.playhead, 10)
    store.previewMode = "Clip"
    store.seekToEnd()
    XCTAssertEqual(store.playhead, 3)
  }

  @MainActor func testFailedRecoveryKeepsCurrentDocumentAndURL() throws {
    let directory = try temporaryDirectory()
    try Data("occupied".utf8).write(to: directory.appendingPathComponent("Recovery"))
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let originalURL = directory.appendingPathComponent("original.weetodd")
    store.projectURL = originalURL
    store.change { $0.name = "Must survive" }
    let before = store.project
    store.newProject()
    XCTAssertEqual(store.project, before)
    XCTAssertEqual(store.projectURL, originalURL)
    XCTAssertTrue(store.dirty)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testDeletedPreflightDestinationReturnsSafelyAndBlocksDuplicateRequest() async throws {
    let fake = SuspendedBridge("describe-generation")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    XCTAssertTrue(store.operationBusy)
    await store.prepareSelected()
    store.deleteClip()
    fake.continuation?.resume(returning: ["fingerprint": "resolved"])
    await task.value
    XCTAssertEqual(fake.calls.map { $0.0 }, ["describe-generation"])
    XCTAssertNil(store.preparedRecipe)
    XCTAssertFalse(store.operationBusy)
  }

  @MainActor func testDeletedRenderDestinationReportsCompletedOutput() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.deleteClip()
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.error?.contains("/tmp/completed.mov") == true)
  }

  @MainActor func testPreflightCanStartWhileBackgroundDescriptionIsPending() async throws {
    let fake = SuspendedBridge("describe-generation"); fake.suspendLimit = 1
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "background description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.describeGeneration() }
    await fulfillment(of: [entered], timeout: 2)
    await store.prepareSelected()
    XCTAssertNotNil(store.preparedRecipe)
    fake.continuation?.resume(returning: [:])
    await task.value
  }

  @MainActor func testSuccessfulAppendRenderPersistsUsableSegment() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.editClip { $0.extensionSource = "/tmp/context.mov"; $0.extensionDirection = "after" }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourceIn, 8)
    XCTAssertEqual(store.selectedClip?.duration, 4)
    XCTAssertEqual(store.selectedClip?.versions.last?.usableSourceIn, 8)
    XCTAssertEqual(store.selectedClip?.versions.last?.usableDuration, 4)
  }

  @MainActor func testLateNativeRenderUsesSubmittedRuntimeToInspectOutput() async throws {
    let fake = SuspendedBridge("render"); fake.inspectRuntimeRoot = "/runtime/submitted"
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.runtime.root = "/runtime/submitted"
    store.addClip()
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.runtime.root = "/runtime/next"
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.versions.last?.path, "/tmp/completed.mov")
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
  }

  @MainActor func testOpeningProjectImmediatelyReplacesActiveRestoreSnapshot() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Dirty A" }
    let departed = store.project
    try ProjectStorage.write(departed, to: directory.appendingPathComponent("Autosave.weetodd"))
    var opened = StudioProject(); opened.name = "Saved B"
    let url = directory.appendingPathComponent("B.weetodd")
    try ProjectStorage.write(opened, to: url)
    store.load(url)
    let restarted = StudioStore(dataDirectory: directory, restoreSession: false)
    restarted.restoreAutosavedProject()
    XCTAssertEqual(restarted.project, opened)
    let recovery = try FileManager.default.contentsOfDirectory(at: directory.appendingPathComponent("Recovery"), includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "weetodd" }.map { try ProjectStorage.read($0) }
    XCTAssertTrue(recovery.contains(departed))
  }

  @MainActor func testNewMovieImmediatelyReplacesActiveRestoreSnapshot() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Dirty A" }
    try ProjectStorage.write(store.project, to: directory.appendingPathComponent("Autosave.weetodd"))
    store.newProject()
    let blank = store.project
    let restarted = StudioStore(dataDirectory: directory, restoreSession: false)
    restarted.restoreAutosavedProject()
    XCTAssertEqual(restarted.project, blank)
  }

  @MainActor func testGenerateUsesAlreadyReviewedPreparedRecipe() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/reviewed/recipe.json"
    store.preparedPrompt = "The reviewed prompt."
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertFalse(fake.calls.contains { ["describe-generation", "prepare"].contains($0.0) })
    XCTAssertEqual(store.selectedClip?.versions.last?.recipePath, "/tmp/reviewed/recipe.json")
    XCTAssertEqual(store.selectedClip?.versions.last?.prompt, "The reviewed prompt.")
  }

  @MainActor func testGenerateRepreparesStaleRecipe() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/reviewed/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    store.runtime.root = "/runtime/changed"
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(fake.calls.prefix(3).map { $0.0 }, ["describe-generation", "prepare", "render"])
    XCTAssertEqual(store.selectedClip?.versions.last?.recipePath, "/tmp/prepared/recipe.json")
  }

}
