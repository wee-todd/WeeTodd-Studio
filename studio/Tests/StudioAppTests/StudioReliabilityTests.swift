import Foundation
import CryptoKit
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
  @MainActor func testPreparedReferenceAttachesAtomicallyAndUndoRestoresShot() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: { command, _, _, _ in
        XCTAssertEqual(command, "prepare-reference")
        return ["path": "/tmp/reference-sheet.png", "kind": "image", "width": 1152, "height": 480]
      })
    store.addClip()
    store.editClip { $0.engine = .ltx25 }
    let source = MediaAsset(name: "Story", kind: .video, path: "/tmp/story.mov")
    store.change { $0.assets.append(source) }
    let before = store.project
    let action = try XCTUnwrap(store.selectedClip?.referenceActions(for: source).first)
    await store.useReference(source, action: action)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.inferredTask, "control")
    XCTAssertEqual(store.selectedClip?.attachments.first?.controlType, "ingredients_reference_sheet")
    XCTAssertEqual(store.project.assets.count, before.assets.count + 1)
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testLateReferencePreparationDoesNotAttachAfterModelChange() async throws {
    let fake = SuspendedBridge("prepare-reference")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.engine = .ltx25 }
    let source = MediaAsset(name: "Story", kind: .video, path: "/tmp/story.mov")
    store.change { $0.assets.append(source) }
    let action = try XCTUnwrap(store.selectedClip?.referenceActions(for: source).first)
    let entered = expectation(description: "reference preparation suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.useReference(source, action: action) }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.engine = .h3 }
    fake.continuation?.resume(returning: ["path": "/tmp/reference-sheet.png", "kind": "image"])
    await task.value
    XCTAssertTrue(store.selectedClip?.attachments.isEmpty == true)
    XCTAssertEqual(store.project.assets.last?.scope, .project)
    XCTAssertEqual(store.project.assets.last?.path, "/tmp/reference-sheet.png")
  }

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
    store.previewMode = "Timeline"
    store.seekToEnd()
    XCTAssertEqual(store.playhead, 10)
  }

  @MainActor func testMoviePreviewAcceptsUnchangedMultiShotProject() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: { _, _, _, _ in [:] })
    for index in 1...6 {
      var clip = Clip(name: "Shot \(index)", engine: .h3)
      clip.selectLocalModel(.ltx25)
      clip.duration = 5
      clip.sourcePath = "/tmp/shot-\(index).mp4"
      clip.continuity = ClipContinuity(mode: index == 1 ? "independent" : "frame")
      store.project.clips.append(clip)
    }
    let snapshot = store.project
    await store.previewMovie()
    XCTAssertEqual(store.project, snapshot)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.previewMode, "Movie")
    XCTAssertEqual(store.effectivePreviewDuration, 30)
  }

  @MainActor func testMoviePreviewRejectsEditsWhileRendering() async throws {
    let fake = SuspendedBridge("preview")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "preview suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.previewMovie() }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.duration = 7 }
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertNotEqual(store.previewMode, "Movie")
    XCTAssertNotNil(store.error)
  }

  @MainActor func testMoviePreviewRejectsReopenedIdenticalDocument() async throws {
    let directory = try temporaryDirectory()
    let fake = SuspendedBridge("preview")
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let url = directory.appendingPathComponent("same-project.weetodd")
    try ProjectStorage.write(store.project, to: url)
    let entered = expectation(description: "preview suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.previewMovie() }
    await fulfillment(of: [entered], timeout: 2)
    store.load(url)
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertNotEqual(store.previewMode, "Movie")
    XCTAssertNotNil(store.error)
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

  @MainActor func testNativeDurationRoundingKeepsAcceptedRenderCurrent() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(.h3)
    store.editClip { $0.duration = 5.17; $0.prompt = "A robot lifts a lantern." }
    await store.prepareSelected()
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov",
      "usable_source_in": 0.0, "usable_duration": 124.0 / 24])
    await task.value
    await store.describeGeneration()
    let clip = try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(clip.duration, 124.0 / 24)
    XCTAssertEqual(clip.renderedSignature, store.signature(for: clip))
  }

  @MainActor func testDurationRefreshCannotMarkAReplacedTakeCurrent() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(.h3)
    store.editClip { $0.duration = 5.17; $0.prompt = "A robot lifts a lantern." }
    await store.prepareSelected()
    let rendered = expectation(description: "render suspended")
    fake.entered = { rendered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [rendered], timeout: 2)
    let refreshed = expectation(description: "duration refresh suspended")
    fake.command = "describe-generation"
    fake.entered = { refreshed.fulfill() }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov",
      "usable_source_in": 0.0, "usable_duration": 124.0 / 24])
    await fulfillment(of: [refreshed], timeout: 2)
    store.editClip { $0.sourcePath = "/tmp/older-take.mov"; $0.renderedSignature = "" }
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/older-take.mov")
    XCTAssertEqual(store.selectedClip?.renderedSignature, "")
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

extension StudioReliabilityTests {
  @MainActor func testPredecessorTrimInvalidatesSuspendedContinuityPreparation() async throws {
    let fake = SuspendedBridge("prepare")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip { $0.continuity = ClipContinuity(mode: "frame") }
    let oldKey = store.generationRequestKey(for: store.selectedClip!)
    let entered = expectation(description: "prepare suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    store.change { $0.clips[0].sourceIn = 1 }
    XCTAssertNotEqual(oldKey, store.generationRequestKey(for: store.selectedClip!))
    fake.continuation?.resume(returning: ["recipePath": "/tmp/prepared/recipe.json", "prompt": "prompt", "report": [:]])
    await task.value
    XCTAssertNil(store.preparedRecipe)
  }

  @MainActor func testPredecessorTakeChangeRetainsLateContinuityRenderAsInactiveVersion() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip { $0.continuity = ClipContinuity(mode: "frame") }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.change { $0.clips[0].sourcePath = "/tmp/new-accepted.mov" }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov", "usable_source_in": 2.0,
      "usable_duration": 4.0, "continuation_artifact": ["manifest": "/tmp/context/manifest.json", "manifest_sha256": "a", "payload_sha256": "b"]])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
    XCTAssertEqual(store.selectedClip?.duration, 5)
    let version = try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertEqual(version.path, "/tmp/completed.mov")
    XCTAssertEqual(version.usableSourceIn, 2)
    XCTAssertEqual(version.usableDuration, 4)
    XCTAssertEqual(version.continuationArtifact?.manifest, "/tmp/context/manifest.json")
  }
}

extension StudioReliabilityTests {
  @MainActor func testFrameContinuityDoesNotRequireReplacedStoredFirstAttachment() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip {
      $0.continuity = ClipContinuity(mode: "frame")
      $0.attachments = [Attachment(assetID: UUID(), role: .first)]
    }
    XCTAssertFalse(store.issues(for: store.selectedClip!).contains("Relink a missing attachment"))
    XCTAssertEqual(store.selectedClip?.attachments.count, 1)
  }
}

extension StudioReliabilityTests {
  @MainActor func testDrawThingsSignaturePreservesLegacyPartsWithoutContinuity() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.engine = .drawThings }
    let clip = try XCTUnwrap(store.selectedClip)
    let parts = [clip.generationFingerprint, "generationFPS:\(clip.settings(in: store.project).fps)"]
    let expected = SHA256.hash(data: Data(parts.joined(separator: "\n").utf8))
      .map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(store.signature(for: clip), expected)
  }
}
