import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

@MainActor private final class SceneBridgeFixture {
  var scene: [String: Any] = [:]
  var video = ""
  var anchor = ""
  var suspend = false
  var entered: (() -> Void)?
  var continuation: CheckedContinuation<[String: Any], Error>?
  var calls: [String] = []
  var renderOutputs: [URL?] = []
  func invoke(_ command: String, _ runtime: RuntimeSettings, _ payload: [String: Any], _ output: URL?) async throws -> [String: Any] {
    calls.append(command)
    switch command {
    case "describe-generation":
      let project = payload["project"] as? [String: Any] ?? [:]
      let clips = project["clips"] as? [[String: Any]] ?? []
      return ["fingerprint": "scene-resolved-" + clips.map { String(describing: $0["duration"]) }.joined(),
        "sourcePaths": ["/tmp/scene-component.weights"], "readinessErrors": []]
    case "prepare": return ["recipePath": "/tmp/scene/prepared/recipe.json", "prompt": "Entire scene prompt", "report": ["scene": scene, "resolvedFingerprint": "scene-resolved"]]
    case "render":
      renderOutputs.append(output)
      if suspend {
        return try await withCheckedThrowingContinuation { continuation = $0; entered?() }
      }
      return ["video": video, "scene": scene]
    case "inspect": return ["duration": 10.0, "fps": 24.0, "num_frames": 240]
    case "freeze-continuity-frame": return ["path": anchor]
    default: return [:]
    }
  }
}

final class ContinuousSceneInteractionTests: XCTestCase {
  @MainActor func testRetryKeepsCandidateFilesSeparateAndDefersAcceptanceWhileRendering() async throws {
    let (store, bridge) = try fixture()
    await store.generateSelected()
    let previous = try XCTUnwrap(store.pendingContinuousScene)
    bridge.suspend = true
    let started = expectation(description: "Retry started")
    bridge.entered = { started.fulfill() }
    let retry = Task { await store.renderPrepared() }
    await fulfillment(of: [started], timeout: 2)
    XCTAssertEqual(store.pendingContinuousScene?.id, previous.id)
    XCTAssertFalse(store.canAcceptContinuousScene)
    XCTAssertEqual(bridge.renderOutputs.count, 2)
    XCTAssertNotEqual(bridge.renderOutputs[0], bridge.renderOutputs[1])
    bridge.continuation?.resume(returning: ["video": bridge.video, "scene": bridge.scene])
    await retry.value
    XCTAssertTrue(store.canAcceptContinuousScene)
    XCTAssertNotEqual(store.pendingContinuousScene?.id, previous.id)
  }

  @MainActor func testReviewSummaryShowsWholeSceneRequestedAndResolvedTiming() throws {
    let clip = Clip(name: "Second shot", engine: .ltx25)
    let scene: [String: Any] = ["version": 1, "frame_rate": 24.0,
      "publication_mode": "single_decode_native_latent_chain", "members": [
        ["clip_id": UUID().uuidString, "source_in": 0.0, "duration": 5.0],
        ["clip_id": clip.id.uuidString, "source_in": 5.0, "duration": 5.0]]]
    let json = try JSONSerialization.data(withJSONObject: ["scene": scene,
      "scenePlan": ["requested_durations": [5.1, 5.0]]])
    let summary = RenderSettingsSummary(clip: clip, report: String(decoding: json, as: UTF8.self))
    XCTAssertEqual(summary.requestedDuration, 10.1, accuracy: 0.0001)
    XCTAssertEqual(summary.sceneReport?.duration, 10.0)
    XCTAssertEqual(summary.sceneReport?.members.count, 2)
    XCTAssertEqual(RenderSettingsSummary(clip: clip, report: "{}").requestedDuration, clip.duration)
  }

  @MainActor private func fixture() throws -> (StudioStore, SceneBridgeFixture) {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
    let bridge = SceneBridgeFixture()
    bridge.video = folder.appendingPathComponent("scene.mp4").path
    try Data("test-artifact".utf8).write(to: URL(fileURLWithPath: bridge.video))
    let store = StudioStore(dataDirectory: folder, restoreSession: false, invocation: bridge.invoke)
    var first = Clip(name: "Start", engine: .ltx25)
    first.prompt = "Robot raises the lantern."
    first.sourcePath = "/tmp/prior-start.mp4"
    var second = Clip(name: "Finish", engine: .ltx25)
    second.prompt = "Robot sets the lantern down."
    second.sourcePath = "/tmp/prior-finish.mp4"
    second.continuity = ClipContinuity(mode: "scene", sourceClipID: first.id)
    store.project.clips = [first, second]
    store.select(second.id)
    bridge.scene = ["version": 1, "frame_rate": 24.0,
      "publication_mode": "single_decode_native_latent_chain", "members": [
        ["clip_id": first.id.uuidString, "source_in": 0.0, "duration": 5.0],
        ["clip_id": second.id.uuidString, "source_in": 5.0, "duration": 5.0]]]
    return (store, bridge)
  }

  @MainActor func testInteriorMemberRendersOneReviewableSceneAndAcceptsAtomically() async throws {
    let (store, bridge) = try fixture()
    let old = store.project.clips
    await store.generateSelected()
    XCTAssertEqual(bridge.calls.filter { $0 == "render" }.count, 1)
    XCTAssertEqual(store.project.clips.map(\.sourcePath), old.map(\.sourcePath), "A scene render requires one explicit review/accept action")
    XCTAssertTrue(store.project.clips.allSatisfy { $0.versions.isEmpty })
    XCTAssertNotNil(store.pendingContinuousScene)
    XCTAssertTrue(store.canAcceptContinuousScene)
    let beforeAcceptance = store.project.clips
    await store.acceptContinuousScene()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.clips.map(\.sourcePath), [bridge.video, bridge.video])
    XCTAssertEqual(store.project.clips.map(\.sourceIn), [0, 5])
    XCTAssertEqual(store.project.clips.map { $0.versions.count }, [2, 2], "Retain the previous accepted sources as history")
    XCTAssertEqual(store.project.clips[0].versions.last?.sceneTakeID, store.project.clips[1].versions.last?.sceneTakeID)
    store.undo()
    XCTAssertEqual(store.project.clips, beforeAcceptance)
  }

  @MainActor func testOtherMemberEditInvalidatesPreparedScene() async throws {
    let (store, _) = try fixture()
    await store.prepareSelected()
    XCTAssertTrue(store.canGenerateSelected)
    store.change { $0.clips[0].prompt = "Changed opening" }
    XCTAssertFalse(store.canGenerateSelected)
  }

  @MainActor func testMemberEditAfterRenderPreventsAcceptance() async throws {
    let (store, _) = try fixture()
    await store.generateSelected()
    store.change { $0.clips[0].seed += 1 }
    XCTAssertFalse(store.canAcceptContinuousScene)
    await store.acceptContinuousScene()
    XCTAssertEqual(store.project.clips[0].sourcePath, "/tmp/prior-start.mp4")
    XCTAssertTrue(store.project.clips.allSatisfy { $0.versions.isEmpty })
    XCTAssertNotNil(store.error)
    XCTAssertNotNil(store.pendingContinuousScene, "Retain the artifact for review")
  }

  @MainActor func testLateSceneCannotPopulateReplacementDocument() async throws {
    let (store, bridge) = try fixture()
    bridge.suspend = true
    let entered = expectation(description: "scene generation entered")
    bridge.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 3)
    store.newProject()
    bridge.continuation?.resume(returning: ["video": bridge.video, "scene": bridge.scene])
    await task.value
    XCTAssertTrue(store.project.clips.isEmpty)
    XCTAssertNil(store.pendingContinuousScene)
    XCTAssertTrue(store.error?.contains(bridge.video) == true)
  }

  @MainActor func testEditingDifferentMemberWhileRenderingRetainsUnacceptedArtifact() async throws {
    let (store, bridge) = try fixture()
    bridge.suspend = true
    let entered = expectation(description: "scene generation entered")
    bridge.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 3)
    store.change { $0.clips[0].prompt = "New opening action" }
    bridge.continuation?.resume(returning: ["video": bridge.video, "scene": bridge.scene])
    await task.value
    XCTAssertNotNil(store.pendingContinuousScene)
    XCTAssertFalse(store.canAcceptContinuousScene)
    XCTAssertTrue(store.project.clips.allSatisfy { $0.versions.isEmpty })
  }

  @MainActor func testSceneResultMustMatchPreparedMemberRanges() async throws {
    let (store, bridge) = try fixture()
    await store.prepareSelected()
    bridge.scene["members"] = [["clip_id": UUID().uuidString, "source_in": 0.0, "duration": 10.0]]
    await store.renderPrepared()
    XCTAssertNil(store.pendingContinuousScene)
    XCTAssertTrue(store.project.clips.allSatisfy { $0.versions.isEmpty })
    XCTAssertNotNil(store.error)
  }

  @MainActor func testGroupedHistoryRestoresEveryShotAndRejectsIndividualOldTake() async throws {
    let (store, bridge) = try fixture()
    await store.generateSelected()
    await store.acceptContinuousScene()
    let firstTake = store.project.clips[1].versions.last!
    let originalPath = bridge.video
    let next = URL(fileURLWithPath: bridge.video).deletingLastPathComponent().appendingPathComponent("second.mp4")
    try Data("second-artifact".utf8).write(to: next)
    bridge.video = next.path
    await store.generateSelected()
    await store.acceptContinuousScene()
    XCTAssertEqual(store.project.clips.map(\.sourcePath), [next.path, next.path])
    store.activateRenderVersion(firstTake, for: store.project.clips[1])
    XCTAssertEqual(store.project.clips.map(\.sourcePath), [originalPath, originalPath])
    let before = store.project
    store.activateRenderVersion(store.project.clips[0].versions.first!, for: store.project.clips[0])
    XCTAssertEqual(store.project, before)
    XCTAssertTrue(store.error?.contains("Disconnect") == true)
  }

  @MainActor func testChangedSceneAssetInvalidatesReviewWithoutClipEdits() async throws {
    let (store, _) = try fixture()
    let image = store.dataDirectory.appendingPathComponent("anchor.png")
    try Data("first-image".utf8).write(to: image)
    let asset = MediaAsset(name: "Anchor", kind: .image, path: image.path)
    store.project.assets.append(asset)
    store.project.clips[0].attachments.append(Attachment(assetID: asset.id, role: .first))
    await store.generateSelected()
    XCTAssertTrue(store.canAcceptContinuousScene)
    try Data("changed-image-bytes".utf8).write(to: image)
    XCTAssertFalse(store.canAcceptContinuousScene)
  }

  @MainActor func testAcceptedQuantizedSceneStaysCurrentWhenEveryMemberIsSelected() async throws {
    let (store, _) = try fixture()
    store.project.clips[0].duration = 5.1
    store.project.clips[1].duration = 4.9
    await store.generateSelected()
    await store.acceptContinuousScene()
    XCTAssertNil(store.error)
    for id in store.project.clips.map(\.id) {
      store.select(id)
      let before = store.selectedClip!.renderedSignature
      await store.describeGeneration()
      XCTAssertEqual(store.signature(for: store.selectedClip!), before,
        "Selecting and describing a freshly accepted member must not invalidate its take")
    }
  }

  @MainActor func testFrameMatchConversionFreezesAnchorAndRetainsOriginalAsset() async throws {
    let (store, bridge) = try fixture()
    let firstID = store.project.clips[0].id
    let targetID = store.project.clips[1].id
    store.project.clips[0].sourcePath = bridge.video
    store.project.clips[1].continuity = ClipContinuity(mode: "frame", sourceClipID: firstID)
    store.project.clips[1].generationSelection = GenerationSelection()
    store.project.clips[1].generationSelection?.steps = 12
    let original = MediaAsset(name: "Original first image", kind: .image, path: "/tmp/original.png")
    store.project.assets.append(original)
    store.project.clips[1].attachments = [Attachment(assetID: original.id, role: .first)]
    let anchor = store.dataDirectory.appendingPathComponent("frozen.png")
    try Data("frozen-image".utf8).write(to: anchor)
    bridge.anchor = anchor.path
    await store.connectContinuousScene(clipID: targetID, preserveFrameMatch: true)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.clips[1].continuityMode, "scene")
    XCTAssertEqual(store.project.clips[1].continuity?.sourceClipID, firstID)
    XCTAssertTrue(store.project.assets.contains(original))
    let selected = store.project.clips[1].attachments.first { $0.role == .first }!
    XCTAssertNotEqual(selected.assetID, original.id)
    XCTAssertEqual(store.project.assets.first { $0.id == selected.assetID }?.path, anchor.path)
    XCTAssertEqual(bridge.calls.filter { $0 == "freeze-continuity-frame" }.count, 1)
    XCTAssertEqual(store.project.clips[1].generationSelection?.task, "i2v")
    XCTAssertEqual(store.project.clips[1].generationSelection?.steps, 12)
  }

  @MainActor func testFrameMatchConversionKeepsExplicitLastFrameTask() async throws {
    let (store, bridge) = try fixture()
    let targetID = store.project.clips[1].id
    store.project.clips[0].sourcePath = bridge.video
    store.project.clips[1].continuity = ClipContinuity(mode: "frame")
    store.project.clips[1].generationSelection = GenerationSelection()
    store.project.clips[1].generationSelection?.refinementSteps = 4
    let last = MediaAsset(name: "Last frame", kind: .image, path: "/tmp/last.png")
    store.project.assets.append(last)
    store.project.clips[1].attachments = [Attachment(assetID: last.id, role: .last)]
    let anchor = store.dataDirectory.appendingPathComponent("frozen.png")
    try Data("frozen-image".utf8).write(to: anchor)
    bridge.anchor = anchor.path
    await store.connectContinuousScene(clipID: targetID, preserveFrameMatch: true)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.clips[1].generationSelection?.task, "fflf")
    XCTAssertEqual(store.project.clips[1].generationSelection?.refinementSteps, 4)
  }
}
