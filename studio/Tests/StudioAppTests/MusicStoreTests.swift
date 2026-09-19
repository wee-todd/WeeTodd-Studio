import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class MusicStoreTests: XCTestCase {
  func directory() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
    return folder
  }
  @MainActor func testGenerationCapturesExactSettingsAndDoesNotPlaceWithoutChoice() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { command, _, body, _ in
        XCTAssertEqual(command, "music-generate")
        XCTAssertEqual((body["music"] as? [String: Any])?["steps"] as? Int, 32)
        return ["audio": "/tmp/song.wav", "duration": 60.0, "sample_rate": 48000,
                "channels": 2, "artifacts": "/tmp/take", "truncated": ["semantic": true]]
      })
    store.openMusic()
    store.change { $0.musicDraft?.modelPath = "/model" }
    await store.generateMusic()
    XCTAssertNil(store.error)
    let asset = try XCTUnwrap(store.project.assets.last)
    XCTAssertEqual(asset.kind, .audio)
    XCTAssertEqual(asset.musicGeneration?.request.modelPath, "/model")
    XCTAssertEqual(asset.musicGeneration?.truncated, true)
    XCTAssertTrue(store.project.audio.isEmpty)
    store.playhead = 4
    store.placeMusicTake(asset)
    XCTAssertEqual(store.project.audio.first?.start, 4)
  }
  @MainActor func testLateMusicResultDoesNotMutateReplacedDocument() async throws {
    let entered = expectation(description: "music request started")
    var continuation: CheckedContinuation<[String: Any], Error>?
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { _, _, _, _ in
        try await withCheckedThrowingContinuation { continuation = $0; entered.fulfill() }
      })
    store.openMusic(); store.change { $0.musicDraft?.modelPath = "/model" }
    let task = Task { await store.generateMusic() }
    await fulfillment(of: [entered], timeout: 2)
    store.project = StudioProject()
    continuation?.resume(returning: ["audio": "/tmp/song.wav", "duration": 60.0,
      "sample_rate": 48000, "channels": 2, "artifacts": "/tmp/take"])
    await task.value
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.notice.contains("/tmp/song.wav"))
  }
  @MainActor func testUnsupportedVideoBackendCannotReceiveMusicDriver() async throws {
    var calls = 0
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { _, _, _, _ in calls += 1; return [:] })
    var clip = Clip(engine: .drawThings); clip.duration = 5
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var asset = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); asset.duration = 60
    await store.useMusicDriver(asset, sourceIn: 12, muteClipAudio: false)
    XCTAssertEqual(calls, 0)
    XCTAssertTrue(store.selectedClip?.attachments.isEmpty == true)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testDownloadDoesNotOverwriteModelEditedWhileWaiting() async throws {
    let entered = expectation(description: "download started")
    var continuation: CheckedContinuation<[String: Any], Error>?
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { _, _, _, _ in
        try await withCheckedThrowingContinuation { continuation = $0; entered.fulfill() }
      })
    store.openMusic(); store.change { $0.musicDraft?.modelPath = "/first" }
    let task = Task { await store.downloadMusicModel(to: URL(fileURLWithPath: "/library")) }
    await fulfillment(of: [entered], timeout: 2)
    store.change { $0.musicDraft?.modelPath = "/chosen"; $0.musicDraft?.precision = "bf16" }
    continuation?.resume(returning: ["model_path": "/downloaded"])
    await task.value
    XCTAssertEqual(store.project.musicDraft?.modelPath, "/chosen")
    XCTAssertEqual(store.project.musicDraft?.precision, "bf16")
    XCTAssertTrue(store.notice.contains("/downloaded"))
  }

  @MainActor func testScoreOnlyUpdatesDraftWithoutCreatingAudioAsset() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { command, _, body, _ in
        XCTAssertEqual(command, "music-plan")
        XCTAssertNil((body["music"] as? [String: Any])?["abc"])
        return ["abc": "X:1\nK:C\nCDEF|"]
      })
    store.openMusic(); store.change { $0.musicDraft?.modelPath = "/model" }
    await store.planMusic()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.musicDraft?.abc, "X:1\nK:C\nCDEF|")
    XCTAssertTrue(store.project.assets.isEmpty)
  }

  @MainActor func testResynthesisUsesSavedCompositionAndCurrentAcousticControls() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { command, _, body, _ in
        XCTAssertEqual(command, "music-resynthesize")
        XCTAssertEqual(body["source_artifacts"] as? String, "/saved/take")
        XCTAssertEqual(body["steps"] as? Int, 8)
        XCTAssertEqual(body["seed"] as? Int, 123)
        return ["audio": "/tmp/new.wav", "duration": 60.0, "sample_rate": 48000,
                "channels": 2, "artifacts": "/tmp/replay"]
      })
    var original = MusicDraft(); original.modelPath = "/model"; original.genre = "Jazz"
    var take = MediaAsset(name: "Original", kind: .audio, path: "/tmp/song.wav")
    take.musicGeneration = MusicGeneration(draft: original, request: try original.request(), artifacts: "/saved/take")
    store.openMusic()
    store.change { $0.musicDraft?.modelPath = "/model"; $0.musicDraft?.steps = 8; $0.musicDraft?.seed = 123 }
    await store.generateMusic(reusing: take)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.assets.last?.musicGeneration?.draft.genre, "Jazz")
    XCTAssertEqual(store.project.assets.last?.musicGeneration?.request.steps, 8)
    XCTAssertEqual(store.project.assets.last?.musicGeneration?.request.seed, 123)
  }
}
