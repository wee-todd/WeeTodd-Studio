import XCTest
import StudioCore
@testable import WeeToddStudio

final class MusicModelStoreTests: XCTestCase {
  @MainActor func testLegacyMusicPathMigratesToRuntimeAndPreservesTake() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    defer { store.invalidateTimelinePlayback() }
    var draft = MusicDraft(); draft.modelPath = "/models/legacy-yue"; draft.vaePath = "/models/vae"
    store.project.musicDraft = draft
    var asset = MediaAsset(name: "Old take", kind: .audio, path: "/music.wav")
    asset.musicGeneration = MusicGeneration(draft: draft, request: try draft.request(), artifacts: "/take")
    store.project.assets = [asset]
    store.openMusic()
    XCTAssertEqual(store.project.musicDraft?.modelPath, "")
    XCTAssertEqual(store.project.assets[0].musicGeneration?.request.modelPath, "/models/legacy-yue")
    XCTAssertEqual(try store.resolvedMusicRequest(store.project.musicDraft!).vaePath, "/models/vae")
    XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("runtime.json").path))
    store.migrateMusicModelLocation()
    XCTAssertEqual(store.installedMusicModels.count, 1)
  }
  @MainActor func testUseReferenceScoreIsExplicitUndoableAndRejectsChangedRange() {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    defer { store.invalidateTimelinePlayback() }
    var draft = MusicDraft(); draft.abc = "old"
    var reference = MusicReference(path: "/song.wav")
    reference.analysis = MusicReferenceAnalysis(abc: "X:1\nK:C\nCDEF", artifacts: "/analysis", sourceSHA256: "hash",
      sourceIn: 0, duration: 30, melodyOnly: true, eventCount: 4)
    draft.reference = reference; store.project.musicDraft = draft
    XCTAssertEqual(store.project.musicDraft?.abc, "old")
    store.useMusicReferenceScore()
    XCTAssertEqual(store.project.musicDraft?.abc, reference.analysis?.abc)
    XCTAssertEqual(store.project.musicDraft?.cot, "melody")
    store.undo()
    XCTAssertEqual(store.project.musicDraft?.abc, "old")
    store.project.musicDraft?.reference?.sourceIn = 5
    store.useMusicReferenceScore()
    XCTAssertEqual(store.project.musicDraft?.abc, "old")
    XCTAssertNotNil(store.error)
  }
  @MainActor func testLateReferenceAnalysisPreservesChangedDraft() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    var store: StudioStore!
    let invocation: Bridge.Invocation = { command, _, _, _ in
      XCTAssertEqual(command, "music-reference-analyze")
      store.project.musicDraft?.direction = "Changed while analyzing"
      return ["abc": "X:1\nK:C\nCDEF", "artifacts": "/saved-analysis", "source_sha256": "hash"]
    }
    store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: invocation)
    defer { store.invalidateTimelinePlayback() }
    let encoder = InstalledMusicModel(kind: .sheetsage, name: "Encoder", path: "/model")
    var library = MusicModelSettings(); library.models = [encoder]; store.runtime.musicModels = library
    var draft = MusicDraft(); var reference = MusicReference(path: "/song.wav"); reference.encoderID = encoder.id
    draft.reference = reference; draft.abc = "original"; store.project.musicDraft = draft
    await store.analyzeMusicReference()
    XCTAssertEqual(store.project.musicDraft?.abc, "original")
    XCTAssertNil(store.project.musicDraft?.reference?.analysis)
    XCTAssertEqual(store.project.musicDraft?.direction, "Changed while analyzing")
    XCTAssertTrue(store.notice.contains("preserved"))
  }

  @MainActor func testTakePinsImplicitModelBeforeGlobalDefaultChanges() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { _, _, _, _ in
      ["audio": "/take.wav", "duration": 8.0, "sample_rate": 48000, "channels": 2, "artifacts": "/take"]
    })
    defer { store.invalidateTimelinePlayback() }
    let first = InstalledMusicModel(kind: .generator, name: "A", path: "/model/A")
    let second = InstalledMusicModel(kind: .generator, name: "B", path: "/model/B")
    var settings = MusicModelSettings(); settings.register(first); store.runtime.musicModels = settings
    store.project.musicDraft = MusicDraft()
    await store.generateMusic()
    let take = try XCTUnwrap(store.project.assets.last?.musicGeneration)
    settings.register(second); store.runtime.musicModels = settings
    XCTAssertEqual(try store.resolvedMusicRequest(take.draft).modelPath, first.path)
  }

}
