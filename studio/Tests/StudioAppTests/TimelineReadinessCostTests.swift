import Foundation
import Combine
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TimelineReadinessCostTests: XCTestCase {
  @MainActor func testPlaybackTicksDoNotInvalidateTheWholeEditor() {
    let store = StudioStore(restoreSession: false)
    store.project.clips = [Clip(engine: .ltx25)]
    store.isPlaying = true
    var editorUpdates = 0
    var positionUpdates = 0
    let observation = store.objectWillChange.sink { editorUpdates += 1 }
    let positionObservation = store.playbackPosition.objectWillChange.sink { positionUpdates += 1 }
    for tick in 1...30 { store.updatePlaybackPosition(Double(tick) / 30) }
    XCTAssertEqual(store.playhead, 1, accuracy: 0.0001)
    XCTAssertEqual(editorUpdates, 0, "Playback positions must not invalidate inspectors, assets and every timeline tile")
    XCTAssertEqual(positionUpdates, 30, "Viewport, titles and the playhead must continue receiving the full playback clock")
    store.pausePlayback()
    let pausedUpdates = editorUpdates
    store.seek(0.5)
    XCTAssertGreaterThan(editorUpdates, pausedUpdates, "Manual seeking must still refresh position-dependent editor controls")
    withExtendedLifetime((observation, positionObservation)) {}
  }

  @MainActor func testPlaybackTileBadgesDoNotRepeatModelFileValidationOnEveryTick() {
    let store = StudioStore(restoreSession: false)
    store.project.clips = (0..<50).map { Clip(name: "Shot \($0)", engine: .ltx25) }
    store.profiles = (0..<50).map {
      ModelProfile(id: "/nonexistent/tile-cost-\($0).json", name: "Model \($0)", engine: "ltx25", task: "i2v")
    }
    store.isPlaying = true
    for clip in store.project.clips { _ = store.timelineClipState(clip) }
    let start = ProcessInfo.processInfo.systemUptime
    for _ in 0..<10 {
      for clip in store.project.clips { _ = store.timelineClipState(clip) }
    }
    XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - start, 0.05,
      "Playback tile badges must reuse their display state instead of checking model files every tick")
  }

  @MainActor func testPlaybackTileSnapshotDoesNotReplaceValidationAndExpiresOnPause() {
    let store = StudioStore(restoreSession: false)
    var clip = Clip(engine: .ltx25)
    clip.prompt = "A quiet sea."
    store.project.clips = [clip]
    store.generationDescriptions[clip.id] = ["studioInput": store.generationRequestKey(for: clip)]
    XCTAssertEqual(store.timelineClipState(clip), .ready)
    store.isPlaying = true
    XCTAssertEqual(store.timelineClipState(clip), .ready)
    store.validationErrors[clip.id] = "The model became unavailable."
    XCTAssertEqual(store.timelineClipState(clip), .ready)
    XCTAssertEqual(store.clipState(clip), .attention)
    store.pausePlayback()
    XCTAssertEqual(store.timelineClipState(clip), .attention)
    store.isPlaying = true
    XCTAssertEqual(store.timelineClipState(clip), .attention)
  }

  @MainActor func testPlaybackActionButtonDoesNotRevalidateTheWholeMovieOnEveryTick() {
    let store = StudioStore(restoreSession: false)
    store.project.clips = (0..<50).map { Clip(name: "Shot \($0)", engine: .ltx25) }
    store.profiles = (0..<50).map {
      ModelProfile(id: "/nonexistent/playback-cost-\($0).json", name: "Model \($0)", engine: "ltx25", task: "i2v")
    }
    store.isPlaying = true
    let start = ProcessInfo.processInfo.systemUptime
    for _ in 0..<10 { _ = store.actionButtonTitle }
    XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - start, 0.05,
      "Playback chrome must not scan all clip/model inputs on every playhead update")
    store.pausePlayback()
    XCTAssertTrue(store.actionButtonTitle.hasPrefix("Actions "))
  }

  @MainActor func testOrdinaryClipSceneLookupDoesNotScaleWithUnrelatedModelLibrary() {
    let store = StudioStore(restoreSession: false)
    let clip = Clip(engine: .ltx25)
    store.project.clips = [clip]
    func elapsed() -> Double {
      let start = ProcessInfo.processInfo.systemUptime
      for _ in 0..<100 { XCTAssertEqual(store.continuousSceneDependencyKey(for: clip), "") }
      return ProcessInfo.processInfo.systemUptime - start
    }
    _ = elapsed() // Warm encoding/runtime startup before measuring.
    let small = elapsed()
    store.profiles = (0..<500).map {
      ModelProfile(id: "/nonexistent/timeline-cost-\($0).json", name: "Model \($0)", engine: "ltx25", task: "i2v")
    }
    let large = elapsed()
    // Broad relative allowance avoids enforcing machine-specific millisecond budgets.
    // Ordinary shots have no scene dependency, regardless of the installed library.
    XCTAssertLessThan(large, max(0.02, small * 10))
  }

  @MainActor func testSceneLookupStillTracksLeaderFollowerAndRuntimeChanges() {
    let store = StudioStore(restoreSession: false)
    let first = Clip(engine: .ltx25)
    var second = Clip(engine: .ltx25)
    second.continuity = ClipContinuity(mode: "scene", sourceClipID: first.id)
    store.project.clips = [first, second]
    let key = store.continuousSceneDependencyKey(for: first)
    XCTAssertFalse(key.isEmpty)
    XCTAssertFalse(key.hasPrefix("invalid-scene:"))
    XCTAssertEqual(key, store.continuousSceneDependencyKey(for: second))
    store.project.clips[1].prompt = "The lantern goes out."
    XCTAssertNotEqual(key, store.continuousSceneDependencyKey(for: first))
    let edited = store.continuousSceneDependencyKey(for: first)
    store.runtime.pythonPath = "/different/runtime/python"
    XCTAssertNotEqual(edited, store.continuousSceneDependencyKey(for: first))
    store.project.clips.removeAll()
    XCTAssertTrue(store.continuousSceneDependencyKey(for: first).hasPrefix("invalid-scene:"))
  }
}
