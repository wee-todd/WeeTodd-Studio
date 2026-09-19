import AVFoundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TimelinePlaybackTests: XCTestCase {
  @MainActor func makeStore() throws -> StudioStore {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    var first = Clip(name: "First", engine: .movie)
    first.duration = 3; first.sourceIn = 2
    var second = Clip(name: "Second", engine: .movie)
    second.duration = 7; second.sourceIn = 20
    store.project.clips = [first, second]
    store.select(first.id)
    return store
  }

  @MainActor func testDefaultTransportSeeksAcrossClipsWithoutChangingInspectorSelection() throws {
    let store = try makeStore()
    let selected = store.selectedClipID
    store.seek(6)
    XCTAssertEqual(store.effectivePreviewDuration, 10)
    XCTAssertEqual(store.playhead, 6)
    XCTAssertEqual(store.selectedClipID, selected)
    store.seekToEnd()
    XCTAssertEqual(store.playhead, 10)
  }

  @MainActor func testSelectingLaterClipMovesToItsTimelineStart() throws {
    let store = try makeStore()
    store.select(store.project.clips[1].id)
    XCTAssertEqual(store.playhead, 3)
    store.seek(5)
    store.refreshPreview()
    XCTAssertEqual(store.playhead, 5)
  }

  @MainActor func testEndpointSlotsSeekWithinLaterClipUsingMovieTime() throws {
    let store = try makeStore()
    store.project.settings.fps = 24
    let secondID = store.project.clips[1].id
    store.selectTimelineEndpoint(secondID, role: .first)
    XCTAssertEqual(store.selectedClipID, secondID)
    XCTAssertEqual(store.playhead, 3, accuracy: 0.000001)
    store.selectTimelineEndpoint(secondID, role: .last)
    XCTAssertEqual(store.selectedClipID, secondID)
    XCTAssertEqual(store.playhead, 9.958333333333334, accuracy: 0.000001)
    XCTAssertEqual(store.previewClip?.id, secondID)
  }

  @MainActor func testEndpointSlotsRespectTransitionOverlapAndClipFrameRate() throws {
    let store = try makeStore()
    store.project.clips[1].transition = "dissolve"
    store.project.clips[1].transitionDuration = 0.5
    store.project.clips[1].settingsOverride = store.project.settings
    store.project.clips[1].settingsOverride?.fps = 30
    let secondID = store.project.clips[1].id
    store.selectTimelineEndpoint(secondID, role: .first)
    XCTAssertEqual(store.playhead, 2.5, accuracy: 0.000001)
    store.selectTimelineEndpoint(secondID, role: .last)
    XCTAssertEqual(store.playhead, 9.466666666666667, accuracy: 0.000001)
  }

  @MainActor func testSplitUsesClipUnderGlobalPlayheadAndLocalTrimOffset() throws {
    let store = try makeStore()
    store.project.clips[1].sourcePath = "/tmp/source.mov"
    store.seek(6)
    store.split()
    XCTAssertEqual(store.project.clips.count, 3)
    guard store.project.clips.count == 3 else { return }
    XCTAssertEqual(store.project.clips[1].duration, 3)
    XCTAssertEqual(store.project.clips[2].duration, 4)
    XCTAssertEqual(store.project.clips[2].sourceIn, 23)
    XCTAssertEqual(store.playhead, 6)
  }
}

extension TimelinePlaybackTests {
  @MainActor func testRulerAndHandleScrubbingKeepInspectorStableAndRestorePlayingState() async throws {
    let store = try makeStore()
    await store.timelineBuildTask?.value
    let selection = store.selectedClipID
    store.togglePlayback()
    XCTAssertTrue(store.isPlaying)
    store.scrubTimeline(to: 12)
    XCTAssertNil(store.scrubWasPlaying)
    XCTAssertTrue(store.isPlaying)
    store.scrubTimeline(to: 6)
    XCTAssertFalse(store.isPlaying)
    XCTAssertEqual(store.playhead, 6)
    XCTAssertEqual(store.previewClip?.name, "Second")
    XCTAssertEqual(store.selectedClipID, selection)
    XCTAssertNil(store.selectedClipPlayhead)
    store.endTimelineScrub()
    XCTAssertTrue(store.isPlaying)
    store.scrubTimeline(to: -20, clamped: true)
    XCTAssertEqual(store.playhead, 0)
    store.scrubTimeline(to: 30, clamped: true)
    XCTAssertEqual(store.playhead, 10)
    store.endTimelineScrub()
    XCTAssertFalse(store.isPlaying)
  }

  @MainActor func testStillAndUnrenderedTimelineAdvancesAndReplaysFromEnd() async throws {
    let store = try makeStore()
    store.project.clips[0].duration = 0.1
    store.project.clips[1].duration = 0.1
    store.refreshPreview()
    await store.timelineBuildTask?.value
    store.togglePlayback()
    try await Task.sleep(nanoseconds: 400_000_000)
    XCTAssertEqual(store.playhead, 0.2, accuracy: 0.0001)
    XCTAssertFalse(store.isPlaying)
    store.togglePlayback()
    XCTAssertEqual(store.playhead, 0)
    XCTAssertTrue(store.isPlaying)
    store.pausePlayback()
  }

  @MainActor func testReopeningDocumentCancelsPreparationAndPendingScrubResume() async throws {
    let store = try makeStore()
    store.togglePlayback()
    store.scrubTimeline(to: 6)
    let task = store.timelineBuildTask
    store.newProject()
    await task?.value
    store.endTimelineScrub()
    XCTAssertNil(store.player.currentItem)
    XCTAssertEqual(store.playhead, 0)
    XCTAssertFalse(store.isPlaying)
    XCTAssertFalse(store.preparingTimelinePlayback)
  }

  @MainActor func testKeyframeOffsetIsLocalToSelectedClip() throws {
    let store = try makeStore()
    store.select(store.project.clips[1].id)
    store.seek(6)
    XCTAssertEqual(store.selectedClipPlayhead, 3)
    store.seek(2)
    XCTAssertNil(store.selectedClipPlayhead)
  }
}
