import XCTest
import AVFoundation
import StudioCore
@testable import WeeToddStudio

final class AudioFirstStoreTests: XCTestCase {
  @MainActor func testImportSongUsesNativeMediaInspectionAndPlacesFullTakeWithoutPython() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let song = directory.appendingPathComponent("Song.wav")
    let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1))
    let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000))
    buffer.frameLength = 48000
    do {
      let file = try AVAudioFile(forWriting: song, settings: format.settings)
      try file.write(from: buffer)
    }
    var calls = 0
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { _, _, _, _ in calls += 1; throw StudioError.invalid("Python is unavailable") })
    defer { store.invalidateTimelinePlayback() }
    store.runtime.pythonPath = "/unavailable/python"
    await store.addSong(from: song)
    XCTAssertNil(store.error)
    let region = try XCTUnwrap(store.project.audio.first)
    XCTAssertEqual(region.path, song.path)
    XCTAssertEqual(region.duration, 2, accuracy: 0.001)
    XCTAssertEqual(store.project.playbackDuration, 2, accuracy: 0.001)
    XCTAssertEqual(store.project.audioTracks.first(where: { $0.id == region.trackID })?.role, .music)
    XCTAssertEqual(calls, 0)
  }

  @MainActor func testPlaceFullVoiceReturnsToTimelineAndUndoRestoresEmptyProject() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:directory) }
    let store = StudioStore(dataDirectory:directory,restoreSession:false)
    defer { store.invalidateTimelinePlayback() }
    store.showVoice = true
    var asset = MediaAsset(name:"Narration",kind:.audio,path:"/tmp/voice.wav"); asset.duration = 79
    store.placeAudioTake(asset,role:.voice)
    XCTAssertFalse(store.showVoice)
    XCTAssertEqual(store.project.audio.first?.duration,79)
    XCTAssertEqual(store.selectedAudioID,store.project.audio.first?.id)
    XCTAssertNil(store.selectedClip)
    XCTAssertEqual(store.project.audioPresentation?.style,"waveform")
    store.undo()
    XCTAssertTrue(store.project.audio.isEmpty)
    XCTAssertEqual(store.project.playbackDuration,0)
  }
}
