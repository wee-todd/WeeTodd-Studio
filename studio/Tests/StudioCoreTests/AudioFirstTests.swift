import XCTest
@testable import StudioCore

final class AudioFirstTests: XCTestCase {
  func testEmptyTimelineAcceptsFullVoiceAndCanSeekWithoutVideo() throws {
    var p = StudioProject(); var a = MediaAsset(name: "Narration", kind: .audio, path: "/tmp/voice.wav")
    a.duration = 79
    let r = try p.placeTimelineAudio(a, role: .voice, at: 3)
    XCTAssertEqual(r.duration, 79); XCTAssertNil(r.anchor)
    XCTAssertEqual(p.duration, 0); XCTAssertEqual(p.playbackDuration, 82)
    XCTAssertEqual(TimelinePlaybackPlan(project: p).seekPosition(40, clamped: false), 40)
    XCTAssertEqual(p.audioTracks.first(where: { $0.id == r.trackID })?.role, .voice)
  }
  func testClipAttachmentOffersExplicitTrimAndPreservesFullTake() throws {
    var p = StudioProject(); var c = Clip(); c.duration = 5; p.clips = [c]
    var a = MediaAsset(name: "Voice", kind: .audio, path: "/tmp/voice.wav"); a.duration = 79
    let full = try p.placeVoice(a, clipID: c.id, trimToClip: false)
    XCTAssertEqual(full.duration, 79)
    let trimmed = try p.placeVoice(a, clipID: c.id, trimToClip: true)
    XCTAssertEqual(trimmed.duration, 5); XCTAssertEqual(a.duration, 79)
  }
  func testPlacementRejectsInvalidTimingAndSongNeverUsesVoiceTrack() throws {
    var p = StudioProject(); p.audioTracks = [AudioTrack(name: "Voice", role: .voice)]
    var a = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); a.duration = 10
    XCTAssertThrowsError(try p.placeTimelineAudio(a, role: .music, at: .nan))
    let r = try p.placeTimelineAudio(a, role: .music, at: 0)
    XCTAssertEqual(p.audioTracks.first(where: { $0.id == r.trackID })?.role, .music)
  }
}
