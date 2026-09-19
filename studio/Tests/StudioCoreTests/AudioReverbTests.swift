import Foundation
import XCTest
@testable import StudioCore

final class AudioReverbTests: XCTestCase {
  func testOldTracksStayDryAndSettingsRoundTrip() throws {
    var track = AudioTrack(name: "Voice", role: .voice)
    XCTAssertNil(track.reverb)
    let dry = try JSONDecoder().decode(AudioTrack.self, from: JSONEncoder().encode(track))
    XCTAssertNil(dry.reverb)
    var effect = AudioReverb(); effect.applyPreset(.hall); effect.mix = 0.23; effect.enabled = false
    track.reverb = effect
    XCTAssertEqual(try JSONDecoder().decode(AudioTrack.self, from: JSONEncoder().encode(track)), track)
    XCTAssertEqual(track.reverb?.decay, 2.6)
    XCTAssertEqual(track.reverb?.mix, 0.23)
  }
  func testReverbValidationAndDriverInvalidation() throws {
    var project = StudioProject(); var clip = Clip(); clip.audioDriverSelection = AudioDriverSelection(mode: .voice)
    project.clips = [clip]
    var track = AudioTrack(name: "Voice", role: .voice); project.audioTracks = [track]
    let before = project.audioDriverRevision(for: clip)
    track.reverb = AudioReverb(); project.audioTracks = [track]
    XCTAssertNotEqual(before, project.audioDriverRevision(for: clip))
    XCTAssertNoThrow(try project.validateAudio())
    for value in [Double.nan, -Double.infinity, -1, 6.1] {
      project.audioTracks[0].reverb?.decay = value
      XCTAssertThrowsError(try project.validateAudio())
    }
  }
  func testPresetsRetainAmountAndBypassState() {
    var value = AudioReverb(); value.mix = 0.4; value.enabled = false
    value.applyPreset(.plate)
    XCTAssertEqual(value.mix, 0.4); XCTAssertFalse(value.enabled)
    XCTAssertEqual(value.decay, 1.6); XCTAssertEqual(value.preDelay, 0.008)
  }
}
