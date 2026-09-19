import Foundation
import XCTest
@testable import StudioCore

final class AudioMixTests: XCTestCase {
  func testLegacyTrackAndRegionDecodeNeutralControls() throws {
    let id = UUID(), asset = UUID()
    let track = try JSONDecoder().decode(AudioTrack.self, from: Data("""
    {"id":"\(id)","name":"Music","muted":false,"solo":false,"replacesSource":false}
    """.utf8))
    XCTAssertEqual(track.role, .music); XCTAssertEqual(track.gainDb, 0); XCTAssertEqual(track.pan, 0)
    let region = try JSONDecoder().decode(AudioRegion.self, from: Data("""
    {"id":"\(id)","assetID":"\(asset)","path":"a.wav","start":0,"sourceIn":0,"duration":5,"volume":0.4,"fade":0.3}
    """.utf8))
    XCTAssertEqual(region.effectiveFadeIn, 0.3); XCTAssertEqual(region.effectiveFadeOut, 0.3)
    XCTAssertEqual(region.volume, 0.4); XCTAssertNil(region.anchor)
  }
  func testVoiceFollowsClipAndMusicDoesNotChooseVoiceTrack() throws {
    var p = StudioProject(); var first = Clip(); first.duration = 5
    let second = Clip(); p.clips = [first, second]; p.audioTracks = []
    var voice = MediaAsset(name: "voice", kind: .audio, path: "voice.wav"); voice.duration = 3
    p.assets = [voice]
    let region = try p.placeVoice(voice, clipID: second.id, offset: 0.5)
    XCTAssertEqual(try resolvedAudioStart(region, in: p), 5.5)
    let music = try p.placeMusic(voice, at: 0, trackID: nil)
    XCTAssertNotEqual(music.trackID, region.trackID)
    p.move(second.id, before: first.id)
    XCTAssertEqual(try resolvedAudioStart(region, in: p), 0.5)
  }
  func testSplitAnchoredVoicePreservesSourceAndInternalEnvelope() throws {
    var p = StudioProject(); var clip = Clip(); clip.sourcePath = "movie.mp4"
    p.clips = [clip]
    var voice = MediaAsset(name: "voice", kind: .audio, path: "voice.wav"); voice.duration = 5
    p.assets = [voice]
    _ = try p.placeVoice(voice, clipID: clip.id, offset: 0)
    let second = try p.split(clip.id, at: 2)
    XCTAssertEqual(p.audio.count, 2)
    XCTAssertEqual(p.audio[0].duration, 2); XCTAssertEqual(p.audio[0].effectiveFadeOut, 0)
    XCTAssertEqual(p.audio[1].sourceIn, 2); XCTAssertEqual(p.audio[1].effectiveFadeIn, 0)
    XCTAssertEqual(p.audio[1].anchor?.clipID, second)
    XCTAssertEqual(try resolvedAudioStart(p.audio[1], in: p), 2)
  }
  func testInvalidAudioFailsValidation() throws {
    var p = StudioProject(); p.audioTracks[0].pan = .nan
    XCTAssertThrowsError(try p.validate())
    p.audioTracks[0].pan = 0; p.audioTracks[0].gainDb = 13
    XCTAssertThrowsError(try p.validate())
  }
}

extension AudioMixTests {
  func testSplitRetainsOriginalFadeWindowAndPhase() throws {
    var p = StudioProject(); var clip = Clip(); clip.sourcePath = "movie.mp4"; p.clips = [clip]
    var asset = MediaAsset(name: "voice", kind: .audio, path: "voice.wav"); asset.duration = 5
    _ = try p.placeVoice(asset, clipID: clip.id)
    p.audio[0].fadeIn = 2; p.audio[0].fadeOut = 2
    _ = try p.split(clip.id, at: 0.5)
    XCTAssertEqual(p.audio[0].envelope?.duration, 5)
    XCTAssertEqual(p.audio[1].envelope?.offset, 0.5)
    XCTAssertEqual(p.audio[1].envelope?.fadeIn, 2)
  }

  func testPreviewAudioUsesCurrentMotionOutputAndTrim() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("source.mp4"), output = folder.appendingPathComponent("motion.mp4")
    try Data([1]).write(to: source); try Data([2]).write(to: output)
    func modified(_ url: URL) throws -> Double { (try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as! Date).timeIntervalSince1970 }
    var clip = Clip(); clip.sourcePath = source.path; clip.sourceIn = 2
    var settings = MotionFidelitySettings(); settings.enabled = true
    clip.motionFidelity = settings
    clip.motionResult = MotionFidelityResult(path: output.path, sourcePath: source.path, sourceIn: 2,
      duration: clip.duration, outputIn: 0.25, recipeID: "", settings: settings, sourceSHA256: "", sha256: "",
      report: "", sourceSize: 1, sourceModified: try modified(source), outputSize: 1, outputModified: try modified(output))
    XCTAssertTrue(clip.motionIsCurrent)
    var p = StudioProject(); p.clips = [clip]
    let resolved = p.audioPlaybackProject
    XCTAssertEqual(resolved.clips[0].sourcePath, output.path)
    XCTAssertEqual(resolved.clips[0].sourceIn, 0.25)
    XCTAssertEqual(p.clips[0].sourcePath, source.path)
  }
}
