import Foundation
import XCTest
@testable import StudioCore

final class MusicGenerationTests: XCTestCase {
  func testDirectionCompilesIntoInspectableStyleWithoutLosingCustomText() {
    var draft = MusicDraft()
    draft.genre = "Indie pop"; draft.mood = "Hopeful"; draft.instruments = "piano, strings"
    draft.language = "English"; draft.bpm = 112; draft.direction = "A memorable chorus"
    XCTAssertTrue(draft.style.contains("112 BPM"))
    XCTAssertTrue(draft.style.contains("A memorable chorus"))
    XCTAssertTrue(draft.style.contains("piano, strings"))
    draft.instrumental = true
    XCTAssertTrue(draft.style.contains("instrumental"))
  }

  func testRequestMapsSeparateScoreAndMusicSamplingAndLengthBudget() throws {
    var draft = MusicDraft(); draft.modelPath = "/models/yue/8bit"
    draft.scoreSampling.temperature = 0.65; draft.musicSampling.topK = 80
    draft.musicSampling.maxTokens = 1500; draft.steps = 8
    let request = try draft.request()
    let data = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    XCTAssertEqual(data["steps"] as? Int, 8)
    XCTAssertEqual(data["model_path"] as? String, "/models/yue/8bit")
    XCTAssertEqual((data["abc_sampling"] as? [String: Any])?["temperature"] as? Double, 0.65)
    XCTAssertEqual((data["semantic_sampling"] as? [String: Any])?["top_k"] as? Int, 80)
    XCTAssertEqual(draft.maximumSeconds, 60)
  }

  func testInvalidRequestsFailBeforeSubmission() {
    var draft = MusicDraft(); draft.modelPath = "/models/yue"
    draft.cot = "off"; draft.abc = "X:1"
    XCTAssertThrowsError(try draft.request())
    draft.abc = ""; draft.musicSampling.temperature = .nan
    XCTAssertThrowsError(try draft.request())
    draft.musicSampling.temperature = 1; draft.musicSampling.minTokens = 5000
    draft.musicSampling.maxTokens = 20
    XCTAssertThrowsError(try draft.request())
  }

  func testShortLengthBudgetKeepsMinimumWithinRequestedMaximum() throws {
    var draft = MusicDraft(); draft.modelPath = "/model"
    draft.setMaximumSeconds(2)
    XCTAssertEqual(draft.musicSampling.maxTokens, 50)
    XCTAssertEqual(draft.musicSampling.minTokens, 50)
    XCTAssertNoThrow(try draft.request())
    draft.setMaximumSeconds(.nan)
    XCTAssertEqual(draft.maximumSeconds, 2)
  }

  func testMusicPlacementPreservesMainAudioAndUsesPlayhead() throws {
    var project = StudioProject(); var clip = Clip(); clip.volume = 0.75
    project.clips = [clip]
    var song = MediaAsset(name: "Song", kind: .audio, path: "/music.wav"); song.duration = 90
    project.assets = [song]
    let region = try project.placeMusic(song, at: 12, trackID: nil, sourceIn: 5, duration: 20)
    XCTAssertEqual(project.audio.count, 1)
    XCTAssertEqual(region.start, 12); XCTAssertEqual(region.sourceIn, 5)
    XCTAssertEqual(region.duration, 20)
    XCTAssertEqual(region.trackID, project.audioTracks.first?.id)
    XCTAssertEqual(project.clips[0].volume, 0.75)
    XCTAssertThrowsError(try project.placeMusic(song, at: 0, trackID: nil, sourceIn: 89, duration: 2))
  }

  func testOlderProjectAndAssetDecodeWithoutMusicMetadata() throws {
    let data = try JSONEncoder().encode(StudioProject())
    let project = try JSONDecoder().decode(StudioProject.self, from: data)
    XCTAssertNil(project.musicDraft)
    let asset = MediaAsset(name: "Legacy", kind: .audio)
    XCTAssertNil(try JSONDecoder().decode(MediaAsset.self, from: JSONEncoder().encode(asset)).musicGeneration)
  }

  func testDefaultMusicPlacementAvoidsSourceReplacementTrack() throws {
    var project = StudioProject()
    var replacement = AudioTrack(); replacement.replacesSource = true
    project.audioTracks = [replacement]
    var song = MediaAsset(name: "Song", kind: .audio, path: "/music.wav"); song.duration = 10
    let region = try project.placeMusic(song, at: 0, trackID: nil)
    XCTAssertNotEqual(region.trackID, replacement.id)
    XCTAssertEqual(project.audioTracks.count, 2)
    XCTAssertFalse(try XCTUnwrap(project.audioTracks.last).replacesSource)
    let explicit = try project.placeMusic(song, at: 20, trackID: replacement.id)
    XCTAssertEqual(explicit.trackID, replacement.id)
  }

  func testCollectedMusicArtifactsSurviveProjectRelocation() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("take")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data("saved tokens".utf8).write(to: source.appendingPathComponent("tokens.json"))
    var draft = MusicDraft(); draft.modelPath = "/shared/models/yue2"
    let generation = MusicGeneration(draft: draft, request: try draft.request(), artifacts: source.path)
    let destination = root.appendingPathComponent("portable/Media/take")
    let collected = try ProjectStorage.collectMusicArtifacts(generation, to: destination)
    XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("tokens.json")), Data("saved tokens".utf8))
    var project = StudioProject()
    var asset = MediaAsset(name: "Song", kind: .audio, path: "Media/take/master.wav")
    asset.musicGeneration = collected; asset.musicGeneration?.artifacts = "Media/take"
    project.assets = [asset]
    let projectURL = root.appendingPathComponent("portable/song.weetodd")
    try ProjectStorage.write(project, to: projectURL)
    let restored = try ProjectStorage.read(projectURL)
    XCTAssertEqual(restored.assets[0].musicGeneration?.artifacts, destination.path)
    XCTAssertEqual(restored.assets[0].musicGeneration?.request.modelPath, "/shared/models/yue2")
  }
}
