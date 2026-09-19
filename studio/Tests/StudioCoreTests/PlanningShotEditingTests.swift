import Foundation
import XCTest
@testable import StudioCore

final class PlanningShotEditingTests: XCTestCase {
  private func plan() -> ProjectPlanning {
    var p = ProjectPlanning()
    var shot = PlanningShot(name: "Arrival", frameCount: 289)
    shot.action = "Approach and enter"; shot.direction = "Full authored direction"
    shot.firstFrame = "Outside"; shot.lastFrame = "Inside"
    shot.firstAssetID = UUID(); shot.lastAssetID = UUID()
    shot.sourceKey = "director:shot:1"
    shot.musicSource = MusicShotSource(path: "/tmp/song.wav", sha256: String(repeating: "a", count: 64),
      start: 30, duration: 12.03, task: "t2v")
    p.shots = [shot]; return p
  }
  func testSplitPreservesSongEndpointsReferencesAndAuthoredText() throws {
    var p = plan(); let original = p.shots[0]
    try p.approveShot(original.id)
    let next = try p.splitShot(original.id, afterFrames: 96, boundary: "At the door")
    XCTAssertEqual(p.shots.map(\.frameCount), [96, 193])
    XCTAssertEqual(p.shots.map { $0.musicSource!.start }, [30, 34])
    XCTAssertEqual(p.shots[0].musicSource!.duration, 4)
    XCTAssertEqual(p.shots[1].musicSource!.start + p.shots[1].musicSource!.duration, 42.03, accuracy: 0.000001)
    XCTAssertEqual(p.shots[0].firstAssetID, original.firstAssetID)
    XCTAssertNil(p.shots[0].lastAssetID); XCTAssertNil(p.shots[1].firstAssetID)
    XCTAssertEqual(p.shots[1].lastAssetID, original.lastAssetID)
    XCTAssertEqual(p.shots[0].lastFrame, "At the door"); XCTAssertEqual(p.shots[1].firstFrame, "At the door")
    XCTAssertEqual(p.shots[0].id, original.id); XCTAssertEqual(p.shots[1].id, next)
    XCTAssertEqual(p.shots[0].sourceKey, original.sourceKey)
    XCTAssertEqual(p.shots.map(\.direction), [original.direction, original.direction])
    XCTAssertTrue(p.shots.allSatisfy { $0.approvedRevision == nil })
    p = try JSONDecoder().decode(ProjectPlanning.self, from: JSONEncoder().encode(p))
    // Removing endpoint references avoids image existence checks for this application test.
    for i in p.shots.indices { p.shots[i].firstAssetID = nil; p.shots[i].lastAssetID = nil }
    for id in p.shots.map(\.id) { try p.approveShot(id) }
    var movie = StudioProject(); movie.planning = p
    try movie.applyPlanningShots(Set(p.shots.map(\.id)))
    XCTAssertEqual(movie.audio.count, 1)
    XCTAssertEqual(movie.audio[0].duration, 12.03, accuracy: 0.000001)
  }
  func testInvalidSplitDoesNotMutatePlan() throws {
    for badFrame in [0, 289, -1] {
      var p = plan(); let before = p
      XCTAssertThrowsError(try p.splitShot(p.shots[0].id, afterFrames: badFrame, boundary: "Door"))
      XCTAssertEqual(p, before)
    }
    for kind in 0..<4 {
      var p = plan()
      if kind == 0 { p.shots[0].linkedClipID = UUID() }
      if kind == 1 { p.shots[0].combinedShots = [p.shots[0]] }
      if kind == 2 { p.shots[0].musicSource?.duration = 2 }
      let before = p
      XCTAssertThrowsError(try p.splitShot(p.shots[0].id, afterFrames: 96, boundary: kind == 3 ? " " : "Door"))
      XCTAssertEqual(p, before)
    }
  }
  func testGenerationSettingsPersistAndReachTimelineWithoutChangingSong() throws {
    var p = plan(); let source = p.shots[0].musicSource
    try p.approveShot(p.shots[0].id)
    try p.setGenerationSettings([p.shots[0].id], engine: .ltx25, width: 1344, height: 768)
    XCTAssertFalse(p.isShotApproved(p.shots[0].id))
    XCTAssertEqual(p.shots[0].musicSource, source)
    p = try JSONDecoder().decode(ProjectPlanning.self, from: JSONEncoder().encode(p))
    p.shots[0].firstAssetID = nil; p.shots[0].lastAssetID = nil
    try p.approveShot(p.shots[0].id)
    var movie = StudioProject(); movie.planning = p
    try movie.applyPlanningShots([p.shots[0].id])
    XCTAssertEqual(movie.clips[0].generationWidth, 1344)
    XCTAssertEqual(movie.clips[0].generationHeight, 768)
    XCTAssertEqual(movie.clips[0].duration, 289.0 / 24)
    XCTAssertEqual(movie.clips[0].engine, .ltx25)
  }
  func testSettingsRejectInvalidSizesAndIncompatibleCombinedSettingsAtomically() throws {
    var p = plan(); let before = p
    XCTAssertThrowsError(try p.setGenerationSettings([p.shots[0].id], engine: .ltx25, width: 1345, height: 768))
    XCTAssertEqual(p, before)
    let second = try p.splitShot(p.shots[0].id, afterFrames: 96, boundary: "Door")
    try p.setGenerationSettings([second], engine: .ltx25, width: 1344, height: 768)
    let split = p
    XCTAssertThrowsError(try p.combineShots(Set(p.shots.map(\.id))))
    XCTAssertEqual(p, split)
  }
  func testAudioDrivenSettingsCannotSwitchToUnsupportedBackend() throws {
    var p = plan(); p.shots[0].musicSource?.task = "a2v"
    let before = p
    XCTAssertThrowsError(try p.setGenerationSettings([p.shots[0].id], engine: .drawThings, width: 1344, height: 768))
    XCTAssertEqual(p, before)
  }
  func testReuseExistingOpeningAndFullSongWithoutDuplicatingEither() throws {
    var p = plan(); p.shots[0].musicSource?.start = 0
    p.shots[0].firstAssetID = nil; p.shots[0].lastAssetID = nil
    let second = try p.splitShot(p.shots[0].id, afterFrames: 96, boundary: "Door")
    for id in p.shots.map(\.id) { try p.approveShot(id) }
    var movie = StudioProject(); movie.planning = p
    var opening = Clip(name: "Existing take", engine: .ltx25)
    opening.sourcePath = "/tmp/opening.mp4"; opening.duration = 4.01
    opening.prompt = "Original reproducible prompt"; movie.clips = [opening]
    var song = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); song.duration = 12.03
    movie.assets = [song]
    _ = try movie.placeMusic(song, at: 0, trackID: nil, sourceIn: 0, duration: 12.03)
    try movie.reuseTimelineClip(opening.id, forPlanningShot: p.shots[0].id)
    XCTAssertTrue(movie.planning!.isShotApproved(p.shots[0].id, assets: movie.assets))
    try movie.applyPlanningShots([second])
    XCTAssertTrue(movie.planning!.isShotApproved(second, assets: movie.assets))
    XCTAssertEqual(movie.clips.count, 2)
    XCTAssertEqual(movie.clips[0].id, opening.id)
    XCTAssertEqual(movie.clips[0].duration, 4)
    XCTAssertEqual(movie.clips[0].sourcePath, opening.sourcePath)
    XCTAssertEqual(movie.clips[0].prompt, opening.prompt)
    XCTAssertEqual(movie.clips[0].musicSource?.start, 0)
    XCTAssertTrue(movie.clips[0].hasReviewedReusedTake)
    movie.clips[0].prompt += " Changed action"
    XCTAssertFalse(movie.clips[0].hasReviewedReusedTake)
    XCTAssertEqual(movie.audio.count, 1)
    XCTAssertEqual(movie.audio[0].duration, 12.03)
  }
  func testReusedTakeMarkerRoundTripsAndDoesNotAcceptDifferentFootage() throws {
    var p = plan(); p.shots[0].firstAssetID = nil; p.shots[0].lastAssetID = nil
    try p.approveShot(p.shots[0].id)
    var movie = StudioProject(); movie.planning = p
    var take = Clip(engine: .ltx25); take.duration = 289.0 / 24; take.sourcePath = "/tmp/take.mp4"
    movie.clips = [take]
    try movie.reuseTimelineClip(take.id, forPlanningShot: p.shots[0].id)
    movie = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(movie))
    XCTAssertTrue(movie.clips[0].hasReviewedReusedTake)
    movie.clips[0].sourceIn = 1
    XCTAssertFalse(movie.clips[0].hasReviewedReusedTake)
  }
  func testReuseRejectsWrongPositionWithoutMutatingTheMovie() throws {
    var p = plan(); p.shots[0].musicSource = nil
    try p.approveShot(p.shots[0].id)
    var movie = StudioProject(); movie.planning = p
    var clip = Clip(); clip.duration = 289.0 / 24; clip.sourcePath = "/tmp/take.mp4"
    movie.clips = [Clip(), clip]
    let before = movie
    XCTAssertThrowsError(try movie.reuseTimelineClip(clip.id, forPlanningShot: p.shots[0].id))
    XCTAssertEqual(movie, before)
  }
  func testPartialSongOverlapIsRejectedWithoutMutation() throws {
    var p = plan(); p.shots[0].musicSource?.start = 0
    p.shots[0].firstAssetID = nil; p.shots[0].lastAssetID = nil
    try p.approveShot(p.shots[0].id)
    var movie = StudioProject(); movie.planning = p
    var song = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); song.duration = 12.03
    movie.assets = [song]
    _ = try movie.placeMusic(song, at: 1, trackID: nil, sourceIn: 1, duration: 2)
    let before = movie
    XCTAssertThrowsError(try movie.applyPlanningShots([p.shots[0].id]))
    XCTAssertEqual(movie, before)
  }
}
