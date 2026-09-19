import Foundation
import XCTest
@testable import StudioCore

final class MusicIntervalRealignmentTests: XCTestCase {
  private func movie() throws -> StudioProject {
    var plan = ProjectPlanning()
    for (index, frames, start, duration) in [(0, 144, 0.0, 6.0), (1, 144, 6.0, 6.0), (2, 49, 12.0, 2.02)] {
      var shot = PlanningShot(name: "Shot \(index)", frameCount: frames)
      shot.action = "Hold the sword"
      shot.firstFrame = "Hands on the hilt"; shot.lastFrame = "Hands braced"
      shot.musicSource = MusicShotSource(path: "/tmp/song.wav", sha256: String(repeating: "a", count: 64), start: start, duration: duration, task: "t2v")
      plan.shots.append(shot)
      try plan.approveShot(shot.id)
    }
    var movie = StudioProject(); movie.planning = plan
    var song = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); song.duration = 14.02
    movie.assets = [song]
    _ = try movie.placeMusic(song, at: 0, trackID: nil, sourceIn: 0, duration: 14.02)
    try movie.applyPlanningShots([plan.shots[0].id])
    movie.clips[0].sourcePath = "/tmp/grip.mp4"
    movie.clips[0].duration = 4.25
    movie.planning!.shots[0].frameCount = 102
    movie.planning!.shots[1].frameCount = 186
    return movie
  }

  func testRealignmentPreservesSongAndFootageAndAllowsReviewedNextShot() throws {
    var movie = try movie(); let original = movie
    XCTAssertEqual(try movie.realignPlanningMusicIntervals(), 2)
    let plan = try XCTUnwrap(movie.planning)
    XCTAssertEqual(plan.shots.map { $0.musicSource!.start }, [0, 4.25, 12])
    XCTAssertEqual(plan.shots[0].musicSource!.duration, 4.25)
    XCTAssertEqual(plan.shots[1].musicSource!.duration, 7.75)
    XCTAssertEqual(plan.shots[2].musicSource!.duration, 2.02)
    XCTAssertEqual(movie.audio, original.audio)
    XCTAssertEqual(movie.clips[0].sourcePath, original.clips[0].sourcePath)
    XCTAssertEqual(movie.clips[0].prompt, original.clips[0].prompt)
    XCTAssertFalse(plan.isShotApproved(plan.shots[0].id))
    XCTAssertFalse(plan.isShotApproved(plan.shots[1].id))
    XCTAssertTrue(plan.isShotApproved(plan.shots[2].id))
    XCTAssertFalse(movie.clips[0].hasReviewedReusedTake)
    for id in plan.shots.map(\.id) { try movie.planning!.approveShot(id) }
    try movie.reuseTimelineClip(movie.clips[0].id, forPlanningShot: plan.shots[0].id)
    try movie.applyPlanningShots([plan.shots[1].id])
    XCTAssertTrue(movie.clips[0].hasReviewedReusedTake)
    XCTAssertEqual(movie.clips[1].duration, 7.75)
    XCTAssertEqual(movie.audio, original.audio)
    let saved = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(movie))
    XCTAssertEqual(saved, movie)
    XCTAssertEqual(try movie.realignPlanningMusicIntervals(), 0)
  }

  func testRealignmentRejectsChangedSongLengthAtomically() throws {
    var movie = try movie(); movie.planning!.shots[1].frameCount = 144
    let before = movie
    XCTAssertThrowsError(try movie.realignPlanningMusicIntervals())
    XCTAssertEqual(movie, before)
  }

  func testRealignmentRequiresExistingTimelineTrim() throws {
    var movie = try movie(); movie.clips[0].duration = 6
    let before = movie
    XCTAssertThrowsError(try movie.realignPlanningMusicIntervals())
    XCTAssertEqual(movie, before)
  }

  func testRealignmentRejectsAudioDrivenTakeAndDiscontinuousSources() throws {
    for kind in 0..<4 {
      var movie = try movie()
      if kind == 0 { movie.planning!.shots[0].musicSource?.task = "a2v"; movie.clips[0].musicSource?.task = "a2v" }
      if kind == 1 { movie.planning!.shots[1].musicSource?.start = 6.1 }
      if kind == 2 { movie.planning!.shots[1].musicSource?.sha256 = String(repeating: "b", count: 64) }
      if kind == 3 { movie.planning!.shots[1].combinedShots = [movie.planning!.shots[1]] }
      let before = movie
      XCTAssertThrowsError(try movie.realignPlanningMusicIntervals())
      XCTAssertEqual(movie, before)
    }
  }
}
