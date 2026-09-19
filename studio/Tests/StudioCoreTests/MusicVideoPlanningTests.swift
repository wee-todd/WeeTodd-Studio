import XCTest
@testable import StudioCore

final class MusicVideoPlanningTests: XCTestCase {
  func testCanonicalDriverRoundTripAndSceneValidation() throws {
    var p = StudioProject()
    var song = MediaAsset(name: "Song", kind: .audio, path: "/tmp/song.wav"); song.duration = 60
    p.assets = [song]
    p.clips = (0..<2).map { i in
      var c = Clip(name: "Shot", engine: .ltx25); c.duration = 4
      var a = Attachment(assetID: song.id, role: .audioDriver)
      a.audioSourceStart = 10 + Double(i) * 4; a.audioSourceDuration = 4
      c.attachments = [a]
      if i > 0 { c.continuity = ClipContinuity(mode: "scene") }
      return c
    }
    XCTAssertTrue(p.continuousSceneIssues(for: p.clips[0]).isEmpty)
    let restored = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(p))
    XCTAssertEqual(restored.clips[1].attachments[0].audioSourceStart, 14)
    p.clips[1].attachments[0].audioSourceStart = 15
    XCTAssertFalse(p.continuousSceneIssues(for: p.clips[0]).isEmpty)
    p.clips[1].attachments = []
    XCTAssertFalse(p.continuousSceneIssues(for: p.clips[0]).isEmpty)
  }
  func testCombineRetainsOrderedOriginalsAndRoundTrips() throws {
    var p = ProjectPlanning()
    var a = PlanningShot(name: "Walk", frameCount: 48); a.action = "Walk to door"; a.firstFrame = "Outside"; a.lastFrame = "At door"
    var b = PlanningShot(name: "Enter", frameCount: 72); b.action = "Open door and enter"; b.firstFrame = "At door"; b.lastFrame = "Inside"
    a.subjectIDs = [UUID()]; b.subjectIDs = [UUID()]
    p.shots = [a,b]
    let originals = p.shots
    let id = try p.combineShots(Set(originals.map(\.id)), maximumSeconds: 15)
    XCTAssertEqual(p.shots.count, 1); XCTAssertEqual(p.shots[0].frameCount, 120)
    XCTAssertEqual(p.shots[0].firstFrame, "Outside"); XCTAssertEqual(p.shots[0].lastFrame, "Inside")
    XCTAssertEqual(Set(p.shots[0].subjectIDs), Set(a.subjectIDs + b.subjectIDs))
    XCTAssertTrue(p.shots[0].direction.contains("Walk to door")); XCTAssertTrue(p.shots[0].direction.contains("Open door and enter"))
    p = try JSONDecoder().decode(ProjectPlanning.self, from: JSONEncoder().encode(p))
    try p.uncombineShot(id)
    XCTAssertEqual(p.shots, originals)
  }
  func testCombineRejectsNonadjacentLinkedAndOverlongWithoutMutation() throws {
    var p = ProjectPlanning(); p.shots = (0..<3).map { PlanningShot(name: "Shot \($0)", frameCount: 240) }
    var before = p
    XCTAssertThrowsError(try p.combineShots([p.shots[0].id,p.shots[2].id], maximumSeconds: 30)); XCTAssertEqual(p,before)
    XCTAssertThrowsError(try p.combineShots([p.shots[0].id,p.shots[1].id], maximumSeconds: 15)); XCTAssertEqual(p,before)
    p.shots[0].linkedClipID = UUID(); before = p
    XCTAssertThrowsError(try p.combineShots([p.shots[0].id,p.shots[1].id], maximumSeconds: 30)); XCTAssertEqual(p,before)
  }
}

extension MusicVideoPlanningTests {
  func testMusicPlanImportsLyricsExactSourceAndAppliesTransactionally() throws {
    let hash = String(repeating: "a", count: 64)
    let timing: [String: Any] = ["sourceAudio": ["path": "/tmp/song.wav", "sha256": hash, "sourceStartSeconds": 10.0, "sourceEndSeconds": 14.0],
      "fps": 24, "totalFrames": 96, "suppliedLyrics": "Original lyric\nSecond line", "lyricStatus": "supplied_unaligned", "engine": "ltx25", "task": "a2v",
      "clips": [["clipID": "clip-1", "startFrame": 0, "frameCount": 96, "sourceStartSeconds": 10.0, "sourceDurationSeconds": 4.0]]]
    let clips: [String: Any] = ["fps": 24, "totalFrames": 96, "characters": [], "clips": [
      ["id": "clip-1", "startFrame": 0, "frameCount": 96, "action": "Walk", "startState": "Outside", "endState": "At door", "location": "", "characters": [], "continuity": "cut"]]]
    let raw: [String: Any] = ["status": "completed", "totalSeconds": 0, "outputs": ["music_timing": timing], "steps": [
      "clips": ["name": "Shots", "status": "completed", "approved": true, "outputs": ["clips": clips]],
      "timing": ["name": "Timing", "status": "completed", "outputs": ["music_timing": timing]]]]
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: JSONSerialization.data(withJSONObject: raw))
    var p = StudioProject(); p.planning = ProjectPlanning()
    try p.planning!.importRun(run, sourceID: "music-run", sourceText: "A walk")
    XCTAssertEqual(p.planning!.musicTimings?["music-run"]?.suppliedLyrics, "Original lyric\nSecond line")
    XCTAssertEqual(p.planning!.shots[0].musicSource?.start, 10)
    let id = p.planning!.shots[0].id
    try p.planning!.approveShot(id)
    try p.applyPlanningShots([id])
    XCTAssertEqual(p.clips.count, 1); XCTAssertEqual(p.clips[0].duration, 4)
    XCTAssertEqual(p.clips[0].attachments.first?.audioSourceStart, 10)
    XCTAssertEqual(p.clips[0].volume, 0)
    XCTAssertEqual(p.audio.count, 1); XCTAssertEqual(p.audio[0].sourceIn, 10)
    XCTAssertEqual(p.audio[0].duration, 4)
    let applied = p
    XCTAssertThrowsError(try p.applyPlanningShots([id])); XCTAssertEqual(p, applied)
  }
}

extension MusicVideoPlanningTests {
  func testMusicSourcePathsFollowPortableProjectMappingIncludingCombinedChildren() throws {
    var p = StudioProject(); p.planning = ProjectPlanning()
    var shot = PlanningShot(name: "Song", frameCount: 48)
    shot.musicSource = MusicShotSource(path: "Media/song.wav", sha256: String(repeating: "b", count: 64), start: 0, duration: 2, task: "a2v")
    var combined = shot; combined.combinedShots = [shot]; p.planning!.shots = [combined]
    var clip = Clip(); clip.musicSource = shot.musicSource; p.clips = [clip]
    ProjectStorage.mapPaths(&p) { $0.hasPrefix("Media/") ? "/portable/" + $0 : $0 }
    XCTAssertEqual(p.clips[0].musicSource?.path, "/portable/Media/song.wav")
    XCTAssertEqual(p.planning!.shots[0].combinedShots?[0].musicSource?.path, "/portable/Media/song.wav")
  }
}

extension MusicVideoPlanningTests {
  func testCombineHonorsH3BackendMaximum() throws {
    var p = ProjectPlanning()
    p.shots = (0..<2).map { i in var s = PlanningShot(name: "Shot \(i)", frameCount: 240); s.engine = .h3; return s }
    let before = p
    XCTAssertThrowsError(try p.combineShots(Set(p.shots.map(\.id)), maximumSeconds: 30))
    XCTAssertEqual(p, before)
  }
}

extension MusicVideoPlanningTests {
  func testDisabledScenePreferenceSurvivesApplyingGridAlignedShots() throws {
    var p = StudioProject(); var plan = ProjectPlanning()
    for i in 0..<2 {
      var shot = PlanningShot(name: "Shot \(i)", frameCount: 48)
      shot.action = "Walk"; shot.firstFrame = "Walking"; shot.lastFrame = "Walking"
      shot.task = "a2v"; shot.continuity = i == 0 ? "cut" : "continue"
      shot.musicSource = MusicShotSource(path: "/tmp/song.wav", sha256: String(repeating: "a", count: 64), start: Double(i * 2), duration: 2, task: "a2v")
      shot.musicSource?.sceneEligible = false
      plan.shots.append(shot)
    }
    for id in plan.shots.map(\.id) { try plan.approveShot(id) }
    p.planning = plan
    try p.applyPlanningShots(Set(plan.shots.map(\.id)))
    XCTAssertEqual(p.clips[1].continuityMode, "independent")
    XCTAssertEqual(p.audio.count, 1)
  }
}

extension MusicVideoPlanningTests {
  func testNestedCombinedShotsKeepInteriorFrameReferences() throws {
    var project = StudioProject(); var plan = ProjectPlanning()
    let reference = MediaAsset(name: "Interior", kind: .image, path: "/tmp/interior.png")
    project.assets = [reference]
    plan.shots = (0..<3).map { i in
      var shot = PlanningShot(name: "Shot \(i)", frameCount: 48)
      shot.action = "Walk"; shot.firstFrame = "Walking"; shot.lastFrame = "Walking"
      return shot
    }
    plan.shots[1].firstAssetID = reference.id
    _ = try plan.combineShots(Set(plan.shots.prefix(2).map(\.id)))
    let combined = try plan.combineShots(Set(plan.shots.map(\.id)))
    try plan.approveShot(combined)
    project.planning = plan
    try project.applyPlanningShots([combined])
    let frames = project.clips[0].attachments.filter { $0.assetID == reference.id }
    XCTAssertEqual(frames.count, 1)
    XCTAssertEqual(frames.first?.role, .keyframe)
    XCTAssertEqual(frames.first?.time, 2)
  }
}
