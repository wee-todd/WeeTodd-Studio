import Foundation
import XCTest
@testable import StudioCore

final class ContinuousSceneTests: XCTestCase {
  func testBoundaryPolicyRoundTripLegacyDefaultAndSceneInvalidation() throws {
    let legacy = try JSONDecoder().decode(ClipContinuity.self, from: Data(#"{"mode":"scene"}"#.utf8))
    XCTAssertEqual(legacy.boundaryImagePolicy, "balanced")
    let legacyFields = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(legacy)) as? [String: Any])
    XCTAssertNil(legacyFields["boundaryImagePolicy"])
    var project = scene()
    let before = try project.continuousSceneInputFingerprint(for: project.clips[1])
    project.clips[0].continuity = ClipContinuity(boundaryImagePolicy: "strict")
    XCTAssertNotEqual(before, try project.continuousSceneInputFingerprint(for: project.clips[1]))
    let restored = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(project))
    XCTAssertEqual(restored.clips[0].continuity?.boundaryImagePolicy, "strict")
    XCTAssertEqual(try restored.continuousSceneMembers(for: restored.clips[1]).count, 3)
    project.clips[0].continuity?.boundaryImagePolicy = "invalid"
    XCTAssertTrue(project.continuousSceneIssues(for: project.clips[1]).joined().contains("Automatic or Strict"))
  }

  private func scene(count: Int = 3) -> StudioProject {
    var project = StudioProject()
    project.clips = (0..<count).map { index in
      var clip = Clip(name: "Shot \(index + 1)", engine: .ltx25)
      if index > 0 { clip.continuity = ClipContinuity(mode: "scene") }
      return clip
    }
    return project
  }

  func testSceneIssuesDoNotRequireAnAcceptedSourceMovie() {
    let project = scene()
    for clip in project.clips {
      XCTAssertTrue(project.continuityIssues(for: clip).isEmpty)
    }
  }

  func testEveryMemberResolvesTheCompleteMaximalGroup() throws {
    var project = scene(count: 6)
    let ids = project.clips.map(\.id)
    for clip in project.clips {
      XCTAssertEqual(try project.continuousSceneMembers(for: clip).map(\.id), ids)
    }
    let ordinary = Clip()
    project.clips.append(ordinary)
    XCTAssertTrue(try project.continuousSceneMembers(for: ordinary).isEmpty)
  }

  func testSceneRejectsInvalidPredecessorsMixedEnginesAndLimits() throws {
    let original = scene()
    var project = original
    project.clips[1].continuity?.sourceClipID = original.clips[2].id
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[2]))
    project = original
    project.clips[1].continuity?.sourceClipID = UUID()
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[0]))
    project = original
    project.clips[1].continuity?.sourceClipID = original.clips[0].id
    XCTAssertEqual(try project.continuousSceneMembers(for: project.clips[1]).count, 3)
    for mode in ["frame", "motion", "scene"] {
      project = original
      project.clips[0].continuity = ClipContinuity(mode: mode)
      XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[2]))
    }
    for engine in [Engine.h3, .ltx23, .drawThings, .movie] {
      project = original
      project.clips[1].engine = engine
      XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[0]))
      XCTAssertFalse(project.continuityIssues(for: project.clips[0]).isEmpty)
    }
    project = scene(count: 7)
    project.clips = project.clips.map { var clip = $0; clip.duration = 1; return clip }
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[3]))
    for duration in [0, -1, .infinity, .nan, 21] as [Double] {
      project = original; project.clips[0].duration = duration
      XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[1]))
    }
  }

  func testRemovedReorderedAndDuplicatedMembersFailSafely() throws {
    var project = scene()
    let absent = project.clips.removeFirst()
    XCTAssertThrowsError(try project.continuousSceneMembers(for: absent))
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[0]))
    project = scene()
    let predecessorID = project.clips[1].id
    project.clips[2].continuity?.sourceClipID = predecessorID
    project.clips.swapAt(0, 1)
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[2]))
    project = scene()
    project.clips[2].id = project.clips[1].id
    XCTAssertThrowsError(try project.continuousSceneMembers(for: project.clips[0]))
  }

  func testSceneIssuesRejectUnsupportedAttachmentsAndExtensions() {
    for role in [MediaRole.reference, .audioDriver, .control] {
      var project = scene()
      project.clips[1].attachments = [Attachment(assetID: UUID(), role: role)]
      XCTAssertFalse(project.continuityIssues(for: project.clips[0]).isEmpty)
    }
    var project = scene()
    project.clips[2].extensionDirection = "after"
    XCTAssertFalse(project.continuityIssues(for: project.clips[1]).isEmpty)
    project = scene()
    for role in [MediaRole.first, .last, .keyframe, .lora] {
      project.clips[1].attachments = [Attachment(assetID: UUID(), role: role)]
      XCTAssertTrue(project.continuityIssues(for: project.clips[0]).isEmpty)
    }
  }

  func testFingerprintExcludesAcceptedMediaAndUnrelatedEditingState() throws {
    var project = scene()
    let original = try project.continuousSceneInputFingerprint(for: project.clips[1])
    XCTAssertEqual(original, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    project.clips[0].sourcePath = "/accepted.mp4"
    project.clips[0].sourceIn = 3
    project.clips[0].versions = [RenderVersion(path: "/accepted.mp4", seed: 2, prompt: "old", recipePath: "")]
    project.clips[0].renderedSignature = "rendered"
    project.clips[0].validatedSignature = "validated"
    project.clips[0].name = "Renamed"
    project.clips[0].volume = 0
    project.clips[0].transition = "fade"
    project.name = "Other name"
    project.clips.append(Clip())
    XCTAssertEqual(original, try project.continuousSceneInputFingerprint(for: project.clips[2]))
    XCTAssertEqual(original, project.continuityDependencyFingerprint(for: project.clips[0]))
  }

  func testFingerprintTracksEveryMemberGenerationInputAndResolvedSettings() throws {
    let original = scene()
    let fingerprint = try original.continuousSceneInputFingerprint(for: original.clips[0])
    let edits: [(inout Clip) -> Void] = [
      { $0.prompt = "Changed" }, { $0.seed += 1 }, { $0.duration += 1 },
      { $0.generationWidth += 64 }, { $0.generationHeight += 64 },
      { $0.profileID = "custom-profile" },
      { $0.generationSelection = GenerationSelection(preset: .lowMemory) },
      { $0.negativePrompt = "Changed negative" },
      { $0.attachments = [Attachment(assetID: UUID(), role: .last)] }
    ]
    for index in original.clips.indices {
      for edit in edits {
        var project = original; edit(&project.clips[index])
        XCTAssertNotEqual(fingerprint, try project.continuousSceneInputFingerprint(for: project.clips[0]))
      }
    }
    var project = original; project.settings.fps = 25
    XCTAssertNotEqual(fingerprint, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    project = original; project.clips[2].settingsOverride = MovieSettings()
    project.clips[2].settingsOverride?.fps = 25
    XCTAssertNotEqual(fingerprint, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    project = original; project.clips[0].soundscape = "Rain"
    XCTAssertNotEqual(fingerprint, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    project = original; project.clips[0].music = "Slow piano"
    XCTAssertNotEqual(fingerprint, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    XCTAssertNotEqual(fingerprint, try original.continuousSceneInputFingerprint(for: original.clips[0], runtimeIdentity: "changed-runtime"))
  }

  func testFingerprintTracksAssetRelinkAndCheapFileReplacementMetadata() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try Data([0]).write(to: url)
    var project = scene()
    let asset = MediaAsset(name: "Endpoint", kind: .image, path: url.path)
    project.assets = [asset]
    project.clips[1].attachments = [Attachment(assetID: asset.id, role: .last)]
    let original = try project.continuousSceneInputFingerprint(for: project.clips[0])
    try Data([0, 1]).write(to: url)
    XCTAssertNotEqual(original, try project.continuousSceneInputFingerprint(for: project.clips[1]))
    let replaced = try project.continuousSceneInputFingerprint(for: project.clips[0])
    project.assets[0].path = "/relinked.png"
    XCTAssertNotEqual(replaced, try project.continuousSceneInputFingerprint(for: project.clips[1]))
    XCTAssertNotEqual(replaced, try project.continuousSceneInputFingerprint(for: project.clips[1], assets: []))
  }

  func testFingerprintDetectsSameSizeInputReplacedWithinOneSecond() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try Data([0]).write(to: url)
    let modified = Date(timeIntervalSince1970: 1_700_000_000.1)
    try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: url.path)
    var project = scene()
    let asset = MediaAsset(name: "Endpoint", kind: .image, path: url.path)
    project.assets = [asset]
    project.clips[1].attachments = [Attachment(assetID: asset.id, role: .last)]
    let original = try project.continuousSceneInputFingerprint(for: project.clips[0])
    try FileManager.default.setAttributes([.modificationDate: modified.addingTimeInterval(0.5)], ofItemAtPath: url.path)
    XCTAssertNotEqual(original, try project.continuousSceneInputFingerprint(for: project.clips[0]))
  }

  private func ranges(_ project: StudioProject) -> [ContinuousSceneMember] {
    var start = 0.0
    return project.clips.map { clip in
      defer { start += clip.duration }
      return ContinuousSceneMember(clipID: clip.id, sourceIn: start, duration: clip.duration)
    }
  }

  private func versions(_ project: StudioProject, path: String) -> [UUID: RenderVersion] {
    Dictionary(uniqueKeysWithValues: project.clips.map {
      ($0.id, RenderVersion(path: path, seed: $0.seed, prompt: $0.prompt, recipePath: "scene.json"))
    })
  }

  func testAcceptIsAtomicPreservesOldMediaAndSupportsValueUndo() throws {
    var project = scene()
    project.clips[0].sourcePath = "/previous-import.mp4"
    let previous = RenderVersion(path: "/previous-take.mp4", seed: 1, prompt: "Old", recipePath: "old.json")
    project.clips[1].versions = [previous]; project.clips[1].sourcePath = previous.path
    let original = project
    let input = try project.continuousSceneInputFingerprint(for: project.clips[0])
    let members = ranges(project)
    try project.acceptContinuousScene(versions: versions(project, path: "/scene.mp4"), members: members)
    let groupID = try XCTUnwrap(project.clips[0].versions.last?.sceneTakeID)
    for (index, clip) in project.clips.enumerated() {
      XCTAssertEqual(clip.sourcePath, "/scene.mp4")
      XCTAssertEqual(clip.sourceIn, Double(index) * 5)
      XCTAssertEqual(clip.duration, 5)
      XCTAssertEqual(clip.versions.last?.sceneMembers, members)
      XCTAssertEqual(clip.versions.last?.sceneTakeID, groupID)
      XCTAssertEqual(clip.versions.last?.usableSourceIn, Double(index) * 5)
      XCTAssertEqual(clip.versions.last?.usableDuration, 5)
    }
    XCTAssertEqual(project.clips[0].versions.first?.path, "/previous-import.mp4")
    XCTAssertEqual(project.clips[1].versions.first, previous)
    XCTAssertEqual(input, try project.continuousSceneInputFingerprint(for: project.clips[0]))
    project = original
    XCTAssertEqual(project.clips[0].sourcePath, "/previous-import.mp4")
    XCTAssertEqual(project.clips[1].versions, [previous])
  }

  func testAcceptInvalidRangesOrPartialResultNeverChangesAnyMember() throws {
    let original = scene()
    let valid = ranges(original)
    let badRanges: [[ContinuousSceneMember]] = [
      Array(valid.dropLast()), Array(valid.reversed()),
      [ContinuousSceneMember(clipID: valid[0].clipID, sourceIn: 1, duration: 5)] + valid.dropFirst(),
      [ContinuousSceneMember(clipID: valid[0].clipID, sourceIn: 0, duration: .nan)] + valid.dropFirst(),
      [ContinuousSceneMember(clipID: valid[0].clipID, sourceIn: 0, duration: 6)] + valid.dropFirst(),
      [ContinuousSceneMember(clipID: UUID(), sourceIn: 0, duration: 5)] + valid.dropFirst()
    ]
    for members in badRanges {
      var project = original
      XCTAssertThrowsError(try project.acceptContinuousScene(versions: versions(original, path: "/scene.mp4"), members: members))
      XCTAssertEqual(project, original)
    }
    var project = original
    var takes = versions(original, path: "/scene.mp4")
    takes.removeValue(forKey: original.clips[1].id)
    XCTAssertThrowsError(try project.acceptContinuousScene(versions: takes, members: valid))
    XCTAssertEqual(project, original)
    takes = versions(original, path: "/scene.mp4")
    takes[original.clips[1].id]?.path = "/other.mp4"
    XCTAssertThrowsError(try project.acceptContinuousScene(versions: takes, members: valid))
    XCTAssertEqual(project, original)
  }

  func testHistoricalTakeActivatesEveryMemberAndRejectsIndividualSwitch() throws {
    var project = scene()
    try project.acceptContinuousScene(versions: versions(project, path: "/first.mp4"), members: ranges(project))
    let first = project.clips[1].versions[0]
    try project.acceptContinuousScene(versions: versions(project, path: "/second.mp4"), members: ranges(project))
    XCTAssertThrowsError(try project.clips[1].activateVersion(first))
    try project.activateContinuousSceneVersion(selectedClipID: project.clips[1].id, version: first)
    XCTAssertEqual(project.clips.map(\.sourcePath), Array(repeating: "/first.mp4", count: 3))
    XCTAssertEqual(project.clips.map { $0.versions.count }, [2, 2, 2])
  }

  func testAcceptRejectsReusingAnExistingSceneTakeIDAtomically() throws {
    var project = scene()
    try project.acceptContinuousScene(versions: versions(project, path: "/first.mp4"), members: ranges(project))
    let original = project
    var incoming = versions(project, path: "/second.mp4")
    for id in incoming.keys { incoming[id]?.sceneTakeID = project.clips[0].versions[0].sceneTakeID }
    XCTAssertThrowsError(try project.acceptContinuousScene(versions: incoming, members: ranges(project)))
    XCTAssertEqual(project, original)
  }

  func testHistoricalTakeRejectsMissingMovedAndObsoleteGroupsWithoutMutation() throws {
    var accepted = scene()
    try accepted.acceptContinuousScene(versions: versions(accepted, path: "/first.mp4"), members: ranges(accepted))
    let selectedID = accepted.clips[1].id, version = accepted.clips[1].versions[0]
    var examples: [StudioProject] = []
    var changed = accepted; changed.clips.removeLast(); examples.append(changed)
    changed = accepted; changed.clips.swapAt(1, 2); examples.append(changed)
    changed = accepted; changed.clips[2].continuity = nil; examples.append(changed)
    changed = accepted; changed.clips[0].versions = []; examples.append(changed)
    changed = accepted; changed.clips[0].versions[0].path = "/different.mp4"; examples.append(changed)
    for original in examples {
      var project = original
      XCTAssertThrowsError(try project.activateContinuousSceneVersion(selectedClipID: selectedID, version: version))
      XCTAssertEqual(project, original)
    }
  }

  func testSceneRangesAndGroupedVersionsRoundTripWhileOldProjectsStayIndependent() throws {
    let oldClip = Clip()
    let decoded = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(oldClip))
    XCTAssertEqual(decoded.continuityMode, "independent")
    let oldVersion = RenderVersion(path: "/old.mp4", seed: 1, prompt: "", recipePath: "")
    let oldDecoded = try JSONDecoder().decode(RenderVersion.self, from: JSONEncoder().encode(oldVersion))
    XCTAssertNil(oldDecoded.sceneMembers); XCTAssertNil(oldDecoded.sceneTakeID)
    var project = scene()
    try project.acceptContinuousScene(versions: versions(project, path: "/scene.mp4"), members: ranges(project))
    let data = try JSONEncoder().encode(project)
    let json = String(decoding: data, as: UTF8.self)
    XCTAssertTrue(json.contains("clip_id")); XCTAssertTrue(json.contains("source_in"))
    XCTAssertEqual(try JSONDecoder().decode(StudioProject.self, from: data), project)
  }
}
