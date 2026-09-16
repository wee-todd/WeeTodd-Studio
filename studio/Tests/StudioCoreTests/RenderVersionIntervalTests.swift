import Foundation
import XCTest
@testable import StudioCore

final class RenderVersionIntervalTests: XCTestCase {
  func version(_ path: String, start: Double?, duration: Double?) throws -> RenderVersion {
    let version = RenderVersion(path: path, seed: 1, prompt: "", recipePath: "")
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(version)) as! [String: Any]
    object["usableSourceIn"] = start
    object["usableDuration"] = duration
    return try JSONDecoder().decode(RenderVersion.self, from: JSONSerialization.data(withJSONObject: object))
  }
  func testReselectingLegacyActiveVersionPreservesExtensionOffsetAndTrim() throws {
    var clip = Clip(); clip.sourcePath = "extension.mov"; clip.sourceIn = 12; clip.duration = 3
    let legacy = try version("extension.mov", start: nil, duration: nil)
    clip.versions = [legacy]
    try clip.activateVersion(legacy)
    XCTAssertEqual(clip.sourceIn, 12); XCTAssertEqual(clip.duration, 3)
  }
  func testSwitchingVersionRetainsSplitRelativeTrimAndClampsToUsableInterval() throws {
    let a = try version("a.mov", start: 10, duration: 5)
    let b = try version("b.mov", start: 20, duration: 4)
    var clip = Clip(); clip.sourcePath = "a.mov"; clip.sourceIn = 10; clip.duration = 5; clip.versions = [a, b]
    var project = StudioProject(); project.clips = [clip]
    let id = try project.split(clip.id, at: 2)
    let index = project.clips.firstIndex { $0.id == id }!
    try project.clips[index].activateVersion(b)
    XCTAssertEqual(project.clips[index].sourceIn, 22)
    XCTAssertEqual(project.clips[index].duration, 2)
    let roundtrip = try JSONSerialization.jsonObject(with: JSONEncoder().encode(b)) as! [String: Any]
    XCTAssertEqual(roundtrip["usableSourceIn"] as? Double, 20)
    XCTAssertEqual(roundtrip["usableDuration"] as? Double, 4)
  }
  func testVersionWithNoRemainingTrimRangeIsRejectedWithoutChangingClip() throws {
    let a = try version("a.mov", start: 10, duration: 8)
    let b = try version("b.mov", start: 20, duration: 2)
    var clip = Clip(); clip.sourcePath = "a.mov"; clip.sourceIn = 13; clip.duration = 2; clip.versions = [a, b]
    let before = clip
    XCTAssertThrowsError(try clip.activateVersion(b))
    XCTAssertEqual(clip, before)
  }
  func testLegacyAppendCanSwitchToKnownUsableSegmentWithoutCountingContextAsTrim() throws {
    let legacy = try version("legacy.mov", start: nil, duration: nil)
    let current = try version("current.mov", start: 8, duration: 4)
    var clip = Clip(); clip.sourcePath = legacy.path; clip.sourceIn = 8; clip.duration = 4
    clip.extensionSource = "context.mov"; clip.extensionDirection = "after"
    clip.versions = [legacy, current]
    try clip.activateVersion(current)
    XCTAssertEqual(clip.sourcePath, "current.mov")
    XCTAssertEqual(clip.sourceIn, 8)
    XCTAssertEqual(clip.duration, 4)
  }

  func testExplicitVersionSwitchDoesNotInventTrimFromAmbiguousLegacyContext() throws {
    let legacy = try version("legacy.mov", start: nil, duration: nil)
    let current = try version("current.mov", start: 8, duration: 4)
    var clip = Clip(); clip.sourcePath = legacy.path; clip.sourceIn = 12; clip.duration = 2
    clip.extensionSource = "context.mov"; clip.extensionDirection = "after"
    clip.versions = [legacy, current]
    try clip.activateVersion(current)
    XCTAssertEqual(clip.sourceIn, 8)
    XCTAssertEqual(clip.duration, 2)
  }

  func testSelectingLegacyAppendRetainsKnownContextBoundaryAndTrim() throws {
    let legacy = try version("legacy.mov", start: nil, duration: nil)
    let current = try version("current.mov", start: 8, duration: 4)
    var clip = Clip(); clip.sourcePath = current.path; clip.sourceIn = 9; clip.duration = 2
    clip.extensionSource = "context.mov"; clip.extensionDirection = "after"
    clip.versions = [legacy, current]
    try clip.activateVersion(legacy)
    XCTAssertEqual(clip.sourcePath, "legacy.mov")
    XCTAssertEqual(clip.sourceIn, 9)
    XCTAssertEqual(clip.duration, 2)
  }

}
