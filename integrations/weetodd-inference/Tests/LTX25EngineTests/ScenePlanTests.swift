import XCTest
@testable import LTX25Engine

final class ScenePlanTests: XCTestCase {
  func testCumulativeShotQuantizationAndAVOverlaps() throws {
    let plan = try LTX25ScenePlan(durations: [2, 3], fps: 24)
    XCTAssertEqual(plan.segmentFrames, [48, 72])
    XCTAssertEqual(plan.windowFrames, [49, 97])
    XCTAssertEqual(plan.windowStarts, [0, 24])
    XCTAssertEqual(plan.segmentStarts, [0, 48])
    XCTAssertEqual(plan.totalFrames, 121)
    XCTAssertEqual(plan.videoOverlapLatentFrames, 4)
    XCTAssertEqual(plan.windowAudioTokens, [52, 102])
    XCTAssertEqual(plan.joinAudioTokens, [27])
    XCTAssertEqual(plan.expectedAudioTokens, 127)
    XCTAssertEqual(plan.windowAudioTokens[0], try AVGeometry(width: 768,
      height: 448, frames: plan.windowFrames[0], fps: 24).audioFrames)
  }

  func testRejectsTooShortOverlongAndNonfiniteScenesBeforeInference() throws {
    for durations in [[1.0], [1.0, 0.0], [0.1, 0.1], [20, 20],
      [1, 1, 1, 1, 1, 1, 1], [Double.nan, 2]] {
      XCTAssertThrowsError(try LTX25ScenePlan(durations: durations, fps: 24))
    }
    XCTAssertThrowsError(try LTX25ScenePlan(durations: [2, 3], fps: 0))
    XCTAssertThrowsError(try LTX25ScenePlan(durations: [2, 3], fps: 24,
      overlapFrames: 24))
  }
}
