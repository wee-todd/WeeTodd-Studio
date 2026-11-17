import XCTest
@testable import ReCaVSRMLX

final class ReCaStreamStateTests: XCTestCase {
  func testGeometryRoundsVisibleAxesAndKeepsPaddingInternal() throws {
    let g = try ReCaGeometry(sourceWidth: 321, sourceHeight: 181, scale: 1.5)
    XCTAssertEqual([g.width, g.height, g.paddedWidth, g.paddedHeight], [482, 272, 512, 288])
    XCTAssertEqual([g.tokenWidth, g.tokenHeight], [16, 9])
    XCTAssertEqual(g.sourceIndex(482, axisSize: 482), 480)
    let tiny = try ReCaGeometry(sourceWidth: 1, sourceHeight: 1, scale: 2)
    XCTAssertEqual(tiny.sourceIndex(31, axisSize: 2), 1)
    XCTAssertThrowsError(try ReCaGeometry(sourceWidth: 0, sourceHeight: 181, scale: 4))
    XCTAssertThrowsError(try ReCaGeometry(sourceWidth: Int.max, sourceHeight: 181, scale: 4))
  }

  func testScheduleRepeatsOnlyInternalTailAndRestartsPrefixAtCuts() throws {
    for total in [1, 20, 21, 22, 28, 29, 30, 101] {
      let blocks = try ReCaBlockSchedule.blocks(frameCount: total)
      XCTAssertEqual(blocks.reduce(0) { $0 + $1.validFrames }, total)
      XCTAssertEqual(blocks[0].rgbFrames, 21)
      XCTAssertEqual(blocks.dropFirst().map(\.rgbFrames), Array(repeating: 8, count: blocks.count - 1))
    }
    let blocks = try ReCaBlockSchedule.blocks(frameCount: 30, resetFrames: [22])
    XCTAssertEqual(blocks.map(\.sourceStart), [0, 21, 22])
    XCTAssertEqual(blocks.map(\.validFrames), [21, 1, 8])
    XCTAssertEqual(blocks.map(\.latentFrames), [6, 2, 6])
    XCTAssertEqual(blocks.map(\.shotStart), [0, 0, 22])
    XCTAssertThrowsError(try ReCaBlockSchedule.blocks(frameCount: 0))
    XCTAssertThrowsError(try ReCaBlockSchedule.blocks(frameCount: 30, resetFrames: [30]))
  }

  func testRouterDeduplicatesAnchorsAndNeverResurrectsEvictedFrames() throws {
    XCTAssertEqual(try ReCaKVRouter.retainedIDs(candidates: Array(0...11), layer: 9), [5, 10, 11])
    XCTAssertEqual(try ReCaKVRouter.retainedIDs(candidates: Array(0...23), layer: 14), [17, 23])
    XCTAssertEqual(try ReCaKVRouter.retainedIDs(candidates: Array(0...7), layer: 0), [])
    for layer in 0..<30 {
      var ids: [Int] = []
      for start in stride(from: 0, to: 10_000, by: 2) {
        let candidates = ids + [start, start + 1]
        ids = try ReCaKVRouter.retainedIDs(candidates: candidates, layer: layer)
        XCTAssertTrue(ids.allSatisfy(candidates.contains))
        XCTAssertLessThanOrEqual(ids.count, ReCaKVRouter.capacity(layer: layer))
      }
    }
    XCTAssertEqual((0..<30).reduce(0) { $0 + ReCaKVRouter.capacity(layer: $1) }, 66)
  }

  func testRollingOriginUsesAbsoluteLatentIDsInsteadOfSlots() throws {
    let prefix = try ReCaRollingRoPE.positions(start: 0, frames: 6)
    XCTAssertEqual(prefix.origin, 0); XCTAssertEqual(prefix.positions, [0, 1, 2, 3, 4, 5])
    let rolled = try ReCaRollingRoPE.positions(start: 22, frames: 2)
    XCTAssertEqual(rolled.origin, 2); XCTAssertEqual(rolled.positions, [20, 21])
    XCTAssertThrowsError(try ReCaRollingRoPE.positions(start: -1, frames: 2))
    XCTAssertThrowsError(try ReCaRollingRoPE.positions(start: 0, frames: 23))
  }
}
