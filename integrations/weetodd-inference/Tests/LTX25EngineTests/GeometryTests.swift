import XCTest

@testable import LTX25Engine

final class GeometryTests: XCTestCase {
  func testCausalMidpointsAndAudioTiming() throws {
    let g = try AVGeometry(width: 64, height: 32, frames: 9, fps: 24)
    XCTAssertEqual(g.videoShape, [1, 128, 2, 1, 2])
    XCTAssertEqual(g.audioFrames, 10)
    let first: Float = 0.5 / 24
    let later: Float = 5.0 / 24
    let expected: [Float] = [first, 16, 16, first, 16, 48, later, 16, 16, later, 16, 48]
    XCTAssertEqual(g.videoPositions, expected)
    XCTAssertEqual(g.audioPositions[0], 0.005, accuracy: 1e-7)
    XCTAssertEqual(g.audioPositions[1], 0.03, accuracy: 1e-7)
    XCTAssertEqual(g.audioPositions[2], 0.07, accuracy: 1e-7)
  }
  func testUnpackingKeepsVideoChannelsAndAudioFrequencyInCorrectOrder() throws {
    let g = try AVGeometry(width: 64, height: 32, frames: 9, fps: 24)
    let v = (0..<(g.videoTokens * 128)).map(Float.init)
    let a = (0..<(g.audioFrames * 128)).map(Float.init)
    let uv = try g.unpackVideo(v)
    let ua = try g.unpackAudio(a)
    XCTAssertEqual(Array(uv.prefix(4)), [0, 128, 256, 384])
    XCTAssertEqual(Array(uv[4..<8]), [1, 129, 257, 385])
    XCTAssertEqual(Array(ua.prefix(16)), Array(a.prefix(16)))
    XCTAssertEqual(ua[16], 128)
    XCTAssertEqual(ua[g.audioFrames * 16], 16)
  }
  func testRejectsUnsafeOrUnsupportedGeometryAndUnpackShape() throws {
    for x in [
      (63, 64, 9, 24.0), (64, 64, 8, 24), (64, 64, 9, 0), (Int.max, 64, 9, 24),
      (64, 64, 9, Double.nan),
    ] {
      XCTAssertThrowsError(try AVGeometry(width: x.0, height: x.1, frames: x.2, fps: x.3))
    }
    let g = try AVGeometry(width: 32, height: 32, frames: 1, fps: 24)
    XCTAssertThrowsError(try g.unpackVideo([1]))
    XCTAssertThrowsError(try g.unpackAudio(Array(repeating: .nan, count: g.audioFrames * 128)))
  }
  func testNativeNoiseReplaysAndAdvancesWithoutGlobalState() throws {
    var first = GaussianNoise(seed: 123)
    var replay = GaussianNoise(seed: 123)
    var other = GaussianNoise(seed: 124)
    let prefix = try first.values(count: 3)
    XCTAssertEqual(prefix + (try first.values(count: 5)), try replay.values(count: 8))
    XCTAssertNotEqual(prefix, try other.values(count: 3))
    XCTAssertThrowsError(try first.values(count: Int.max))
    XCTAssertEqual(try first.values(count: 0), [])
  }
}
