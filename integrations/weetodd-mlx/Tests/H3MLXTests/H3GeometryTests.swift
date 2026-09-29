import XCTest
@testable import H3MLX

final class H3GeometryTests: XCTestCase {
  func testFiveSecondRequestSharesAlignedVideoAndAudioClock() throws {
    let geometry = try H3Geometry(width: 768, height: 448, durationSeconds: 5)
    XCTAssertEqual(geometry.frames, 124)
    XCTAssertEqual(geometry.videoLatentFrames, 37)
    XCTAssertEqual(geometry.audioLatentFrames, 207)
    XCTAssertEqual(geometry.videoRows, 12_432)
    XCTAssertEqual(geometry.audioRows, 414)
    XCTAssertEqual(try geometry.packedRows(textRows: 100,
      conditionVideoRows: 20, conditionAudioRows: 40), 13_006)
  }

  func testDurationAndCanvasFailBeforeAnyAllocation() throws {
    XCTAssertThrowsError(try H3Geometry(width: 770, height: 448, durationSeconds: 5))
    XCTAssertThrowsError(try H3Geometry(width: 768, height: 448, durationSeconds: 2.49))
    XCTAssertThrowsError(try H3Geometry(width: 768, height: 448, durationSeconds: 15.01))
    XCTAssertThrowsError(try H3Geometry(width: 768, height: 448, durationSeconds: .infinity))
    let geometry = try H3Geometry(width: 768, height: 448, durationSeconds: 5)
    XCTAssertThrowsError(try geometry.packedRows(textRows: 0,
      conditionVideoRows: 0, conditionAudioRows: 0))
    XCTAssertThrowsError(try geometry.packedRows(textRows: 1,
      conditionVideoRows: .max, conditionAudioRows: 0))
  }
}
