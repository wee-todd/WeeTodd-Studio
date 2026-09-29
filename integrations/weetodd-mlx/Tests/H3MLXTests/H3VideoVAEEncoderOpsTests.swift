import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEEncoderOpsTests: XCTestCase {
  func testCausalConvolutionCannotReadFutureFrames() throws {
    let input = MLXArray([Float](repeating: 1, count: 9)
      + [Float](repeating: 2, count: 9)
      + [Float](repeating: 3, count: 9), [1, 3, 3, 3, 1])
    var values = [Float](repeating: 0, count: 27)
    values[1 * 9 + 1 * 3 + 1] = 1 // previous frame, center pixel
    let weight = MLXArray(values, [1, 3, 3, 3, 1])
    let result = try H3VideoVAEEncoderOps.causalConv(input, weight: weight,
      bias: MLXArray([Float(0)]), spatialPadding: 1,
      temporalPadding: 2)
    XCTAssertEqual(result.shape, [1, 3, 3, 3, 1])
    XCTAssertEqual(result[0, 0, 1, 1, 0].item(Float.self), 0)
    XCTAssertEqual(result[0, 1, 1, 1, 0].item(Float.self), 1)
    XCTAssertEqual(result[0, 2, 1, 1, 0].item(Float.self), 2)
  }

  func testSpatialPaddingReflectsWithoutRepeatingBorder() throws {
    let input = MLXArray([Float(1), 2, 3,
      4, 5, 6,
      7, 8, 9], [1, 1, 3, 3, 1])
    let padded = try H3VideoVAEEncoderOps.reflectSpatial(input,
      axis: 3, before: 1, after: 1)
    XCTAssertEqual(padded.shape, [1, 1, 3, 5, 1])
    XCTAssertEqual(padded[0, 0, 0, 0, 0].item(Float.self), 2)
    XCTAssertEqual(padded[0, 0, 0, 1, 0].item(Float.self), 1)
    XCTAssertEqual(padded[0, 0, 0, 4, 0].item(Float.self), 2)
  }

  func testGroupNormalizationKeepsFrameStatisticsIsolated() throws {
    let input = MLXArray([Float](repeating: 1, count: 32)
      + [Float](repeating: 3, count: 32)
      + [Float](repeating: 10, count: 32)
      + [Float](repeating: 14, count: 32), [1, 2, 2, 1, 32])
    let output = try H3VideoVAEEncoderOps.temporallyIsolatedGroupNorm(input,
      weight: MLXArray([Float](repeating: 2, count: 32)),
      bias: MLXArray([Float](repeating: 3, count: 32)))
    XCTAssertEqual(output[0, 0, 0, 0, 0].item(Float.self), 1, accuracy: 0.001)
    XCTAssertEqual(output[0, 0, 1, 0, 0].item(Float.self), 5, accuracy: 0.001)
    XCTAssertEqual(output[0, 1, 0, 0, 0].item(Float.self), 1, accuracy: 0.001)
    XCTAssertEqual(output[0, 1, 1, 0, 0].item(Float.self), 5, accuracy: 0.001)
  }
}
