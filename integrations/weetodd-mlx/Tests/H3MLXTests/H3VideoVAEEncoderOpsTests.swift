import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEEncoderOpsTests: XCTestCase {
  func testSequentialConvolutionTermsMatchWholeVolumeAndPreserveOtherStrideFallbacks() throws {
    let pixels: [Float] = (0..<4480).map { Float(($0 % 17) - 8) / 8 }
    let weights: [Float] = (0..<6912).map { Float(($0 % 13) - 6) / 32 }
    let input = MLXArray(pixels, [1, 5, 7, 8, 16])
    let weight = MLXArray(weights, [16, 3, 3, 3, 16])
    for dtype in [DType.float32, .float16] {
      for stride in [IntOrTriple(1), IntOrTriple((2, 2, 2))] {
        let value = input.asType(dtype), kernel = weight.asType(dtype)
        let bias = MLXArray((0..<16).map { Float($0) / 16 }).asType(dtype)
        let expected = try H3VideoVAEEncoderOps.causalConv(value, weight: kernel,
          bias: bias, spatialPadding: 1, temporalPadding: 2, stride: stride)
        let bounded = try H3VideoVAEEncoderOps.causalConv(value, weight: kernel,
          bias: bias, spatialPadding: 1, temporalPadding: 2, stride: stride,
          releaseTemporalTerms: true)
        XCTAssertEqual(bounded.shape, expected.shape)
        XCTAssertLessThan(max(abs(bounded.asType(.float32) - expected.asType(.float32))).item(Float.self),
          dtype == .float32 ? 0.00001 : 0.004)
      }
    }
  }

  func testSequentialConvolutionTermsPreserveFP16SumOrder() throws {
    let values: [Float] = [10000, -10000, 1]
    let output = try H3VideoVAEEncoderOps.sumEvaluatedConvolutionTerms(count: 3) {
      MLXArray([values[$0]], [1]).asType(.float16)
    }
    XCTAssertEqual(output.item(Float.self), 1)
  }

  func testCancellationBetweenConvolutionTermsStopsAndRestoresCacheLimit() async {
    let result = await Task.detached { () -> (Int, Bool) in
      let cacheLimit = Memory.cacheLimit
      var calls = 0
      do {
        _ = try H3VideoVAEEncoderOps.sumEvaluatedConvolutionTerms(count: 3) { _ in
          calls += 1
          withUnsafeCurrentTask { $0?.cancel() }
          return MLXArray.ones([1, 3, 3, 16], dtype: .float16)
        }
        return (-1, false)
      } catch is CancellationError {
        return (calls, Memory.cacheLimit == cacheLimit)
      } catch { return (-1, false) }
    }.value
    XCTAssertEqual(result.0, 1)
    XCTAssertTrue(result.1)
  }

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
