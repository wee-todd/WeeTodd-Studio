import Foundation
import MLX
import XCTest
@testable import MOSSImageMLX

final class MathTests: XCTestCase {
  func testOptInProfilerRetainsValuesAndCountsEvaluations() {
    Device.withDefaultDevice(.cpu) {
      let profiler = MOSSProfiler()
      let input = MLXArray([Float(1), 2, 3])
      let first = profiler.evaluate("fixture", input * 2)
      let second = profiler.evaluate("fixture", first + 1)
      XCTAssertEqual(second.asArray(Float.self), [3, 5, 7])
      XCTAssertEqual(profiler.records["fixture"]?.count, 2)
      XCTAssertGreaterThanOrEqual(profiler.records["fixture"]?.seconds ?? -1, 0)
    }
  }
  func testSiTUUsesBothSaturatingBranches() {
    Device.withDefaultDevice(.cpu) {
      let out = MOSSMath.situ(MLXArray([Float(25), -25, 100]), MLXArray([Float(4), -4, 8]))
      assertClose(out.asArray(Float.self), [56.95932, 1.0432463, 96.30579], tolerance: 0.00003)
    }
  }
  func testTimeEmbeddingRetainsDoublePhaseBeforeFloatOutput() {
    assertClose(MOSSMath.timeEmbedding(1000, dimension: 4), [0.5623791, -0.8390715, 0.82687956, -0.5440211], tolerance: 1e-7)
  }
  func testFourAxisRoPEOrdersImageThenTextAndCentersSpatialPositions() {
    Device.withDefaultDevice(.cpu) {
      let rope = MOSSMath.rope(height: 2, width: 2, textLength: 1, headDimension: 8)
      XCTAssertEqual(rope.shape, [1, 5, 8])
      assertClose(rope[0, 0].asArray(Float.self), [0, 0.84147096, -0.47942555, -0.47942555, 1, 0.5403023, 0.87758255, 0.87758255], tolerance: 1e-6)
      assertClose(rope[0, 4].asArray(Float.self), [0, 0, 0, 0, 1, 1, 1, 1], tolerance: 1e-7)
    }
  }
  func testQwenZeroCenteredNormAndDeltaRuleEpsilon() {
    Device.withDefaultDevice(.cpu) {
      let normalized = MOSSQwenMath.norm(MLXArray([Float(3), 4]), weight: MLXArray([Float(0), 1]))
      assertClose(normalized.asArray(Float.self), [0.8485281, 2.2627417], tolerance: 1e-6)
      let q = MLXArray([Float(1), 0, 1, 0]).reshaped(1, 2, 1, 2)
      let k = q, v = MLXArray([Float(2), 4]).reshaped(1, 2, 1, 1)
      let zeros = MLXArray.zeros([1, 2, 1])
      let result = MOSSQwenMath.delta(q: q, k: k, v: v, a: zeros, b: zeros,
        aLog: MLXArray.zeros([1]), dtBias: MLXArray.zeros([1]), useKernel: false)
      assertClose(result.asArray(Float.self), [0.7071061, 1.590989], tolerance: 3e-6)
    }
  }
  func testQwenTemplateDoesNotGenerateOrPadTokens() {
    XCTAssertEqual(MOSSQwenTokenizer.render(""), "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n")
    XCTAssertEqual(MOSSQwenTokenizer.render(" 茶\n "), "<|im_start|>user\n 茶\n <|im_end|>\n<|im_start|>assistant\n<think>\n")
  }
  private func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float,
    file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: tolerance, file: file, line: line) }
  }
}
