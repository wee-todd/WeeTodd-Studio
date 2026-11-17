import MLX
import XCTest
@testable import MOSSImageMLX

final class SamplerTests: XCTestCase {
  func testSimpleScheduleUsesTrainingIndicesAndDiffersFromRelease() throws {
    let values = try MOSSSchedule.simple(steps: 4, shift: 5)
    XCTAssertEqual(values.count, 5)
    let expected: [Float] = [1, 0.9375, 5.0 / 6, 0.625, 0]
    for (a, b) in zip(values, expected) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    XCTAssertNotEqual(values, try MOSSSchedule.sigmas(steps: 4))
  }
  func testERSDEUsesFlowAlphaAndInjectedNoise() throws {
    try Device.withDefaultDevice(.cpu) {
      var steps: [Float] = []
      let result = try MOSSERSDESampler.sample(initial: MLXArray([Float(1)]), sigmas: [0.8, 0.4, 0],
        guidance: 1, skipLayers: [], skipStart: 0, skipStop: 1, noise: { _ in MLXArray([Float(0.25)]) },
        predict: { _, _, _, _ in MLXArray([Float(0.5)]) }, onStep: { _, _, x in steps.append(x.item(Float.self)) })
      // Independent scalar first-order VP update: lambda=sigma/(1-sigma), phi=lambda*(exp(lambda^.3)+10).
      let s: Double = 4, t: Double = 2.0 / 3
      let phi: (Double) -> Double = { $0 * (exp(pow($0, 0.3)) + 10) }
      let ratio = phi(t) / phi(s)
      let first = 3 * ratio + 0.6 * (1 - ratio) * 0.6 + 0.6 * 0.25 * sqrt(t*t-s*s*ratio*ratio)
      XCTAssertEqual(steps[0], Float(first), accuracy: 2e-6)
      XCTAssertEqual(result.item(Float.self), Float(first) - 0.2, accuracy: 2e-6)
      XCTAssertNotEqual(result.item(Float.self), 0.6)
    }
  }
  func testNonfinitePredictionFailsInsteadOfSavingABlackImage() throws {
    try Device.withDefaultDevice(.cpu) {
      XCTAssertThrowsError(try MOSSEulerSampler.sample(initial: MLXArray([Float(0)]), sigmas: [1, 0], guidance: 1,
        skipLayers: [], skipStart: 0, skipStop: 1, predict: { _, _, _, _ in MLXArray([Float.nan]) }))
    }
  }
  func testCFGAndEulerUseBothBranchesWithNegativeOnlyLayerSkipping() throws {
    try Device.withDefaultDevice(.cpu) {
      var negativeSkips: [Set<Int>] = [], calls = 0
      let result = try MOSSEulerSampler.sample(initial: MLXArray([Float(0)]), sigmas: [1, 0.5, 0],
        guidance: 3, skipLayers: [22], skipStart: 0, skipStop: 0.5,
        predict: { _, _, negative, skip in
          calls += 1
          if negative { negativeSkips.append(skip) } else { XCTAssertTrue(skip.isEmpty) }
          return MLXArray([negative ? Float(1) : Float(2)])
        })
      XCTAssertEqual(result.item(Float.self), -4)
      XCTAssertEqual(calls, 4)
      XCTAssertEqual(negativeSkips, [Set([22]), Set<Int>()])
    }
  }
  func testCFGOneNeverEncodesOrEvaluatesNegativeBranch() throws {
    try Device.withDefaultDevice(.cpu) {
      var calls = 0
      let result = try MOSSEulerSampler.sample(initial: MLXArray([Float(1)]), sigmas: [1, 0], guidance: 1,
        skipLayers: [22], skipStart: 0, skipStop: 1, predict: { _, _, negative, skip in
          calls += 1; XCTAssertFalse(negative); XCTAssertTrue(skip.isEmpty); return MLXArray([Float(2)])
        })
      XCTAssertEqual(result.item(Float.self), -1); XCTAssertEqual(calls, 1)
    }
  }
}
