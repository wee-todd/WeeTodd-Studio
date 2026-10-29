import MLX
import XCTest

@testable import Drama3MLX

final class NumericalTests: XCTestCase {
  func testNormMatchesIndependentFloatReference() throws {
    let result = Drama3Math.norm(
      MLXArray([Float(3), 4]).reshaped([1, 2]), weight: MLXArray([Float(1), 1]), epsilon: 0)
    XCTAssertEqual(result[0, 0].item(Float.self), Float(3 / sqrt(12.5)), accuracy: 0.00001)
  }
  func testGreedySamplingAndEOSCandidates() throws {
    let id = try Drama3Sampler.sample(
      logits: MLXArray([Float(-8), 9, 1]), temperature: 0, topK: 3, topP: 1)
    XCTAssertEqual(id.item(Int.self), 1)
    XCTAssertThrowsError(
      try Drama3Sampler.sample(
        logits: MLXArray([Float.nan, 0]), temperature: 0.7, topK: 2, topP: 1))
  }
}
