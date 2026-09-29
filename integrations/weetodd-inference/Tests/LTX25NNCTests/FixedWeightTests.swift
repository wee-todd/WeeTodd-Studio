import XCTest
import InferenceTestSupport
@testable import LTX25NNC

final class FixedWeightTests: XCTestCase {
  func testPreflightRequiresEveryFixedTargetAndShapeBeforeReading() throws {
    let c = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let shapes = DenoiserLayout.weightShapes(c)
    let tensors = shapes.map { ("model.diffusion_model." + $0.key, $0.value, "BF16") }
    try withTensorFile(tensors: tensors) { url in
      let source = try FixedWeightSource(url: url, configuration: c)
      XCTAssertEqual(try source.read("proj_out.bias", shape: [128]), [Float](repeating: 0, count: 128))
      XCTAssertThrowsError(try source.read("proj_out.bias", shape: [64]))
    }
    try withTensorFile(tensors: Array(tensors.dropLast())) { url in
      XCTAssertThrowsError(try FixedWeightSource(url: url, configuration: c))
    }
  }

  func testOutputHeadNormalizesMeanRatherThanRMS() throws {
    let stage = FixedStage.projection("proj_out", rows: 1, width: 4, output: 2, table: "scale_shift_table")
    let output = try stage.evaluate([[2, 2, 2, 2], [0, 0, 0, 0]]) { name, shape in
      name == "proj_out.weight" ? [Float](repeating: 1, count: shape.reduce(1, *)) : [Float](repeating: 0, count: shape.reduce(1, *))
    }
    XCTAssertEqual(output[0], [0, 0])
  }

  func testDenoiserInvalidInputsFailBeforeCallingAnyWeightProvider() throws {
    let c = try AVBlockConfiguration(videoTokens: 5, audioTokens: 3, textTokens: 4)
    let runner = try DenoiserRunner(configuration: c)
    XCTAssertThrowsError(try runner.evaluate([:], sigma: 0.5, fixedWeights: { _, _ in
      XCTFail("Invalid request reached fixed weights"); return []
    }, blockWeights: { _, _, _ in XCTFail("Invalid request reached blocks"); return [] }))
  }
}
