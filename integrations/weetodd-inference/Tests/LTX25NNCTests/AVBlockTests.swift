import XCTest
@testable import LTX25NNC

final class AVBlockTests: XCTestCase {
  func testFiniteInputsThatOverflowDoNotReturnSuccessfulOutput() throws {
    let configuration = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let runner = try AVBlockRunner(configuration: configuration)
    let inputs = runner.inputShapes.mapValues { [Float](repeating: 1, count: $0.reduce(1, *)) }
    try runner.load { _, shape in [Float](repeating: 1e20, count: shape.reduce(1, *)) }
    XCTAssertThrowsError(try runner.evaluate(inputs))
  }

  @MainActor
  func testCancellationDuringFinalWeightProviderDoesNotCommitLoadedState() async throws {
    let canceled = try await Task {
      let configuration = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
        videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
      let runner = try AVBlockRunner(configuration: configuration)
      let total = runner.weightShapes.count
      var calls = 0
      do {
        try runner.load { _, shape in
          calls += 1
          if calls == total { withUnsafeCurrentTask { $0?.cancel() } }
          return [Float](repeating: 0, count: shape.reduce(1, *))
        }
        return false
      } catch is CancellationError { return true }
    }.value
    XCTAssertTrue(canceled)
  }

  func testNonzeroJointBlockMatchesIndependentMLXReference() throws {
    struct Fixture: Decodable {
      let configuration: AVBlockConfiguration
      let inputs: [String: [Float]]
      let expected: [String: [Float]]
    }
    let url = Bundle.module.url(forResource: "block-reference", withExtension: "json", subdirectory: "Fixtures")!
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    let runner = try AVBlockRunner(configuration: fixture.configuration)
    try runner.load { name, shape in
      let seed = name.utf8.reduce(0) { $0 + Int($1) }
      let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
    }
    let actual = try runner.evaluate(fixture.inputs)
    for (name, values) in [("video", actual.video), ("audio", actual.audio)] {
      let expected = fixture.expected[name]!
      XCTAssertEqual(values.count, expected.count)
      let error = zip(values, expected).map { abs($0 - $1) }.max()!
      XCTAssertLessThan(error, 0.00003, "\(name) differs from Float32 MLX block; max absolute error \(error)")
    }
    // A reusable block slot must not retain folded/cached parameter values after reload.
    try runner.load { _, shape in [Float](repeating: 0, count: shape.reduce(1, *)) }
    var zeroGateInputs = fixture.inputs
    for name in zeroGateInputs.keys where name.contains("modulation") || name.hasSuffix("gate") {
      zeroGateInputs[name] = zeroGateInputs[name]!.map { _ in 0 }
    }
    let reset = try runner.evaluate(zeroGateInputs)
    XCTAssertEqual(reset.video, fixture.inputs["video"])
    XCTAssertEqual(reset.audio, fixture.inputs["audio"])
    XCTAssertThrowsError(try runner.load { _, _ in [] })
    XCTAssertThrowsError(try runner.evaluate(zeroGateInputs), "A failed reload must invalidate partially installed weights")
  }

  func testZeroResidualGatesPreserveBothStreams() throws {
    let configuration = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let runner = try AVBlockRunner(configuration: configuration)
    var inputs = runner.inputShapes.mapValues { [Float](repeating: 0, count: $0.reduce(1, *)) }
    inputs["video"] = (0..<160).map { Float($0 % 17 - 8) / 8 }
    inputs["audio"] = (0..<48).map { Float($0 % 13 - 6) / 8 }
    for key in inputs.keys where key.hasSuffix("cos") { inputs[key] = inputs[key]!.map { _ in 1 } }
    try runner.load { _, shape in [Float](repeating: 0, count: shape.reduce(1, *)) }
    let result = try runner.evaluate(inputs)
    XCTAssertEqual(result.video, inputs["video"])
    XCTAssertEqual(result.audio, inputs["audio"])
  }

  func testRejectsMissingInputsAndBadWeightShapesBeforeGraphExecution() throws {
    let configuration = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let runner = try AVBlockRunner(configuration: configuration)
    XCTAssertThrowsError(try runner.evaluate([:]))
    XCTAssertThrowsError(try runner.load { _, _ in [] })
  }

  func testConfigurationRejectsInconsistentHeadsAndUnsupportedBatching() throws {
    XCTAssertThrowsError(try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 3,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4))
    XCTAssertThrowsError(try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 0, audioTokens: 3, textTokens: 4))
  }

  func testRejectsOverBudgetBlockBeforeCompile() throws {
    let configuration = try AVBlockConfiguration(videoTokens: 131072, audioTokens: 3, textTokens: 4)
    XCTAssertThrowsError(try AVBlockRunner.validateAllocation(configuration: configuration))
    let small = try AVBlockConfiguration(videoTokens: 5, audioTokens: 3, textTokens: 4)
    XCTAssertNoThrow(try AVBlockRunner.validateAllocation(configuration: small))
    XCTAssertThrowsError(try AVBlockRunner.validateAllocation(configuration: small, maximumDecodedWeightBytes: 16))
  }

  func testDecodedConfigurationRejectsUnknownExecutionControls() throws {
    let configuration = try AVBlockConfiguration(videoTokens: 5, audioTokens: 3, textTokens: 4)
    let data = try JSONEncoder().encode(configuration)
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    json["batchSize"] = 2
    XCTAssertThrowsError(try JSONDecoder().decode(AVBlockConfiguration.self,
      from: JSONSerialization.data(withJSONObject: json)))
    json.removeValue(forKey: "batchSize")
    json["videoTokens"] = 0
    XCTAssertThrowsError(try JSONDecoder().decode(AVBlockConfiguration.self,
      from: JSONSerialization.data(withJSONObject: json)))
  }
}
