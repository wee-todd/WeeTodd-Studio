import XCTest
@testable import LTX25NNC

final class DenoiserTests: XCTestCase {
  struct Fixture: Decodable {
    let configuration: AVBlockConfiguration
    let inputs: [String: [Float]]
    let sigma: Float
    let expected: [String: [Float]]
  }
  private func fixture() throws -> Fixture {
    let url = Bundle.module.url(forResource: "denoiser-reference", withExtension: "json", subdirectory: "Fixtures")!
    return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
  }
  private func weights(_ name: String, _ shape: [Int]) -> [Float] {
    let seed = name.utf8.reduce(0) { $0 + Int($1) }
    let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
  }
  private func block(_ index: Int, _ name: String, _ shape: [Int]) -> [Float] {
    weights("transformer_blocks.\(index)." + name, shape)
  }
  func testPackedLatentsToVelocityMatchIndependentMLX() throws {
    let f = try fixture(), runner = try DenoiserRunner(configuration: f.configuration, blockCount: 1)
    var stages: [String] = []
    let output = try runner.evaluate(f.inputs, sigma: f.sigma, fixedWeights: weights, blockWeights: block) {
      stages.append($0.stage)
    }
    for (name, actual) in [("video", output.videoVelocity), ("audio", output.audioVelocity)] {
      let expected = try XCTUnwrap(f.expected[name])
      XCTAssertEqual(actual.count, expected.count)
      let maxError = zip(actual, expected).map { abs($0 - $1) }.max()!
      let relative = sqrt(zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) }
        / expected.reduce(0.0) { $0 + Double($1) * Double($1) })
      XCTAssertLessThan(maxError, 0.0001, "\(name): \(maxError)")
      XCTAssertLessThan(relative, 0.0001, "\(name): \(relative)")
    }
    XCTAssertEqual(stages.count, 14)
    XCTAssertEqual(Array(stages.suffix(3)), ["transformer_released", "proj_out", "audio_proj_out"])
  }

  func testObserverFailureReleasesStageAndAllowsRetry() throws {
    enum Failure: Error { case stop }
    let f = try fixture(), runner = try DenoiserRunner(configuration: f.configuration, blockCount: 1)
    var blockCalls = 0
    XCTAssertThrowsError(try runner.evaluate(f.inputs, sigma: f.sigma, fixedWeights: weights, blockWeights: {
      blockCalls += 1; return self.block($0, $1, $2)
    }) { _ in throw Failure.stop })
    XCTAssertEqual(blockCalls, 0)
    let retried = try runner.evaluate(f.inputs, sigma: f.sigma, fixedWeights: weights, blockWeights: block)
    XCTAssertEqual(retried.videoVelocity.count, 640)
    for sigma: Float in [-1, 2, .nan, .infinity] {
      XCTAssertThrowsError(try runner.evaluate(f.inputs, sigma: sigma, fixedWeights: { _, _ in
        XCTFail("Invalid sigma reached weights"); return []
      }, blockWeights: block))
    }
  }

  func testOverflowingPositionsFailBeforeReadingAnyWeights() throws {
    let f = try fixture(), runner = try DenoiserRunner(configuration: f.configuration, blockCount: 1)
    var invalid = f.inputs
    invalid["video_positions"]![0] = Float.greatestFiniteMagnitude
    XCTAssertThrowsError(try runner.evaluate(invalid, sigma: f.sigma, fixedWeights: { _, _ in
      XCTFail("Invalid positions reached fixed weights"); return []
    }, blockWeights: { _, _, _ in XCTFail("Invalid positions reached blocks"); return [] }))
    let tiny = try AVBlockConfiguration(videoDimension: 2, audioDimension: 2, heads: 1,
      videoHeadDimension: 2, audioHeadDimension: 2, videoTokens: 1, audioTokens: 1, textTokens: 1)
    let tinyRunner = try DenoiserRunner(configuration: tiny)
    let tinyInputs = tinyRunner.inputShapes.mapValues { [Float](repeating: 0, count: $0.reduce(1, *)) }
    XCTAssertThrowsError(try tinyRunner.evaluate(tinyInputs, sigma: f.sigma, fixedWeights: { _, _ in
      XCTFail("Unsupported rotary layout reached fixed weights"); return []
    }, blockWeights: { _, _, _ in XCTFail("Unsupported rotary layout reached blocks"); return [] }))
  }

  @MainActor
  func testCancellationAtStageBoundaryStopsBeforeNextHead() async throws {
    let canceled = try await Task {
      let f = try fixture(), runner = try DenoiserRunner(configuration: f.configuration, blockCount: 1)
      var heads: Set<String> = []
      do {
        _ = try runner.evaluate(f.inputs, sigma: f.sigma, fixedWeights: { name, shape in
          heads.insert(String(name.split(separator: ".")[0])); return self.weights(name, shape)
        }, blockWeights: block) { _ in withUnsafeCurrentTask { $0?.cancel() } }
        return false
      } catch is CancellationError { return heads == ["adaln_single"] }
    }.value
    XCTAssertTrue(canceled)
  }
}
