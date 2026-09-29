import Foundation
import XCTest
import LTX25Engine
@testable import LTX25NNC

final class PrecisionPolicyTests: XCTestCase {
  struct Fixture: Decodable { let configuration: AVBlockConfiguration; let inputs: [String: [Float]] }
  func fixture() throws -> Fixture {
    try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
      Bundle.module.url(forResource: "block-reference", withExtension: "json", subdirectory: "Fixtures")!))
  }
  func weight(_ name: String, _ shape: [Int]) -> [Float] {
    let seed = name.utf8.reduce(0) { $0 + Int($1) }
    let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
  }
  func testDefaultPolicyPreservesReferenceAndMixedProjectionsRetainFloatOutputs() throws {
    let f = try fixture()
    let baseline = try AVBlockRunner(configuration: f.configuration, diagnostics: true)
    try baseline.load(weight)
    let expected = try baseline.evaluate(f.inputs)
    for policy in LTXPrecisionPolicy.allCases {
      let runner = try AVBlockRunner(configuration: f.configuration, diagnostics: true, precision: policy)
      try runner.load(weight)
      let actual = try runner.evaluate(f.inputs)
      XCTAssertEqual(actual.intermediates.keys.sorted(), expected.intermediates.keys.sorted())
      for (name, values) in actual.intermediates {
        let reference = expected.intermediates[name]!
        if policy == .float32 { XCTAssertEqual(values, reference) }
        else {
          let error = zip(values, reference).map { abs($0 - $1) }.max()!
          // A synthetic contract check, not the installed-model qualification gate.
          XCTAssertLessThan(error, 0.02, "\(policy): \(name)")
        }
      }
    }
  }
  func testHalfOverflowFailsRatherThanPublishingNonfiniteOutput() throws {
    let f = try fixture()
    let runner = try AVBlockRunner(configuration: f.configuration, precision: .float16Projections)
    XCTAssertThrowsError(try runner.load { name, shape in
      name == "attn1.to_q.weight" ? [Float](repeating: 100_000, count: shape.reduce(1,*)) : self.weight(name, shape)
    })
    XCTAssertThrowsError(try runner.evaluate(f.inputs))
  }
  func testOverflowingHalfActivationsRejectAndCleanRetryStillWorks() throws {
    let f = try fixture()
    let runner = try AVBlockRunner(configuration: f.configuration, precision: .float16Projections)
    try runner.load(weight)
    var invalid = f.inputs
    invalid["video_prompt_modulation"] = invalid["video_prompt_modulation"]!.map { _ in 1e8 }
    XCTAssertThrowsError(try runner.evaluate(invalid))
    XCTAssertTrue(try runner.evaluate(f.inputs).video.allSatisfy(\.isFinite))
  }
  func testPrecisionNamesAreExplicitAndUnknownPoliciesReject() throws {
    XCTAssertNil(LTXPrecisionPolicy(rawValue: "automatic"))
    for policy in LTXPrecisionPolicy.allCases {
      XCTAssertEqual(try JSONDecoder().decode(LTXPrecisionPolicy.self, from: JSONEncoder().encode(policy)), policy)
    }
    XCTAssertEqual(LTXPrecisionPolicy.float32.rawValue, "float32")
  }
  func testSamplerPropagatesPrecisionThroughSessionAndStagedRoutes() throws {
    let url = Bundle.module.url(forResource: "denoiser-reference", withExtension: "json", subdirectory: "Fixtures")!
    let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    let schedule = try SamplingSchedule(sigmas: [0.5, 0], eta: 0)
    let half = try LTXSamplingRunner(configuration: f.configuration, blockCount: 1, experimentalPrecision: .float16Projections)
    let baseline = try LTXSamplingRunner(configuration: f.configuration, blockCount: 1)
    let reference = try baseline.evaluate(f.inputs, schedule: schedule, fixedWeights: weight,
      blockWeights: { self.weight($1, $2) })
    let session = try half.evaluate(f.inputs, schedule: schedule, fixedWeights: weight,
      blockWeights: { self.weight($1, $2) })
    let staged = try half.evaluate(f.inputs, schedule: schedule, reuseSession: false, fixedWeights: weight,
      blockWeights: { self.weight($1, $2) })
    XCTAssertEqual(session.video, staged.video)
    XCTAssertEqual(session.audio, staged.audio)
    XCTAssertNotEqual(session.video, reference.video)
  }
}
