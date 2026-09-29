import LTX25Engine
import XCTest

@testable import LTX25NNC

final class SamplingTests: XCTestCase {
  func testThreeStepJointAncestralTrajectoryMatchesMLX() throws {
    struct Fixture: Decodable {
      struct Schedule: Decodable {
        let sigmas: [Double]
        let eta: Double
      }
      let configuration: AVBlockConfiguration
      let inputs: [String: [Float]]
      let schedule: Schedule
      let noise: [String: [Float]]
      let expected: [String: [Float]]
    }
    let url = Bundle.module.url(
      forResource: "trajectory-reference", withExtension: "json", subdirectory: "Fixtures")!
    let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    func weights(_ name: String, _ shape: [Int]) -> [Float] {
      let seed = name.utf8.reduce(0) { $0 + Int($1) }
      let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
    }
    let runner = try LTXSamplingRunner(configuration: f.configuration, blockCount: 1)
    var completed: [Int] = []
    var blocks: [Int] = []
    let actual = try runner.evaluate(
      f.inputs,
      schedule: SamplingSchedule(sigmas: f.schedule.sigmas, eta: f.schedule.eta),
      fixedWeights: weights, blockWeights: { weights("transformer_blocks.\($0)." + $1, $2) },
      noise: { index, modality, _ in f.noise["\(index).\(modality.rawValue)"]! },
      stageProgress: { index, event in if event.stage == "transformer" { blocks.append(index) } },
      progress: { completed.append($0.completedSteps) })
    XCTAssertEqual(runner.lastGraphBuildCount, 1)
    XCTAssertEqual(completed, [1, 2, 3])
    XCTAssertEqual(blocks, [1, 2, 3])
    for (name, values) in [("video", actual.video), ("audio", actual.audio)] {
      let reference = f.expected[name]!
      XCTAssertEqual(values.count, reference.count)
      XCTAssertLessThan(zip(values, reference).map { abs($0 - $1) }.max()!, 0.0001)
    }
  }
}
