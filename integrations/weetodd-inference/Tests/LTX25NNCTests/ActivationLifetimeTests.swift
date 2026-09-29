import Foundation
import XCTest
@testable import LTX25NNC

final class ActivationLifetimeTests: XCTestCase {
  func testSequentialQKVPreservesBothCrossModalStreamsExactly() throws {
    struct Fixture: Decodable { let configuration: AVBlockConfiguration; let inputs: [String: [Float]] }
    let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
      Bundle.module.url(forResource: "block-reference", withExtension: "json", subdirectory: "Fixtures")!))
    func weight(_ name: String, _ shape: [Int]) -> [Float] {
      let seed = name.utf8.reduce(0) { $0 + Int($1) }
      let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
    }
    let original = try AVBlockRunner(configuration: f.configuration, diagnostics: true, sequenceAttention: false)
    let scheduled = try AVBlockRunner(configuration: f.configuration, diagnostics: true, sequenceAttention: true)
    try original.load(weight); try scheduled.load(weight)
    let expected = try original.evaluate(f.inputs), actual = try scheduled.evaluate(f.inputs)
    XCTAssertEqual(expected.intermediates, actual.intermediates)
  }
}
