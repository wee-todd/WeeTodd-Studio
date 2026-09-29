import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3TimeEmbeddingTests: XCTestCase {
  func testPrunedFL2VACurveInterpolatesAndClampsWithoutActivation() throws {
    var table = [Float](repeating: 0, count: 1001 * 64)
    for row in 0..<1001 {
      table[row * 64] = Float(row) / 1000
      table[row * 64 + 1] = -Float(row) / 500
    }
    let result = try H3TimeEmbedding.interpolateCurve(table: table,
      timesteps: [-1, 0, 0.0005, 0.42, 1, 2])
    XCTAssertEqual(result[0], 0)
    XCTAssertEqual(result[64], 0)
    XCTAssertEqual(result[128], 0.0005, accuracy: 0.000001)
    XCTAssertEqual(result[128 + 1], -0.001, accuracy: 0.000001)
    XCTAssertEqual(result[192], 0.42, accuracy: 0.000001)
    XCTAssertEqual(result[256], 1)
    XCTAssertEqual(result[320], 1)
    var nonlinear = [Float](repeating: 0, count: 1001 * 64)
    for row in 0..<1001 {
      for column in 0..<64 {
        let angle = Double(row) * 0.017 + Double(column) * 0.11
        nonlinear[row * 64 + column] = Float(sin(angle))
      }
    }
    let curved = try H3TimeEmbedding.interpolateCurve(table: nonlinear,
      timesteps: [0.0005, 0.42137, 0.9997])
    // Independent Python interpolation over the same sinusoidal table.
    XCTAssertEqual(curved[0], 0.008499591, accuracy: 0.000002)
    XCTAssertEqual(curved[13], 0.9912258, accuracy: 0.000002)
    XCTAssertEqual(curved[64], 0.7707796, accuracy: 0.000002)
    XCTAssertEqual(curved[64 + 63], 0.9990039, accuracy: 0.000002)
    XCTAssertEqual(curved[128], -0.95995253, accuracy: 0.000002)
    XCTAssertEqual(curved[128 + 13], -0.41199473, accuracy: 0.000002)
    XCTAssertThrowsError(try H3TimeEmbedding.interpolateCurve(table: [0], timesteps: [0.5]))
  }
  func testInstalledTimestepMLPMatchesMixedPrecisionOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint and timestep oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array(_ name: String, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: root.appendingPathComponent(name))
      return data.withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
    }
    var differences: [(String, Float)] = []
    let output = try H3TimeEmbedding.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint),
      timesteps: MLXArray([Float(1), 0.5, 0.1])) { name, value in
      let width = name == "sinusoid" ? 256 : name == "output" ? 2688 : 5376
      let expected = try array(name + ".f32", shape: [3, width])
      differences.append((name,
        max(abs(value.asType(.float32) - expected)).item(Float.self)))
    }
    XCTAssertEqual(output.shape, [3, 2688])
    XCTAssertLessThan(differences.map(\.1).max() ?? 1, 0.00001)
  }
}
