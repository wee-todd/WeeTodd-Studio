import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVisionBlockTests: XCTestCase {
  func testInstalledVisionBlockZeroMatchesMLXOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"] else {
      throw XCTSkip("Set compact Qwen and vision-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array(_ name: String, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: root.appendingPathComponent(name))
      return data.withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
    }
    let input = try array("input.f32", shape: [16, 1152]).asType(.bfloat16)
    let rotary = try array("rotary.f32", shape: [16, 36])
    let expected = try array("output.f32", shape: [16, 1152])
    var stages: [(String, Float)] = []
    let actual = try H3QwenVisionBlock.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      index: 0, input: input, rotary: rotary, boundaries: [0, 16]) { name, value in
      let oracle = try array(name + ".f32", shape: value.shape)
      let error = max(abs(value.asType(.float32) - oracle)).item(Float.self)
      stages.append((name, error))
    }
    XCTAssertLessThan(stages.map(\.1).max() ?? 1, 0.0001)
    let difference = max(abs(actual.asType(.float32) - expected)).item(Float.self)
    XCTAssertEqual(difference, 0)
  }

  func testLaterBlocksFromExactOracleInputs() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"] else {
      throw XCTSkip("Set compact Qwen and vision-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array(_ name: String, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: root.appendingPathComponent(name))
      return data.withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
    }
    let rotary = try array("rotary.f32", shape: [16, 36])
    for index in [2, 3, 26] {
      let input = try array(String(format: "layer-%02d.f32", index - 1),
        shape: [16, 1152]).asType(.bfloat16)
      let expected = try array(String(format: "layer-%02d.f32", index), shape: [16, 1152])
      let actual = try H3QwenVisionBlock.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
        index: index, input: input, rotary: rotary, boundaries: [0, 16])
      let error = max(abs(actual.asType(.float32) - expected)).item(Float.self)
      XCTAssertEqual(error, 0, "vision block \(index)")
    }
  }
}
