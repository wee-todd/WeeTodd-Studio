import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenQ8ProjectionTests: XCTestCase {
  func testPackedAffineGroupsProjectWithoutDenseExpansion() throws {
    let packed = MLXArray([UInt32](repeating: 0x01010101, count: 16)
      + [UInt32](repeating: 0x02020202, count: 16), [2, 16])
    let projection = try H3QwenQ8Projection(packed: packed,
      scales: MLXArray([Float(0.5), 2], [2, 1]),
      biases: MLXArray([Float(0), 0], [2, 1]), columns: 64)
    let result = try projection.project(MLXArray([Float](repeating: 1, count: 64), [1, 64]))
      .asArray(Float.self)
    XCTAssertEqual(result, [32, 256])
    XCTAssertEqual(projection.storageBytes, 128 + 16)
  }

  func testDeferredPackedWeightsKeepOwnedBytesAndExactBF16Projection() throws {
    let words = [UInt32](repeating: 0x01010101, count: 16)
      + [UInt32](repeating: 0x02020202, count: 16)
    var bytes = words.withUnsafeBytes { Data($0) }
    let packed = bytes.withUnsafeBytes { MLXArray($0, [2, 16], type: UInt32.self) }
    bytes.resetBytes(in: 0..<bytes.count)
    let scales = MLXArray([UInt16(0x3f00), 0x4000], [2, 1]).view(dtype: .bfloat16)
    let biases = MLXArray([UInt16(0), 0], [2, 1]).view(dtype: .bfloat16)
    let eager = try H3QwenQ8Projection(packed: packed, scales: scales,
      biases: biases, columns: 64)
    let deferred = try H3QwenQ8Projection(packed: packed, scales: scales,
      biases: biases, columns: 64, materializeWeights: false)
    XCTAssertEqual(deferred.parametersToMaterialize.count, 3)
    XCTAssertEqual(deferred.parametersToMaterialize.map(\.shape), [[2, 16], [2, 1], [2, 1]])
    XCTAssertEqual(deferred.storageBytes, 136)
    eval(deferred.parametersToMaterialize)
    let input = MLXArray([Float](repeating: 1, count: 64), [1, 64]).asType(.bfloat16)
    let expected = try eager.project(input).view(dtype: .uint16).asArray(UInt16.self)
    let actual = try deferred.project(input).view(dtype: .uint16).asArray(UInt16.self)
    XCTAssertEqual(actual, expected)
    XCTAssertEqual(actual, [0x4200, 0x4380])
  }

  func testDeferredPackedWeightsPreserveCompanionShapeAdmission() {
    let packed = MLXArray([UInt32](repeating: 0, count: 32), [2, 16])
    XCTAssertThrowsError(try H3QwenQ8Projection(packed: packed,
      scales: MLXArray.zeros([2, 2]), biases: MLXArray.zeros([2, 2]),
      columns: 64, materializeWeights: false))
  }

  func testInstalledPagedQwenProjectionMatchesAffineReference() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"] else {
      throw XCTSkip("Set WEETODD_H3_QWEN_PAGED for the optional real projection check.")
    }
    let projection = try H3QwenQ8Projection(checkpointURL: URL(fileURLWithPath: root)
      .appendingPathComponent("pages/layer-000.safetensors"),
      name: "model.layers.0.self_attn.q_proj.weight")
    let output = try projection.project(MLXArray([Float](repeating: 1, count: 5120), [1, 5120]))
      .asArray(Float.self)
    let expected: [Float] = [-0.7441654, -0.5353718, -0.4477043, 0.73174286]
    for index in expected.indices {
      XCTAssertEqual(output[index], expected[index], accuracy: 0.02, "row \(index)")
    }
  }
}
