import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FinalLayerTests: XCTestCase {
  private func rejectBeforeCheckpoint(dtype: DType,
    precision: H3FinalLayer.ResidualPrecision = .bfloat16) {
    let input = MLXArray([Float](repeating: 0, count: 5376), [1, 1, 5376]).asType(dtype)
    let time = MLXArray([Float](repeating: 0, count: 2688), [1, 2688])
    let indices = MLXArray([Int32(0)])
    XCTAssertThrowsError(try H3FinalLayer.evaluate(
      checkpointURL: URL(fileURLWithPath: "/nonexistent/weetodd-final-residual.safetensors"),
      input: input, timeEmbeddings: time, timestepIndices: indices,
      videoIndices: indices, audioIndices: indices,
      residualPrecision: precision, observe: { _, _ in })) {
      XCTAssertEqual($0 as? H3CheckpointError, .invalid("Invalid H3 final layer inputs."))
    }
  }

  func testDefaultFinalHeadRejectsFloat32ResidualBeforeCheckpoint() {
    rejectBeforeCheckpoint(dtype: .float32)
    rejectBeforeCheckpoint(dtype: .float16)
  }

  func testExplicitFloat32ResidualPolicyRejectsRoundedInputsBeforeCheckpoint() {
    rejectBeforeCheckpoint(dtype: .bfloat16, precision: .float32)
    rejectBeforeCheckpoint(dtype: .float16, precision: .float32)
  }

  func testFloat32ResidualNormMatchesIndependentScalarOracleWithoutBF16Round() {
    let width = 5376, rows = 3
    let values = (0..<(rows * width)).map { index -> Float in
      let centered = Float(index % 37 - 18)
      return centered * 0.07137 + Float(index / width) * 0.00313 + 0.00191
    }
    let weights = (0..<width).map { index -> Float in [0.5, 1, 2][index % 3] }
    let shifts: [Float] = [-0.5, 0.25, 0.75]
    let input = MLXArray(values, [1, rows, width])
    let norm = MLXArray(weights).asType(.bfloat16)
    let shift = MLXArray(shifts, [rows, 1]).asType(.bfloat16)
    let scale = MLXArray([Float](repeating: 0.25, count: rows), [rows, 1]).asType(.bfloat16)
    let actual = H3FinalLayer.modulatedNormalization(input: input,
      normWeight: norm, shift: shift, scale: scale)
    XCTAssertEqual(actual.dtype, .float32)
    let result = actual.asArray(Float.self)
    var expected: [Float] = []
    expected.reserveCapacity(values.count)
    for row in 0..<rows {
      let offset = row * width
      let squares = values[offset..<(offset + width)].reduce(0.0) {
        $0 + Double($1) * Double($1)
      }
      let inverse = 1.0 / sqrt(squares / Double(width) + 1e-5)
      for column in 0..<width {
        expected.append(Float(Double(values[offset + column]) * inverse
          * Double(weights[column]) * 1.25 + Double(shifts[row])))
      }
    }
    XCTAssertTrue(result.allSatisfy(\.isFinite))
    XCTAssertLessThanOrEqual(zip(result, expected).map { abs($0 - $1) }.max() ?? .infinity,
      0.00001)
    let prematurelyRounded = H3FinalLayer.modulatedNormalization(
      input: input.asType(.bfloat16), normWeight: norm, shift: shift, scale: scale)
      .asType(.float32).asArray(Float.self)
    XCTAssertGreaterThan(zip(result, prematurelyRounded).map { abs($0 - $1) }.max() ?? 0,
      0.0001, "An early BF16 conversion must not masquerade as the retained FP32 residual policy.")
  }

  func testBF16NormalizationRetainsOriginalExpressionAndOutputWords() {
    let width = 5376, rows = 2
    let input = MLXArray((0..<(rows * width)).map { Float($0 % 29 - 14) * 0.13 },
      [1, rows, width]).asType(.bfloat16)
    let weight = MLXArray((0..<width).map { Float($0 % 5 + 1) * 0.25 }).asType(.bfloat16)
    let shift = MLXArray([Float(-0.25), 0.5], [rows, 1]).asType(.bfloat16)
    let scale = MLXArray([Float(0.5), -0.25], [rows, 1]).asType(.bfloat16)
    let original = MLXFast.rmsNorm(input, weight: weight, eps: 1e-5) * (1 + scale) + shift
    let actual = H3FinalLayer.modulatedNormalization(input: input,
      normWeight: weight, shift: shift, scale: scale)
    XCTAssertEqual(actual.dtype, .bfloat16)
    XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self),
      original.view(dtype: .uint16).asArray(UInt16.self))
  }

  func testInstalledOutputHeadsMatchReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let refiner = ProcessInfo.processInfo.environment["WEETODD_H3_REFINER_ORACLE"],
      let time = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_FINAL_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint, refiner, time and final-layer oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array<T: HasDType>(_ url: URL, shape: [Int], type: T.Type) throws -> MLXArray {
      let data = try Data(contentsOf: url)
      return data.withUnsafeBytes { MLXArray($0, shape, type: type) }
    }
    let input = try array(URL(fileURLWithPath: refiner)
      .appendingPathComponent("final.u16"), shape: [1, 4, 5376],
      type: UInt16.self).view(dtype: .bfloat16)
    let timeEmbedding = try array(URL(fileURLWithPath: time)
      .appendingPathComponent("output.f32"), shape: [3, 2688], type: Float.self)
    let timestepIndices = try array(root.appendingPathComponent("timestep-indices.i32"),
      shape: [4], type: Int32.self)
    let videoIndices = try array(root.appendingPathComponent("video-indices.i32"),
      shape: [2], type: Int32.self)
    let audioIndices = try array(root.appendingPathComponent("audio-indices.i32"),
      shape: [2], type: Int32.self)
    let output = try H3FinalLayer.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      input: input, timeEmbeddings: timeEmbedding,
      timestepIndices: timestepIndices, videoIndices: videoIndices,
      audioIndices: audioIndices) { name, value in
      let shape = name == "modulation" ? [3, 10752] : [1, 4, 5376]
      let expected = try array(root.appendingPathComponent("\(name).u16"),
        shape: shape, type: UInt16.self).view(dtype: .bfloat16)
      XCTAssertEqual(max(abs(value.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, name)
    }
    let video = try array(root.appendingPathComponent("video.f32"),
      shape: [1, 2, 96], type: Float.self)
    let audio = try array(root.appendingPathComponent("audio.f32"),
      shape: [1, 2, 32], type: Float.self)
    XCTAssertEqual(max(abs(output.video - video)).item(Float.self), 0)
    XCTAssertEqual(max(abs(output.audio - audio)).item(Float.self), 0)
  }
}
