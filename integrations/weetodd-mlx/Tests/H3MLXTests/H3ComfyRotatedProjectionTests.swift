import Foundation
import MLX
import MLXNN
import TensorIO
import XCTest
@testable import H3MLX

final class H3ComfyRotatedProjectionTests: XCTestCase {
  func testInstalledConvRotProjectionNumericalDiagnostic() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_ROTATED_PROBE"] == "1",
      let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Enable the installed ConvRot numerical probe explicitly.")
    }
    let checkpoint = URL(fileURLWithPath: path)
    let file = try SafeTensorFile(url: checkpoint)
    let prefix = "model.diffusion_model.blocks.0."
    for (suffix, rows, columns, reorder) in [
      ("attn.qkv_proj", 21504, 5376, true),
      ("attn.out_proj", 5376, 7168, false),
      ("mlp.fc1", 28672, 5376, false),
      ("mlp.fc2", 5376, 14336, false),
    ] {
      let values = (0..<(4 * columns)).map { Float(($0 * 31) % 257 - 128) / 256 }
      let input = MLXArray(values, [1, 4, columns]).asType(.bfloat16)
      let name = prefix + suffix + ".weight"
      let decoded = try H3ComfyDecodedProjection.load(file: file,
        checkpointURL: checkpoint, name: name, rows: rows,
        columns: columns, reorderQKV: reorder)
      let expected = matmul(input, decoded.T)
      let rotated = try H3ComfyRotatedProjection(file: file,
        name: name, rows: rows, columns: columns)
      let actual = try rotated.project(input, reorderQKV: reorder)
      let error = abs(actual.asType(.float32) - expected.asType(.float32))
      let maximum = max(error).item(Float.self)
      let meanError = mean(error).item(Float.self)
      let referenceScale = max(abs(expected.asType(.float32))).item(Float.self)
      print("H3_ROTATED_PROBE projection=\(suffix) max=\(maximum) mean=\(meanError) reference_scale=\(referenceScale)")
      XCTAssertTrue(maximum.isFinite)
    }
  }

  func testInstalledQKVAndAdaLNProjectWithoutDenseConvRotExpansion() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let blockFixture = ProcessInfo.processInfo.environment["WEETODD_H3_BLOCK_ORACLE"],
      let timeFixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint and QKV/AdaLN oracle paths.")
    }
    func bf16(_ path: URL, shape: [Int]) throws -> MLXArray {
      try Data(contentsOf: path).withUnsafeBytes {
        MLXArray($0, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let file = try SafeTensorFile(url: URL(fileURLWithPath: checkpoint))
    let base = "model.diffusion_model.blocks.0."
    let qkvInput = try bf16(URL(fileURLWithPath: blockFixture)
      .appendingPathComponent("norm1_adaln.u16"), shape: [1, 4, 5376])
    let qkv = try H3ComfyRotatedProjection(file: file,
      name: base + "attn.qkv_proj.weight", rows: 21504, columns: 5376)
    let qkvOutput = try qkv.project(qkvInput, reorderQKV: true)
      .reshaped([1, 4, 56, 3, 128])
    let expectedQKV = try bf16(URL(fileURLWithPath: blockFixture)
      .appendingPathComponent("qkv.u16"), shape: [1, 4, 56, 3, 128])
    let qkvError = abs(qkvOutput.asType(.float32) - expectedQKV.asType(.float32))
    let qkvMaximum = max(qkvError).item(Float.self)
    let qkvMean = mean(qkvError).item(Float.self)
    print("H3 ConvRot QKV max=\(qkvMaximum) mean=\(qkvMean)")
    XCTAssertLessThan(qkvMaximum, 1)

    let time = try Data(contentsOf: URL(fileURLWithPath: timeFixture)
      .appendingPathComponent("output.f32"))
      .withUnsafeBytes { MLXArray($0, [3, 2688], type: Float.self) }
    let activated = silu(time).asType(.bfloat16)
    let adaln = try H3ComfyRotatedProjection(file: file,
      name: base + "adaln_proj.linear.weight", rows: 96768,
      columns: 2688, biasName: base + "adaln_proj.linear.bias")
    let adaOutput = try adaln.project(activated)
    let expectedAda = try bf16(URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16"), shape: [3, 96768])
    let adaError = abs(adaOutput.asType(.float32) - expectedAda.asType(.float32))
    let adaMaximum = max(adaError).item(Float.self)
    let adaMean = mean(adaError).item(Float.self)
    print("H3 ConvRot AdaLN max=\(adaMaximum) mean=\(adaMean)")
    XCTAssertLessThan(adaMaximum, 1)
  }
}
