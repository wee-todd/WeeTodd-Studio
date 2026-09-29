import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3LoRATests: XCTestCase {
  func testActivationDeltaUsesPerTargetAlphaAndRank() throws {
    let input = MLXArray([Float(1), 2], [1, 1, 2])
    let base = MLXArray([Float(10), 20], [1, 1, 2])
    let a = MLXArray([Float(1), 0, 0, 2], [2, 2])
    let b = MLXArray([Float(3), 4, 5, 6], [2, 2])
    let result = H3LoRAProjection.apply(base: base, input: input,
      a: a, b: b, alpha: 2, strength: 0.5, reorderQKV: false)
    XCTAssertEqual(result.asArray(Float.self), [19.5, 34.5])
  }

  func testContiguousQKVAdapterReordersHeadsBeforeAddition() throws {
    let input = MLXArray([Float(2)], [1, 1, 1])
    let base = MLXArray.zeros([1, 1, 12], dtype: .float32)
    let a = MLXArray([Float(1)], [1, 1])
    let b = MLXArray((1...12).map(Float.init), [12, 1])
    let result = H3LoRAProjection.apply(base: base, input: input,
      a: a, b: b, alpha: 1, strength: 1,
      reorderQKV: true, qkvHeads: 2, qkvHeadSize: 2)
    XCTAssertEqual(result.asArray(Float.self),
      [2, 4, 10, 12, 18, 20, 6, 8, 14, 16, 22, 24])
  }

  func testInstalledTurboLoRAHasCompleteSupportedLayout() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TURBO_LORA"],
      FileManager.default.isReadableFile(atPath: path) else {
      throw XCTSkip("Set WEETODD_H3_TURBO_LORA to the installed Turbo adapter.")
    }
    let adapter = try H3LoRAFile(url: URL(fileURLWithPath: path), strength: 1)
    XCTAssertEqual(adapter.targetCount, 208)
    let input = MLXArray.ones([1, 4, 5376], dtype: .bfloat16)
    let base = MLXArray.zeros([1, 4, 21504], dtype: .bfloat16)
    let output = try adapter.apply(base: base, input: input,
      target: "diffusion_model.blocks.0.attn.qkv_proj",
      reorderQKV: true)
    XCTAssertEqual(output.shape, [1, 4, 21504])
    XCTAssertTrue(output[0, 0, 0].item(Float.self).isFinite)
    XCTAssertGreaterThan(max(abs(output.asType(.float32))).item(Float.self), 0)
  }
}
