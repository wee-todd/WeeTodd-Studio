import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3LoRATests: XCTestCase {
  private func makeSparseAdapter() throws -> URL {
    let target = "diffusion_model.blocks.0.mlp.fc2"
    var down = [UInt16](repeating: 0, count: 14336)
    var up = [UInt16](repeating: 0, count: 5376)
    down[0] = 0x3f80
    up[0] = 0x3f80
    var payload = Data()
    let downBytes = down.withUnsafeBytes { Data($0) }
    let upBytes = up.withUnsafeBytes { Data($0) }
    payload.append(downBytes)
    payload.append(upBytes)
    let alphaStart = payload.count
    var alpha = Float(1).bitPattern.littleEndian
    payload.append(withUnsafeBytes(of: &alpha) { Data($0) })
    let header: [String: Any] = [
      "__metadata__": ["target_format": "ComfyUI generic LoRA",
        "qkv_fusion": "block diagonal B"],
      target + ".lora_A.weight": ["dtype": "BF16", "shape": [1, 14336],
        "data_offsets": [0, downBytes.count]],
      target + ".lora_B.weight": ["dtype": "BF16", "shape": [5376, 1],
        "data_offsets": [downBytes.count, alphaStart]],
      target + ".alpha": ["dtype": "F32", "shape": [],
        "data_offsets": [alphaStart, payload.count]]]
    let headerData = try JSONSerialization.data(withJSONObject: header)
    var headerLength = UInt64(headerData.count).littleEndian
    var data = withUnsafeBytes(of: &headerLength) { Data($0) }
    data.append(headerData)
    data.append(payload)
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".safetensors")
    try data.write(to: url)
    return url
  }

  func testSparseRankOneAdapterUpdatesOnlyItsDeclaredProjection() throws {
    let target = "diffusion_model.blocks.0.mlp.fc2"
    let url = try makeSparseAdapter()
    defer { try? FileManager.default.removeItem(at: url) }
    let adapter = try H3LoRAFile(url: url, strength: 1)
    XCTAssertEqual(adapter.targetCount, 1)
    var source = [Float](repeating: 0, count: 14336)
    source[0] = 2
    let input = MLXArray(source, [1, 1, 14336])
    let base = MLXArray.ones([1, 1, 5376], dtype: .float32)
    let result = try adapter.apply(base: base, input: input, target: target)
    XCTAssertEqual(result[0, 0, 0].item(Float.self), 3)
    XCTAssertEqual(result[0, 0, 1].item(Float.self), 1)
    let unchanged = try adapter.apply(base: base, input: input,
      target: "diffusion_model.blocks.1.mlp.fc2")
    XCTAssertEqual(unchanged[0, 0, 0].item(Float.self), 1)
  }

  func testOrderedAdapterStackAddsEachStrengthWithoutMergingWeights() throws {
    let firstURL = try makeSparseAdapter()
    let secondURL = try makeSparseAdapter()
    defer {
      try? FileManager.default.removeItem(at: firstURL)
      try? FileManager.default.removeItem(at: secondURL)
    }
    let stack = try H3LoRAStack(adapters: [
      H3LoRAAdapter(url: firstURL, strength: 1),
      H3LoRAAdapter(url: secondURL, strength: 0.5),
    ])
    var source = [Float](repeating: 0, count: 14336)
    source[0] = 2
    let input = MLXArray(source, [1, 1, 14336])
    let base = MLXArray.ones([1, 1, 5376], dtype: .float32)
    let result = try stack.apply(base: base, input: input,
      target: "diffusion_model.blocks.0.mlp.fc2")
    XCTAssertEqual(result[0, 0, 0].item(Float.self), 4)
    XCTAssertEqual(result[0, 0, 1].item(Float.self), 1)
  }

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
