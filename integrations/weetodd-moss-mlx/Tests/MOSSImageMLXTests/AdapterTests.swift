import Foundation
import ImageLayout
import MLX
import XCTest
@testable import MOSSImageMLX

final class AdapterTests: XCTestCase {
  func testSharedStudioTargetAllowlistMatchesNativeLinearContract() throws {
    let native = Set(MOSSModelContract.transformerShapes().filter { $0.value.count == 2 && $0.key.hasSuffix(".weight") }.keys.map { String($0.dropLast(7)) })
    XCTAssertEqual(MOSSAdapterTargets.names, native)
    XCTAssertEqual(try MOSSAdapterTargets.canonical("base_model.model.text_embedder.blocks.0.qkv"), "text_embedder.blocks.0.qkv")
    XCTAssertEqual(try MOSSAdapterTargets.canonical("lora_unet_layers_41_proj_ffn_up_gate"), "layers.41.proj.ffn_up_gate")
  }
  func testLoRAKeepsSmallFP32DeltaAndRejectsUnknownTargets() throws {
    try Device.withDefaultDevice(.cpu) {
      let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: folder) }
      let file = folder.appendingPathComponent("adapter.safetensors")
      try MLX.save(arrays: ["diffusion_model.layers.0.proj.q.lora_A.weight": MLXArray([Float(0.001), 0]).reshaped(1, 2),
        "diffusion_model.layers.0.proj.q.lora_B.weight": MLXArray([Float(1), 0]).reshaped(2, 1),
        "diffusion_model.layers.0.proj.q.alpha": MLXArray(Float(1))], url: file)
      let adapter = try MOSSLoRA(descriptor: .init(path: file.path), shapes: ["layers.0.proj.q.weight": [2, 2]])
      let delta = try adapter.delta(MLXArray([Float(1), 1]).reshaped(1, 2), target: "layers.0.proj.q")
      XCTAssertEqual(try XCTUnwrap(delta).asArray(Float.self)[0], 0.001, accuracy: 1e-8)
      let profiler = MOSSProfiler()
      let profiled = try MOSSLoRA(descriptor: .init(path: file.path), shapes: ["layers.0.proj.q.weight": [2, 2]], profiler: profiler)
      XCTAssertEqual(try XCTUnwrap(profiled.delta(MLXArray([Float(1), 1]).reshaped(1, 2), target: "layers.0.proj.q")).asArray(Float.self),
        try XCTUnwrap(delta).asArray(Float.self))
      XCTAssertEqual(profiler.records["adapter/FP32 projection"]?.count, 1)
      XCTAssertGreaterThan(profiler.records["adapter/tensor read + upload"]?.bytes ?? 0, 0)
      XCTAssertThrowsError(try MOSSLoRA(descriptor: .init(path: file.path), shapes: ["other.weight": [2, 2]]))
      let disabled = try MOSSLoRA(descriptor: .init(path: file.path, strength: 0), shapes: ["layers.0.proj.q.weight": [2, 2]])
      XCTAssertNil(try disabled.delta(.ones([1, 2]), target: "layers.0.proj.q"))
    }
  }
  func testIncompleteAndDuplicatePairSchemasFail() throws {
    try Device.withDefaultDevice(.cpu) {
      let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
      defer { try? FileManager.default.removeItem(at: file) }
      try MLX.save(arrays: ["layers.0.proj.q.lora_down.weight": .ones([1, 2])], url: file)
      XCTAssertThrowsError(try MOSSLoRA(descriptor: .init(path: file.path), shapes: ["layers.0.proj.q.weight": [2, 2]]))
    }
  }
}
