import XCTest
@testable import H3MLX

final class H3QwenCheckpointLayoutTests: XCTestCase {
  private func layer(_ index: Int) -> [String: H3TensorInfo] {
    let root = "model.layers.\(index)."
    var tensors: [String: H3TensorInfo] = [:]
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
      tensors[root + name] = .init(dtype: "BF16", shape: [5120])
    }
    for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
      tensors[root + name] = .init(dtype: "BF16", shape: [128])
    }
    for (name, rows, columns) in [
      ("self_attn.q_proj", 8192, 5120), ("self_attn.k_proj", 1024, 5120),
      ("self_attn.v_proj", 1024, 5120), ("self_attn.o_proj", 5120, 8192),
      ("mlp.gate_proj", 25600, 5120), ("mlp.up_proj", 25600, 5120),
      ("mlp.down_proj", 5120, 25600),
    ] {
      let stem = root + name
      tensors[stem + ".weight"] = .init(dtype: "U32", shape: [UInt64(rows), UInt64(columns / 4)])
      tensors[stem + ".scales"] = .init(dtype: "BF16", shape: [UInt64(rows), UInt64(columns / 64)])
      tensors[stem + ".biases"] = .init(dtype: "BF16", shape: [UInt64(rows), UInt64(columns / 64)])
    }
    return tensors
  }

  func testPagedLayerAcceptsAffineQ8AndRejectsMissingCompanion() throws {
    XCTAssertNoThrow(try H3QwenCheckpointLayout.validateLayer(index: 0, tensors: layer(0)))
    var tensors = layer(49)
    tensors.removeValue(forKey: "model.layers.49.self_attn.q_proj.scales")
    XCTAssertThrowsError(try H3QwenCheckpointLayout.validateLayer(index: 49, tensors: tensors))
  }

  func testVisionHeaderRejectsPartialTower() {
    let partial: [String: H3TensorInfo] = [
      "visual.patch_embed.proj.weight": .init(dtype: "BF16", shape: [1152, 3, 2, 16, 16]),
      "visual.blocks.26.attn.qkv.weight": .init(dtype: "U32", shape: [3456, 288]),
    ]
    XCTAssertThrowsError(try H3QwenCheckpointLayout.validateVision(tensors: partial))
  }

  func testInstalledPagedAndCompactQwenLayoutsReuseWeightsInPlace() throws {
    guard let paged = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let compact = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"] else {
      throw XCTSkip("Set both installed Qwen checkpoint paths for header-only qualification.")
    }
    let pagedLayout = try H3QwenCheckpointLayout.inspect(root: URL(fileURLWithPath: paged))
    XCTAssertEqual(pagedLayout.layerFiles.count, 50)
    XCTAssertFalse(pagedLayout.hasVision)
    let compactLayout = try H3QwenCheckpointLayout.inspect(root: URL(fileURLWithPath: compact))
    XCTAssertEqual(compactLayout.layerFiles.count, 50)
    XCTAssertTrue(compactLayout.hasVision)
  }
}
