import XCTest
@testable import H3MLX

final class H3CheckpointLayoutTests: XCTestCase {
  private let prefix = "model.diffusion_model."

  private func fixture() -> [String: H3TensorInfo] {
    var tensors: [String: H3TensorInfo] = [
      prefix + "video_patch_proj.weight": .init(dtype: "BF16", shape: [5376, 96]),
      prefix + "video_patch_proj.bias": .init(dtype: "F32", shape: [5376]),
      prefix + "audio_patch_proj.weight": .init(dtype: "BF16", shape: [5376, 32]),
      prefix + "audio_patch_proj.bias": .init(dtype: "F32", shape: [5376]),
      prefix + "condition_proj.weight": .init(dtype: "BF16", shape: [5376, 5120]),
      prefix + "condition_proj.bias": .init(dtype: "BF16", shape: [5376]),
      prefix + "final_layer.video_out.weight": .init(dtype: "BF16", shape: [96, 5376]),
      prefix + "final_layer.video_out.bias": .init(dtype: "F32", shape: [96]),
      prefix + "final_layer.audio_out.weight": .init(dtype: "BF16", shape: [32, 5376]),
      prefix + "final_layer.audio_out.bias": .init(dtype: "F32", shape: [32]),
      prefix + "final_layer.norm.weight": .init(dtype: "BF16", shape: [5376]),
      prefix + "final_layer.adaln_proj.linear.weight": .init(dtype: "BF16", shape: [10752, 2688]),
      prefix + "final_layer.adaln_proj.linear.bias": .init(dtype: "BF16", shape: [10752]),
      prefix + "time_embedder.proj_in.weight": .init(dtype: "BF16", shape: [5376, 256]),
      prefix + "time_embedder.proj_in.bias": .init(dtype: "F32", shape: [5376]),
      prefix + "time_embedder.proj_out.weight": .init(dtype: "BF16", shape: [2688, 5376]),
      prefix + "time_embedder.proj_out.bias": .init(dtype: "F32", shape: [2688]),
      prefix + "rope.inv_freq": .init(dtype: "F32", shape: [16]),
      prefix + "token_refiner.final_norm.weight": .init(dtype: "BF16", shape: [5376]),
    ]
    for index in 0..<2 {
      let base = prefix + "token_refiner.blocks.\(index)."
      for (name, shape) in [
        ("attn.qkv_proj.weight", [UInt64(21504), 5376]),
        ("attn.out_proj.weight", [5376, 7168]),
        ("mlp.fc1.weight", [28672, 5376]),
        ("mlp.fc2.weight", [5376, 14336]),
        ("attn.q_norm.weight", [128]), ("attn.k_norm.weight", [128]),
        ("norm1.weight", [5376]), ("norm2.weight", [5376]),
      ] { tensors[base + name] = .init(dtype: "BF16", shape: shape) }
    }
    let projections: [(String, [UInt64])] = [
      ("attn.qkv_proj", [21504, 5376]), ("attn.out_proj", [5376, 7168]),
      ("mlp.fc1", [28672, 5376]), ("mlp.fc2", [5376, 14336]),
      ("adaln_proj.linear", [96768, 2688]),
    ]
    for index in 0..<50 {
      let base = prefix + "blocks.\(index)."
      for name in ["attn.q_norm.weight", "attn.k_norm.weight"] {
        tensors[base + name] = .init(dtype: "BF16", shape: [128])
      }
      for name in ["norm1.weight", "norm2.weight"] {
        tensors[base + name] = .init(dtype: "BF16", shape: [5376])
      }
      tensors[base + "adaln_proj.linear.bias"] = .init(dtype: "BF16", shape: [96768])
      for (name, shape) in projections {
        let projection = base + name
        tensors[projection + ".weight"] = .init(dtype: "I8", shape: shape)
        tensors[projection + ".weight_scale"] = .init(dtype: "F32", shape: [shape[0], 1])
        tensors[projection + ".comfy_quant"] = .init(dtype: "U8", shape: [72])
      }
    }
    return tensors
  }

  func testAcceptsDirectComfyInt8H3WithoutWeightConversion() throws {
    let result = try H3CheckpointLayout(tensors: fixture())
    XCTAssertEqual(result.prefix, prefix)
    XCTAssertEqual(result.quantizedProjections, 250)
    XCTAssertEqual(result.blockCount, 50)
  }

  func testMissingScaleOrWrongAttentionShapeFailsBeforeLoadingWeights() throws {
    var tensors = fixture()
    tensors.removeValue(forKey: prefix + "blocks.17.attn.qkv_proj.weight_scale")
    XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors))
    tensors = fixture()
    tensors[prefix + "blocks.49.attn.qkv_proj.weight"] = .init(dtype: "I8", shape: [5376, 5376])
    XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors))
  }

  func testMissingNonProjectionTensorFailsBeforeLoadingWeights() throws {
    var tensors = fixture()
    tensors.removeValue(forKey: prefix + "token_refiner.blocks.1.attn.q_norm.weight")
    XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors))
    tensors = fixture()
    tensors.removeValue(forKey: prefix + "blocks.24.adaln_proj.linear.bias")
    XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors))
  }

  func testComfyMarkerHeaderBoundsRemainStrictBeforeBufferedAcquisition() throws {
    let name = prefix + "blocks.17.attn.qkv_proj.comfy_quant"
    for count: UInt64 in [1, 4096] {
      var tensors = fixture()
      tensors[name] = .init(dtype: "U8", shape: [count])
      XCTAssertEqual(try H3CheckpointLayout(tensors: tensors).quantizedProjections, 250)
    }
    for invalid: H3TensorInfo in [
      .init(dtype: "U8", shape: [0]), .init(dtype: "U8", shape: [4097]),
      .init(dtype: "F32", shape: [72]), .init(dtype: "U8", shape: [1, 72]),
    ] {
      var tensors = fixture()
      tensors[name] = invalid
      XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors)) { error in
        XCTAssertEqual(error as? H3CheckpointError,
          .invalid("Incomplete Comfy INT8 metadata: \(self.prefix)blocks.17.attn.qkv_proj"))
      }
    }
  }

  func testInstalledFL2VACurveCheckpointHeaderWithoutReadingWeightsWhenProvided() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_FL2VA_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_FL2VA_CHECKPOINT for released FL2VA curve header admission.")
    }
    let layout = try H3CheckpointLayout(url: URL(fileURLWithPath: path))
    XCTAssertEqual(layout.curveRank, 64)
    XCTAssertEqual(layout.quantizedProjections, 0)
    XCTAssertEqual(layout.blockCount, 50)
  }

  func testInstalledDirectComfyCheckpointHeaderWithoutReadingWeights() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_TEST_CHECKPOINT to inspect an installed direct checkpoint.")
    }
    let layout = try H3CheckpointLayout(url: URL(fileURLWithPath: path))
    XCTAssertEqual(layout.quantizedProjections, 250)
    XCTAssertEqual(layout.blockCount, 50)
  }
}
