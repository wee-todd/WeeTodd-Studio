import XCTest
import TensorIO
import InferenceTestSupport
@testable import LTX25Engine

final class LTXAdapterCompatibilityTests: XCTestCase {
  func testPixelSpatialDFRRejectsIncompleteOrMisclassifiedAdapter() throws {
    let tensors: [(String, [Int], String)] = [
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", [32, 4096], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 32], "BF16")]
    for metadata in [
      ["model_version":"2.5","reference_downscale_factor":"2","reference_spatial_scale_factor":"2"],
      ["model_version":"2.3","reference_downscale_factor":"2","reference_spatial_scale_factor":"2"],
      ["model_version":"2.5","reference_downscale_factor":"1","reference_spatial_scale_factor":"2"],
      ["model_version":"2.5","reference_downscale_factor":"2","reference_spatial_scale_factor":"1"]] {
      try withTensorFile(metadata:metadata,tensors:tensors) { url in
        let file=try SafeTensorFile(url:url)
        XCTAssertThrowsError(try LTXAdapterCompatibility.pixelSpatialDFRPlan(file:file,strength:0.5))
        XCTAssertThrowsError(try LTXAdapterCompatibility.standardPlan(file:file,strength:0.5))
      }
    }
  }
  func testLTX23MetadataAndComfyNamesRemainCompatibleWith25() throws {
    try withTensorFile(metadata: ["model_version": "2.3.0", "lora_alpha": "4"], tensors: [
      ("diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight", [2, 4096], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight", [4096, 2], "BF16"),
      ("diffusion_model.transformer_blocks.47.audio_ff.net.0.proj.lora_down.weight", [2, 2048], "F16"),
      ("diffusion_model.transformer_blocks.47.audio_ff.net.0.proj.lora_up.weight", [8192, 2], "F16")]) { url in
      let plan = try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 0.5)
      XCTAssertEqual(plan.pairs.map(\.target), ["transformer_blocks.0.attn1.to_out", "transformer_blocks.47.audio_ff.proj_in"])
      XCTAssertEqual(plan.pairs.map(\.scale), [1, 1])
    }
  }

  func testDistilledAdapterIncludesValidatedNonBlockModulationTargets() throws {
    try withTensorFile(metadata: ["model_version": "2.5.0"], tensors: [
      ("diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight", [2, 256], "BF16"),
      ("diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_B.weight", [4096, 2], "BF16"),
      ("diffusion_model.audio_prompt_adaln_single.linear.lora_A.weight", [2, 2048], "BF16"),
      ("diffusion_model.audio_prompt_adaln_single.linear.lora_B.weight", [4096, 2], "BF16")]) { url in
      let plan = try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 1)
      XCTAssertEqual(plan.pairs.count, 2)
      XCTAssertEqual(plan.pairs[0].target, "adaln_single.emb.timestep_embedder.linear1")
    }
  }

  func testAllSixAttentionGateTargetsUseHeadCountInsteadOfHiddenWidth() throws {
    let names = ["attn1", "attn2", "audio_to_video_attn", "audio_attn1", "audio_attn2", "video_to_audio_attn"]
    let dims = [4096, 4096, 4096, 2048, 2048, 2048]
    var tensors: [(String, [Int], String)] = []
    for (name, dim) in zip(names, dims) {
      let stem = "transformer_blocks.0.\(name).to_gate_logits"
      tensors += [(stem + ".lora_A.weight", [2, dim], "BF16"),
                  (stem + ".lora_B.weight", [32, 2], "BF16")]
    }
    try withTensorFile(metadata: ["model_version": "2.5.0"], tensors: tensors) { url in
      let plan = try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 1)
      XCTAssertEqual(plan.pairs.count, 6)
      XCTAssertTrue(plan.pairs.allSatisfy { $0.shape[0] == 32 })
    }
  }

  func testVersionTagCannotOverrideWrongShapeOrEnableLegacyBaseArchitecture() throws {
    for version in ["2.2.0", "2.5.0", "nonsense"] {
      try withTensorFile(metadata: ["model_version": version], tensors: [
        ("transformer_blocks.0.attn1.to_q.lora_A.weight", [2, 2048], "F16"),
        ("transformer_blocks.0.attn1.to_q.lora_B.weight", [2048, 2], "F16")]) { url in
        XCTAssertThrowsError(try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 1))
      }
    }
  }

  func testReferenceAdapterCannotSilentlyBecomeOrdinaryStyleLoRA() throws {
    try withTensorFile(metadata: ["model_version": "2.3.0", "reference_downscale_factor": "2"], tensors: [
      ("transformer_blocks.0.attn1.to_q.lora_A.weight", [2, 4096], "F16"),
      ("transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 2], "F16")]) { url in
      XCTAssertThrowsError(try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 1))
    }
  }

  func testUnionControlRejectsIncompleteOrMisclassifiedTaskAdapter() throws {
    let tensors: [(String, [Int], String)] = [
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", [64, 4096], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 64], "BF16")]
    for metadata in [
      ["model_version":"2.3.0","reference_downscale_factor":"2"],
      ["model_version":"2.3.0","reference_downscale_factor":"1"],
      ["model_version":"2.3.0","reference_downscale_factor":"2",
        "reference_spatial_scale_factor":"2"]] {
      try withTensorFile(metadata:metadata,tensors:tensors) { url in
        let file=try SafeTensorFile(url:url)
        XCTAssertThrowsError(try LTXAdapterCompatibility.unionControlPlan(file:file,strength:1))
      }
    }
  }

  func testIngredientsRejectsIncompleteOrWrongScaleTaskAdapter() throws {
    let tensors: [(String, [Int], String)] = [
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", [128, 4096], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 128], "BF16")]
    for metadata in [
      ["model_version":"2.3","reference_downscale_factor":"1"],
      ["model_version":"2.3","reference_downscale_factor":"2"],
      ["model_version":"2.3","reference_downscale_factor":"1",
        "reference_spatial_scale_factor":"2"]] {
      try withTensorFile(metadata:metadata,tensors:tensors) { url in
        XCTAssertThrowsError(try LTXAdapterCompatibility.ingredientsPlan(file:SafeTensorFile(url:url),strength:1.2))
      }
    }
  }

  func testMSRRejectsIncompleteSlotAndLoRASignatures() throws {
    let tensors: [(String, [Int], String)] = [
      ("diffusion_model.reference_slot_embedding.frequencies", [16], "BF16"),
      ("diffusion_model.reference_slot_embedding.net.0.weight", [256, 33], "BF16"),
      ("diffusion_model.reference_slot_embedding.net.0.bias", [256], "BF16"),
      ("diffusion_model.reference_slot_embedding.net.2.weight", [128, 256], "BF16"),
      ("diffusion_model.reference_slot_embedding.net.2.bias", [128], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", [128, 4096], "BF16"),
      ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 128], "BF16")]
    let metadata=["reference_slot_embedding_type":"fourier_mlp",
      "reference_token_order":"prepend", "reference_slot_time_offsets":"pic1_based_negative_time"]
    try withTensorFile(metadata:metadata,tensors:tensors) { url in
      XCTAssertThrowsError(try LTXAdapterCompatibility.msrPlan(file:SafeTensorFile(url:url),strength:1))
    }
    var wrong=metadata;wrong["reference_token_order"]="append"
    try withTensorFile(metadata:wrong,tensors:tensors) { url in
      XCTAssertThrowsError(try LTXAdapterCompatibility.msrPlan(file:SafeTensorFile(url:url),strength:1))
    }
  }

  func testDifferentSourceAliasesCannotApplyTwiceToSameDestination() throws {
    var tensors: [(String, [Int], String)] = []
    for prefix in ["diffusion_model.", "model.diffusion_model."] {
      tensors += [(prefix + "transformer_blocks.0.attn1.to_q.lora_A.weight", [2, 4096], "F16"),
                  (prefix + "transformer_blocks.0.attn1.to_q.lora_B.weight", [4096, 2], "F16")]
    }
    try withTensorFile(tensors: tensors) { url in
      XCTAssertThrowsError(try LTXAdapterCompatibility.standardPlan(file: SafeTensorFile(url: url), strength: 1))
    }
  }
}
