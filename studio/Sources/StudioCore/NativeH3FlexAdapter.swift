import Foundation

/// Release header identities, independent of filenames. Full payload checksums
/// are verified at installation; library inspection only reads bounded headers.
enum NativeH3FlexAdapter {
  static func profile(headerSHA256: String, fileBytes: Int64) -> H3LoRAProfile? {
    switch (fileBytes, headerSHA256) {
    case (1_956_172_176, "d1a1a019dc4bf95f90d38b6a669b52ae776d41a92356712a6ce90b93539240f1"): return .pikaFlexFL2VA
    case (1_956_172_176, "b4f84aa4ea051ccf3dbfb0ba728e00fc3ac5d45800d0e5d4cd78b4cf80408754"): return .pikaFlexRef2VA
    case (559_266_144, "b29a94ff979d024491dc9cb007f322d9c9b11e1963f659d94757cf1418a21a74"): return .pikaFlexFL2VA
    case (553_202_008, "bdc5bdcd9d3ff2ade9e58d2b171abe676dc2e791465ff3ab0ce5b255cd243689"): return .pikaFlexRef2VA
    default: return nil
    }
  }
  static func isReducedNativeFormat(_ metadata: [String:String]) -> Bool {
    // Released reduced exports carry no separate sampler/profile/layout overrides.
    // An explicit task selection must not hide contradictory declarations.
    guard ["adapter_profile", "profile", "distillation_profile", "adapter_role",
      "inference_steps", "num_inference_steps", "steps", "transformer_evaluations",
      "schedule_points", "qkv_layout", "qkv_fusion"].allSatisfy({ metadata[$0] == nil }) else { return false }
    return metadata["converted_layout"] == "comfyui_minimax_h3" && metadata["format"] == "pt"
      && metadata["base_model"] == "MiniMaxAI/MiniMax-H3"
      && metadata["base_revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
      && metadata["modified_by"] == "Mellis, Inc."
      && metadata["ss_network_dim"] == "Dynamic" && metadata["ss_network_alpha"] == "Dynamic"
      && metadata["baked_scale"] == "1.0"
      && metadata["conversion"] == "qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier"
  }
  static func isNativeFormat(_ metadata: [String: String]) -> Bool {
    metadata["converted_for"] == "comfyui-minimax-h3"
      && metadata["base_model"] == "MiniMaxAI/MiniMax-H3"
      && metadata["base_revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
      && metadata["modified_by"] == "Mellis, Inc."
      && metadata["source_adapter"] == "adapter.safetensors"
      && metadata["format"] == "pt"
  }
  static func validate(profile: H3LoRAProfile, task: String, specialized: Bool) throws {
    let matching = profile == .pikaFlexFL2VA ? ["t2v", "t2va", "i2v", "fflf", "fl2va"].contains(task) : task == "ref2va"
    guard profile.isFlex, matching, !specialized else {
      throw StudioError.invalid("Pika Flex requires its matching ordinary FL2VA or Ref2VA task. Continuation, stills, controls and refinement are not qualified.")
    }
  }
}
