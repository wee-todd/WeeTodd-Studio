import CryptoKit
import Foundation
import TensorIO

/// Pinned publisher headers identify the two otherwise identically labelled
/// task adapters. Installation verifies their full published SHA-256 hashes;
/// runtime header identity is not a fresh checksum of the 1.96 GB payload.
enum H3FlexAdapter {
  enum Variant: String { case fl2va, ref2va
    var profile: H3LoRAProfile { self == .fl2va ? .pikaFlexFL2VA : .pikaFlexRef2VA }
  }

  static func variant(headerSHA256: String, fileBytes: UInt64) -> Variant? {
    switch (fileBytes, headerSHA256) {
    case (1_956_172_176, "d1a1a019dc4bf95f90d38b6a669b52ae776d41a92356712a6ce90b93539240f1"): return .fl2va
    case (1_956_172_176, "b4f84aa4ea051ccf3dbfb0ba728e00fc3ac5d45800d0e5d4cd78b4cf80408754"): return .ref2va
    case (559_266_144, "b29a94ff979d024491dc9cb007f322d9c9b11e1963f659d94757cf1418a21a74"): return .fl2va
    case (553_202_008, "bdc5bdcd9d3ff2ade9e58d2b171abe676dc2e791465ff3ab0ce5b255cd243689"): return .ref2va
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

  static func isNativeFormat(_ metadata: [String:String]) -> Bool {
    metadata["converted_for"] == "comfyui-minimax-h3"
      && metadata["base_model"] == "MiniMaxAI/MiniMax-H3"
      && metadata["base_revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
      && metadata["modified_by"] == "Mellis, Inc."
      && metadata["source_adapter"] == "adapter.safetensors"
      && metadata["format"] == "pt"
  }

  static func resolvedProfile(url: URL, selected: H3LoRAProfile) throws -> H3LoRAProfile {
    let file = try SafeTensorFile(url: url)
    var official: Variant?
    if isNativeFormat(file.metadata) || isReducedNativeFormat(file.metadata) {
      let handle = try FileHandle(forReadingFrom: url)
      defer { try? handle.close() }
      let prefix = try handle.read(upToCount: 8) ?? Data()
      guard prefix.count == 8 else { throw H3CheckpointError.invalid("Truncated Flex header.") }
      let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
      guard length > 0, length <= 4 * 1024 * 1024 else { throw H3CheckpointError.invalid("Unbounded Flex header.") }
      let bytes = try handle.read(upToCount: Int(length)) ?? Data()
      guard bytes.count == Int(length) else { throw H3CheckpointError.invalid("Truncated Flex header.") }
      let digest = SHA256.hash(data: prefix + bytes).map { String(format: "%02x", $0) }.joined()
      official = variant(headerSHA256: digest, fileBytes: file.fileByteCount)
      try file.checkUnchanged(at: url)
    }
    if let official {
      guard selected == .auto || selected == official.profile else {
        throw H3CheckpointError.invalid("The selected H3 LoRA profile conflicts with the released Pika Flex task adapter.")
      }
      return official.profile
    }
    if selected.isFlex, !isNativeFormat(file.metadata), !isReducedNativeFormat(file.metadata) {
      throw H3CheckpointError.invalid("Pika Flex requires its native Comfy-format adapter.")
    }
    return selected
  }

  static func validate(variant: Variant, task: String, specialized: Bool) throws {
    guard !specialized, variant == .fl2va ? ["t2va", "fl2va"].contains(task) : task == "ref2va" else {
      throw H3CheckpointError.invalid("Pika Flex requires the matching ordinary FL2VA or Ref2VA task; specialized H3 pipelines are not qualified.")
    }
  }

  static func validateBase(partition: String?, variant: Variant) throws {
    guard partition?.lowercased() == (variant == .fl2va ? "fl2va" : "ref2va") else {
      throw H3CheckpointError.invalid("Pika Flex requires a base checkpoint declaring the matching FL2VA or Ref2VA partition.")
    }
  }

  static func admit(adapters: [H3LoRAAdapter], steps: Int, method: H3SamplingMethod,
    task: String, specialized: Bool, checkpoint: URL) throws {
    let profiles = try adapters.map { try resolvedProfile(url: $0.url, selected: $0.profile) }
    let flex = profiles.enumerated().filter { $0.element.isFlex }
    if flex.isEmpty {
      guard method != .pikaFlex else { throw H3CheckpointError.invalid("Pika Flex sampling requires one matching Flex adapter.") }
      return
    }
    guard flex.count == 1, !profiles.contains(.turbo), method == .pikaFlex, steps == 7 else {
      throw H3CheckpointError.invalid("Use one Pika Flex adapter with its six-evaluation sampler and no Turbo adapter.")
    }
    let entry = flex[0], adapter = adapters[entry.offset]
    guard adapter.strength == 1, adapter.startAfterEvaluations == 0 else {
      throw H3CheckpointError.invalid("Pika Flex requires strength 1 and immediate activation.")
    }
    try validate(variant: entry.element == .pikaFlexFL2VA ? .fl2va : .ref2va,
      task: task, specialized: specialized)
    try validateBase(partition: SafeTensorFile(url: checkpoint).metadata["partition"],
      variant: entry.element == .pikaFlexFL2VA ? .fl2va : .ref2va)
  }
}
