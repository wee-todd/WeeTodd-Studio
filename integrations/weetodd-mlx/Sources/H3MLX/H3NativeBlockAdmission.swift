import CoreFoundation
import Foundation
import TensorIO

public enum H3TransformerBackend: String, Sendable {
  case mlx
  case nncExperimental = "nnc_experimental"
}

/// CPU admission only. Experimental NNC has a separate numerical/backend
/// qualification; matching this contract does not imply MLX output parity.
enum H3NativeBlockAdmission {
  private static let coreTargets: [(String, UInt64, UInt64)] = [
    ("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168),
    ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336),
  ]

  static func validateSettings(task: String, steps: Int,
    samplingMethod: H3SamplingMethod, adapters: [H3LoRAAdapter],
    contextFrames: Int, isRefinement: Bool, packedRows: Int) throws {
    try Task.checkCancellation()
    guard task == "ref2va", steps == 5, samplingMethod == .euler,
      contextFrames == 0, !isRefinement, (1...40000).contains(packedRows),
      adapters.count == 1, let adapter = adapters.first,
      adapter.strength == 1, adapter.profile != .standard,
      adapter.qkvLayout != .nativeInterleaved,
      adapter.startAfterEvaluations == 0 else {
      throw H3CheckpointError.invalid("Experimental NNC requires ordinary Ref2VA, five Euler schedule points, one strength-1 contiguous Turbo adapter, no context/refinement and at most 40000 packed rows.")
    }
    try adapter.validate(requestedSteps: steps, samplingMethod: samplingMethod)
    try Task.checkCancellation()
  }

  static func validateAdapterHeader(tensors: [String: TensorDescriptor]) throws {
    try Task.checkCancellation()
    var allowed = Set<String>()
    var ranks: [String: UInt64] = [:]
    for index in 0..<50 {
      try Task.checkCancellation()
      for (target, rows, columns) in coreTargets {
        let base = "diffusion_model.blocks.\(index)." + target
        let names = [base + ".lora_A.weight", base + ".lora_B.weight", base + ".alpha"]
        allowed.formUnion(names)
        guard let a = tensors[names[0]], a.dtype == "BF16", a.shape.count == 2,
          let rank = a.shape.first, (1...512).contains(rank), a.shape[1] == columns,
          ranks[target] == nil || ranks[target] == rank,
          let b = tensors[names[1]], b.dtype == "BF16", b.shape == [rows, rank],
          let alpha = tensors[names[2]], alpha.dtype == "F32",
          alpha.shape.isEmpty || alpha.shape == [1] else {
          throw H3CheckpointError.invalid("Experimental NNC requires complete uniform BF16 core adapter pairs: \(base)")
        }
        ranks[target] = rank
      }
    }
    for name in tensors.keys {
      try Task.checkCancellation()
      guard name.hasPrefix("diffusion_model."),
        !name.hasPrefix("diffusion_model.transformer_blocks.") else {
        throw H3CheckpointError.invalid("Experimental NNC requires exact ComfyUI adapter names: \(name)")
      }
      if name.hasPrefix("diffusion_model.blocks."), !allowed.contains(name),
        !name.contains(".adaln_proj.") {
        throw H3CheckpointError.invalid("Experimental NNC does not support this core adapter target: \(name)")
      }
    }
    try Task.checkCancellation()
  }

  /// Both file descriptors remain open until all CPU guards finish. Only small
  /// scalar/marker payloads are acquired; no MLX arrays or core weights are read.
  static func inspect(checkpoint: URL, adapter: URL) throws {
    try Task.checkCancellation()
    let checkpointFile = try SafeTensorFile(url: checkpoint)
    let layout = try H3CheckpointLayout(url: checkpoint)
    guard layout.prefix == "model.diffusion_model.", layout.curveRank == nil,
      layout.fastVariant == nil else {
      throw H3CheckpointError.invalid("Experimental NNC requires the original monolithic Comfy H3 checkpoint.")
    }
    for (name, descriptor) in checkpointFile.tensors where descriptor.dtype == "I8" {
      try Task.checkCancellation()
      guard name.hasSuffix(".weight"), descriptor.shape.count == 2 else {
        throw H3CheckpointError.invalid("Unsupported experimental NNC signed-I8 checkpoint tensor: \(name)")
      }
      let base = String(name.dropLast(".weight".count))
      let markerName = base + ".comfy_quant"
      guard let marker = checkpointFile.tensors[markerName], marker.dtype == "U8",
        marker.shape.count == 1, (1...4096).contains(marker.byteCount) else {
        throw H3CheckpointError.invalid("Invalid experimental NNC quantization marker: \(base)")
      }
      let bytes = try H3TensorPayload.withTensorBytes(file: checkpointFile, name: markerName,
        maximumBufferedBytes: 4096) { Data($0) }
      try validateQuantizationMarker(bytes, inputWidth: descriptor.shape[1])
    }
    let adapterFile = try SafeTensorFile(url: adapter)
    try validateAdapterHeader(tensors: adapterFile.tensors)
    // H3LoRAFile enforces supported metadata, target sets, sampling declarations
    // and the existing explicit-alpha layout. Its stricter scalar-[] and AdaLN
    // restrictions remain unchanged by this experimental admission helper.
    for key in ["adapter_profile", "profile", "distillation_profile"] {
      let value = adapterFile.metadata[key]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      guard !["standard", "base", "quality"].contains(value ?? "") else {
        throw H3CheckpointError.invalid("Experimental NNC requires a Turbo adapter, not a declared ordinary adapter.")
      }
    }
    guard adapterFile.metadata["adapter_role"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "standard" else {
      throw H3CheckpointError.invalid("Experimental NNC requires a Turbo adapter, not a declared ordinary adapter.")
    }
    let validated = try H3LoRAFile(url: adapter, strength: 1, requestedSteps: 5,
      samplingMethod: .euler, qkvLayout: .contiguousQKV, profile: .turbo,
      startAfterEvaluations: 0)
    for index in 0..<50 {
      for (target, _, _) in coreTargets {
        let name = "diffusion_model.blocks.\(index).\(target).alpha"
        let alpha = try H3TensorPayload.withTensorBytes(file: adapterFile, name: name,
          maximumBufferedBytes: 4) { $0.loadUnaligned(as: Float.self) }
        try validateCoreAlpha(alpha)
      }
    }
    try validated.checkUnchanged()
    try adapterFile.checkUnchanged(at: adapter)
    try checkpointFile.checkUnchanged(at: checkpoint)
    try Task.checkCancellation()
  }

  static func validateCoreAlpha(_ alpha: Float) throws {
    try Task.checkCancellation()
    guard alpha.isFinite, alpha > 0 else {
      throw H3CheckpointError.invalid("Experimental NNC requires finite positive core adapter alpha.")
    }
  }

  /// Match the packaged worker's marker type/group admission before Qwen load.
  static func validateQuantizationMarker(_ bytes: Data, inputWidth: UInt64) throws {
    try Task.checkCancellation()
    guard let marker = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
      marker["format"] as? String == "int8_tensorwise",
      Set(marker.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]) else {
      throw H3CheckpointError.invalid("Unsupported experimental NNC quantization marker.")
    }
    var rotated = false
    if let raw = marker["convrot"] {
      guard CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID(), let value = raw as? Bool else {
        throw H3CheckpointError.invalid("Invalid experimental NNC ConvRot flag.")
      }
      rotated = value
    }
    var group = 0
    if rotated {
      group = 256
      if let raw = marker["convrot_groupsize"] {
        guard CFGetTypeID(raw as CFTypeRef) != CFBooleanGetTypeID(),
          let number = raw as? NSNumber, let integer = Int(exactly: number.doubleValue) else {
          throw H3CheckpointError.invalid("Invalid experimental NNC ConvRot group type.")
        }
        group = integer
      }
    }
    guard group == 0 || ([4, 16, 64, 256, 1024].contains(group)
      && inputWidth % UInt64(group) == 0) else {
      throw H3CheckpointError.invalid("Invalid experimental NNC ConvRot group.")
    }
    try Task.checkCancellation()
  }
}
