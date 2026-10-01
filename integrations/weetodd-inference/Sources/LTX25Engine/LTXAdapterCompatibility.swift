import Foundation
import AdapterRuntime
import TensorIO

public enum LTXAdapterCompatibility {
  public static func weightStack(_ adapters: [LoRAAdapter]) throws -> LoRAWeightStack {
    try LoRAWeightStack(adapters: adapters,plan: standardPlan)
  }

  /// First native qualification scope: dense standard 22B transformer adapters. Keep the
  /// 2.3 source family independent of the retired 2.3 generation engine. IC/MSR and
  /// other task adapters must use their own conditioning validator before execution.
  public static func standardPlan(file: SafeTensorFile, strength: Float) throws -> LoRAPlan {
    if let version = file.metadata["model_version"], !version.isEmpty {
      let parts = version.split(separator: ".")
      guard parts.count >= 2, let major = Int(parts[0]), let minor = Int(parts[1]),
            major > 2 || (major == 2 && minor >= 3) else {
        throw LTXError.invalid("LTX LoRA must declare version 2.3 or newer, or prove compatibility without version metadata.")
      }
    }
    guard !file.metadata.keys.contains(where: { $0.hasPrefix("reference_") }),
          file.metadata["adapter_family"].map({ $0 == "standard" }) ?? true else {
      throw LTXError.invalid("This LTX adapter requires a separately qualified conditioning task; ordinary LoRA application is not sufficient.")
    }
    return try LoRAPlan(file: file, strength: strength, targetShapes: targetShapes,
      normalize: normalize)
  }

  /// A task adapter needs both compatible matrices and a complete, typed
  /// conditioning signature. Never admit it through `standardPlan` alone.
  public static func unionControlPlan(file: SafeTensorFile, strength: Float) throws -> LoRAPlan {
    guard file.metadata["model_version"] == "2.3.0",
      file.metadata["reference_downscale_factor"] == "2",
      file.metadata["reference_temporal_scale_factor"].map({ $0 == "1" }) ?? true,
      file.metadata["reference_spatial_scale_factor"] == nil,
      file.metadata["adapter_family"].map({ $0 == "union_control" }) ?? true else {
      throw LTXError.invalid("Union Control needs the compatible LTX 2.3 IC-LoRA reference metadata.")
    }
    let plan = try LoRAPlan(file: file, strength: strength,
      targetShapes: targetShapes, normalize: normalize)
    let expected = Set((0..<48).flatMap { block in
      ["attn1.to_k", "attn1.to_out", "attn1.to_q", "attn1.to_v",
       "attn2.to_k", "attn2.to_out", "attn2.to_q", "attn2.to_v",
       "ff.proj_in", "ff.proj_out"].map { "transformer_blocks.\(block).\($0)" }
    })
    guard plan.pairs.count == 480, Set(plan.pairs.map(\.target)) == expected,
      plan.pairs.allSatisfy({ $0.rank == 64 }) else {
      throw LTXError.invalid("Union Control needs its complete 48-block, rank-64 task adapter.")
    }
    return plan
  }

  /// Ingredients uses a full-resolution static reference sheet. Its complete
  /// rank-128 signature is distinct from the half-resolution Union guide.
  public static func ingredientsPlan(file: SafeTensorFile, strength: Float) throws -> LoRAPlan {
    guard ["2.3", "2.3.0"].contains(file.metadata["model_version"] ?? ""),
      file.metadata["reference_downscale_factor"] == "1",
      file.metadata["reference_temporal_scale_factor"].map({ $0 == "1" }) ?? true,
      file.metadata["reference_spatial_scale_factor"] == nil,
      file.metadata["adapter_family"].map({ $0 == "ingredients_reference_sheet" }) ?? true else {
      throw LTXError.invalid("Ingredients needs compatible full-resolution LTX 2.3 reference metadata.")
    }
    let plan = try LoRAPlan(file: file, strength: strength,
      targetShapes: targetShapes, normalize: normalize)
    let expected = Set((0..<48).flatMap { block in
      ["attn1.to_k", "attn1.to_out", "attn1.to_q", "attn1.to_v",
       "attn2.to_k", "attn2.to_out", "attn2.to_q", "attn2.to_v",
       "ff.proj_in", "ff.proj_out"].map { "transformer_blocks.\(block).\($0)" }
    })
    guard plan.pairs.count == 480, Set(plan.pairs.map(\.target)) == expected,
      plan.pairs.allSatisfy({ $0.rank == 128 }) else {
      throw LTXError.invalid("Ingredients needs its complete 48-block, rank-128 task adapter.")
    }
    return plan
  }

  public static func normalize(_ source: String) -> String? {
    var key = source
    let prefixes = ["base_model.model.model.diffusion_model.", "base_model.model.diffusion_model.",
      "base_model.model.transformer.", "base_model.model.", "model.diffusion_model.",
      "diffusion_model.", "transformer."]
    if let prefix = prefixes.first(where: key.hasPrefix) { key.removeFirst(prefix.count) }
    // Append a delimiter so replacement also handles an adapter stem with no .weight suffix.
    key += "."
    for (before, after) in [(".to_out.0.", ".to_out."),
      (".ff.net.0.proj.", ".ff.proj_in."), (".ff.net.2.", ".ff.proj_out."),
      (".audio_ff.net.0.proj.", ".audio_ff.proj_in."), (".audio_ff.net.2.", ".audio_ff.proj_out."),
      (".linear_1.", ".linear1."), (".linear_2.", ".linear2.")] {
      key = key.replacingOccurrences(of: before, with: after)
    }
    key.removeLast()
    return key
  }

  public static let targetShapes: [String: [UInt64]] = {
    var result = blockTargetShapes
    // Dimensions verified against the released 2.5 22B transformer header. The
    // metadata version never bypasses shape validation, including older adapters.
    for (name, dim, outputs): (String, UInt64, UInt64) in [
      ("adaln_single", 4096, 9), ("audio_adaln_single", 2048, 9),
      ("prompt_adaln_single", 4096, 2), ("audio_prompt_adaln_single", 2048, 2),
      ("av_ca_a2v_gate_adaln_single", 4096, 1), ("av_ca_v2a_gate_adaln_single", 2048, 1),
      ("av_ca_video_scale_shift_adaln_single", 4096, 4), ("av_ca_audio_scale_shift_adaln_single", 2048, 4),
    ] {
      result[name + ".emb.timestep_embedder.linear1"] = [dim, 256]
      result[name + ".emb.timestep_embedder.linear2"] = [dim, dim]
      result[name + ".linear"] = [outputs * dim, dim]
    }
    result["patchify_proj"] = [4096, 128]; result["proj_out"] = [128, 4096]
    result["audio_patchify_proj"] = [2048, 128]; result["audio_proj_out"] = [128, 2048]
    return result
  }()

  public static let blockTargetShapes: [String: [UInt64]] = {
    var tails: [String: [UInt64]] = [:]
    for name in ["attn1", "attn2", "audio_attn1", "audio_attn2"] {
      let dimension: UInt64 = name.hasPrefix("audio_") ? 2048 : 4096
      for projection in ["to_q", "to_k", "to_v", "to_out"] {
        tails[name + "." + projection] = [dimension, dimension]
      }
      tails[name + ".to_gate_logits"] = [32, dimension]
    }
    for (name, shape): (String, [UInt64]) in [
      ("ff.proj_in", [16384, 4096]), ("ff.proj_out", [4096, 16384]),
      ("audio_ff.proj_in", [8192, 2048]), ("audio_ff.proj_out", [2048, 8192]),
      ("audio_to_video_attn.to_q", [2048, 4096]), ("audio_to_video_attn.to_k", [2048, 2048]),
      ("audio_to_video_attn.to_v", [2048, 2048]), ("audio_to_video_attn.to_out", [4096, 2048]),
      ("audio_to_video_attn.to_gate_logits", [32, 4096]),
      ("video_to_audio_attn.to_q", [2048, 2048]), ("video_to_audio_attn.to_k", [2048, 4096]),
      ("video_to_audio_attn.to_v", [2048, 4096]), ("video_to_audio_attn.to_out", [2048, 2048]),
      ("video_to_audio_attn.to_gate_logits", [32, 2048]),
    ] { tails[name] = shape }
    var result: [String: [UInt64]] = [:]
    for block in 0..<48 {
      for (name, shape) in tails { result["transformer_blocks.\(block).\(name)"] = shape }
    }
    return result
  }()
}
