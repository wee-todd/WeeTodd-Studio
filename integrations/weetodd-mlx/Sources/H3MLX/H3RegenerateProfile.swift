import Foundation

/// Header-only release-layout identity. This is not a base-checkpoint or quality qualification.
public struct H3RegenerateTensor: Equatable, Sendable {
  public let dtype: String
  public let shape: [Int]
  public init(dtype: String, shape: [Int]) { self.dtype = dtype; self.shape = shape }
}

public enum H3RegenerateCompanion: String, Sendable, Hashable {
  case turbo, sharpLinear, sharpCurve
}

/// The official merge already includes Turbo and Sharp's linear update. The
/// extracted files contain only the refiner; the rank-16 file is an approximation.
public enum H3RegenerateProfile: String, Sendable, CaseIterable {
  case publisherThreeStep, extractedRefiner384, extractedRefiner16

  /// These two prepared files have identical complete tensor payloads. The
  /// source release, FL2VA and fused/Turbo checkpoints are separate contracts.
  public static func validateBaseSHA256(_ sha256:String) throws {
    guard ["600307eba6a67388b333aa5a94537a9ea83da65a6f37664ba22cf079e40a1f3b",
      "37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff"].contains(sha256) else {
      throw H3CheckpointError.invalid("H3 regeneration requires the verified Ref2VA base preparation; this checkpoint identity is not qualified.")
    }
  }

  private static let extractedMetadata = [
    "format":"pt", "split":"refiner ranks 0:384 of the rank-576 merge (refiner|turbo|sharp); alpha == rank, strength folded",
    "source":"Efficient-Large-Model/Open-H3-Regenerate-2K lora/open_h3_regenerate_2k_lora.safetensors",
    "converted_layout":"comfyui_minimax_h3", "baked_scale":"1.0",
    "conversion":"qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier"]

  private var metadata: [String:String] {
    if self == .publisherThreeStep {
      return ["format":"lora", "scale":"alpha == rank for every module (strengths folded)", "dtype":"float32"]
    }
    var result = Self.extractedMetadata
    if self == .extractedRefiner16 {
      result["ss_network_dim"] = "16"; result["ss_network_alpha"] = "16.0"
      result["ss_training_comment"] = "dimension is resized from 1152 to 16; "
    }
    return result
  }

  public static func admit(metadata: [String:String], tensors: [String:H3RegenerateTensor]) throws -> Self {
    guard let profile = allCases.first(where: { $0.metadata == metadata }),
      tensors == profile.expectedTensors else {
      throw H3CheckpointError.invalid("Unsupported or incomplete Open-H3 regenerate release layout.")
    }
    return profile
  }

  public func validateCompanions(_ roles: [H3RegenerateCompanion]) throws {
    let required: Set<H3RegenerateCompanion> = self == .publisherThreeStep
      ? [.sharpCurve] : [.turbo,.sharpLinear,.sharpCurve]
    guard Set(roles) == required, roles.count == required.count else {
      throw H3CheckpointError.invalid("Regenerate companion roles are missing, duplicated, or already included in the official merge.")
    }
  }

  /// Official split factors remain three independent projections. No fused
  /// rank-1728/block-diagonal allocation is needed at this mapping boundary.
  public func sources(for nativeTarget: String) throws -> [String] {
    guard let entry = Self.nativeTargets[nativeTarget] else {
      throw H3CheckpointError.invalid("Unsupported regenerate projection target: \(nativeTarget)")
    }
    guard self == .publisherThreeStep else { return [nativeTarget] }
    let prefix = entry.block < 50 ? "transformer.transformer_blocks.\(entry.block)."
      : "transformer.token_refiner.refiner_blocks.\(entry.block - 50)."
    switch entry.suffix {
    case "attn.qkv_proj": return ["attn.to_q","attn.to_k","attn.to_v"].map { prefix + $0 }
    case "attn.out_proj": return [prefix + "attn.to_out.0"]
    case "mlp.fc1": return [prefix + "ff.net.0.proj"]
    default: return [prefix + "ff.net.2"]
    }
  }

  static let nativeTargets: [String:(block:Int,suffix:String,rows:Int,columns:Int)] = {
    var result: [String:(Int,String,Int,Int)] = [:]
    for block in 0..<52 {
      let prefix = block < 50 ? "diffusion_model.blocks.\(block)." : "diffusion_model.token_refiner.blocks.\(block - 50)."
      for (suffix,rows,columns) in [("attn.qkv_proj",21504,5376),("attn.out_proj",5376,7168),
        ("mlp.fc1",28672,5376),("mlp.fc2",5376,14336)] {
        result[prefix + suffix] = (block,suffix,rows,columns)
      }
    }
    return result
  }()

  private var expectedTensors: [String:H3RegenerateTensor] {
    var result: [String:H3RegenerateTensor] = [:]
    for (target,entry) in Self.nativeTargets {
      // Every key above is admitted; this call cannot fail.
      guard let sources = try? sources(for:target) else { continue }
      let rank = self == .publisherThreeStep ? 576 : self == .extractedRefiner16 ? 16 : entry.suffix == "attn.qkv_proj" ? 1152 : 384
      let dtype = self == .publisherThreeStep ? "F32":"BF16"
      for source in sources {
        result[source + ".lora_A.weight"] = .init(dtype:dtype,shape:[rank,entry.columns])
        result[source + ".lora_B.weight"] = .init(dtype:dtype,shape:[sources.count == 3 ? entry.rows / 3:entry.rows,rank])
        if self == .extractedRefiner16 { result[source + ".alpha"] = .init(dtype:"BF16",shape:[]) }
      }
    }
    return result
  }
}
