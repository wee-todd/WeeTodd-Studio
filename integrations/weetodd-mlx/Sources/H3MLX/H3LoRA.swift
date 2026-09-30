import Foundation
import MLX
import TensorIO

public struct H3LoRAAdapter: Sendable {
  public let url: URL
  public let strength: Float

  public init(url: URL, strength: Float) throws {
    guard url.isFileURL, url.path.hasPrefix("/"), strength.isFinite,
      (0...2).contains(strength) else {
      throw H3CheckpointError.invalid("Invalid H3 LoRA path or strength.")
    }
    self.url = url
    self.strength = strength
  }
}

protocol H3LoRAApplying {
  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool) throws -> MLXArray
}

/// Ordered activation-space updates. Each file maps only the active projection;
/// no merged checkpoint or resident adapter-weight stack is created.
final class H3LoRAStack: H3LoRAApplying {
  private let files: [H3LoRAFile]

  init(adapters: [H3LoRAAdapter]) throws {
    guard (1...4).contains(adapters.count),
      Set(adapters.map { $0.url.standardizedFileURL.path }).count == adapters.count else {
      throw H3CheckpointError.invalid("H3 supports one to four distinct ordered LoRAs.")
    }
    files = try adapters.map { try H3LoRAFile(url: $0.url,
      strength: $0.strength) }
  }

  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    var value = base
    for file in files {
      value = try file.apply(base: value, input: input,
        target: target, reorderQKV: reorderQKV)
    }
    return value
  }
}

/// Apply a LoRA in activation space. H3's installed transformer projections
/// are weight-decoded one at a time, so merging a 2 GB adapter into the base
/// checkpoint would create another large resident copy.
enum H3LoRAProjection {
  static func apply(base: MLXArray, input: MLXArray,
    a: MLXArray, b: MLXArray, alpha: Float, strength: Float,
    reorderQKV: Bool, qkvHeads: Int = 56,
    qkvHeadSize: Int = 128) -> MLXArray {
    let rank = a.shape[0]
    var outputWeight = b
    if reorderQKV {
      outputWeight = b.reshaped([3, qkvHeads, qkvHeadSize, rank])
        .transposed(1, 0, 2, 3).reshaped([b.shape[0], rank])
    }
    let source = input.asType(a.dtype)
    let lowRank = matmul(source, a.T)
    let delta = matmul(lowRank, outputWeight.T)
      * (alpha * strength / Float(rank))
    return base + delta.asType(base.dtype)
  }
}

/// Compatible ComfyUI H3 adapters are validated once from their safetensors
/// headers. Declared projections are mapped only while their block executes,
/// preserving H3's staged weight residency.
final class H3LoRAFile: H3LoRAApplying {
  private enum ScaleLayout {
    case perTargetAlpha
    case bakedIntoB
  }
  let url: URL
  let strength: Float
  let targetCount: Int
  private let file: SafeTensorFile
  private let targets: [String: (rank: Int, alpha: Float)]

  init(url: URL, strength: Float) throws {
    guard url.isFileURL, strength.isFinite, (0...2).contains(strength) else {
      throw H3CheckpointError.invalid("Invalid H3 LoRA path or strength.")
    }
    let file = try SafeTensorFile(url: url)
    let explicitAlpha = file.metadata["target_format"] == "ComfyUI generic LoRA"
      && file.metadata["qkv_fusion"]?.contains("block diagonal B") == true
    let conversion = file.metadata["conversion"] ?? ""
    let bakedScale = file.metadata["converted_layout"] == "comfyui_minimax_h3"
      && file.metadata["format"] == "pt"
      && file.metadata["floating_dtype"] == "bfloat16"
      && file.metadata["baked_scale"] == "0.125"
      && conversion.contains("mlp.fc1 swiglu halves swapped")
      && (conversion.contains("qkv fused contiguous q|k|v (block-diagonal")
        || conversion.contains("qkv block-diag fused (per-projection ranks)"))
    guard explicitAlpha != bakedScale else {
      throw H3CheckpointError.invalid("H3 LoRA metadata or QKV layout is unsupported.")
    }
    let layout: ScaleLayout = explicitAlpha ? .perTargetAlpha : .bakedIntoB
    var supported: [String: (rows: Int, columns: Int)] = [:]
    for group in 0..<52 {
      let prefix = group < 50
        ? "diffusion_model.blocks.\(group)."
        : "diffusion_model.token_refiner.blocks.\(group - 50)."
      for (suffix, rows, columns) in [
        ("attn.qkv_proj", 21504, 5376),
        ("attn.out_proj", 5376, 7168),
        ("mlp.fc1", 28672, 5376),
        ("mlp.fc2", 5376, 14336),
      ] {
        supported[prefix + suffix] = (rows, columns)
      }
    }
    var targets: [String: (rank: Int, alpha: Float)] = [:]
    for aName in file.tensors.keys where aName.hasSuffix(".lora_A.weight") {
        let target = String(aName.dropLast(".lora_A.weight".count))
        guard let dimensions = supported[target], let a = file.tensors[aName],
          a.dtype == "BF16", a.shape.count == 2,
          let rankValue = a.shape.first, (1...512).contains(rankValue),
          a.shape[1] == UInt64(dimensions.columns) else {
          throw H3CheckpointError.invalid("Unsupported H3 LoRA target or down projection: \(target)")
        }
        let rank = Int(rankValue)
        let bName = target + ".lora_B.weight"
        let alphaName = target + ".alpha"
        guard let b = file.tensors[bName], b.dtype == "BF16",
          b.shape == [UInt64(dimensions.rows), rankValue] else {
          throw H3CheckpointError.invalid("Incomplete H3 LoRA target: \(target)")
        }
        let alpha: Float
        switch layout {
        case .perTargetAlpha:
          guard let descriptor = file.tensors[alphaName],
            descriptor.dtype == "F32", descriptor.shape.isEmpty else {
            throw H3CheckpointError.invalid("Incomplete H3 LoRA target: \(target)")
          }
          alpha = try file.withTensorBytes(named: alphaName) {
            $0.loadUnaligned(as: Float.self)
          }
        case .bakedIntoB:
          guard file.tensors[alphaName] == nil else {
            throw H3CheckpointError.invalid("Baked-scale H3 LoRA must not add a second alpha: \(target)")
          }
          // The installed conversion already multiplied B by its training
          // scale. alpha == rank makes activation-space application exactly B@A.
          alpha = Float(rank)
        }
        guard alpha.isFinite, alpha >= 0 else {
          throw H3CheckpointError.invalid("Invalid H3 LoRA alpha: \(target)")
        }
        targets[target] = (rank, alpha)
    }
    let tensorsPerTarget = layout == .perTargetAlpha ? 3 : 2
    guard !targets.isEmpty, file.tensors.count == targets.count * tensorsPerTarget else {
      throw H3CheckpointError.invalid("H3 LoRA contains unsupported extra targets.")
    }
    self.url = url
    self.strength = strength
    self.targetCount = targets.count
    self.file = file
    self.targets = targets
    try file.checkUnchanged(at: url)
  }

  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    guard let info = targets[target] else { return base }
    let aName = target + ".lora_A.weight"
    let bName = target + ".lora_B.weight"
    let aShape = [info.rank, input.shape.last!]
    let bShape = [base.shape.last!, info.rank]
    let a = try file.withTensorBytes(named: aName) { bytes in
      MLXArray(bytes, aShape, type: UInt16.self).view(dtype: .bfloat16)
    }
    let b = try file.withTensorBytes(named: bName) { bytes in
      MLXArray(bytes, bShape, type: UInt16.self).view(dtype: .bfloat16)
    }
    let output = H3LoRAProjection.apply(base: base, input: input,
      a: a, b: b, alpha: info.alpha, strength: strength,
      reorderQKV: reorderQKV)
    eval(output)
    try file.checkUnchanged(at: url)
    try Task.checkCancellation()
    return output
  }
}
