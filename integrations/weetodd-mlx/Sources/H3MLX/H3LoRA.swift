import Foundation
import MLX
import TensorIO

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

/// The released ComfyUI Turbo adapter is validated once from its safetensors
/// header. Projection pairs are then mapped only while their matching block
/// executes, preserving H3's staged weight residency.
final class H3LoRAFile {
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
    guard file.metadata["target_format"] == "ComfyUI generic LoRA",
      file.metadata["qkv_fusion"]?.contains("block diagonal B") == true else {
      throw H3CheckpointError.invalid("H3 Turbo LoRA metadata or QKV layout is unsupported.")
    }
    var targets: [String: (rank: Int, alpha: Float)] = [:]
    for group in 0..<52 {
      let prefix = group < 50
        ? "diffusion_model.blocks.\(group)."
        : "diffusion_model.token_refiner.blocks.\(group - 50)."
      for (suffix, rows, columns, rank) in [
        ("attn.qkv_proj", 21504, 5376, 384),
        ("attn.out_proj", 5376, 7168, 128),
        ("mlp.fc1", 28672, 5376, 128),
        ("mlp.fc2", 5376, 14336, 128),
      ] {
        let target = prefix + suffix
        let aName = target + ".lora_A.weight"
        let bName = target + ".lora_B.weight"
        let alphaName = target + ".alpha"
        guard let a = file.tensors[aName], a.dtype == "BF16",
          a.shape == [UInt64(rank), UInt64(columns)],
          let b = file.tensors[bName], b.dtype == "BF16",
          b.shape == [UInt64(rows), UInt64(rank)],
          let alphaDescriptor = file.tensors[alphaName],
          alphaDescriptor.dtype == "F32", alphaDescriptor.shape.isEmpty else {
          throw H3CheckpointError.invalid("Incomplete H3 Turbo LoRA target: \(target)")
        }
        let alpha = try file.withTensorBytes(named: alphaName) {
          $0.loadUnaligned(as: Float.self)
        }
        guard alpha.isFinite, alpha >= 0 else {
          throw H3CheckpointError.invalid("Invalid H3 Turbo LoRA alpha: \(target)")
        }
        targets[target] = (rank, alpha)
      }
    }
    guard file.tensors.count == targets.count * 3 else {
      throw H3CheckpointError.invalid("H3 Turbo LoRA contains unsupported extra targets.")
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
    guard let info = targets[target] else {
      throw H3CheckpointError.invalid("H3 LoRA target is missing: \(target)")
    }
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
