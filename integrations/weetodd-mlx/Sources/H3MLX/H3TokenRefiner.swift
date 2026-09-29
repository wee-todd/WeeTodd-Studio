import Foundation
import MLX
import MLXNN
import TensorIO

/// Two pre-norm text-refiner blocks from the installed H3 transformer.
/// Projections are read and released one at a time; no duplicate checkpoint is
/// created and the large QKV/FFN matrices are never resident together.
public enum H3TokenRefiner {
  public static func evaluate(checkpointURL: URL,
    input: MLXArray) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, input: input, lora: nil)
  }

  static func evaluate(checkpointURL: URL, input: MLXArray,
    lora: H3LoRAFile?) throws -> MLXArray {
    var value = input
    for index in 0..<2 {
      value = try evaluateBlock(checkpointURL: checkpointURL,
        index: index, input: value, lora: lora,
        observe: { _, _ in })
    }
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    let name = layout.prefix + "token_refiner.final_norm.weight"
    guard let descriptor = file.tensors[name], descriptor.dtype == "BF16",
      descriptor.shape == [5376] else {
      throw H3CheckpointError.invalid("Missing H3 token-refiner final norm.")
    }
    let weight = try file.withTensorBytes(named: name) { bytes in
      MLXArray(bytes, [5376], type: UInt16.self).view(dtype: .bfloat16)
    }
    let result = MLXFast.rmsNorm(value, weight: weight, eps: 1e-5)
    eval(result)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return result
  }

  public static func evaluateBlock(checkpointURL: URL, index: Int,
    input: MLXArray) throws -> MLXArray {
    try evaluateBlock(checkpointURL: checkpointURL, index: index,
      input: input, lora: nil, observe: { _, _ in })
  }

  static func evaluateBlock(checkpointURL: URL, index: Int,
    input: MLXArray, lora: H3LoRAFile? = nil,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<2).contains(index), input.ndim == 3,
      input.shape[0] == 1, (1...1024).contains(input.shape[1]),
      input.shape[2] == 5376, input.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("Invalid H3 text refiner block or input.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    let prefix = layout.prefix + "token_refiner.blocks.\(index)."
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    func read(_ suffix: String, shape: [Int]) throws -> MLXArray {
      let name = prefix + suffix
      guard let descriptor = file.tensors[name], descriptor.dtype == "BF16",
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 text refiner tensor: \(suffix)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
      eval(value)
      return value
    }
    func project(_ activation: MLXArray, _ suffix: String,
      rows: Int, columns: Int, qkv: Bool = false) throws -> MLXArray {
      var weight = try read(suffix + ".weight", shape: [rows, columns])
      if qkv {
        weight = H3QKVRowOrder.forHeadMajorAttention(weight,
          heads: 56, headWidth: 128, groupedSource: layout.curveRank == nil)
      }
      let base = matmul(activation, weight.T)
      let value = try lora?.apply(base: base, input: activation,
        target: "diffusion_model.token_refiner.blocks.\(index).\(suffix)",
        reorderQKV: qkv) ?? base
      eval(value)
      Memory.clearCache()
      try Task.checkCancellation()
      return value
    }
    let count = input.shape[1]
    let firstNorm = try read("norm1.weight", shape: [5376])
    let normalized = MLXFast.rmsNorm(input, weight: firstNorm, eps: 1e-5)
    try observe("norm1", normalized)
    let qkv = try project(normalized, "attn.qkv_proj",
      rows: 21504, columns: 5376, qkv: true)
      .reshaped([1, count, 56, 3, 128])
    try observe("qkv", qkv)
    let qNorm = try read("attn.q_norm.weight", shape: [128])
    let kNorm = try read("attn.k_norm.weight", shape: [128])
    let query = MLXFast.rmsNorm(qkv[.ellipsis, 0, 0..<128],
      weight: qNorm, eps: 1e-5).transposed(0, 2, 1, 3)
    let key = MLXFast.rmsNorm(qkv[.ellipsis, 1, 0..<128],
      weight: kNorm, eps: 1e-5).transposed(0, 2, 1, 3)
    let value = qkv[.ellipsis, 2, 0..<128].transposed(0, 2, 1, 3)
    try observe("query", query)
    try observe("key", key)
    try observe("value", value)
    let attended = MLXFast.scaledDotProductAttention(queries: query,
      keys: key, values: value, scale: 1 / Float(128).squareRoot(),
      mask: nil).transposed(0, 2, 1, 3).reshaped([1, count, 7168])
    try observe("attended", attended)
    let attention = try project(attended, "attn.out_proj",
      rows: 5376, columns: 7168)
    try observe("attention", attention)
    let residual = input + attention
    eval(residual)
    try observe("residual", residual)
    let secondNorm = try read("norm2.weight", shape: [5376])
    let feedInput = MLXFast.rmsNorm(residual, weight: secondNorm, eps: 1e-5)
    try observe("norm2", feedInput)
    let fused = try project(feedInput, "mlp.fc1",
      rows: 28672, columns: 5376)
    try observe("fused", fused)
    let gate = fused[.ellipsis, 0..<14336]
    let gated = silu(gate) * fused[.ellipsis, 14336..<28672]
    try observe("gated", gated)
    let feed = try project(gated,
      "mlp.fc2", rows: 5376, columns: 14336)
    try observe("feed", feed)
    let output = residual + feed
    eval(output)
    try observe("output", output)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return output
  }
}
