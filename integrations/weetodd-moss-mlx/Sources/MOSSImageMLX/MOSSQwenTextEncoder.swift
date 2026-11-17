import Foundation
import MLX
import MLXNN

final class MOSSQwenTextEncoder {
  let weights: any MOSSWeightAccess
  init(weights: any MOSSWeightAccess) { self.weights = weights }
  func encode(tokens: [Int], progress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
    guard !tokens.isEmpty, tokens.count <= 4096 else { throw MOSSError("Invalid conditioner token count.") }
    let ids = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
    var x = try weights.tensor("language_model.embed_tokens.weight")[ids]
    for layer in 0..<32 {
      try Task.checkCancellation()
      let prefix = "language_model.layers.\(layer)."
      let y = MOSSQwenMath.norm(x, weight: try weights.tensor(prefix + "input_layernorm.weight"))
      let mixed = layer % 4 == 3 ? try attention(y, prefix: prefix + "self_attn.") : try delta(y, prefix: prefix + "linear_attn.")
      x = x + mixed
      let normed = MOSSQwenMath.norm(x, weight: try weights.tensor(prefix + "post_attention_layernorm.weight"))
      let gate = try weights.linear(normed, prefix + "mlp.gate_proj")
      let up = try weights.linear(normed, prefix + "mlp.up_proj")
      let activation = silu(gate) * up
      let activated = weights.profiler?.evaluate("qwen/FFN activation", activation) ?? activation
      x = x + (try weights.linear(activated, prefix + "mlp.down_proj"))
      eval(x); progress?(layer + 1, 32)
    }
    let output = MOSSQwenMath.norm(x, weight: try weights.tensor("language_model.norm.weight"))
    eval(output); return output
  }
  private func attention(_ x: MLXArray, prefix p: String) throws -> MLXArray {
    let n = x.dim(1)
    let qAndGate = try weights.linear(x, p + "q_proj").reshaped(1, n, 16, 512)
    let gate = qAndGate[.ellipsis, 256...].reshaped(1, n, 4096)
    var q = MOSSQwenMath.norm(qAndGate[.ellipsis, 0..<256], weight: try weights.tensor(p + "q_norm.weight")).transposed(0, 2, 1, 3)
    var k = MOSSQwenMath.norm(try weights.linear(x, p + "k_proj").reshaped(1, n, 4, 256), weight: try weights.tensor(p + "k_norm.weight")).transposed(0, 2, 1, 3)
    q = MLXFast.RoPE(q, dimensions: 64, traditional: false, base: 10_000_000, scale: 1, offset: 0)
    k = MLXFast.RoPE(k, dimensions: 64, traditional: false, base: 10_000_000, scale: 1, offset: 0)
    let v = try weights.linear(x, p + "v_proj").reshaped(1, n, 4, 256).transposed(0, 2, 1, 3)
    let row = MLXArray(0..<n)
    let mask = row.expandedDimensions(axis: 0) .<= row.expandedDimensions(axis: 1)
    let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / 16,
      mask: mask).transposed(0, 2, 1, 3).reshaped(1, n, 4096)
    let attention = weights.profiler?.evaluate("qwen/full attention + RoPE", attended) ?? attended
    return try weights.linear(attention * sigmoid(gate), p + "o_proj")
  }
  private func delta(_ x: MLXArray, prefix p: String) throws -> MLXArray {
    let n = x.dim(1)
    let qkv = try weights.linear(x, p + "in_proj_qkv")
    let padded = concatenated([MLXArray.zeros([1, 3, 8192], dtype: qkv.dtype), qkv], axis: 1)
    let kernel = try weights.tensor(p + "conv1d.weight").transposed(0, 2, 1)
    let mixed = silu(conv1d(padded, kernel, groups: 8192))
    let q = mixed[.ellipsis, 0..<2048].reshaped(1, n, 16, 128)
    let k = mixed[.ellipsis, 2048..<4096].reshaped(1, n, 16, 128)
    let v = mixed[.ellipsis, 4096...].reshaped(1, n, 32, 128)
    let a = try weights.linear(x, p + "in_proj_a"), b = try weights.linear(x, p + "in_proj_b")
    var out = MOSSQwenMath.delta(q: q, k: k, v: v, a: a, b: b,
      aLog: try weights.tensor(p + "A_log"), dtBias: try weights.tensor(p + "dt_bias"))
    if let profiler = weights.profiler { out = profiler.evaluate("qwen/GDN + causal conv", out) }
    let gate = try weights.linear(x, p + "in_proj_z").reshaped(1, n, 32, 128)
    let normalized = MOSSQwenMath.gatedNorm(out.asType(x.dtype), weight: try weights.tensor(p + "norm.weight"), gate: gate)
    return try weights.linear(normalized.reshaped(1, n, 4096), p + "out_proj")
  }
}
