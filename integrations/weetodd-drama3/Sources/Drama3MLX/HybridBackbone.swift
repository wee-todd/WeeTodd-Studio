import Drama3Contracts
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

final class Drama3State {
  var attention: [String: KVCacheSimple] = [:]
  var recurrent: [String: MLXArray] = [:]
  var convolution: [String: MLXArray] = [:]
  func cache(_ name: String) -> KVCacheSimple {
    if let c = attention[name] { return c }
    let c = KVCacheSimple()
    attention[name] = c
    return c
  }
  func evaluate(_ extra: [MLXArray] = []) {
    eval(
      attention.values.flatMap { $0.state } + Array(recurrent.values) + Array(convolution.values)
        + extra)
  }
}
final class Drama3HybridBackbone {
  let w: Drama3Weights
  init(_ w: Drama3Weights) { self.w = w }
  func embedding(_ frames: MLXArray) throws -> MLXArray {
    let text = try w.get("backbone.model.embed_tokens.weight")[frames[.ellipsis, 0]]
    let semantic = (frames[.ellipsis, 0] .>= 248099) .&& (frames[.ellipsis, 0] .<= 252194)
    let residual = try w.get("tower.codebook_embeddings.weight")[frames[.ellipsis, 2] + 4096]
    return MLX.where(semantic[.ellipsis, .newAxis], (text + residual) * Float(1 / sqrt(2.0)), text)
  }
  func forward(_ frames: MLXArray, state: Drama3State) throws -> MLXArray {
    var x = try embedding(frames)
    for layer in 0..<40 { x = try block(layer, x, state: state) }
    return try w.norm("backbone.model.norm", x)
  }
  func block(_ layer: Int, _ input: MLXArray, state: Drama3State) throws -> MLXArray {
    var x = input
    let prefix = "backbone.model.layers.\(layer)"
    let normalized = try w.norm(prefix + ".input_layernorm", x)
    if layer % 4 == 3 {
      x = x + (try attention(normalized, prefix: prefix + ".self_attn", state: state, tower: false))
    } else {
      x = x + (try delta(normalized, prefix: prefix + ".linear_attn", state: state))
    }
    let y = try w.norm(prefix + ".post_attention_layernorm", x)
    let p = prefix + ".mlp"
    let probabilities = softmax(try w.linear(p + ".gate", y), axis: -1, precise: true)
    let ids = argPartition(probabilities, kth: 248, axis: -1)[.ellipsis, 248...]
    var scores = takeAlong(probabilities, ids, axis: -1)
    scores = scores / scores.sum(axis: -1, keepDims: true)
    let expanded = y[.ellipsis, .newAxis, .newAxis, 0...]
    let gate = try w.linear(p + ".switch_mlp.gate_proj", expanded, indices: ids)
    let up = try w.linear(p + ".switch_mlp.up_proj", expanded, indices: ids)
    let experts = try w.linear(p + ".switch_mlp.down_proj", silu(gate) * up, indices: ids).squeezed(
      axis: -2)
    let routed = (experts * scores[.ellipsis, .newAxis]).sum(axis: -2)
    let shared =
      try w.mlp(p + ".shared_expert", y) * sigmoid(w.linear(p + ".shared_expert_gate", y))
    x = x + (routed + shared)

    return x
  }
  func attention(_ x: MLXArray, prefix: String, state: Drama3State, tower: Bool) throws -> MLXArray
  {
    let b = x.dim(0)
    let t = x.dim(1)
    let cache = state.cache(prefix)
    let offset = cache.offset
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let gate: MLXArray
    if tower {
      let projection = try w.linear(prefix + ".wqkv", x)
      let split = MLX.split(projection, indices: [4096, 8192, 8704], axis: -1)
      q = split[0].reshaped(b, t, 16, 256)
      gate = split[1]
      k = split[2].reshaped(b, t, 2, 256)
      v = split[3].reshaped(b, t, 2, 256)
    } else {
      let projection = try w.linear(prefix + ".q_proj", x).reshaped(b, t, 16, 512)
      q = projection[.ellipsis, ..<256]
      gate = projection[.ellipsis, 256...].reshaped(b, t, 4096)
      k = try w.linear(prefix + ".k_proj", x).reshaped(b, t, 2, 256)
      v = try w.linear(prefix + ".v_proj", x).reshaped(b, t, 2, 256)
    }
    let qnorm = try w.norm(prefix + ".q_norm", q).transposed(0, 2, 1, 3)
    let knorm = try w.norm(prefix + ".k_norm", k).transposed(0, 2, 1, 3)
    let qr = MLXFast.RoPE(
      tower ? qnorm.asType(.float32) : qnorm, dimensions: 64, traditional: tower, base: 10_000_000,
      scale: 1, offset: offset
    ).asType(x.dtype)
    let kr = MLXFast.RoPE(
      tower ? knorm.asType(.float32) : knorm, dimensions: 64, traditional: tower, base: 10_000_000,
      scale: 1, offset: offset
    ).asType(x.dtype)
    let (keys, values) = cache.update(keys: kr, values: v.transposed(0, 2, 1, 3))
    let attended = MLXFast.scaledDotProductAttention(
      queries: contiguous(qr), keys: t > 1 ? contiguous(keys) : keys,
      values: t > 1 ? contiguous(values) : values, scale: 1 / 16, mask: t > 1 ? .causal : .none)
    let out = attended.transposed(0, 2, 1, 3).reshaped(b, t, 4096) * sigmoid(gate)
    return try w.linear(prefix + (tower ? ".wo" : ".o_proj"), out)
  }
  private func delta(_ x: MLXArray, prefix: String, state: Drama3State) throws -> MLXArray {
    let b = x.dim(0)
    let t = x.dim(1)
    let qkv = try w.linear(prefix + ".in_proj_qkv", x)
    let old = state.convolution[prefix] ?? MLXArray.zeros([b, 3, 8192], dtype: x.dtype)
    let combined = concatenated([old, qkv], axis: 1)
    let convolved = silu(try conv1d(combined, w.get(prefix + ".conv1d.weight"), groups: 8192))
    state.convolution[prefix] = combined[0..., t..<(t + 3), 0...]
    let parts = MLX.split(convolved, indices: [2048, 4096], axis: -1)
    let q = parts[0].reshaped(b, t, 16, 128)
    let k = parts[1].reshaped(b, t, 16, 128)
    let v = parts[2].reshaped(b, t, 32, 128)
    let scale = Float(1 / sqrt(128.0))
    let epsilon: Float = 1e-6
    let qn = MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: epsilon) * (scale * scale)
    let kn = MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: epsilon) * scale
    let result = try gatedDeltaUpdate(
      q: qn, k: kn, v: v, a: w.linear(prefix + ".in_proj_a", x),
      b: w.linear(prefix + ".in_proj_b", x), aLog: w.get(prefix + ".A_log"),
      dtBias: w.get(prefix + ".dt_bias"), state: state.recurrent[prefix])
    state.recurrent[prefix] = result.1
    let z = try w.linear(prefix + ".in_proj_z", x).reshaped(b, t, 32, 128)
    let normalized = MLXFast.rmsNorm(
      result.0, weight: try w.get(prefix + ".norm.weight"), eps: 1e-6)
    let gated = (normalized.asType(.float32) * silu(z.asType(.float32))).asType(result.0.dtype)
    return try w.linear(prefix + ".out_proj", gated.reshaped(b, t, 4096))
  }
}
