import Foundation
import MLX
import MLXNN
import ImageLayout

struct MOSSPreparedConditioning {
  let text: MLXArray
  let rope: MLXArray
  let height, width: Int
  var attentionPlan: MOSSPreparedAttentionPlan? = nil
}

final class MOSSTransformer {
  let weights: any MOSSWeightAccess
  let d: MOSSDimensions
  init(weights: any MOSSWeightAccess, dimensions: MOSSDimensions = .init()) { self.weights = weights; d = dimensions }
  private func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
    let value = MOSSMath.rms(x, weight: try weights.tensor(name + ".weight"))
    return weights.profiler?.evaluate("transformer/RMSNorm", value) ?? value
  }
  private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, prefix: String, rope: MLXArray, plan: MOSSPreparedAttentionPlan? = nil) throws -> MLXArray {
    let b = q.dim(0), n = q.dim(1), head = d.width / d.heads
    let queries = MOSSMath.rotate(try norm(q.reshaped(b, n, d.heads, head), prefix + "q_norm"), rope: rope).transposed(0, 2, 1, 3).asType(v.dtype)
    let keys = MOSSMath.rotate(try norm(k.reshaped(b, n, d.heads, head), prefix + "k_norm"), rope: rope).transposed(0, 2, 1, 3).asType(v.dtype)
    let values = v.reshaped(b,n,d.heads,head).transposed(0,2,1,3), scale = Float(1 / sqrt(Double(head)))
    let attended = plan.map { MOSSRegionalAttention.apply(queries: queries, keys: keys, values: values, scale: scale, plan: $0) }
      ?? MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: nil)
    let value = attended.transposed(0, 2, 1, 3).reshaped(b, n, d.width)
    return weights.profiler?.evaluate("transformer/attention + RoPE", value) ?? value
  }
  func prepare(context: MLXArray, height: Int, width: Int) throws -> MOSSPreparedConditioning {
    let dtype = try weights.tensor("video_embedder.weight").dtype
    let rope = MOSSMath.rope(height: height, width: width, textLength: context.dim(1), headDimension: d.width / d.heads).asType(dtype)
    let textRope = rope[0..., (height * width)..., 0...]
    var x = try weights.linear(context, "text_embedder.proj").asType(.float32)
    for i in 0..<2 {
      let p = "text_embedder.blocks.\(i)."
      let qkv = try weights.linear(norm(x, p + "norm1"), p + "qkv")
      let q = qkv[.ellipsis, 0..<d.width], k = qkv[.ellipsis, d.width..<(2 * d.width)], v = qkv[.ellipsis, (2 * d.width)...]
      x = x + (try weights.linear(attention(q, k, v, prefix: p, rope: textRope), p + "proj")).asType(.float32)
      let y = try norm(x, p + "norm2")
      var ffn = MOSSMath.situ(try weights.linear(y, p + "ffn_up"), try weights.linear(y, p + "ffn_gate"))
      if let profiler = weights.profiler { ffn = profiler.evaluate("transformer/SiTU", ffn) }
      x = x + (try weights.linear(ffn, p + "ffn_down")).asType(.float32)
    }
    eval(x, rope)
    return .init(text: x, rope: rope, height: height, width: width)
  }
  func prepareRegional(contexts: [MLXArray], groups: [MOSSRegionalPrompt], height: Int, width: Int) throws -> MOSSPreparedConditioning {
    guard contexts.count == groups.count, !contexts.isEmpty else { throw MOSSError("Regional contexts do not match their boxes.") }
    let refined = try contexts.map { try prepare(context: $0, height: height, width: width) }
    let lengths = contexts.map { $0.dim(1) }
    let plan = try MOSSRegionalAttentionPlan(height: height, width: width, groups: groups, textLengths: lengths)
    // Right alignment preserves image/text relative RoPE: total-(total-len+j)=len-j.
    let combinedRope = MOSSMath.regionalRope(height: height, width: width, textLengths: lengths, headDimension: d.width/d.heads).asType(refined[0].rope.dtype)
    var result = MOSSPreparedConditioning(text: concatenated(refined.map(\.text), axis: 1), rope: combinedRope, height: height, width: width)
    result.attentionPlan = .init(plan)
    eval(result.text, result.rope)
    return result
  }
  func predict(latents: MLXArray, timestep: Double, conditioning c: MOSSPreparedConditioning, skipLayers: Set<Int>) throws -> MLXArray {
    try Task.checkCancellation()
    let rows = c.height * c.width
    var time = MLXArray(MOSSMath.timeEmbedding(timestep, dimension: d.frequencyWidth)).reshaped(1, d.frequencyWidth)
    time = try weights.linear(silu(weights.linear(time, "time_embedding.0")), "time_embedding.2")
    let mod = try weights.linear(time, "shared_block_modulation").asType(.float32).reshaped(1, 6, d.width)
    let head = try weights.linear(time, "video_head_modulation").asType(.float32).reshaped(1, 2, d.width)
    let image = latents.reshaped(1, d.latentChannels, rows).transposed(0, 2, 1)
    var x = concatenated([try weights.linear(image, "video_embedder").asType(.float32), c.text], axis: 1)
    for i in 0..<d.depth where !skipLayers.contains(i) {
      try Task.checkCancellation()
      let p = "layers.\(i)."
      let y = try norm(x, p + "norm1") * (1 + mod[0..., 1..<2, 0...]) + mod[0..., 0..<1, 0...]
      let attended = try attention(weights.linear(y, p + "proj.q"), weights.linear(y, p + "proj.k"),
        weights.linear(y, p + "proj.v"), prefix: p, rope: c.rope, plan: c.attentionPlan)
      x = x + (try weights.linear(attended, p + "proj.o")).asType(.float32) * mod[0..., 2..<3, 0...]
      let z = try norm(x, p + "norm2") * (1 + mod[0..., 4..<5, 0...]) + mod[0..., 3..<4, 0...]
      let up = try weights.linear(z, p + "proj.ffn_up_gate")
      var ffn = MOSSMath.situ(up[.ellipsis, 0..<d.feedForward], up[.ellipsis, d.feedForward...])
      if let profiler = weights.profiler { ffn = profiler.evaluate("transformer/SiTU", ffn) }
      x = x + (try weights.linear(ffn, p + "proj.ffn_down")).asType(.float32) * mod[0..., 5..<6, 0...]
      // Resolve all references to a streamed block before opening the next one.
      if let profiler = weights.profiler { x = profiler.evaluate("transformer/residual", x) } else { eval(x) }
    }
    let weight = try weights.tensor("final_norm_video.weight")
    var output = MOSSMath.rms(x[0..., 0..<rows, 0...].asType(weight.dtype), weight: weight).asType(.float32)
    output = output * (1 + head[0..., 1..<2, 0...]) + head[0..., 0..<1, 0...]
    return try weights.linear(output, "final_linear_video").transposed(0, 2, 1).reshaped(1, d.latentChannels, c.height, c.width).asType(.float32)
  }
}
