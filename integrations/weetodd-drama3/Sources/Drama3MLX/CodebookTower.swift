import Drama3Contracts
import Foundation
import MLX
import MLXNN

final class Drama3CodebookTower {
  var diagnostic: ((Int, MLXArray) -> Void)?
  let w: Drama3Weights
  let backbone: Drama3HybridBackbone
  init(_ w: Drama3Weights, backbone: Drama3HybridBackbone) {
    self.w = w
    self.backbone = backbone
  }
  func semantic(_ tokens: MLXArray) -> MLXArray { (tokens .>= 248099) .&& (tokens .<= 252194) }
  func layer(
    _ name: String, hidden: MLXArray, codes: MLXArray, mask: MLXArray, state: Drama3State,
    extra: MLXArray? = nil, enrichment: Bool
  ) throws -> MLXArray {
    let embedding = try w.get(
      name + (enrichment ? ".semantic_embedding.weight" : ".codebook_embedding.weight"))[
        clip(codes, min: 0, max: 4095)]
    var e = MLX.where(mask[.ellipsis, .newAxis], embedding, MLXArray.zeros(like: embedding))
    if let extra { e = e + extra }
    var x = try w.linear(
      name + ".eh_proj",
      concatenated([w.norm(name + ".enorm", e), w.norm(name + ".hnorm", hidden)], axis: -1))
    let p = name + ".transformer_layer"
    x =
      x
      + (try backbone.attention(
        w.norm(p + ".attention_norm", x), prefix: p + ".attention", state: state, tower: true))
    let n = try w.norm(p + ".ffn_norm", x)
    x =
      x
      + (try w.linear(
        p + ".feed_forward.w2",
        silu(w.linear(p + ".feed_forward.w1", n)) * w.linear(p + ".feed_forward.w3", n)))
    return try w.norm(name + ".final_layernorm", x)
  }
  private func enrich(
    _ hidden: MLXArray, c0: MLXArray, mask: MLXArray, frame: MLXArray, state: Drama3State
  ) throws -> MLXArray {
    let indices = frame[.ellipsis, 3..<11] + MLXArray((2..<10).map { Int32($0 * 4096) })
    let embed = try w.get("tower.codebook_embeddings.weight")[indices].sum(axis: -2)
    let extra = MLX.where(
      semantic(frame[.ellipsis, 0])[.ellipsis, .newAxis], embed, MLXArray.zeros(like: embed))
    var h = hidden
    for i in 0..<4 {
      h = try layer(
        "tower.shared_layers.\(i)", hidden: h, codes: c0, mask: mask, state: state,
        extra: i == 0 ? extra : nil, enrichment: true)
    }
    return h
  }
  func prefill(hidden: MLXArray, frames: MLXArray, next: MLXArray, state: Drama3State) throws {
    let mask = semantic(next[.ellipsis, 0])
    var h = try enrich(hidden, c0: next[.ellipsis, 1], mask: mask, frame: frames, state: state)
    for i in 0..<9 {
      h = try layer(
        "tower.residual_layers.\(i)", hidden: h, codes: next[.ellipsis, i + 1], mask: mask,
        state: state, enrichment: false)
    }
  }
  func generate(
    hidden: MLXArray, main: MLXArray, previous: MLXArray, state: Drama3State,
    sampling: Drama3Sampling, teacher: MLXArray? = nil
  ) throws -> MLXArray {
    let mask = semantic(main)
    let c0 = clip(main - 248099, min: 0, max: 4095)
    var h = try enrich(hidden, c0: c0, mask: mask, frame: previous, state: state)
    var codes = [c0]
    for i in 0..<9 {
      h = try layer(
        "tower.residual_layers.\(i)", hidden: h, codes: codes.last!, mask: mask, state: state,
        enrichment: false)
      let logits = try w.linear("tower.codebook_heads.\(i)", h)
      diagnostic?(i, logits)
      if let teacher {
        codes.append(teacher[0..., 0..., i + 2])
      } else {
        codes.append(
          try Drama3Sampler.sample(
            logits: logits, temperature: sampling.temperature, topK: sampling.topK,
            topP: sampling.topP, checked: false))
      }
    }
    return stacked([main] + codes, axis: -1).asType(.int32)
  }
}
