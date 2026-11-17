import Foundation
import MLX
import MLXNN

struct ReCaFlashBlock {
  let query: ReCaLinear, key: ReCaLinear, value: ReCaLinear, output: ReCaLinear
  let gate: ReCaLinear, up: ReCaLinear, down: ReCaLinear
  let attentionNorm: MLXArray, feedNorm: MLXArray, keyNorm: MLXArray, valueNorm: MLXArray
  init(_ c: ReCaCheckpoint, _ prefix: String) throws {
    query = try c.linear(prefix + ".attention.query"); key = try c.linear(prefix + ".attention.key")
    value = try c.linear(prefix + ".attention.value"); output = try c.linear(prefix + ".attention.output")
    gate = try c.linear(prefix + ".feed_forward.gate"); up = try c.linear(prefix + ".feed_forward.up")
    down = try c.linear(prefix + ".feed_forward.down")
    attentionNorm = try c.tensor(prefix + ".attention_norm.weight")
    feedNorm = try c.tensor(prefix + ".feed_forward_norm.weight")
    keyNorm = try c.tensor(prefix + ".attention.key_norm.weight")
    valueNorm = try c.tensor(prefix + ".attention.value_norm.weight")
  }
  func step(_ x: MLXArray, history: ReCaKV?, queryRoPE: ReCaRotary, keyRoPE: ReCaRotary,
    plan: ReCaAttentionPlan) throws -> (MLXArray, ReCaKV) {
    let norm = ReCaMath.rms(x, weight: attentionNorm, epsilon: 1e-5, promote: true)
    let q = query(norm).reshaped([-1, 8, 64])
    let k = ReCaMath.rms(key(norm).reshaped([-1, 2, 64]), weight: keyNorm, epsilon: 1e-5, promote: true)
    let v = ReCaMath.rms(value(norm).reshaped([-1, 2, 64]), weight: valueNorm, epsilon: 1e-5, promote: true)
    let keys = history.map { concatenated([$0.key, k], axis: 0) } ?? k
    let values = history.map { concatenated([$0.value, v], axis: 0) } ?? v
    let attended = try plan.apply(query: queryRoPE.apply(q), key: keyRoPE.apply(keys), value: values)
    let residual = x + output(attended.reshaped([-1, 512]))
    let feed = ReCaMath.rms(residual, weight: feedNorm, epsilon: 1e-5, promote: true)
    let result = residual + down(silu(gate(feed)) * up(feed))
    eval(result, keys, values)
    return (result, .init(key: ReCaMath.own(keys, range: 0..<keys.dim(0)),
      value: ReCaMath.own(values, range: 0..<values.dim(0))))
  }
}

private struct ReCaFlashWeights {
  let latent: ReCaLinear, temporal: ReCaLinear, spatialUp: ReCaLinear, spatialOut: ReCaLinear
  let lrStem: ReCaLinear, lrEarly: ReCaLinear, lrLate: ReCaLinear
  let backbone: [ReCaFlashBlock], refinement: [ReCaFlashBlock]
  init(_ c: ReCaCheckpoint) throws {
    latent = try c.linear("latent_projection"); temporal = try c.linear("temporal_projection")
    spatialUp = try c.linear("spatial_projection.0"); spatialOut = try c.linear("spatial_projection.2")
    lrStem = try c.linear("lr_stem"); lrEarly = try c.linear("lr_early_projection", bias: false)
    lrLate = try c.linear("lr_late_projection", bias: false)
    backbone = try (0..<12).map { try ReCaFlashBlock(c, "backbone.blocks.\($0)") }
    refinement = try (0..<2).map { try ReCaFlashBlock(c, "temporal_refinement.blocks.\($0)") }
  }
}

struct ReCaVAENormalization: Decodable {
  let latents_mean: [Float], latents_std: [Float]
  func validate() throws {
    guard latents_mean.count == 48, latents_std.count == 48,
      latents_mean.allSatisfy(\.isFinite), latents_std.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw ReCaError.invalid("ReCaVSR needs the 48-channel Wan2.2 latent normalization.")
    }
  }
}

final class ReCaFlashDecoder {
  let geometry: ReCaGeometry
  private let mean: MLXArray, std: MLXArray
  private var weights: ReCaFlashWeights?
  private var identity: String?
  private(set) var frameIndex = 0
  private var backbone = Array<ReCaKV?>(repeating: nil, count: 12)
  private var refinement = Array<ReCaKV?>(repeating: nil, count: 2)
  private var failed = false
  private let attentionPlan: ReCaAttentionPlan
  init(geometry: ReCaGeometry, normalization: ReCaVAENormalization,
    scheduling: ReCaAttentionScheduling = .automatic) throws {
    try normalization.validate(); self.geometry = geometry
    mean = MLXArray(normalization.latents_mean).asType(.bfloat16)
    std = MLXArray(normalization.latents_std).asType(.bfloat16)
    attentionPlan = ReCaAttentionPlan(height: geometry.latentHeight, width: geometry.latentWidth,
      windowHeight: 24, windowWidth: 24, slidesAtEdge: false, scheduling:scheduling)
  }
  func loadWeights(from checkpoint: ReCaCheckpoint) throws {
    guard !failed, identity == nil || identity == checkpoint.asset.sha256 else { throw ReCaError.invalid("Decoder identity changed; reset the ReCaVSR stream.") }
    if weights != nil { return }
    weights = try ReCaFlashWeights(checkpoint); identity = checkpoint.asset.sha256
  }
  func unloadWeights() { weights = nil; Memory.clearCache() }
  func reset() {
    frameIndex = 0; failed = false
    backbone = Array(repeating: nil, count: 12); refinement = Array(repeating: nil, count: 2)
    attentionPlan.reset()
  }
  func close() { weights = nil; reset(); Memory.clearCache() }
  static func pixelShuffle(_ x: MLXArray, factor: Int = 16) throws -> MLXArray {
    guard x.ndim == 4, factor > 0, x.dim(3) == 3 * factor * factor else { throw ReCaError.invalid("Invalid decoder subpixel layout.") }
    return x.reshaped([x.dim(0), x.dim(1), x.dim(2), 3, factor, factor])
      .transposed(0, 1, 4, 2, 5, 3).reshaped([x.dim(0), x.dim(1) * factor, x.dim(2) * factor, 3])
  }
  static func groupLRFeatures(_ features: MLXArray, first: Bool) throws -> MLXArray {
    guard features.ndim == 4, features.dim(0) == (first ? 1 : 4) else { throw ReCaError.invalid("LR decoder conditioning must contain one first frame or four body frames.") }
    return first ? repeated(features, count: 4, axis: 0) : features
  }
  private func stack(_ x: MLXArray, blocks: [ReCaFlashBlock], state: inout [ReCaKV?], frames: Int) throws -> MLXArray {
    let h = geometry.latentHeight, w = geometry.latentWidth, span = h * w
    let past = state[0] == nil ? 0 : frames
    let qrope = ReCaRotary(height: h, width: w, times: Array(past..<(past + frames)), axes: [16, 24, 24], floatFrequencies: true)
    let krope = ReCaRotary(height: h, width: w, times: Array(0..<(past + frames)), axes: [16, 24, 24], floatFrequencies: true)
    var hidden = x
    for i in blocks.indices {
      try Task.checkCancellation()
      let old = state[i].map { ReCaKV(key: ReCaMath.own($0.key, range: ($0.key.dim(0) - past * span)..<$0.key.dim(0)),
        value: ReCaMath.own($0.value, range: ($0.value.dim(0) - past * span)..<$0.value.dim(0))) }
      let (next, kv) = try blocks[i].step(hidden, history: old, queryRoPE: qrope, keyRoPE: krope,
        plan: attentionPlan)
      state[i] = kv; hidden = next
    }
    return hidden
  }
  func decode(latents: MLXArray, lq: MLXArray, emit: (MLXArray) throws -> Void) throws {
    guard !failed, let weights else { throw ReCaError.invalid("ReCaVSR decoder weights are not loaded or the stream failed.") }
    let expected = latents.dim(0) * 4 - (frameIndex == 0 ? 3 : 0)
    guard latents.ndim == 4, latents.shape == [latents.dim(0), geometry.latentHeight, geometry.latentWidth, 48],
      lq.shape == [expected, geometry.paddedHeight, geometry.paddedWidth, 3] else { throw ReCaError.invalid("Decoder LQ and latent blocks do not align.") }
    var offset = 0
    do {
      for i in 0..<latents.dim(0) {
        try Task.checkCancellation()
        let first = frameIndex == 0, take = first ? 1 : 4
        let multiplied = latents[i..<i + 1].asType(.bfloat16) * std; eval(multiplied)
        let raw = multiplied + mean; eval(raw)
        let features = try weights.lrStem(ReCaLQProjection.pixelUnshuffle(lq[offset..<offset + take].asType(.bfloat16)))
        let grouped = try Self.groupLRFeatures(features, first: first)
        let early = weights.lrEarly(grouped.transposed(1, 2, 0, 3).reshaped([-1, 192]))
        let late = weights.lrLate(grouped.reshaped([-1, 48]))
        var x = weights.latent(raw.reshaped([-1, 48])) + early
        x = try stack(x, blocks: weights.backbone, state: &backbone, frames: 1)
        x = weights.temporal(x).reshaped([geometry.latentHeight, geometry.latentWidth, 4, 512])
          .transposed(2, 0, 1, 3).reshaped([-1, 512]) + late
        x = try stack(x, blocks: weights.refinement, state: &refinement, frames: 4)
        let pixels = weights.spatialOut(silu(weights.spatialUp(x))).reshaped([4, geometry.latentHeight, geometry.latentWidth, 768])
        let shuffled = try Self.pixelShuffle(pixels)
        let rgb = clip(shuffled[first ? 3..<4 : 0..<4], min: -1, max: 1)
        eval(rgb); try emit(rgb)
        frameIndex += 1; offset += take
      }
    } catch { failed = true; weightsCleanup(); throw error }
  }
  private func weightsCleanup() { weights = nil; backbone = Array(repeating: nil, count: 12); refinement = Array(repeating: nil, count: 2); Memory.clearCache() }
}
