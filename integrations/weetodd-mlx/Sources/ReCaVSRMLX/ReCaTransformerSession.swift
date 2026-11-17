import Foundation
import InferenceContracts
import MLX
import MLXRandom

/// Per-shot stream state survives parameter unloading. A failed partial step
/// invalidates the session; replay begins at the shot boundary.
final class ReCaTransformerSession {
  let geometry: ReCaGeometry, seed: UInt64
  private(set) var weights: ReCaTransformerWeights?
  private var identity: String?
  private(set) var caches = (0..<30).map { ReCaKVCache(layer: $0) }
  private var lqState = ReCaLQState()
  private var previous: MLXArray?
  private(set) var offset = 0
  private var block = 0
  private var failed = false
  private var spatialTables: [Int: ReCaRotary] = [:]
  private let attentionPlan: ReCaAttentionPlan
  init(geometry: ReCaGeometry, seed: UInt64, scheduling: ReCaAttentionScheduling = .automatic) {
    self.geometry = geometry; self.seed = seed
    attentionPlan = ReCaAttentionPlan(height: geometry.tokenHeight, width: geometry.tokenWidth,
      windowHeight: 22, windowWidth: 40, slidesAtEdge: true, scheduling:scheduling)
  }
  func loadWeights(from checkpoint: ReCaCheckpoint, prompt: MLXArray) throws {
    guard !failed, identity == nil || identity == checkpoint.asset.sha256 else { throw ReCaError.invalid("ReCaVSR stream weights changed; reset the session.") }
    weights = try ReCaTransformerWeights(checkpoint, prompt: prompt); identity = checkpoint.asset.sha256
  }
  func unloadWeights() { weights = nil; Memory.clearCache() }
  func close() { weights = nil; reset(); Memory.clearCache() }
  func reset() {
    caches = (0..<30).map { ReCaKVCache(layer: $0) }; lqState = ReCaLQState()
    previous = nil; offset = 0; block = 0; failed = false; spatialTables.removeAll()
    attentionPlan.reset()
  }
  static func predictedLatents(noise: MLXArray, prediction: MLXArray) -> MLXArray {
    noise.asType(.float32) - prediction.asType(.float32)
  }
  /// Engine tensors are [T,H,W,C]; permutations at worker/reference boundaries are explicit.
  func step(lq: MLXArray, noise: MLXArray? = nil) throws -> MLXArray {
    guard !failed, let weights else { throw ReCaError.invalid("ReCaVSR transformer weights are not loaded or the stream failed.") }
    let frames = offset == 0 ? 6 : 2, rgb = offset == 0 ? 21 : 8
    guard lq.shape == [rgb, geometry.paddedHeight, geometry.paddedWidth, 3] else { throw ReCaError.invalid("Wrong ReCaVSR prefix/body input shape.") }
    let shape = [frames, geometry.latentHeight, geometry.latentWidth, 48]
    let noise = noise ?? MLXRandom.normal(shape, dtype: .float32, key: MLXRandom.key(seed &+ UInt64(block)))
    guard noise.shape == shape, noise.dtype == .float32 else { throw ReCaError.invalid("Explicit ReCaVSR noise must be FP32 with the admitted block shape.") }
    do {
      var x = weights.patch(noise)
      x = x + (try weights.lq.step(lq, state: &lqState))
      x = x + weights.recycle(weights.patch(previous ?? MLXArray.zeros(shape, dtype: .bfloat16)))
      let positions = try ReCaRollingRoPE.positions(start: offset, frames: frames)
      if spatialTables[frames] == nil {
        spatialTables[frames] = ReCaRotary(height: geometry.tokenHeight, width: geometry.tokenWidth,
          times: Array(repeating: 0, count: frames), axes: [44, 42, 42], temporal: false)
      }
      let spatial = spatialTables[frames]!
      let queryTimes = positions.positions
      let temporalQuery = ReCaRotary(height: geometry.tokenHeight, width: geometry.tokenWidth,
        times: queryTimes, axes: [44, 42, 42], spatial: false)
      var temporalTables: [[Int]: ReCaRotary] = [:]
      for i in 0..<30 {
        try Task.checkCancellation()
        let times = caches[i].ids.map { $0 - positions.origin } + queryTimes
        guard times.allSatisfy({ (0..<22).contains($0) }) else { throw ReCaError.invalid("Routed KV escaped the 22-latent RoPE interval.") }
        if temporalTables[times] == nil {
          temporalTables[times] = ReCaRotary(height: geometry.tokenHeight, width: geometry.tokenWidth,
            times: times, axes: [44, 42, 42], spatial: false)
        }
        let (next, kv) = try weights.blocks[i].step(x, spatial: spatial, temporalQuery: temporalQuery,
          temporalKey: temporalTables[times]!, cache: caches[i], plan: attentionPlan)
        eval(next, kv.key, kv.value); x = next
        try caches[i].commit(key: kv.key, value: kv.value, frames: frames, start: offset)
      }
      let m = weights.headModulation
      x = (MLXFast.layerNorm(x.asType(.float32), weight: nil, bias: nil, eps: 1e-6) * (1 + m[1]) + m[0]).asType(.bfloat16)
      let prediction = weights.head(x).reshaped([frames, geometry.tokenHeight, geometry.tokenWidth, 2, 2, 48])
        .transposed(0, 1, 3, 2, 4, 5).reshaped(shape)
      let result = Self.predictedLatents(noise: noise, prediction: prediction)
      eval(result); previous = ReCaMath.own(result.asType(.bfloat16), range: (frames - 2)..<frames)
      offset += frames; block += 1
      return result
    } catch { failed = true; weightsCleanup(); throw error }
  }
  private func weightsCleanup() { weights = nil; previous = nil; caches = (0..<30).map { ReCaKVCache(layer: $0) }; lqState = ReCaLQState(); Memory.clearCache() }
}
