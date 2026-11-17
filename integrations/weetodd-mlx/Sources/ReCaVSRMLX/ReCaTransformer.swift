import Foundation
import MLX
import MLXNN

struct ReCaKV { let key: MLXArray, value: MLXArray }
struct ReCaKVCache {
  let layer: Int
  private(set) var ids: [Int] = []
  private(set) var planes: [Int: ReCaKV] = [:]
  private var next = 0
  mutating func commit(key: MLXArray, value: MLXArray, frames: Int, start: Int) throws {
    guard start == next, frames > 0, key.shape == value.shape, key.dim(0) % frames == 0 else {
      throw ReCaError.invalid("KV commits must contain consecutive complete latent planes.")
    }
    let selected = try ReCaKVRouter.retainedIDs(candidates: ids + Array(start..<(start + frames)), layer: layer)
    let span = key.dim(0) / frames
    planes = planes.filter { selected.contains($0.key) }
    for id in selected where planes[id] == nil {
      guard id >= start else { throw ReCaError.invalid("Cannot resurrect an evicted KV plane.") }
      let range = ((id - start) * span)..<((id - start + 1) * span)
      planes[id] = .init(key: ReCaMath.own(key, range: range), value: ReCaMath.own(value, range: range))
    }
    ids = selected; next += frames
  }
}

struct ReCaWanAttention {
  let query: ReCaLinear, key: ReCaLinear, value: ReCaLinear, output: ReCaLinear
  let queryNorm: MLXArray, keyNorm: MLXArray
  init(_ c: ReCaCheckpoint, _ prefix: String) throws {
    query = try c.linear(prefix + ".to_q"); key = try c.linear(prefix + ".to_k")
    value = try c.linear(prefix + ".to_v"); output = try c.linear(prefix + ".to_out.0")
    queryNorm = try c.tensor(prefix + ".norm_q.weight"); keyNorm = try c.tensor(prefix + ".norm_k.weight")
  }
  func queries(_ x: MLXArray) -> MLXArray {
    ReCaMath.rms(query(x), weight: queryNorm, epsilon: 1e-6).reshaped([-1, 24, 128])
  }
  func kv(_ x: MLXArray) -> ReCaKV {
    .init(key: ReCaMath.rms(key(x), weight: keyNorm, epsilon: 1e-6).reshaped([-1, 24, 128]),
      value: value(x).reshaped([-1, 24, 128]))
  }
}

struct ReCaWanBlock {
  let attention: ReCaWanAttention, crossAttention: ReCaWanAttention
  let modulation: MLXArray, crossNorm: MLXArray, crossBias: MLXArray
  let expand: ReCaLinear, contract: ReCaLinear
  let cross: ReCaKV
  init(_ c: ReCaCheckpoint, layer: Int, context: MLXArray, time: MLXArray) throws {
    let p = "blocks.\(layer)"
    attention = try ReCaWanAttention(c, p + ".attn1"); crossAttention = try ReCaWanAttention(c, p + ".attn2")
    modulation = try c.tensor(p + ".scale_shift_table").reshaped([6, 3072]).asType(.float32) + time.asType(.float32)
    crossNorm = try c.tensor(p + ".norm2.weight").asType(.float32)
    crossBias = try c.tensor(p + ".norm2.bias").asType(.float32)
    expand = try c.linear(p + ".ffn.net.0.proj"); contract = try c.linear(p + ".ffn.net.2")
    cross = crossAttention.kv(context); eval(cross.key, cross.value, modulation)
  }
  func step(_ x: MLXArray, spatial: ReCaRotary, temporalQuery: ReCaRotary,
    temporalKey: ReCaRotary, cache: ReCaKVCache, plan: ReCaAttentionPlan) throws -> (MLXArray, ReCaKV) {
    let m = modulation
    let normalized = ((MLXFast.layerNorm(x.asType(.float32), weight: nil, bias: nil, eps: 1e-6)
      * (1 + m[1])) + m[0]).asType(.bfloat16)
    let kv = attention.kv(normalized)
    let current = ReCaKV(key: spatial.apply(kv.key), value: kv.value)
    let old = cache.ids.compactMap { cache.planes[$0] }
    let key = concatenated(old.map(\.key) + [current.key], axis: 0)
    let value = concatenated(old.map(\.value) + [current.value], axis: 0)
    let attended = try plan.apply(query: temporalQuery.apply(spatial.apply(attention.queries(normalized))),
      key: temporalKey.apply(key), value: value)
    let selfResult = (x.asType(.float32) + attention.output(attended.reshaped([-1, 3072])).asType(.float32) * m[2]).asType(.bfloat16)
    let crossInput = MLXFast.layerNorm(selfResult.asType(.float32), weight: crossNorm, bias: crossBias, eps: 1e-6).asType(.bfloat16)
    let crossOutput = crossAttention.output(ReCaAttentionPlan.attend(crossAttention.queries(crossInput), cross.key, cross.value).reshaped([-1, 3072]))
    let residual = selfResult + crossOutput
    let ffInput = (MLXFast.layerNorm(residual.asType(.float32), weight: nil, bias: nil, eps: 1e-6)
      * (1 + m[4]) + m[3]).asType(.bfloat16)
    let output = (residual.asType(.float32) + contract(geluApproximate(expand(ffInput))).asType(.float32) * m[5]).asType(.bfloat16)
    return (output, current)
  }
}

final class ReCaTransformerWeights {
  let blocks: [ReCaWanBlock]
  let lq: ReCaLQProjection
  private let patchWeight: MLXArray, patchBias: MLXArray
  let recycle: ReCaLinear, head: ReCaLinear
  let headModulation: MLXArray
  init(_ c: ReCaCheckpoint, prompt: MLXArray) throws {
    lq = try ReCaLQProjection(checkpoint: c)
    patchWeight = try c.tensor("patch_embedding.weight").transposed(0, 2, 3, 4, 1)
    patchBias = try c.tensor("patch_embedding.bias")
    recycle = try c.linear("sr_proj"); head = try c.linear("proj_out")
    let angles = MLXArray((0..<128).map { Float(1000) * exp(-log(Float(10_000)) * Float($0) / 128) })
    let timeInput = concatenated([cos(angles), sin(angles)]).asType(.bfloat16)
    let time = try c.linear("condition_embedder.time_embedder.linear_2")(
      silu(c.linear("condition_embedder.time_embedder.linear_1")(timeInput)))
    let modulation = try c.linear("condition_embedder.time_proj")(silu(time)).reshaped([6, 3072])
    headModulation = try (c.tensor("scale_shift_table").reshaped([2, 3072]) + time).asType(.float32)
    let context = try c.linear("condition_embedder.text_embedder.linear_2")(
      geluApproximate(c.linear("condition_embedder.text_embedder.linear_1")(prompt.reshaped([512, 4096]))))
    var loaded: [ReCaWanBlock] = []
    for layer in 0..<30 { try Task.checkCancellation(); loaded.append(try ReCaWanBlock(c, layer: layer, context: context, time: modulation)) }
    blocks = loaded
  }
  func patch(_ x: MLXArray) -> MLXArray {
    (conv3d(x.asType(.bfloat16).expandedDimensions(axis: 0), patchWeight, stride: [1, 2, 2]) + patchBias).reshaped([-1, 3072])
  }
}
