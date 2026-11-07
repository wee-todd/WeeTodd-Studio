import Foundation
import MLX
import MLXNN

private let wanWidth = 3072
private let wanHeads = 24
private let wanHeadWidth = 128
private let wanEpsilon: Float = 1e-6

private struct WanAttention {
  let query: RelayLinear
  let key: RelayLinear
  let value: RelayLinear
  let output: RelayLinear
  let queryNorm: MLXArray
  let keyNorm: MLXArray

  init(_ weights: RelayWeights, _ path: String, rotation: MLXArray) throws {
    query = try weights.linear(path + ".q", wanWidth, wanWidth, rotation: rotation)
    key = try weights.linear(path + ".k", wanWidth, wanWidth, rotation: rotation)
    value = try weights.linear(path + ".v", wanWidth, wanWidth, rotation: rotation)
    output = try weights.linear(path + ".o", wanWidth, wanWidth, rotation: rotation)
    queryNorm = try weights.tensor(path + ".norm_q.weight", [wanWidth])
    keyNorm = try weights.tensor(path + ".norm_k.weight", [wanWidth])
  }

  func queries(_ x: MLXArray) -> MLXArray {
    MLXFast.rmsNorm(query(x), weight: queryNorm, eps: wanEpsilon)
      .reshaped([-1, wanHeads, wanHeadWidth])
  }

  func keysAndValues(_ x: MLXArray) -> RelayKV {
    RelayKV(key: MLXFast.rmsNorm(key(x), weight: keyNorm, eps: wanEpsilon)
      .reshaped([-1, wanHeads, wanHeadWidth]), value: value(x).reshaped([-1, wanHeads, wanHeadWidth]))
  }
}

private struct WanBlock {
  let selfAttention: WanAttention
  let crossAttention: WanAttention
  let crossNormWeight: MLXArray
  let crossNormBias: MLXArray
  let modulation: MLXArray
  let expand: RelayLinear
  let contract: RelayLinear

  init(_ weights: RelayWeights, index: Int, rotation: MLXArray) throws {
    let path = "blocks.\(index)"
    selfAttention = try WanAttention(weights, path + ".self_attn", rotation: rotation)
    crossAttention = try WanAttention(weights, path + ".cross_attn", rotation: rotation)
    crossNormWeight = try weights.tensor(path + ".norm3.weight", [wanWidth])
    crossNormBias = try weights.tensor(path + ".norm3.bias", [wanWidth])
    modulation = try weights.tensor(path + ".modulation", [1, 6, wanWidth]).squeezed(axis: 0)
    expand = try weights.linear(path + ".ffn.0", wanWidth, 14336, rotation: rotation)
    contract = try weights.linear(path + ".ffn.2", 14336, wanWidth, rotation: rotation)
  }

  func callAsFunction(_ x: MLXArray, time: MLXArray, rope: RelayRotary, cross: RelayKV,
    past: [RelayKV], attention: RelayAttentionPlan) -> (MLXArray, RelayKV)
  {
    let modulation = self.modulation + time
    let selfInput = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: wanEpsilon)
      * (1 + modulation[1]) + modulation[0]
    let kv = selfAttention.keysAndValues(selfInput)
    let current = RelayKV(key: rope.apply(kv.key), value: kv.value)
    let spatial = attention(rope.apply(selfAttention.queries(selfInput)), past + [current])
    let selfResult = x + selfAttention.output(spatial.reshaped([-1, wanWidth])) * modulation[2]
    let crossInput = MLXFast.layerNorm(selfResult, weight: crossNormWeight,
      bias: crossNormBias, eps: wanEpsilon)
    let attended = RelayAttentionPlan.attend(crossAttention.queries(crossInput), cross.key, cross.value)
    let crossResult = selfResult + crossAttention.output(attended.reshaped([-1, wanWidth]))
    let ffInput = MLXFast.layerNorm(crossResult, weight: nil, bias: nil, eps: wanEpsilon)
      * (1 + modulation[4]) + modulation[3]
    let output = crossResult + contract(geluApproximate(expand(ffInput))) * modulation[5]
    return (output, current)
  }
}

final class RelayWan {
  private let blocks: [WanBlock]
  private let cross: [RelayKV]
  private let patch: RelayConv
  private let lqInput: RelayConv
  private let lqDown: RelayConv
  private let lqNorm1: MLXArray
  private let lqNorm2: MLXArray
  private let lqOutput: RelayLinear
  private let head: RelayLinear
  private let times: [(blocks: MLXArray, head: MLXArray)]
  private let plans: [RelayAttentionPlan]
  private let sigmas: [Float]
  private let height: Int
  private let width: Int
  private let seed: UInt64
  private var index = 0
  private var history: [[[RelayKV]]]

  init(weights url: URL, geometry: RelayGeometry, steps: Int, seed: UInt64) throws {
    let weights = try RelayWeights(url: url, format: "relayvsr-wan-x2-v1")
    let sigmas = try RelaySchedule.sigmas(steps: steps)
    self.sigmas = sigmas
    self.seed = seed
    height = geometry.paddedHeight / 16
    width = geometry.paddedWidth / 16
    let window = RelayWindow(height: 22, width: 40, slidesAtEdge: true)
    plans = (1...4).map { RelayAttentionPlan(height: geometry.paddedHeight / 16,
      width: geometry.paddedWidth / 16, windows: Array(repeating: window, count: $0)) }
    let rotation = RelayLinear.rotation()
    blocks = try (0..<30).map { try WanBlock(weights, index: $0, rotation: rotation) }
    history = Array(repeating: Array(repeating: [], count: blocks.count), count: steps)
    patch = try weights.convolution("patch_embedding", 48, wanWidth, kernel: 2,
      stride: 2, temporalAxis: true)
    lqInput = try weights.convolution("lq_proj_in.conv1", 192, 768, kernel: 1)
    lqDown = try weights.convolution("lq_proj_in.conv2", 768, wanWidth, kernel: 3,
      stride: 2, padding: 1)
    lqNorm1 = try weights.tensor("lq_proj_in.norm1.weight", [768])
    lqNorm2 = try weights.tensor("lq_proj_in.norm2.weight", [wanWidth])
    lqOutput = try weights.linear("lq_proj_in.linear", wanWidth, wanWidth)
    head = try weights.linear("head.head", wanWidth, 192)
    let headModulation = try weights.tensor("head.modulation", [1, 2, wanWidth]).squeezed(axis: 0)
    let timeInput = try weights.linear("time_embedding.0", 256, wanWidth)
    let timeOutput = try weights.linear("time_embedding.2", wanWidth, wanWidth)
    let timeProjection = try weights.linear("time_projection.1", wanWidth, 6 * wanWidth)
    let frequencies = (0..<128).map { pow(Float(10_000), -Float($0) / 128) }
    times = sigmas.dropLast().map { sigma in
      let angles = MLXArray(frequencies) * (sigma * 1000)
      let embedding = concatenated([cos(angles), sin(angles)]).asType(.bfloat16)
      let time = timeOutput(silu(timeInput(embedding)))
      return (timeProjection(silu(time)).reshaped([6, wanWidth]), headModulation + time)
    }
    let textInput = try weights.linear("text_embedding.0", 4096, wanWidth)
    let textOutput = try weights.linear("text_embedding.2", wanWidth, wanWidth)
    let prompt = try weights.tensor("positive_conditioning", [512, 4096])
    let context = textOutput(geluApproximate(textInput(prompt)))
    cross = blocks.map { block in
      let kv = block.crossAttention.keysAndValues(context)
      eval(kv.key, kv.value)
      return kv
    }
  }

  func keyframe(_ lowResolution: MLXArray) throws -> MLXArray {
    let normalized = lowResolution.asType(.bfloat16)
    let input = normalized.reshaped([height * 2, 8, width * 2, 8, 3])
      .transposed(0, 2, 4, 1, 3).reshaped([height * 2, width * 2, 192])
    let lowFeatures = silu(MLXFast.rmsNorm(lqInput(input), weight: lqNorm1, eps: wanEpsilon))
    let lowTokens = lqOutput(silu(MLXFast.rmsNorm(lqDown(lowFeatures), weight: lqNorm2,
      eps: wanEpsilon)).reshaped([-1, wanWidth]))
    let rope = RelayRotary(height: height, width: width, time: index, axes: [44, 42, 42])
    let shape = [height * 2, width * 2, 48]
    var latent = MLXRandom.normal(shape, dtype: .float32, key: MLXRandom.key(seed &+ UInt64(index)))
    for step in times.indices {
      var x = patch(latent.asType(.bfloat16)).reshaped([-1, wanWidth]) + lowTokens
      let time = times[step]
      for blockIndex in blocks.indices {
        try Task.checkCancellation()
        var past = history[step][blockIndex]
        let (next, current) = blocks[blockIndex](x, time: time.blocks, rope: rope,
          cross: cross[blockIndex], past: past, attention: plans[past.count])
        eval(next, current.key, current.value)
        x = next
        if past.count == 3 { past.removeFirst() }
        past.append(current)
        history[step][blockIndex] = past
      }
      let normalized = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: wanEpsilon)
        * (1 + time.head[1]) + time.head[0]
      let flow = head(normalized).reshaped([height, width, 2, 2, 48])
        .transposed(0, 2, 1, 3, 4).reshaped(shape).asType(.float32)
      latent = latent + flow * (sigmas[step + 1] - sigmas[step])
      eval(latent)
    }
    index += 1
    return latent.asType(.bfloat16)
  }
}
