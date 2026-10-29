import Drama3Contracts
import Foundation
import MLX
import MLXAudioCodecs
import MLXFast
import MLXNN

/// Process-local causal decoder state. Convolution tails and attention windows are bounded.
final class Drama3CodecStream {
  private let model: FishS1DAC
  private var tails: [ObjectIdentifier: MLXArray] = [:]
  private var keys: [ObjectIdentifier: MLXArray] = [:], values: [ObjectIdentifier: MLXArray] = [:]
  private var positions: [ObjectIdentifier: Int] = [:]
  private var normalizedWeights: [ObjectIdentifier: MLXArray] = [:]
  var retainedElements: Int {
    (Array(tails.values) + Array(keys.values) + Array(values.values)).reduce(0) { $0 + $1.size }
  }
  init(model: FishS1DAC) { self.model = model }
  func decode(_ codes: MLXArray) throws -> MLXArray {
    try Drama3AudioCodec.validate(codes)
    let q = model.quantizer
    let input = codes.expandedDimensions(axis: 0)
    let semantic = q.semanticQuantizer.fromCodes(
      clip(input[0..., 0..<1, 0...], min: 0, max: q.semanticQuantizer.codebookSize - 1)
    ).0
    let residual = q.quantizer.fromCodes(
      clip(input[0..., 1..., 0...], min: 0, max: q.quantizer.codebookSize - 1)
    ).0
    var latent = try run(q.postModule, semantic + residual)
    for stage in q.upsample { latent = try run(stage.block, run(stage.conv, latent)) }
    var audio = latent
    for module in model.decoder.model { audio = try run(module, audio) }
    let result = audio.reshaped(-1)
    eval(Array(tails.values) + Array(keys.values) + Array(values.values) + [result])
    guard result.size == codes.dim(1) * model.frameLength,
      MLX.all(MLX.isFinite(result)).item(Bool.self)
    else { throw Drama3Error.invalid("Invalid streamed codec output.") }
    return result
  }
  private func sequence(_ modules: [Module], _ input: MLXArray) throws -> MLXArray {
    var x = input
    for module in modules { x = try run(module, x) }
    return x
  }
  private func withTail(_ module: Module, _ x: MLXArray, count: Int) -> (MLXArray, Int) {
    let id = ObjectIdentifier(module)
    let old = tails[id]
    let length = old?.dim(2) ?? 0
    let joined = old.map { concatenated([$0, x], axis: 2) } ?? x
    if count > 0 { tails[id] = joined[0..., 0..., max(0, joined.dim(2) - count)...] }
    return (joined, length)
  }
  private func weight(_ module: Module, _ gain: MLXArray, _ vector: MLXArray) -> MLXArray {
    let id = ObjectIdentifier(module)
    if let value = normalizedWeights[id] { return value }
    let axes = Array(1..<vector.ndim)
    let value = gain * vector / (sqrt((vector * vector).sum(axes: axes, keepDims: true)) + 1e-12)
    eval(value)
    normalizedWeights[id] = value
    return value
  }
  private func convolution(
    _ module: Module, _ conv: FishS1CausalConvNet, _ x: MLXArray, _ prepared: MLXArray? = nil,
    _ bias: MLXArray? = nil
  ) throws -> MLXArray {
    guard conv.stride == 1 else {
      throw Drama3Error.invalid("Unexpected decoder convolution stride.")
    }
    let (joined, old) = withTail(module, x, count: conv.padding)
    let padded =
      old < conv.padding
      ? MLX.padded(
        joined, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((conv.padding - old, 0))]) : joined
    let w = prepared ?? conv.conv.weight
    var output = MLX.conv1d(
      padded.transposed(0, 2, 1), w.transposed(0, 2, 1), dilation: conv.conv.dilation,
      groups: conv.conv.groups)
    if let b = bias ?? conv.conv.bias { output = output + b }
    return output.transposed(0, 2, 1)
  }
  private func transpose(
    _ module: Module, _ conv: FishS1CausalTransConvNet, _ x: MLXArray, _ prepared: MLXArray? = nil,
    _ bias: MLXArray? = nil
  ) -> MLXArray {
    let history = max(1, (conv.kernelSize + conv.stride - 1) / conv.stride)
    let (joined, old) = withTail(module, x, count: history)
    let w = prepared ?? conv.conv.weight
    var output = MLX.convTransposed1d(
      joined.transposed(0, 2, 1), w.transposed(1, 2, 0), stride: conv.stride, padding: 0,
      dilation: conv.conv.dilation, groups: conv.conv.groups)
    if let b = bias ?? conv.conv.bias { output = output + b }
    let start = old * conv.stride
    let end = start + x.dim(2) * conv.stride
    return output[0..., start..<end, 0...].transposed(0, 2, 1)
  }
  private func run(_ module: Module, _ x: MLXArray) throws -> MLXArray {
    switch module {
    case let m as FishS1CausalWNConv1d:
      return try convolution(m, m.conv, x, weight(m, m.weight_g, m.weight_v), m.bias)
    case let m as FishS1CausalConvNet: return try convolution(m, m, x)
    case let m as FishS1CausalWNConvTranspose1d:
      return transpose(m, m.conv, x, weight(m, m.weight_g, m.weight_v), m.bias)
    case let m as FishS1CausalTransConvNet: return transpose(m, m, x)
    case let m as FishS1ResidualUnit: return x + (try sequence(m.block, x))
    case let m as FishS1DecoderBlock: return try sequence(m.block, x)
    case let m as FishS1ConvNeXtBlock:
      var y = try run(m.dwconv, x).transposed(0, 2, 1)
      y = m.pwconv2(m.act(m.pwconv1(m.norm(y))))
      if let gamma = m.gamma { y = y * gamma }
      return x + y.transposed(0, 2, 1)
    case let m as FishS1WindowLimitedTransformer: return try transformer(m, x)
    case let unary as UnaryLayer: return unary(x)
    default: throw Drama3Error.invalid("Unsupported native streaming codec layer.")
    }
  }
  private func transformer(_ module: FishS1WindowLimitedTransformer, _ input: MLXArray) throws
    -> MLXArray
  {
    guard module.causal else { throw Drama3Error.invalid("Streaming requires a causal codec.") }
    var x = module.channelsFirst ? input.transposed(0, 2, 1) : input
    x = try run(module.lookAheadConv, run(module.inputProj, x))
    let length = x.dim(1)
    let moduleID = ObjectIdentifier(module)
    let offset = positions[moduleID] ?? 0
    let window = module.windowSize ?? module.config.blockSize
    for layer in module.layers {
      let a = layer.attention
      let id = ObjectIdentifier(a)
      let h = layer.attentionNorm(x)
      let projected = a.wqkv(h)
      let width = a.nLocalHeads * a.headDim
      let parts = MLX.split(projected, indices: [width, 2 * width], axis: -1)
      func heads(_ value: MLXArray, _ count: Int) -> MLXArray {
        value.reshaped(1, length, count, a.headDim).transposed(0, 2, 1, 3)
      }
      var q = heads(parts[0], a.nHead)
      var k = heads(parts[1], a.nLocalHeads)
      let v = heads(parts[2], a.nLocalHeads)
      if a.posEmbedType == "rope" {
        q = MLXFast.RoPE(
          q.asType(.float32), dimensions: a.headDim, traditional: true,
          base: Float(module.config.ropeBase), scale: 1, offset: offset
        ).asType(h.dtype)
        k = MLXFast.RoPE(
          k.asType(.float32), dimensions: a.headDim, traditional: true,
          base: Float(module.config.ropeBase), scale: 1, offset: offset
        ).asType(h.dtype)
      }
      let old = keys[id]?.dim(2) ?? 0
      let allK = keys[id].map { concatenated([$0, k], axis: 2) } ?? k
      let allV = values[id].map { concatenated([$0, v], axis: 2) } ?? v
      let row = MLXArray(offset..<(offset + length))[.newAxis, .newAxis, 0..., .newAxis]
      let col = MLXArray((offset - old)..<(offset + length))[.newAxis, .newAxis, .newAxis, 0...]
      let mask = MLX.where(
        (col .<= row) .&& (col .> row - window), MLXArray(0, dtype: h.dtype),
        MLXArray(-1e9, dtype: h.dtype))
      let attended = MLXFast.scaledDotProductAttention(
        queries: q, keys: allK, values: allV, scale: 1 / sqrt(Float(a.headDim)), mask: mask
      ).transposed(0, 2, 1, 3).reshaped(1, length, a.nHead * a.headDim)
      x = x + layer.attentionLayerScale(a.wo(attended))
      x = x + layer.ffnLayerScale(layer.feedForward(layer.ffnNorm(x)))
      let start = max(0, allK.dim(2) - window + 1)
      keys[id] = allK[0..., 0..., start..., 0...]
      values[id] = allV[0..., 0..., start..., 0...]
    }
    positions[moduleID] = offset + length
    x = try run(module.outputProj, module.norm(x))
    return module.channelsFirst ? x.transposed(0, 2, 1) : x
  }
}
