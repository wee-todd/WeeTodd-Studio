import Foundation
import MLX
import MLXNN
import TensorIO

/// Folded-weight DAC encoder and causal attention projection for reference sound.
/// Stereo channels remain separate batch items; no decoder weights are resident.
public enum H3AudioVAEEncoder {
  /// Header-only admission for every weighted encoder stage. Call before Qwen.
  public static func inspect(checkpointURL: URL) throws {
    _ = try H3AudioVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    try validateHeader(file)
    try file.checkUnchanged(at: checkpointURL)
  }

  private static func validateHeader(_ file: SafeTensorFile) throws {
    func require(_ name: String, _ shape: [Int]) throws {
      guard let tensor = file.tensors[name], tensor.dtype == "F32",
        tensor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 audio encoder tensor: \(name)")
      }
    }
    func conv(_ name: String, out: Int, input: Int, kernel: Int) throws {
      try require(name + ".weight", [out, input, kernel])
      try require(name + ".bias", [out])
    }
    func norm(_ name: String, channels: Int) throws {
      try require(name + ".weight", [channels])
      try require(name + ".bias", [channels])
    }
    func linear(_ name: String, out: Int, input: Int) throws {
      try require(name + ".weight", [out, input])
      try require(name + ".bias", [out])
    }
    try conv("encoder.block.0", out: 64, input: 1, kernel: 7)
    for (stage, rate) in [2, 4, 4, 5, 5].enumerated() {
      let channels = 64 << stage
      let base = "encoder.block.\(stage + 1).block"
      for unit in 0..<3 {
        let prefix = "\(base).\(unit).block"
        try require("\(prefix).0.alpha", [1, channels, 1])
        try conv("\(prefix).1", out: channels,
          input: channels, kernel: 7)
        try require("\(prefix).2.alpha", [1, channels, 1])
        try conv("\(prefix).3", out: channels,
          input: channels, kernel: 1)
      }
      try require("\(base).3.alpha", [1, channels, 1])
      try conv("\(base).4", out: channels * 2,
        input: channels, kernel: 2 * rate)
    }
    try require("encoder.block.6.alpha", [1, 2048, 1])
    try conv("encoder.block.7", out: 2048, input: 2048, kernel: 3)
    for name in ["norm1", "norm3"] {
      try norm("pre_block.\(name)", channels: 2048)
    }
    try norm("pre_block.norm2", channels: 32)
    try norm("pre_block.mlp.norm", channels: 32)
    try require("pre_block.attn.qkv.weight", [6144, 2048])
    for name in ["q_bias", "zero_k_bias", "v_bias"] {
      try require("pre_block.attn.\(name)", [2048])
    }
    try linear("pre_block.attn.proj", out: 32, input: 32)
    try linear("pre_block.proj", out: 32, input: 2048)
    for name in ["w0", "w1"] {
      try linear("pre_block.mlp.\(name)", out: 64, input: 32)
    }
    try linear("pre_block.mlp.w2", out: 32, input: 64)
    try conv("mean_proj", out: 32, input: 32, kernel: 1)
  }

  public static func encode(checkpointURL: URL, waveform: MLXArray) throws -> MLXArray {
    try encode(checkpointURL: checkpointURL, waveform: waveform,
      observe: { _, _ in })
  }

  static func encode(checkpointURL: URL, waveform: MLXArray,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard waveform.ndim == 3, waveform.shape[0] == 2,
      waveform.shape[2] == 1, waveform.dtype == .float32,
      (800...480_000).contains(waveform.shape[1]) else {
      throw H3CheckpointError.invalid("H3 reference audio requires bounded 32 kHz stereo samples.")
    }
    _ = try H3AudioVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    try validateHeader(file)
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    func read(_ name: String) throws -> MLXArray {
      guard let descriptor = file.tensors[name], descriptor.dtype == "F32",
        descriptor.byteCount <= 128 * 1024 * 1024 else {
        throw H3CheckpointError.invalid("Missing or oversized H3 audio encoder tensor: \(name)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, descriptor.shape.map(Int.init), type: Float.self)
      }
      eval(value)
      return value
    }
    func finish(_ value: MLXArray) throws -> MLXArray {
      eval(value)
      try Task.checkCancellation()
      Memory.clearCache()
      return value
    }
    func convolution(_ value: MLXArray, _ name: String,
      kernel: Int, stride: Int = 1, dilation: Int = 1,
      padding: Int = 0) throws -> MLXArray {
      try autoreleasepool {
        let weight = try read(name + ".weight")
        guard weight.shape.count == 3, weight.shape[1] == value.shape[2],
          weight.shape[2] == kernel else {
          throw H3CheckpointError.invalid("H3 audio encoder convolution shape changed: \(name)")
        }
        let result = conv1d(value, weight.transposed(0, 2, 1),
          stride: stride, padding: padding, dilation: dilation)
          + (try read(name + ".bias"))
        return try finish(result)
      }
    }
    func snake(_ value: MLXArray, _ name: String) throws -> MLXArray {
      let alpha = try read(name + ".alpha").transposed(0, 2, 1)
      guard alpha.shape == [1, 1, value.shape[2]] else {
        throw H3CheckpointError.invalid("H3 audio encoder Snake shape changed.")
      }
      let periodic = sin(value * alpha)
      return try finish(value + periodic * periodic / (alpha + 1e-9))
    }
    func layerNorm(_ value: MLXArray, _ name: String) throws -> MLXArray {
      let weight = try read(name + ".weight")
      let bias = try read(name + ".bias")
      guard weight.shape == [value.shape[2]], bias.shape == weight.shape else {
        throw H3CheckpointError.invalid("H3 audio encoder LayerNorm shape changed.")
      }
      return try finish(MLXFast.layerNorm(value, weight: weight,
        bias: bias, eps: 1e-5))
    }
    func linear(_ value: MLXArray, _ name: String) throws -> MLXArray {
      let weight = try read(name + ".weight")
      let bias = try read(name + ".bias")
      guard weight.ndim == 2, weight.shape[1] == value.shape[2],
        bias.shape == [weight.shape[0]] else {
        throw H3CheckpointError.invalid("H3 audio encoder linear shape changed: \(name)")
      }
      return try finish(matmul(value, weight.T) + bias)
    }
    let samples = waveform.shape[1]
    let paddedSamples = ((samples + 799) / 800) * 800
    var signal = paddedSamples == samples ? waveform : padded(waveform,
      widths: [.init(0), .init((0, paddedSamples - samples)), .init(0)])
    signal = try convolution(signal, "encoder.block.0", kernel: 7, padding: 3)
    for stage in 0..<5 {
      try Task.checkCancellation()
      let base = "encoder.block.\(stage + 1)"
      for (unit, dilation) in [1, 3, 9].enumerated() {
        let prefix = "\(base).block.\(unit).block"
        let first = try snake(signal, "\(prefix).0")
        let second = try convolution(first, "\(prefix).1", kernel: 7,
          dilation: dilation, padding: 3 * dilation)
        let third = try snake(second, "\(prefix).2")
        let fourth = try convolution(third, "\(prefix).3", kernel: 1)
        guard signal.shape == fourth.shape else {
          throw H3CheckpointError.invalid("H3 audio residual length changed.")
        }
        signal = try finish(signal + fourth)
      }
      let rate = [2, 4, 4, 5, 5][stage]
      signal = try snake(signal, "\(base).block.3")
      signal = try convolution(signal, "\(base).block.4", kernel: 2 * rate,
        stride: rate, padding: (rate + 1) / 2)
    }
    signal = try snake(signal, "encoder.block.6")
    signal = try convolution(signal, "encoder.block.7", kernel: 3, padding: 1)
    let latentFrames = paddedSamples / 800
    guard signal.shape == [2, latentFrames, 2048] else {
      throw H3CheckpointError.invalid("H3 audio encoder hop count changed.")
    }
    try observe("encoder", signal)

    let normed = try layerNorm(signal, "pre_block.norm1")
    let qkvWeight = try read("pre_block.attn.qkv.weight")
    guard qkvWeight.shape == [6144, 2048] else {
      throw H3CheckpointError.invalid("H3 audio encoder QKV shape changed.")
    }
    let qBias = try read("pre_block.attn.q_bias")
    let kBias = try read("pre_block.attn.zero_k_bias")
    let vBias = try read("pre_block.attn.v_bias")
    let packed = try finish(matmul(normed, qkvWeight.T)
      + concatenated([qBias, kBias, vBias]).reshaped([1, 1, 6144]))
      .reshaped([2, latentFrames, 3, 8, 256])
    let query = packed[0..<2, 0..<latentFrames, 0, 0..<8, 0..<256]
      .transposed(0, 2, 1, 3)
    let key = packed[0..<2, 0..<latentFrames, 1, 0..<8, 0..<256]
      .transposed(0, 2, 1, 3)
    let value = packed[0..<2, 0..<latentFrames, 2, 0..<8, 0..<256]
      .transposed(0, 2, 1, 3)
    let ids = MLXArray((0..<latentFrames).map(Int32.init))
    let causal = lessEqual(ids.reshaped([1, latentFrames]),
      ids.reshaped([latentFrames, 1]))
    let attended = MLXFast.scaledDotProductAttention(queries: query,
      keys: key, values: value, scale: 1 / Float(256).squareRoot(),
      mask: causal).transposed(0, 2, 1, 3)
    let pooled = attended.mean(axis: 2)
      .reshaped([2, latentFrames, 32, 8]).mean(axis: -1)
    let attention = try linear(pooled, "pre_block.attn.proj")
    try observe("attention", attention)
    let projection = try linear(try layerNorm(signal, "pre_block.norm3"),
      "pre_block.proj")
    try observe("projection", projection)
    var features = try finish(projection + attention)
    let mlpInput = try layerNorm(features, "pre_block.norm2")
    let mlpNorm = try layerNorm(mlpInput, "pre_block.mlp.norm")
    let gate = geluApproximate(try linear(mlpNorm, "pre_block.mlp.w0"))
    let up = try linear(mlpNorm, "pre_block.mlp.w1")
    let mlp = try linear(gate * up, "pre_block.mlp.w2")
    try observe("mlp", mlp)
    features = try finish(features + mlp)
    let output = try convolution(features, "mean_proj", kernel: 1)
    try observe("mean", output)
    guard output.shape == [2, latentFrames, 32] else {
      throw H3CheckpointError.invalid("H3 audio encoder posterior geometry changed.")
    }
    try file.checkUnchanged(at: checkpointURL)
    return output
  }
}
