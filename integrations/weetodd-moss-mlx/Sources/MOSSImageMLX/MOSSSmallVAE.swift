import Foundation
import MLX
import MLXNN

final class MOSSSmallVAE {
  let weights: any MOSSWeightAccess
  let maximumBandBytes: Int
  init(weights: any MOSSWeightAccess, maximumBandBytes: Int = 64 * 1024 * 1024) {
    self.weights = weights; self.maximumBandBytes = max(1024, maximumBandBytes)
  }
  static func unpack(_ packed: MLXArray, mean: MLXArray, variance: MLXArray) -> MLXArray {
    let h = packed.dim(2), w = packed.dim(3)
    let z = packed.asType(.float32) * sqrt(variance.reshaped(1, 128, 1, 1) + 1e-4) + mean.reshaped(1, 128, 1, 1)
    return z.reshaped(1, 32, 2, 2, h, w).transposed(0, 4, 2, 5, 3, 1).reshaped(1, h * 2, w * 2, 32)
  }
  static func pack(_ image: MLXArray, mean: MLXArray, variance: MLXArray) -> MLXArray {
    let h = image.dim(1) / 2, w = image.dim(2) / 2
    let packed = image.reshaped(1, h, 2, w, 2, 32).transposed(0, 5, 2, 4, 1, 3).reshaped(1, 128, h, w)
    return (packed.asType(.float32) - mean.reshaped(1, 128, 1, 1)) / sqrt(variance.reshaped(1, 128, 1, 1) + 1e-4)
  }
  static func finalize(_ decoded: MLXArray) throws -> MLXArray {
    guard isFinite(decoded).all().item(Bool.self) else { throw MOSSError("The MOSS VAE produced nonfinite pixels.") }
    let image = clip(decoded, min: -1, max: 1); eval(image); return image
  }
  private func rowLimit(width: Int, channels: Int) -> Int { max(1, maximumBandBytes / max(1, width * channels * 4)) }
  private func conv(_ image: MOSSImageBands, _ name: String) throws -> MOSSImageBands {
    try Task.checkCancellation()
    let weight = try weights.tensor(name + ".weight").asType(.float32).transposed(0, 2, 3, 1)
    let bias = try weights.tensor(name + ".bias").asType(.float32)
    let run = { image.convolved(weight: weight, bias: bias, rowLimit: self.rowLimit(width: image.width, channels: max(image.channels, weight.dim(0)))) }
    return weights.profiler?.measure("vae/convolution", run) ?? run()
  }
  private func norm(_ image: MOSSImageBands, _ name: String) throws -> MOSSImageBands {
    let weight = try weights.tensor(name + ".weight").asType(.float32), bias = try weights.tensor(name + ".bias").asType(.float32)
    let run = { image.groupNormalized(weight: weight, bias: bias) }
    return weights.profiler?.measure("vae/global GroupNorm", run) ?? run()
  }
  private func residual(_ input: MOSSImageBands, _ prefix: String) throws -> MOSSImageBands {
    let first = try conv(norm(input, prefix + ".norm1").mapped(silu), prefix + ".conv1")
    let output = try conv(norm(first, prefix + ".norm2").mapped(silu), prefix + ".conv2")
    let shortcut = weights.contains(prefix + ".conv_shortcut.weight") ? try conv(input, prefix + ".conv_shortcut") : input
    return output.adding(shortcut)
  }
  private func attention(_ input: MOSSImageBands, prefix p: String = "decoder.mid_block.attentions.0.") throws -> MOSSImageBands {
    let normalized = try norm(input, p + "group_norm")
    let tokens = input.height * input.width, channels = input.channels
    // Whole-image K/V context; only queries are chunked. Head dimension 384 otherwise materializes N² scores.
    let flat = normalized.materialized().reshaped(1, tokens, channels)
    let keys = try weights.linear(flat, p + "to_k").reshaped(1, 1, tokens, channels)
    let values = try weights.linear(flat, p + "to_v").reshaped(1, 1, tokens, channels)
    eval(keys, values)
    var rows: [MLXArray] = [], offset = 0
    for band in normalized.bands {
      try Task.checkCancellation()
      let count = band.dim(1) * input.width
      let queries = try weights.linear(band.reshaped(1, count, channels), p + "to_q")
      var chunks: [MLXArray] = []
      for start in stride(from: 0, to: count, by: 256) {
        let end = min(count, start + 256)
        let query = queries[0..., start..<end, 0...].reshaped(1, 1, end - start, channels)
        let attended = MLXFast.scaledDotProductAttention(queries: query, keys: keys, values: values,
          scale: Float(1 / sqrt(Double(channels))), mask: nil).reshaped(1, end - start, channels)
        let output = try weights.linear(attended, p + "to_out.0")
        eval(output); chunks.append(output)
      }
      let result = concatenated(chunks, axis: 1).reshaped(band.shape) + input.rows(offset..<(offset + band.dim(1)))
      eval(result); rows.append(result); offset += band.dim(1)
    }
    return .init(bands: rows)
  }
  func encode(image: MLXArray, progress: ((String) -> Void)? = nil) throws -> MLXArray {
    guard image.ndim == 4, image.dim(0) == 1, image.dim(3) == 3,
      image.dim(1) > 0, image.dim(2) > 0, image.dim(1).isMultiple(of: 16), image.dim(2).isMultiple(of: 16),
      isFinite(image).all().item(Bool.self) else { throw MOSSError("Source encoding requires finite RGB pixels on the 16-pixel grid.") }
    var x = MOSSImageBands(image.asType(.float32), rowLimit: rowLimit(width: image.dim(2), channels: 128))
    x = try conv(x, "encoder.conv_in")
    for block in 0..<4 {
      progress?("VAE encode \(block + 1)/4")
      for layer in 0..<2 { x = try residual(x, "encoder.down_blocks.\(block).resnets.\(layer)") }
      if block < 3 {
        try Task.checkCancellation()
        let name = "encoder.down_blocks.\(block).downsamplers.0.conv"
        let weight = try weights.tensor(name + ".weight").asType(.float32).transposed(0, 2, 3, 1)
        let bias = try weights.tensor(name + ".bias").asType(.float32)
        x = x.downsampled(weight: weight, bias: bias, rowLimit: rowLimit(width: x.width / 2, channels: weight.dim(0)))
      }
    }
    x = try residual(x, "encoder.mid_block.resnets.0")
    x = try attention(x, prefix: "encoder.mid_block.attentions.0.")
    x = try residual(x, "encoder.mid_block.resnets.1")
    x = try conv(norm(x, "encoder.conv_norm_out").mapped(silu), "encoder.conv_out")
    x = try conv(x, "quant_conv")
    // Deterministic posterior mean; sampling the posterior would add avoidable source drift.
    let mean = x.materialized()[0..., 0..., 0..., 0..<32]
    let result = Self.pack(mean, mean: try weights.tensor("bn.running_mean"), variance: try weights.tensor("bn.running_var"))
    guard isFinite(result).all().item(Bool.self) else { throw MOSSError("The source encoder produced nonfinite latents.") }
    eval(result); return result
  }
  func decode(packedLatents: MLXArray, progress: ((String) -> Void)? = nil) throws -> MLXArray {
    guard packedLatents.ndim == 4, packedLatents.dim(0) == 1, packedLatents.dim(1) == 128 else { throw MOSSError("Invalid MOSS packed latents.") }
    let unpacked = Self.unpack(packedLatents, mean: try weights.tensor("bn.running_mean"), variance: try weights.tensor("bn.running_var"))
    var x = MOSSImageBands(unpacked, rowLimit: rowLimit(width: unpacked.dim(2), channels: 32))
    x = try conv(x, "post_quant_conv"); x = try conv(x, "decoder.conv_in")
    progress?("VAE middle blocks")
    x = try residual(x, "decoder.mid_block.resnets.0")
    if let profiler = weights.profiler { x = try profiler.measure("vae/full-context attention") { try attention(x) } } else { x = try attention(x) }
    x = try residual(x, "decoder.mid_block.resnets.1")
    for block in 0..<4 {
      progress?("VAE decode \(block + 1)/4")
      for layer in 0..<3 { x = try residual(x, "decoder.up_blocks.\(block).resnets.\(layer)") }
      if block < 3 { x = try conv(x.upsampled(), "decoder.up_blocks.\(block).upsamplers.0.conv") }
    }
    x = try conv(norm(x, "decoder.conv_norm_out").mapped(silu), "decoder.conv_out")
    return try Self.finalize(x.materialized())
  }
}
