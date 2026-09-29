import Foundation
import MLX
import TensorIO

/// Stage-loaded BigVGAN decoder for H3's two mono batches (left and right).
/// Checkpoint weights stay in their installed safetensors file; each weighted
/// convolution is released before the next one is read.
public enum H3AudioVAEDecoder {
  public static func decode(checkpointURL: URL, latent: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
    try decode(checkpointURL: checkpointURL, latent: latent,
      progress: progress, observe: { _, _ in })
  }

  static func decode(checkpointURL: URL, latent: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in },
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard latent.ndim == 3, latent.shape[0] == 2,
      (1...601).contains(latent.shape[1]), latent.shape[2] == 32,
      latent.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 stereo audio VAE latent geometry.")
    }
    let layout = try H3AudioVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    func read(_ name: String) throws -> MLXArray {
      guard let descriptor = file.tensors[name], descriptor.dtype == "F32",
        descriptor.byteCount <= 64 * 1024 * 1024 else {
        throw H3CheckpointError.invalid("Missing or oversized H3 audio tensor: \(name)")
      }
      let shape = descriptor.shape.map(Int.init)
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: Float.self)
      }
      eval(value)
      return value
    }
    func finish(_ value: MLXArray) throws -> MLXArray {
      eval(value)
      try Task.checkCancellation()
      return value
    }
    func convolution(_ input: MLXArray, _ name: String,
      kernel: Int, dilation: Int = 1, bias: Bool = true) throws -> MLXArray {
      try autoreleasepool {
        let weight = try read(name + ".weight")
        guard weight.shape.count == 3, weight.shape[2] == kernel,
          weight.shape[1] == input.shape[2] else {
          throw H3CheckpointError.invalid("H3 audio convolution shape changed: \(name)")
        }
        let padding = (kernel * dilation - dilation) / 2
        var output = conv1d(input, weight.transposed(0, 2, 1),
          padding: padding, dilation: dilation)
        if bias { output = output + (try read(name + ".bias")) }
        let result = try finish(output)
        Memory.clearCache()
        return result
      }
    }
    func transpose(_ input: MLXArray, stage: Int,
      rate: Int, kernel: Int) throws -> MLXArray {
      try autoreleasepool {
        let base = "decoder.ups.\(stage).0"
        let weight = try read(base + ".weight")
        guard weight.shape == [input.shape[2], input.shape[2] / 2, kernel] else {
          throw H3CheckpointError.invalid("H3 audio upsampler shape changed.")
        }
        let output = convTransposed1d(input,
          weight.transposed(1, 2, 0), stride: rate,
          padding: (kernel - rate) / 2) + (try read(base + ".bias"))
        let result = try finish(output)
        Memory.clearCache()
        return result
      }
    }
    func activation(_ input: MLXArray, _ name: String) throws -> MLXArray {
      try autoreleasepool {
        let batch = input.shape[0]
        let frames = input.shape[1]
        let channels = input.shape[2]
        let upFilter = try read(name + ".upsample.filter")
          .reshaped([1, 12, 1])
        let downFilter = try read(name + ".downsample.lowpass.filter")
          .reshaped([1, 12, 1])
        let alpha = exp(try read(name + ".act.alpha"))
          .reshaped([1, 1, channels])
        let beta = exp(try read(name + ".act.beta"))
          .reshaped([1, 1, channels])
        var signal = input.transposed(0, 2, 1)
          .reshaped([batch * channels, frames, 1])
        signal = padded(signal, widths: [.init(0), .init((5, 5)), .init(0)],
          mode: .edge)
        signal = Float(2) * convTransposed1d(signal, upFilter, stride: 2)
        signal = signal[0..<(batch * channels), 15..<(15 + frames * 2), 0..<1]
          .reshaped([batch, channels, frames * 2])
          .transposed(0, 2, 1)
        let periodic = sin(signal * alpha)
        signal = signal + periodic * periodic / (beta + 1e-9)
        signal = signal.transposed(0, 2, 1)
          .reshaped([batch * channels, frames * 2, 1])
        signal = padded(signal,
          widths: [.init(0), .init((5, 6)), .init(0)], mode: .edge)
        signal = conv1d(signal, downFilter, stride: 2)
        let result = try finish(signal.reshaped([batch, channels, frames])
          .transposed(0, 2, 1))
        Memory.clearCache()
        return result
      }
    }
    func residual(_ input: MLXArray, block: Int,
      kernel: Int) throws -> MLXArray {
      let base = "decoder.resblocks.\(block)"
      var value = input
      for layer in 0..<3 {
        let dilation = [1, 3, 5][layer]
        let first = try convolution(activation(value,
          base + ".activations.\(2 * layer)"),
          base + ".convs1.\(layer)", kernel: kernel,
          dilation: dilation)
        let second = try convolution(activation(first,
          base + ".activations.\(2 * layer + 1)"),
          base + ".convs2.\(layer)", kernel: kernel)
        value = try finish(value + second)
      }
      return value
    }
    var signal = try convolution(latent, "dec_in_proj", kernel: 1)
    try observe("decin", signal)
    signal = try convolution(signal, "decoder.conv_pre", kernel: 7)
    try observe("convpre", signal)
    for stage in layout.upsampleRates.indices {
      let rate = layout.upsampleRates[stage]
      let kernel = [9, 9, 4, 4, 4, 4, 4][stage]
      signal = try transpose(signal, stage: stage, rate: rate, kernel: kernel)
      try observe("up\(stage)", signal)
      var sum: MLXArray?
      for branch in 0..<3 {
        let output = try residual(signal, block: stage * 3 + branch,
          kernel: [3, 7, 11][branch])
        sum = try finish(sum.map { $0 + output } ?? output)
      }
      signal = try finish(sum! / Float(3))
      try observe("res\(stage)", signal)
      progress(stage + 1, layout.upsampleRates.count)
      try Task.checkCancellation()
    }
    signal = try activation(signal, "decoder.activation_post")
    try observe("postact", signal)
    signal = try convolution(signal, "decoder.conv_post", kernel: 7, bias: false)
    try observe("postconv", signal)
    signal = try finish(clip(signal, min: -1, max: 1))
    try observe("wave", signal)
    guard signal.shape == [2, latent.shape[1] * layout.samplesPerLatent, 1] else {
      throw H3CheckpointError.invalid("H3 audio decoder output timing disagrees with latent count.")
    }
    try file.checkUnchanged(at: checkpointURL)
    return signal
  }
}
