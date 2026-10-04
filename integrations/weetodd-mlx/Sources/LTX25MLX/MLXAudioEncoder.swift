import Foundation
import MLX
import TensorIO
import LTX25Audio
import LTX25Engine

/// Released LTX 2.5 stereo audio VAE encoder. One weighted convolution is
/// resident at a time; the public output is normalized mean-only tokens.
public final class MLXAudioEncoder {
  private let file: SafeTensorFile
  private let gate = NSLock()
  public let maximumMelFrames: Int

  public static func expectedShapes() -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    func conv(_ name: String, _ input: Int, _ output: Int, _ kernel: Int) {
      shapes[name + ".weight"] = [output, input, kernel, kernel]
      shapes[name + ".bias"] = [output]
    }
    let prefix = "audio_vae.encoder."
    conv(prefix + "conv_in.conv", 2, 128, 3)
    var channels = 128
    for (stage, output) in [128, 256, 512].enumerated() {
      for block in 0..<2 {
        let root = prefix + "down.\(stage).block.\(block)."
        conv(root + "conv1.conv", channels, output, 3)
        conv(root + "conv2.conv", output, output, 3)
        if channels != output { conv(root + "nin_shortcut.conv", channels, output, 1) }
        channels = output
      }
      if stage < 2 { conv(prefix + "down.\(stage).downsample.conv", output, output, 3) }
    }
    for block in 1...2 {
      let root = prefix + "mid.block_\(block)."
      conv(root + "conv1.conv", 512, 512, 3)
      conv(root + "conv2.conv", 512, 512, 3)
    }
    conv(prefix + "conv_out.conv", 512, 16, 3)
    shapes["audio_vae.per_channel_statistics.mean-of-means"] = [128]
    shapes["audio_vae.per_channel_statistics.std-of-means"] = [128]
    return shapes
  }
  public static func latentFrames(melFrames: Int, maximumMelFrames: Int = 2011) throws -> Int {
    guard (1...MLXAudioMelPlan.maximumMelFrames).contains(maximumMelFrames), (1...maximumMelFrames).contains(melFrames) else { throw LTXError.invalid("Audio mel frame count exceeds the native limit.") }
    return (melFrames + 3) / 4
  }
  public init(checkpoint: URL, maximumMelFrames: Int = 2011) throws {
    guard (1...MLXAudioMelPlan.maximumMelFrames).contains(maximumMelFrames) else { throw LTXError.invalid("Invalid audio encoder frame limit.") }
    let file = try SafeTensorFile(url: checkpoint)
    guard file.metadata["model_version"] == "2.5.0" else {
      throw LTXError.invalid("Audio encoder requires a released LTX 2.5 checkpoint.")
    }
    for (name, shape) in Self.expectedShapes() {
      guard let descriptor = file.tensors[name], descriptor.shape == shape.map(UInt64.init),
        ["BF16", "F16", "F32"].contains(descriptor.dtype) else {
        throw LTXError.invalid("Audio encoder checkpoint tensor mismatch: \(name)")
      }
    }
    self.file = file; self.maximumMelFrames = maximumMelFrames
  }
  private func read(_ name: String) throws -> MLXArray {
    let value = try MLXWeight.read(file, name).asType(.float32)
    eval(value)
    guard MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite audio encoder weight: \(name)") }
    return value
  }
  private func finish(_ value: MLXArray) throws -> MLXArray {
    eval(value); try Task.checkCancellation()
    guard MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Audio encoder produced nonfinite latent values.") }
    return value
  }
  private func convolution(_ input: MLXArray, _ name: String, downsample: Bool = false) throws -> MLXArray {
    try Task.checkCancellation()
    return try autoreleasepool {
      let weights = try read(name + ".weight"), bias = try read(name + ".bias")
      let kernel = weights.shape[2]
      let padded = MLX.padded(input.expandedDimensions(axis: 0),
        widths: [.init(0), .init((kernel - 1, 0)), .init((kernel / 2, kernel / 2)), .init(0)])
      let result = conv2d(padded, weights.transposed(0, 2, 3, 1), stride: downsample ? 2 : 1)[0] + bias
      return try finish(result)
    }
  }
  private func residual(_ input: MLXArray, _ root: String) throws -> MLXArray {
    let first = try convolution(MLXAudioMath.normalizedSiLU(input), root + ".conv1.conv")
    let second = try convolution(MLXAudioMath.normalizedSiLU(first), root + ".conv2.conv")
    let skip = input.shape.last == second.shape.last ? input : try convolution(input, root + ".nin_shortcut.conv")
    return try finish(skip + second)
  }
  /// Input [1, 2, melFrames, 64], output [latentFrames, 128].
  public func encode(mel: MLXArray, maximumOwnedBufferBytes: Int = 2*1024*1024*1024, progress: (Int, Int) throws -> Void = { _, _ in }) throws -> MLXArray {
    guard gate.try() else { throw LTXError.invalid("Audio encoder is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    guard mel.dtype == .float32, mel.shape.count == 4, mel.shape[0] == 1,
      mel.shape[1] == 2, mel.shape[3] == 64, (1...maximumMelFrames).contains(mel.shape[2]) else {
      throw LTXError.invalid("Audio encoder requires finite stereo Float32 mel frames.")
    }
    let admission=try MLXAudioEncodePlan(melFrames:mel.shape[2],maximumMelFrames:maximumMelFrames,maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    try Task.checkCancellation()
    guard MLX.isFinite(mel).all().item(Bool.self) else { throw LTXError.invalid("Audio encoder requires finite mel frames.") }
    let expected=admission.latentFrames
    let prefix = "audio_vae.encoder."
    var x = try convolution(mel.transposed(0, 2, 3, 1)[0], prefix + "conv_in.conv")
    var completed = 0
    for stage in 0..<3 {
      for block in 0..<2 {
        x = try residual(x, prefix + "down.\(stage).block.\(block)")
        completed += 1; try progress(completed, 9)
      }
      if stage < 2 { x = try convolution(x, prefix + "down.\(stage).downsample.conv", downsample: true) }
    }
    for block in 1...2 {
      x = try residual(x, prefix + "mid.block_\(block)")
      completed += 1; try progress(completed, 9)
    }
    x = try convolution(MLXAudioMath.normalizedSiLU(x), prefix + "conv_out.conv")
    guard x.shape == [expected, 16, 16] else { throw LTXError.invalid("Audio encoder latent geometry differs from the causal contract.") }
    let mean = try read("audio_vae.per_channel_statistics.mean-of-means")
    let std = try read("audio_vae.per_channel_statistics.std-of-means")
    guard std.min().item(Float.self) > 0 else { throw LTXError.invalid("Audio normalization standard deviation must be positive.") }
    let raw = x[0..., 0..., 0..<8].transposed(0, 2, 1).reshaped([expected, 128])
    let tokens = try finish((raw - mean) / (std + 1e-8))
    try progress(9, 9)
    return tokens
  }
}
