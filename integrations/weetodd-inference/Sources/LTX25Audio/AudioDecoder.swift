import Foundation
import TensorIO

public struct AudioWaveform: Sendable {
  /// Channel-major [left samples, right samples].
  public let samples: [Float]
  public let sampleRate: Int
  public let channels: Int
  public var frameCount: Int { samples.count / channels }
  public init(samples:[Float],sampleRate:Int,channels:Int) {
    self.samples=samples;self.sampleRate=sampleRate;self.channels=channels
  }
}

/// Released LTX 2.5 stereo VAE + BigVGAN + BWE, batch one, Float32.
/// Convolutions execute as bounded Metal Performance Shaders matrix products.
/// Small transforms run on the CPU. Checkpoint storage is mapped one tensor at a
/// time; weighted stages release all temporary buffers before the next stage.
public final class AudioDecoder {
  private let file: SafeTensorFile
  private let matrix: AudioMatrixEngine
  private var active = false
  public let maximumLatentFrames: Int
  public let maximumResidentBytes: UInt64
  public init(checkpoint: URL, maximumLatentFrames: Int = 501,
    maximumResidentBytes: UInt64 = 2 * 1024 * 1024 * 1024) throws {
    guard (1...1501).contains(maximumLatentFrames) else { throw AudioError.invalid("Audio latent frame limit must be in 1...1501") }
    guard maximumResidentBytes >= 256 * 1024 * 1024 && maximumResidentBytes <= 8 * 1024 * 1024 * 1024 else {
      throw AudioError.invalid("Audio resident memory budget must be between 256 MiB and 8 GiB")
    }
    self.maximumLatentFrames = maximumLatentFrames
    self.maximumResidentBytes = maximumResidentBytes
    file = try SafeTensorFile(url: checkpoint)
    try Self.validate(file)
    matrix = try AudioMatrixEngine()
  }
  public static func sampleCount(latentFrames: Int) throws -> Int {
    guard (1...1501).contains(latentFrames) else { throw AudioError.invalid("Audio latent frame count must be in 1...1501") }
    return (4 * latentFrames - 3) * 480
  }
  /// Header-only validation shared by independent execution backends.
  public static func validateCheckpoint(_ checkpoint:URL) throws {
    try validate(SafeTensorFile(url:checkpoint))
  }
  /// Conservative live host/shared-buffer bound: eight copies of the largest
  /// activation plus 256 MiB for the largest source/reordered/GPU weight, im2col
  /// tile and driver allowance. This is admission accounting, not an OS RSS cap.
  public static func estimatedPeakBytes(latentFrames: Int) throws -> UInt64 {
    _ = try sampleCount(latentFrames: latentFrames)
    let largest = UInt64(4 * latentFrames) * 64 * 256
    return largest * 4 * 8 + 256 * 1024 * 1024
  }
  private static func validate(_ file: SafeTensorFile) throws {
    guard file.metadata["model_version"] == "2.5.0",
      let configString = file.metadata["config"], let data = configString.data(using: .utf8),
      let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let audio = config["audio_vae"] as? [String: Any],
      let model = audio["model"] as? [String: Any],
      let params = model["params"] as? [String: Any],
      let ddconfig = params["ddconfig"] as? [String: Any],
      ddconfig["ch"] as? Int == 128, ddconfig["z_channels"] as? Int == 8,
      ddconfig["ch_mult"] as? [Int] == [1, 2, 4],
      ddconfig["num_res_blocks"] as? Int == 2,
      ddconfig["norm_type"] as? String == "pixel",
      ddconfig["causality_axis"] as? String == "height",
      ddconfig["mid_block_add_attention"] as? Bool == false,
      ddconfig["attn_resolutions"] as? [Int] == [],
      let vocoder = config["vocoder"] as? [String: Any],
      let base = vocoder["vocoder"] as? [String: Any],
      let bwe = vocoder["bwe"] as? [String: Any],
      base["upsample_rates"] as? [Int] == [5, 2, 2, 2, 2, 2],
      base["use_tanh_at_final"] as? Bool == false,
      base["activation"] as? String == "snakebeta", base["stereo"] as? Bool == true,
      bwe["activation"] as? String == "snakebeta", bwe["stereo"] as? Bool == true,
      bwe["n_fft"] as? Int == 512, bwe["num_mels"] as? Int == 64,
      bwe["upsample_rates"] as? [Int] == [6, 5, 2, 2, 2],
      bwe["output_sampling_rate"] as? Int == 48000,
      bwe["input_sampling_rate"] as? Int == 16000,
      bwe["hop_length"] as? Int == 80 else { throw AudioError.invalid("Unsupported released audio checkpoint configuration") }
    for (name, shape) in expectedShapes() {
      guard let descriptor = file.tensors[name], descriptor.shape == shape.map(UInt64.init),
        ["F32", "BF16", "F16"].contains(descriptor.dtype) else { throw AudioError.invalid("Audio checkpoint tensor mismatch: \(name)") }
    }
  }
  public static func expectedShapes() -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    func conv(_ p: String, _ input: Int, _ output: Int, _ kernel: [Int], bias: Bool = true, transpose: Bool = false) {
      shapes[p + ".weight"] = (transpose ? [input, output] : [output, input]) + kernel
      if bias { shapes[p + ".bias"] = [output] }
    }
    let vae = "audio_vae.decoder."
    conv(vae + "conv_in.conv", 8, 512, [3, 3])
    func block(_ p: String, _ input: Int, _ output: Int) {
      conv(p + ".conv1.conv", input, output, [3, 3]); conv(p + ".conv2.conv", output, output, [3, 3])
      if input != output { conv(p + ".nin_shortcut.conv", input, output, [1, 1]) }
    }
    block(vae + "mid.block_1", 512, 512); block(vae + "mid.block_2", 512, 512)
    var channels = 512
    for stage in [2, 1, 0] {
      let output = [128, 256, 512][stage]
      for b in 0..<3 { block(vae + "up.\(stage).block.\(b)", channels, output); channels = output }
      if stage > 0 { conv(vae + "up.\(stage).upsample.conv.conv", channels, channels, [3, 3]) }
    }
    conv(vae + "conv_out.conv", 128, 2, [3, 3])
    shapes["audio_vae.per_channel_statistics.mean-of-means"] = [128]
    shapes["audio_vae.per_channel_statistics.std-of-means"] = [128]
    func activation(_ p: String, _ channels: Int) {
      shapes[p + ".act.alpha"] = [channels]; shapes[p + ".act.beta"] = [channels]
      shapes[p + ".upsample.filter"] = [1, 1, 12]; shapes[p + ".downsample.lowpass.filter"] = [1, 1, 12]
    }
    for (prefix, initial, kernels) in [("vocoder.vocoder", 1536, [11, 4, 4, 4, 4, 4]), ("vocoder.bwe_generator", 512, [12, 11, 4, 4, 4])] {
      conv(prefix + ".conv_pre", 128, initial, [7]); var c = initial
      for stage in kernels.indices {
        conv(prefix + ".ups.\(stage)", c, c / 2, [kernels[stage]], transpose: true); c /= 2
        for (j, kernel) in [3, 7, 11].enumerated() { for layer in 0..<3 { for half in [1, 2] {
          let p = prefix + ".resblocks.\(stage * 3 + j)"
          conv(p + ".convs\(half).\(layer)", c, c, [kernel]); activation(p + ".acts\(half).\(layer)", c)
        } } }
      }
      activation(prefix + ".act_post", c); conv(prefix + ".conv_post", c, 2, [7], bias: false)
    }
    shapes["vocoder.mel_stft.mel_basis"] = [64, 257]
    shapes["vocoder.mel_stft.stft_fn.forward_basis"] = [514, 1, 512]
    return shapes
  }
  private func read(_ key: String) throws -> [Float] {
    let values = try file.readFloat32(named: key, maximumBytes: 96 * 1024 * 1024)
    guard values.allSatisfy(\.isFinite) else { throw AudioError.invalid("Nonfinite audio weight: \(key)") }
    return values
  }
  private func convolution(_ x: AudioTensor, _ prefix: String, causal: Bool = false, dilation: Int = 1) throws -> AudioTensor {
    let shape = file.tensors[prefix + ".weight"]!.shape.map(Int.init)
    let kh = shape[2], kw = shape.count == 4 ? shape[3] : 1
    let weights = try read(prefix + ".weight")
    let bias = file.tensors[prefix + ".bias"] == nil ? [] : try read(prefix + ".bias")
    return try AudioMath.conv(x, weight: weights, bias: bias, outputChannels: shape[0], kernelHeight: kh, kernelWidth: kw,
      padTop: causal ? kh - 1 : (kh - 1) * dilation / 2, padLeft: kw / 2, dilation: dilation, matrix: matrix)
  }
  private func residual(_ x: AudioTensor, _ p: String) throws -> AudioTensor {
    let a = try convolution(AudioMath.normalizedSiLU(x), p + ".conv1.conv", causal: true)
    let b = try convolution(AudioMath.normalizedSiLU(a), p + ".conv2.conv", causal: true)
    let skip = x.channels == b.channels ? x : try convolution(x, p + ".nin_shortcut.conv")
    return AudioMath.add(skip, b)
  }
  /// Latent layout [8, latentFrames, 16], matching the shared renderer contract.
  /// maximumSamples trims to a caller's exact audiovisual duration; never pads or
  /// silently invents samples when the latent is too short.
  public func decode(latent: [Float], latentFrames: Int, maximumSamples: Int? = nil,
    progress: ((String) throws -> Void)? = nil) throws -> AudioWaveform {
    guard !active else { throw AudioError.invalid("Audio decoder is already evaluating") }
    active = true
    defer { active = false }
    let fullSamples = try Self.sampleCount(latentFrames: latentFrames)
    guard latentFrames <= maximumLatentFrames, latent.count == latentFrames * 128,
      latent.allSatisfy(\.isFinite), maximumSamples == nil || (maximumSamples! > 0 && maximumSamples! <= fullSamples) else {
      throw AudioError.invalid("Invalid audio latent, frame budget, or requested duration")
    }
    guard try Self.estimatedPeakBytes(latentFrames: latentFrames) <= maximumResidentBytes else {
      throw AudioError.invalid("Audio decode exceeds its admitted resident memory budget")
    }
    try Task.checkCancellation()
    let mel = try decodeMel(latent: latent, frames: latentFrames)
    try progress?("audio_vae")
    var wave = try vocode(mel, prefix: "vocoder.vocoder", rates: [5, 2, 2, 2, 2, 2], clamp: true)
    try progress?("vocoder")
    let outputLength = wave.height * 3
    if wave.height % 80 != 0 {
      let count = 80 - wave.height % 80
      wave = AudioTensor(wave.values + [Float](repeating: 0, count: count * 2), height: wave.height + count, width: 1, channels: 2)
    }
    let bweMel = try computeMel(wave)
    let residual = try vocode(bweMel, prefix: "vocoder.bwe_generator", rates: [6, 5, 2, 2, 2], clamp: false)
    let skip = AudioMath.resample48k(wave)
    guard residual.height == skip.height, outputLength == fullSamples else { throw AudioError.invalid("Audio BWE sample alignment mismatch") }
    let count = maximumSamples ?? outputLength
    var planar = [Float](repeating: 0, count: count * 2)
    for t in 0..<count { for c in 0..<2 {
      let value = residual.values[t * 2 + c] + skip.values[t * 2 + c]
      guard value.isFinite else { throw AudioError.invalid("Nonfinite audio bandwidth extension result") }
      planar[c * count + t] = min(1, max(-1, value))
    } }
    guard planar.allSatisfy(\.isFinite) else { throw AudioError.invalid("Audio decoder produced nonfinite samples") }
    try progress?("bandwidth_extension")
    try Task.checkCancellation()
    return AudioWaveform(samples: planar, sampleRate: 48000, channels: 2)
  }
  private func decodeMel(latent: [Float], frames: Int) throws -> AudioTensor {
    let mean = try read("audio_vae.per_channel_statistics.mean-of-means"), std = try read("audio_vae.per_channel_statistics.std-of-means")
    var values = [Float](repeating: 0, count: latent.count)
    for t in 0..<frames { for f in 0..<16 { for c in 0..<8 {
      values[(t * 16 + f) * 8 + c] = latent[(c * frames + t) * 16 + f] * std[c * 16 + f] + mean[c * 16 + f]
    } } }
    let p = "audio_vae.decoder."
    var x = try convolution(AudioTensor(values, height: frames, width: 16, channels: 8), p + "conv_in.conv", causal: true)
    x = try residual(x, p + "mid.block_1"); x = try residual(x, p + "mid.block_2")
    for stage in [2, 1, 0] {
      for b in 0..<3 { x = try residual(x, p + "up.\(stage).block.\(b)") }
      if stage > 0 {
        x = try convolution(AudioMath.nearest2x(x), p + "up.\(stage).upsample.conv.conv", causal: true)
        x = AudioTensor(Array(x.values.dropFirst(x.width * x.channels)), height: x.height - 1, width: x.width, channels: x.channels)
      }
    }
    x = try convolution(AudioMath.normalizedSiLU(x), p + "conv_out.conv", causal: true)
    var mel = [Float](repeating: 0, count: x.values.count)
    for t in 0..<x.height { for c in 0..<2 { for f in 0..<64 { mel[t * 128 + c * 64 + f] = x.values[(t * 64 + f) * 2 + c] } } }
    return AudioTensor(mel, height: x.height, width: 1, channels: 128)
  }
  private func activation(_ x: AudioTensor, _ p: String) throws -> AudioTensor {
    let up = try read(p + ".upsample.filter"), down = try read(p + ".downsample.lowpass.filter")
    let alpha = try read(p + ".act.alpha").map { exp($0) }, beta = try read(p + ".act.beta").map { exp($0) }
    var y = AudioMath.upsampleFilter(x, filter: up, ratio: 2, inputPad: 5, cropLeft: 15)
    for t in 0..<y.height { for c in 0..<y.channels {
      let i = t * y.channels + c, s = sin(y.values[i] * alpha[c]); y.values[i] += s * s / (beta[c] + 1e-9)
    } }
    return AudioMath.downsampleFilter(y, filter: down)
  }
  private func vocode(_ input: AudioTensor, prefix: String, rates: [Int], clamp: Bool) throws -> AudioTensor {
    var x = try convolution(input, prefix + ".conv_pre")
    for (stage, rate) in rates.enumerated() {
      let p = prefix + ".ups.\(stage)", shape = file.tensors[p + ".weight"]!.shape.map(Int.init)
      x = try AudioMath.transposeConv(x, weight: read(p + ".weight"), bias: read(p + ".bias"), outputChannels: shape[1],
        kernel: shape[2], stride: rate, padding: (shape[2] - rate) / 2, matrix: matrix)
      var sum: AudioTensor?
      for j in 0..<3 {
        let block = prefix + ".resblocks.\(stage * 3 + j)"
        var branch = x
        for (layer, dilation) in [1, 3, 5].enumerated() {
          let a = try convolution(activation(branch, block + ".acts1.\(layer)"), block + ".convs1.\(layer)", dilation: dilation)
          let b = try convolution(activation(a, block + ".acts2.\(layer)"), block + ".convs2.\(layer)")
          branch = AudioMath.add(branch, b)
        }
        sum = sum.map { AudioMath.add($0, branch) } ?? branch
      }
      x = sum!; for i in x.values.indices { x.values[i] /= 3 }
    }
    x = try convolution(activation(x, prefix + ".act_post"), prefix + ".conv_post")
    if clamp { for i in x.values.indices { x.values[i] = min(1, max(-1, x.values[i])) } }
    return x
  }
  private func computeMel(_ wave: AudioTensor) throws -> AudioTensor {
    let basis = try read("vocoder.mel_stft.stft_fn.forward_basis"), mel = try read("vocoder.mel_stft.mel_basis")
    let frames = wave.height / 80
    var result = [Float](repeating: 0, count: frames * 128)
    for channel in 0..<2 {
      let mono = AudioTensor((0..<wave.height).map { wave.values[$0 * 2 + channel] }, height: wave.height, width: 1, channels: 1)
      let stft = try AudioMath.conv(mono, weight: basis, bias: [], outputChannels: 514, kernelHeight: 512,
        padTop: 432, stride: 80, outputHeight: frames, matrix: matrix)
      var magnitudes = [Float](repeating: 0, count: frames * 257)
      for t in 0..<frames { for f in 0..<257 {
        let real = stft.values[t * 514 + f], imaginary = stft.values[t * 514 + 257 + f]
        magnitudes[t * 257 + f] = sqrt(real * real + imaginary * imaginary)
      } }
      let melValues = try AudioMath.conv(AudioTensor(magnitudes, height: frames, width: 1, channels: 257), weight: mel, bias: [],
        outputChannels: 64, kernelHeight: 1, padTop: 0, matrix: matrix)
      for t in 0..<frames { for f in 0..<64 { result[t * 128 + channel * 64 + f] = log(max(1e-5, melValues.values[t * 64 + f])) } }
    }
    return AudioTensor(result, height: frames, width: 1, channels: 128)
  }
}
