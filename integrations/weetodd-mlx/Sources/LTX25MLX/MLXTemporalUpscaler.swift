import Foundation
import MLX
import TensorIO
import LTX25Engine

/// Streams the released temporal x2 latent network one convolution at a time.
/// The first shuffled frame is discarded because the causal VAE's first latent
/// represents a single image; the remaining output has 2*T-1 latent frames.
public final class MLXTemporalUpscaler {
  private let file: SafeTensorFile
  private let statistics: SafeTensorFile
  private let gate = NSLock()
  public private(set) var residentWeightBytes = 0

  public init(checkpoint: URL, statisticsCheckpoint: URL) throws {
    let source = try SafeTensorFile(url: checkpoint)
    try Self.validate(source)
    let stats = try SafeTensorFile(url: statisticsCheckpoint)
    for name in ["mean-of-means", "std-of-means"] {
      guard let entry = stats.tensors["per_channel_statistics." + name],
        entry.shape == [128], ["F32", "F16", "BF16"].contains(entry.dtype) else {
        throw LTXError.invalid("Temporal upscaler needs 128-channel latent statistics.")
      }
    }
    file = source
    statistics = stats
  }

  public static func estimatedActivationBytes(shape: [Int]) throws -> Int {
    guard shape.count == 4, (2...129).contains(shape[0]), (1...128).contains(shape[1]),
      (1...128).contains(shape[2]), shape[3] == 128 else {
      throw LTXError.invalid("Temporal upscaler needs bounded THWC 128-channel latents.")
    }
    let sites = UInt64(shape[0]) * UInt64(shape[1]) * UInt64(shape[2])
    let high = 2 * sites
    // Full-volume GroupNorm and two convolution outputs remain resident. A
    // conservative im2col workspace and one 512x512x3x3x3 layer are included.
    let bytes = high * 512 * 4 * 6 + sites * 128 * 4 + high * 128 * 4
      + 512 * 512 * 27 * 4 + 64 * 1024 * 1024
    guard bytes <= UInt64(Int.max), high * 1024 <= UInt64(UInt32.max) else {
      throw LTXError.invalid("Temporal upscaler exceeds its activation/workspace allowance.")
    }
    return Int(bytes)
  }

  public static func admit(shape: [Int], maximumActivationBytes: Int) throws -> [Int] {
    guard maximumActivationBytes > 0,
      try estimatedActivationBytes(shape:shape) <= maximumActivationBytes else {
      throw LTXError.invalid("Temporal upscaler exceeds its activation/workspace allowance.")
    }
    return [2 * shape[0] - 1, shape[1], shape[2], 128]
  }

  public func upscale(_ input: MLXArray,
    maximumActivationBytes: Int = 2 * 1024 * 1024 * 1024,
    progress: (String) throws -> Void = { _ in }) throws -> MLXArray {
    let expected = try Self.admit(shape: input.shape, maximumActivationBytes: maximumActivationBytes)
    guard input.dtype == .float32, gate.try() else {
      throw LTXError.invalid("Temporal upscaler needs Float32 input and exclusive execution.")
    }
    defer { gate.unlock() }
    let previousCache = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      residentWeightBytes = 0
      Memory.clearCache()
      Memory.cacheLimit = previousCache
    }
    func finish(_ value: MLXArray) throws -> MLXArray {
      eval(value)
      try Task.checkCancellation()
      guard MLX.isFinite(value).all().item(Bool.self) else {
        throw LTXError.invalid("Temporal upscaler produced nonfinite latents.")
      }
      return value
    }
    func report(_ name: String) throws { try Task.checkCancellation(); try progress(name); try Task.checkCancellation() }
    func read(_ name: String) throws -> MLXArray { try MLXWeight.read(file, name).asType(.float32) }
    func conv(_ value: MLXArray, _ name: String) throws -> MLXArray {
      try Task.checkCancellation()
      return try autoreleasepool {
        let weight = try read(name + ".weight"), bias = try read(name + ".bias")
        residentWeightBytes = weight.nbytes + bias.nbytes
        defer { residentWeightBytes = 0 }
        return try finish(MLXLatentUpscaler.convolution(value, weight: weight, bias: bias))
      }
    }
    func norm(_ value: MLXArray, _ name: String, _ residual: MLXArray? = nil) throws -> MLXArray {
      try finish(MLXLatentUpscaler.normalized(value,
        weight: read(name + ".weight"), bias: read(name + ".bias"), residual: residual))
    }
    let mean = try finish(MLXWeight.read(statistics, "per_channel_statistics.mean-of-means").asType(.float32))
    let std = try finish(MLXWeight.read(statistics, "per_channel_statistics.std-of-means").asType(.float32))
    guard std.min().item(Float.self) > 0 else { throw LTXError.invalid("Invalid latent statistics.") }
    var value = try norm(conv(try finish(input * std + mean), "initial_conv"), "initial_norm")
    try report("initial")
    for stage in ["res_blocks", "post_upsample_res_blocks"] {
      if stage == "post_upsample_res_blocks" {
        let shuffled = Self.temporalShuffle(try conv(value, "upsampler.0"))
        value = try finish(shuffled[1..<shuffled.shape[0]])
        try report("temporal_x2")
      }
      for index in 0..<4 {
        value = try autoreleasepool {
          let stem = "\(stage).\(index)", residual = value
          let first = try norm(conv(value, stem + ".conv1"), stem + ".norm1")
          return try norm(conv(first, stem + ".conv2"), stem + ".norm2", residual)
        }
        try report("\(stage).\(index)")
      }
    }
    value = try finish((conv(value, "final_conv") - mean) / std)
    guard value.shape == expected else { throw LTXError.invalid("Temporal upscaler output geometry differs.") }
    try report("complete")
    return value
  }

  static func temporalShuffle(_ value: MLXArray) -> MLXArray {
    let t = value.shape[0], h = value.shape[1], w = value.shape[2], channels = value.shape[3] / 2
    return value.reshaped([t, h, w, channels, 2]).transposed(0, 4, 1, 2, 3)
      .reshaped([t * 2, h, w, channels])
  }

  private static func validate(_ file: SafeTensorFile) throws {
    guard let json = file.metadata["config"], let data = json.data(using: .utf8),
      let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      config["_class_name"] as? String == "LatentUpsampler",
      config["in_channels"] as? Int == 128, config["mid_channels"] as? Int == 512,
      config["num_blocks_per_stage"] as? Int == 4, config["dims"] as? Int == 3,
      config["spatial_upsample"] as? Bool == false,
      config["temporal_upsample"] as? Bool == true else {
      throw LTXError.invalid("Checkpoint is not the released LTX 2.5 temporal x2 upscaler.")
    }
    var expected: [String: [UInt64]] = [:]
    func add(_ stem: String, _ shape: [UInt64]) {
      expected[stem + ".weight"] = shape
      expected[stem + ".bias"] = [shape[0]]
    }
    add("initial_conv", [512, 128, 3, 3, 3]); add("initial_norm", [512])
    add("upsampler.0", [1024, 512, 3, 3, 3]); add("final_conv", [128, 512, 3, 3, 3])
    for stage in ["res_blocks", "post_upsample_res_blocks"] {
      for index in 0..<4 {
        let stem = "\(stage).\(index)"
        for layer in ["conv1", "conv2"] { add(stem + "." + layer, [512, 512, 3, 3, 3]) }
        for layer in ["norm1", "norm2"] { add(stem + "." + layer, [512]) }
      }
    }
    guard file.tensors.count == expected.count,
      expected.allSatisfy({ name, shape in
        guard let tensor = file.tensors[name] else { return false }
        return tensor.shape == shape && ["BF16", "F16", "F32"].contains(tensor.dtype)
      }) else {
      throw LTXError.invalid("Temporal upscaler tensor layout is incomplete or incompatible.")
    }
  }
}
