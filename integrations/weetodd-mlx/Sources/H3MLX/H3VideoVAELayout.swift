import Foundation
import TensorIO

/// Admit the installed native-layout affine-Q8 H3 video decoder from its
/// safetensors header. All 36 block contracts are checked before GPU work.
public struct H3VideoVAELayout: Sendable {
  struct Metadata {
    let mean: [Float]
    let standardDeviation: [Float]
    let clipLength: Int
    let tokenDrop: Int
  }

  public let blockCount = 36
  public let latentsMean: [Float]
  public let latentsStandardDeviation: [Float]
  public let clipLength: Int
  public let tokenDrop: Int

  static func validateMetadata(_ object: [String: Any]) throws -> Metadata {
    guard object["format"] as? String == "minimax-h3-mlx-video-vae",
      object["format_version"] as? Int == 1,
      object["tensor_layout"] as? String == "ODHWI",
      object["vae_clip_length"] as? Int == 17,
      object["vae_token_drop"] as? Int == 3,
      let quant = object["quantization"] as? [String: Any],
      quant["format"] as? String == "mlx-affine",
      quant["bits"] as? Int == 8,
      quant["group_size"] as? Int == 64,
      quant["scope"] as? String == "decoder-transformer-core",
      quant["quantized_layers"] as? Int == 144,
      let meanValues = object["latents_mean"] as? [NSNumber],
      let stdValues = object["latents_std"] as? [NSNumber],
      meanValues.count == 24, stdValues.count == 24 else {
      throw H3CheckpointError.invalid("Unsupported H3 video VAE format or normalization metadata.")
    }
    let mean = meanValues.map(\.floatValue)
    let standardDeviation = stdValues.map(\.floatValue)
    guard mean.allSatisfy(\.isFinite),
      standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 video VAE latent normalization.")
    }
    return Metadata(mean: mean, standardDeviation: standardDeviation,
      clipLength: 17, tokenDrop: 3)
  }

  public init(url: URL) throws {
    let file = try SafeTensorFile(url: url)
    try self.init(file: file, url: url)
  }

  init(file: SafeTensorFile, url: URL) throws {
    guard let raw = file.metadata["minimax_h3_video_vae"],
      let data = raw.data(using: .utf8),
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3CheckpointError.invalid("Missing H3 video VAE checkpoint metadata.")
    }
    let metadata = try Self.validateMetadata(object)
    func require(_ name: String, _ dtype: String, _ shape: [UInt64]) throws {
      guard let tensor = file.tensors[name], tensor.dtype == dtype,
        tensor.shape == shape else {
        throw H3CheckpointError.invalid("Missing H3 video VAE tensor: \(name)")
      }
    }
    try require("post_quant_conv.weight", "F16", [24, 1, 1, 1, 24])
    try require("post_quant_conv.bias", "F16", [24])
    try require("decoder.x_embedder.weight", "F16", [2048, 24])
    try require("decoder.x_embedder.bias", "F16", [2048])
    try require("decoder.register_tokens", "F16", [1, 4, 2048])
    try require("decoder.norm_out.weight", "F16", [2048])
    try require("decoder.norm_out.bias", "F16", [2048])
    try require("decoder.proj_out.weight", "F16", [3072, 2048])
    try require("decoder.proj_out.bias", "F16", [3072])
    for index in 0..<36 {
      let base = "decoder.transformer_blocks.\(index)."
      for suffix in ["norm1.weight", "norm2.weight", "scale1", "scale2"] {
        try require(base + suffix, "F16", [2048])
      }
      for (suffix, rows, columns) in [
        ("attn.to_qkv", 6144, 2048),
        ("attn.to_out", 2048, 2048),
        ("ff.w1", 16384, 2048),
        ("ff.w2", 2048, 8192),
      ] {
        let packed = [UInt64(rows), UInt64(columns / 4)]
        let groups = [UInt64(rows), UInt64(columns / 64)]
        try require(base + suffix + ".weight", "U32", packed)
        try require(base + suffix + ".scales", "F16", groups)
        try require(base + suffix + ".biases", "F16", groups)
        try require(base + suffix + ".bias", "F16", [UInt64(rows)])
      }
    }
    try file.checkUnchanged(at: url)
    latentsMean = metadata.mean
    latentsStandardDeviation = metadata.standardDeviation
    clipLength = metadata.clipLength
    tokenDrop = metadata.tokenDrop
  }
}
