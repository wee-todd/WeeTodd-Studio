import Foundation
import TensorIO

/// Header-only admission for the installed folded-weight H3 audio VAE. The
/// decoder reads one float32 convolution at a time from this source file.
public struct H3AudioVAELayout: Sendable {
  public let sampleRate: Int
  public let samplesPerLatent: Int
  public let upsampleRates: [Int]
  public let latentsMean: [Float]
  public let latentsStandardDeviation: [Float]

  public init(url: URL) throws {
    let file = try SafeTensorFile(url: url)
    guard let raw = file.metadata["minimax_h3_audio_vae"],
      let data = raw.data(using: .utf8),
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let kwargs = object["kwargs"] as? [String: Any],
      kwargs["decoder_type"] as? String == "bigvgan",
      kwargs["sample_rate"] as? Int == 32_000,
      kwargs["latent_dim"] as? Int == 2048,
      kwargs["vae_latent_channels"] as? Int == 32,
      kwargs["decoder_dim"] as? Int == 1024,
      kwargs["decoder_rates"] as? [Int] == [5, 5, 2, 2, 2, 2, 2],
      kwargs["encoder_rates"] as? [Int] == [2, 4, 4, 5, 5],
      object["output_channel"] as? Int == 2,
      let mean = object["latents_mean"] as? [NSNumber], mean.count == 32,
      let std = object["latents_std"] as? [NSNumber], std.count == 32 else {
      throw H3CheckpointError.invalid("Unsupported H3 folded-weight audio VAE metadata.")
    }
    let meanValues = mean.map(\.floatValue)
    let stdValues = std.map(\.floatValue)
    guard meanValues.allSatisfy(\.isFinite),
      stdValues.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 audio latent statistics.")
    }
    func require(_ name: String, _ shape: [UInt64]) throws {
      guard let tensor = file.tensors[name], tensor.dtype == "F32",
        tensor.shape == shape else {
        throw H3CheckpointError.invalid("Missing H3 audio decoder tensor: \(name)")
      }
    }
    try require("dec_in_proj.weight", [2048, 32, 1])
    try require("dec_in_proj.bias", [2048])
    try require("decoder.conv_pre.weight", [1024, 2048, 7])
    try require("decoder.conv_pre.bias", [1024])
    let rates = [5, 5, 2, 2, 2, 2, 2]
    let kernels = [9, 9, 4, 4, 4, 4, 4]
    for stage in rates.indices {
      let previous = 1024 >> stage
      let channels = previous / 2
      try require("decoder.ups.\(stage).0.weight",
        [UInt64(previous), UInt64(channels), UInt64(kernels[stage])])
      try require("decoder.ups.\(stage).0.bias", [UInt64(channels)])
      for branch in 0..<3 {
        let block = stage * 3 + branch
        let kernel = [3, 7, 11][branch]
        for layer in 0..<3 {
          for bank in ["convs1", "convs2"] {
            let base = "decoder.resblocks.\(block).\(bank).\(layer)"
            try require(base + ".weight",
              [UInt64(channels), UInt64(channels), UInt64(kernel)])
            try require(base + ".bias", [UInt64(channels)])
          }
        }
        for activation in 0..<6 {
          let base = "decoder.resblocks.\(block).activations.\(activation)"
          try require(base + ".act.alpha", [UInt64(channels)])
          try require(base + ".act.beta", [UInt64(channels)])
          try require(base + ".upsample.filter", [1, 1, 12])
          try require(base + ".downsample.lowpass.filter", [1, 1, 12])
        }
      }
    }
    let post = "decoder.activation_post"
    for key in ["act.alpha", "act.beta"] {
      try require(post + "." + key, [8])
    }
    try require(post + ".upsample.filter", [1, 1, 12])
    try require(post + ".downsample.lowpass.filter", [1, 1, 12])
    try require("decoder.conv_post.weight", [1, 8, 7])
    try file.checkUnchanged(at: url)
    sampleRate = 32_000
    samplesPerLatent = 800
    upsampleRates = rates
    latentsMean = meanValues
    latentsStandardDeviation = stdValues
  }
}
