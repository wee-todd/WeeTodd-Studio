import Foundation

struct MOSSDimensions {
  var width = 3072, feedForward = 8192, heads = 24, depth = 42
  var textWidth = 4096, latentChannels = 128, frequencyWidth = 256
}

public enum MOSSModelContract {
  public static func validateTransformerConfig(_ data: Data) throws {
    let config = try object(data)
    let expected = ["dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42,
      "text_dim": 4096, "latent_channels": 128, "freq_dim": 256, "rope_base": 10000]
    guard expected.allSatisfy({ (config[$0.key] as? NSNumber)?.intValue == $0.value }) else {
      throw MOSSError("Choose the 42-layer, width-3072 MOSS-Image Base or SFT transformer.")
    }
  }
  public static func validateVAEConfig(_ data: Data) throws {
    let config = try object(data)
    guard config["decoder_block_out_channels"] as? [Int] == [96, 192, 384, 384],
      config["latent_channels"] as? Int == 32, config["patch_size"] as? [Int] == [2, 2],
      config["norm_num_groups"] as? Int == 32, config["layers_per_block"] as? Int == 2,
      abs(((config["batch_norm_eps"] as? NSNumber)?.doubleValue ?? 0) - 0.0001) < 1e-10 else {
      throw MOSSError("MOSS requires its small FLUX.2 VAE, including the packed batch-normalization statistics.")
    }
  }
  static func validateQwenConfig(_ data: Data) throws {
    let root = try object(data)
    guard root["model_type"] as? String == "qwen3_5", let text = root["text_config"] as? [String: Any] else { throw MOSSError("MOSS requires Qwen3.5-9B conditioning.") }
    let expected = ["hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 16,
      "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 12288, "vocab_size": 248320,
      "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16,
      "linear_num_value_heads": 32, "linear_conv_kernel_dim": 4, "full_attention_interval": 4]
    guard expected.allSatisfy({ text[$0.key] as? Int == $0.value }),
      (text["partial_rotary_factor"] as? NSNumber)?.doubleValue == 0.25,
      let rope = text["rope_parameters"] as? [String: Any], (rope["rope_theta"] as? NSNumber)?.doubleValue == 10_000_000 else {
      throw MOSSError("The selected Qwen model is incompatible with MOSS's frozen 9B text encoder.")
    }
  }
  static func validateEncoderConfig(_ data: Data) throws {
    try validateVAEConfig(data)
    let config = try object(data)
    guard config["block_out_channels"] as? [Int] == [128, 256, 512, 512],
      config["in_channels"] as? Int == 3, config["use_quant_conv"] as? Bool == true,
      config["use_post_quant_conv"] as? Bool == true else {
      throw MOSSError("Source reconstruction requires the complete MOSS FLUX.2 VAE encoder and quantization convolution.")
    }
  }
  static func object(_ data: Data) throws -> [String: Any] {
    guard data.count <= 1024 * 1024, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw MOSSError("Invalid or oversized model config.") }
    return object
  }
  static func transformerShapes(_ d: MOSSDimensions = .init()) -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    func linear(_ key: String, _ out: Int, _ input: Int, _ bias: Bool = true) {
      shapes[key + ".weight"] = [out, input]; if bias { shapes[key + ".bias"] = [out] }
    }
    linear("video_embedder", d.width, d.latentChannels); linear("text_embedder.proj", d.width, d.textWidth)
    linear("time_embedding.0", d.width, d.frequencyWidth); linear("time_embedding.2", d.width, d.width)
    linear("shared_block_modulation", 6 * d.width, d.width); linear("video_head_modulation", 2 * d.width, d.width)
    shapes["final_norm_video.weight"] = [d.width]; linear("final_linear_video", d.latentChannels, d.width, false)
    for i in 0..<2 {
      let p = "text_embedder.blocks.\(i)."
      for n in ["norm1", "norm2"] { shapes[p + n + ".weight"] = [d.width] }
      for n in ["q_norm", "k_norm"] { shapes[p + n + ".weight"] = [d.width / d.heads] }
      linear(p + "qkv", 3 * d.width, d.width); linear(p + "proj", d.width, d.width)
      linear(p + "ffn_gate", d.feedForward, d.width); linear(p + "ffn_up", d.feedForward, d.width)
      linear(p + "ffn_down", d.width, d.feedForward)
    }
    for i in 0..<d.depth {
      let p = "layers.\(i)."
      for n in ["norm1", "norm2"] { shapes[p + n + ".weight"] = [d.width] }
      for n in ["q_norm", "k_norm"] { shapes[p + n + ".weight"] = [d.width / d.heads] }
      for n in ["q", "k", "v", "o"] { linear(p + "proj." + n, d.width, d.width, false) }
      linear(p + "proj.ffn_up_gate", 2 * d.feedForward, d.width, false)
      linear(p + "proj.ffn_down", d.width, d.feedForward, false)
    }
    return shapes
  }
  static func qwenShapes() -> [String: [Int]] {
    var result = ["language_model.embed_tokens.weight": [248320, 4096], "language_model.norm.weight": [4096]]
    for i in 0..<32 {
      let p = "language_model.layers.\(i)."
      result[p + "input_layernorm.weight"] = [4096]; result[p + "post_attention_layernorm.weight"] = [4096]
      for n in ["gate_proj", "up_proj"] { result[p + "mlp." + n + ".weight"] = [12288, 4096] }
      result[p + "mlp.down_proj.weight"] = [4096, 12288]
      if i % 4 == 3 {
        for (n, shape) in ["q_proj": [8192, 4096], "k_proj": [1024, 4096], "v_proj": [1024, 4096], "o_proj": [4096, 4096], "q_norm": [256], "k_norm": [256]] {
          result[p + "self_attn." + n + ".weight"] = shape
        }
      } else {
        let q = p + "linear_attn."
        result[q + "A_log"] = [32]; result[q + "dt_bias"] = [32]; result[q + "conv1d.weight"] = [8192, 1, 4]
        result[q + "norm.weight"] = [128]
        for (n, shape) in ["in_proj_qkv": [8192, 4096], "in_proj_z": [4096, 4096], "in_proj_a": [32, 4096], "in_proj_b": [32, 4096], "out_proj": [4096, 4096]] {
          result[q + n + ".weight"] = shape
        }
      }
    }
    return result
  }
  static func vaeShapes() -> [String: [Int]] {
    var result = ["bn.running_mean": [128], "bn.running_var": [128]]
    func conv(_ name: String, _ output: Int, _ input: Int, _ kernel: Int = 3) {
      result[name + ".weight"] = [output, input, kernel, kernel]; result[name + ".bias"] = [output]
    }
    func norm(_ name: String, _ channels: Int) { result[name + ".weight"] = [channels]; result[name + ".bias"] = [channels] }
    func resnet(_ p: String, _ input: Int, _ output: Int) {
      norm(p + ".norm1", input); norm(p + ".norm2", output)
      conv(p + ".conv1", output, input); conv(p + ".conv2", output, output)
      if input != output { conv(p + ".conv_shortcut", output, input, 1) }
    }
    conv("post_quant_conv", 32, 32, 1); conv("decoder.conv_in", 384, 32)
    for i in 0..<2 { resnet("decoder.mid_block.resnets.\(i)", 384, 384) }
    let attention = "decoder.mid_block.attentions.0."
    norm(attention + "group_norm", 384)
    for name in ["to_q", "to_k", "to_v", "to_out.0"] {
      result[attention + name + ".weight"] = [384, 384]; result[attention + name + ".bias"] = [384]
    }
    var input = 384
    for (block, output) in [384, 384, 192, 96].enumerated() {
      for i in 0..<3 { resnet("decoder.up_blocks.\(block).resnets.\(i)", input, output); input = output }
      if block < 3 { conv("decoder.up_blocks.\(block).upsamplers.0.conv", output, output) }
    }
    norm("decoder.conv_norm_out", 96); conv("decoder.conv_out", 3, 96)
    return result
  }
  static func encoderShapes() -> [String: [Int]] {
    var result: [String: [Int]] = [:]
    func conv(_ name: String, _ output: Int, _ input: Int, _ kernel: Int = 3) {
      result[name + ".weight"] = [output, input, kernel, kernel]; result[name + ".bias"] = [output]
    }
    func norm(_ name: String, _ channels: Int) { result[name + ".weight"] = [channels]; result[name + ".bias"] = [channels] }
    func resnet(_ p: String, _ input: Int, _ output: Int) {
      norm(p + ".norm1", input); norm(p + ".norm2", output)
      conv(p + ".conv1", output, input); conv(p + ".conv2", output, output)
      if input != output { conv(p + ".conv_shortcut", output, input, 1) }
    }
    conv("encoder.conv_in", 128, 3)
    var input = 128
    for (block, output) in [128, 256, 512, 512].enumerated() {
      for layer in 0..<2 { resnet("encoder.down_blocks.\(block).resnets.\(layer)", input, output); input = output }
      if block < 3 { conv("encoder.down_blocks.\(block).downsamplers.0.conv", output, output) }
    }
    for layer in 0..<2 { resnet("encoder.mid_block.resnets.\(layer)", 512, 512) }
    let p = "encoder.mid_block.attentions.0."
    norm(p + "group_norm", 512)
    for name in ["to_q", "to_k", "to_v", "to_out.0"] {
      result[p + name + ".weight"] = [512, 512]; result[p + name + ".bias"] = [512]
    }
    norm("encoder.conv_norm_out", 512); conv("encoder.conv_out", 64, 512); conv("quant_conv", 64, 64, 1)
    return result
  }
}
