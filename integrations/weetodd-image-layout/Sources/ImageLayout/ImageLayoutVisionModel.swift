import Foundation

public enum ImageLayoutVisionModel: String, Codable, Sendable, CaseIterable {
  case qwen35_4b, gemma4_12b
  public var displayName: String {
    switch self {
    case .qwen35_4b: "Qwen3.5-4B"
    case .gemma4_12b: "Gemma4-12B"
    }
  }
  public var requiredAssets: [String] {
    let common = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]
    return self == .qwen35_4b ? common + ["preprocessor_config.json"] : common
  }
  public static func identify(_ root: [String: Any]) throws -> Self {
    guard let text = root["text_config"] as? [String: Any], let vision = root["vision_config"] as? [String: Any],
      let quant = root["quantization"] as? [String: Any], quant["group_size"] as? Int == 64,
      quant["mode"] as? String == "affine" else { throw ImageLayoutError.invalid("Choose a supported affine MLX vision checkpoint.") }
    if root["model_type"] as? String == "qwen3_5", text["hidden_size"] as? Int == 2560,
      text["num_hidden_layers"] as? Int == 32, vision["hidden_size"] as? Int == 1024,
      vision["out_hidden_size"] as? Int == 2560, vision["patch_size"] as? Int == 16,
      [4, 8].contains(quant["bits"] as? Int ?? 0) { return .qwen35_4b }
    if root["model_type"] as? String == "gemma4_unified", text["hidden_size"] as? Int == 3840,
      text["num_hidden_layers"] as? Int == 48, vision["mm_embed_dim"] as? Int == 3840,
      vision["model_patch_size"] as? Int == 48, vision["patch_size"] as? Int == 16,
      quant["bits"] as? Int == 8 { return .gemma4_12b }
    throw ImageLayoutError.invalid("Use MLX Qwen3.5-4B Q4/Q8 or Gemma4-12B unified Q8 vision weights.")
  }
}
