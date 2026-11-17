import Foundation

/// Payload-free target names shared by Studio inspection and native admission.
public enum MOSSAdapterTargets {
  public static let names: Set<String> = {
    var result: Set<String> = ["video_embedder", "text_embedder.proj", "time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation", "final_linear_video"]
    for i in 0..<42 { for projection in ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"] { result.insert("layers.\(i).proj." + projection) } }
    for i in 0..<2 { for projection in ["qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"] { result.insert("text_embedder.blocks.\(i)." + projection) } }
    return result
  }()
  public static func canonical(_ raw: String, allowed: Set<String> = names) throws -> String {
    var key = raw
    for prefix in ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "model."] {
      if key.hasPrefix(prefix) { key = String(key.dropFirst(prefix.count)) }
    }
    if allowed.contains(key) { return key }
    for prefix in ["lora_unet_", "lora_transformer_"] where key.hasPrefix(prefix) { key = String(key.dropFirst(prefix.count)) }
    let matches = allowed.filter { $0.replacingOccurrences(of: ".", with: "_") == key }
    guard matches.count == 1, let match = matches.first else { throw ImageLayoutError.invalid("Unrecognized or ambiguous MOSS adapter target: \(raw)") }
    return match
  }
}
