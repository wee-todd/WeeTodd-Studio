import CoreFoundation
import Foundation

/// Saved selection for the fixed trained Omni release. No weighted state is retained.
public enum H3OmniRecipe {
  public static let contractSHA256 = "68577f23f0c866e329aa8a578b2455e4ca9a26a3ca1d15ddd6d0e99091007e18"
  public struct Selection: Sendable {
    public let contractPath: URL
    public let contractSHA256: String
    public let contract: H3OmniContract
    public var metadata: [String: Any] { ["version": 1, "contract_path": contractPath.path,
      "contract_sha256": contractSHA256, "source_sha256": H3OmniCheckpoint.sourceSHA256,
      "variant": H3OmniVariant.affineQ8V1.rawValue, "schedule_nodes": 9,
      "evaluations": 8, "pdd_steps": 32, "attention": "trained-vsa-128-90",
      "reference_policy": "p2_multi_region", "reference_keep_rate": 0.1] }
    func checkUnchanged() throws { _ = try H3OmniContract.load(url: contractPath, expectedSHA256: contractSHA256) }
  }
  public struct Prepared {
    public let selection: Selection
    public let ordinaryRecipe: Data
  }

  /// Must inspect the untouched recipe before any competing wrapper or image dispatch.
  public static func prepare(data: Data) throws -> Prepared? {
    guard data.count <= 1_048_576,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3CheckpointError.invalid("Invalid H3 recipe.")
    }
    guard let value = root["fasth3_omni"] else { return nil }
    try validateComponents(root["components"])
    func number(_ value: Any?, integer: Bool = false) -> Double? {
      guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else { return nil }
      if integer && ["f", "d"].contains(String(cString:n.objCType)) { return nil }
      return n.doubleValue
    }
    func trueBoolean(_ value: Any?) -> Bool {
      guard let value else { return true }
      guard let n = value as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { return false }
      return n.boolValue
    }
    guard let omni = value as? [String: Any], Set(omni.keys) == ["version", "contract_path", "contract_sha256"],
      number(omni["version"], integer: true) == 1,
      let path = omni["contract_path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
      let digest = omni["contract_sha256"] as? String, digest == contractSHA256,
      digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      Set(root.keys).isSubset(of: ["format", "engine", "candidate", "components", "config", "prompt", "conditioning", "ffmpeg", "block_residency", "negative_prompt", "fasth3_omni"]),
      let config = root["config"] as? [String: Any],
      Set(config.keys).isSubset(of: ["width", "height", "duration_seconds", "steps", "seed", "sampling_method", "drop_adaln", "resolution_mode", "resolution_tier", "aspect_ratio", "memory_mode", "video_decode_precision", "attention_chunk_size", "attention_head_chunk_size", "ffn_row_chunk_size", "projection_backend", "transformer_backend", "inference_optimization", "paging_cache_gb", "transformer_weight_cache_gb"]),
      number(config["steps"], integer:true) == 9, trueBoolean(config["drop_adaln"]),
      let duration = number(config["duration_seconds"]), (5...15).contains(duration),
      let width = number(config["width"], integer:true), let height = number(config["height"], integer:true),
      width > 0, height > 0, width <= 2048, height <= 2048,
      Int(width).isMultiple(of:32), Int(height).isMultiple(of:32),
      let seed = number(config["seed"], integer:true), (0...Double(UInt32.max)).contains(seed),
      config["sampling_method"] == nil || config["sampling_method"] as? String == "euler",
      config["transformer_backend"] == nil || config["transformer_backend"] as? String == "mlx",
      config["inference_optimization"] == nil || config["inference_optimization"] as? String == "off",
      config["paging_cache_gb"] == nil || number(config["paging_cache_gb"]) == 0,
      config["transformer_weight_cache_gb"] == nil || number(config["transformer_weight_cache_gb"]) == 0,
      let conditioning = root["conditioning"] as? [String: Any],
      number(conditioning["version"], integer:true) == 1, conditioning["task"] as? String == "ref2va",
      let inputs = conditioning["inputs"] as? [[String: Any]], (1...12).contains(inputs.count),
      inputs.contains(where: { ["image","video"].contains($0["kind"] as? String ?? "") }),
      inputs.allSatisfy({ $0["frame_index"] == nil && $0["role"] as? String == "reference" &&
        Set($0.keys).isSubset(of:["id","kind","role","path","strength","sha256","soundtrack_path","soundtrack_sha256","image_pixel_budget_percent","size_policy","temporal_density"]) }) else {
      throw H3CheckpointError.invalid("FastH3 Omni requires the fixed eight-evaluation Ref2VA profile without image mode, timed references, adapters, controls, refinement or caches.")
    }
    for input in inputs {
      let controls = try H3ReferencePreparationControls.parse(input, kind: input["kind"] as? String ?? "")
      guard controls?.temporalDensity == nil || controls?.temporalDensity == .full else {
        throw H3CheckpointError.invalid("FastH3 Omni supports full reference movie temporal density only.")
      }
    }
    let url = URL(fileURLWithPath:path)
    let contract = try H3OmniContract.load(url:url, expectedSHA256:digest)
    root.removeValue(forKey:"fasth3_omni")
    return Prepared(selection:Selection(contractPath:url, contractSHA256:digest, contract:contract),
      ordinaryRecipe:try JSONSerialization.data(withJSONObject:root))
  }

  /// Validate before the common Ref compiler removes vision/compatibility
  /// fields or substitutes the text encoder for an omitted vision encoder.
  /// The legacy FL weight compatibility flag is not part of Omni's schema.
  private static func validateComponents(_ value: Any?) throws {
    let requiredPaths: Set<String> = ["transformer", "text_encoder", "tokenizer", "video_vae", "audio_vae"]
    let optionalPaths: Set<String> = ["vision_encoder", "checkpoint", "processor"]
    func validPath(_ value: Any?) -> Bool {
      guard let path = value as? String else { return false }
      return path.hasPrefix("/") && !path.utf8.contains(0)
    }
    guard let components = value as? [String: Any],
      Set(components.keys).isSubset(of: requiredPaths.union(optionalPaths).union(["task", "loras"])),
      components["task"] as? String == "ref2va",
      requiredPaths.allSatisfy({ validPath(components[$0]) }),
      optionalPaths.allSatisfy({ components[$0] == nil || validPath(components[$0]) }),
      components["loras"] == nil || (components["loras"] as? [Any])?.isEmpty == true else {
      throw H3CheckpointError.invalid("FastH3 Omni components require native Ref2VA paths and no compatibility controls.")
    }
  }

}
