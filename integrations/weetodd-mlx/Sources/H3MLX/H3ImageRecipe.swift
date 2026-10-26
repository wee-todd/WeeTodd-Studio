import Foundation

/// Image output is an explicit contract around the shared audiovisual sampler.
/// The encoder VAE remains in the base request; the output decoder is separate.
public enum H3ImageRecipe {
  public struct Prepared {
    public let text: H3T2VARequest?
    public let reference: H3Ref2VAStillRequest?
    public let decoder: URL
    public let mode: H3StillImageMode
    public let selection: H3ImageFrameSelection
    public let sourceTask: String
    public var geometry: H3Geometry { text?.geometry ?? reference!.geometry }
  }

  public static func compile(data: Data,
    resolveImage: (String, String, H3Geometry) throws -> H3StillReference) throws -> Prepared {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(root.keys).isSubset(of: ["format", "engine", "candidate", "components", "config",
        "prompt", "conditioning", "block_residency", "negative_prompt", "loras", "ffmpeg", "still_image"]),
      let fields = root.removeValue(forKey: "still_image") as? [String: Any],
      Set(fields.keys).isSubset(of: ["version", "mode", "frame_selection", "image_vae", "request_id", "expected_fingerprint"]),
      let version = fields["version"] as? NSNumber,
      CFGetTypeID(version) != CFBooleanGetTypeID(), version.doubleValue == 1,
      let mode = (fields["mode"] as? String).flatMap(H3StillImageMode.init(rawValue:)),
      let selection = (fields["frame_selection"] as? String).flatMap(H3ImageFrameSelection.init(rawValue:)),
      var components = root["components"] as? [String: Any],
      let sourceTask = components["task"] as? String, ["t2va", "fl2va", "ref2va"].contains(sourceTask),
      components["allow_fl2va_weights_for_ref2va"] == nil || components["allow_fl2va_weights_for_ref2va"] as? Bool == false,
      let originalVAE = components["video_vae"] as? String,
      let conditioning = root["conditioning"] as? [String: Any],
      let inputs = conditioning["inputs"] as? [[String: Any]], inputs.count <= 9,
      let config = root["config"] as? [String: Any],
      (config["attention_policy"] as? String ?? "dense") == "dense",
      (config["transformer_backend"] as? String ?? "mlx") == "mlx" else {
      throw H3CheckpointError.invalid("H3 stills require an ordinary image recipe, at most nine image references, and no video-only controls.")
    }
    guard fields["request_id"] == nil || (fields["request_id"] as? String).flatMap(UUID.init(uuidString:)) != nil,
      fields["expected_fingerprint"] == nil || (fields["expected_fingerprint"] as? String).map({
        $0.hasPrefix("sha256:") && $0.utf8.count == 71 && $0.dropFirst(7).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
      }) == true else { throw H3CheckpointError.invalid("Invalid H3 image request identity.") }
    let decoder: String
    if mode == .singleLatent {
      guard let value = fields["image_vae"] as? String, value.hasPrefix("/"), !value.utf8.contains(0) else {
        throw H3CheckpointError.invalid("Choose a prepared H3 image-only VAE for single-latent generation.")
      }
      decoder = value
    } else {
      guard fields["image_vae"] == nil else {
        throw H3CheckpointError.invalid("Quality stills use the original video VAE.")
      }
      decoder = originalVAE
    }
    let text: H3T2VARequest?
    let reference: H3Ref2VAStillRequest?
    if inputs.isEmpty {
      guard ["t2v", "ref2va"].contains(conditioning["task"] as? String ?? ""),
        Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
        conditioning["version"] as? Int == 1,
        (conditioning["audio_policy"] as? String ?? "generated") == "generated" else {
        throw H3CheckpointError.invalid("H3 text-to-image needs an empty generated conditioning selection.")
      }
      components["task"] = "t2va"
      components.removeValue(forKey: "vision_encoder")
      components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
      root["components"] = components
      if sourceTask != "t2va" {
        _ = try H3ReferenceNoiseControls.parse(config)
        var textConfig = config
        textConfig.removeValue(forKey: "visual_condition_strength")
        textConfig.removeValue(forKey: "audio_condition_strength")
        root["config"] = textConfig
      }
      root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated"]
      // The selected transformer never changes when there are no reference rows.
      text = try H3StudioRecipe.compile(data: JSONSerialization.data(withJSONObject: root), stillImageMode: mode)
      reference = nil
    } else {
      guard sourceTask == "ref2va", inputs.allSatisfy({ $0["kind"] as? String == "image" && $0["placement"] == nil }) else {
        throw H3CheckpointError.invalid("H3 image editing requires a Ref2VA model and untimed image references.")
      }
      text = nil
      reference = try H3StudioRecipe.compileMediaReferences(data: JSONSerialization.data(withJSONObject: root), stillImageMode: mode) {
        path, kind, digest, geometry, controls in
        guard kind == "image", controls == nil else {
          throw H3CheckpointError.invalid("H3 still editing does not accept movie, audio or video preparation controls.")
        }
        return .image(try resolveImage(path, digest, geometry))
      }
    }
    return Prepared(text: text, reference: reference, decoder: URL(fileURLWithPath: decoder),
      mode: mode, selection: selection, sourceTask: sourceTask)
  }
}
