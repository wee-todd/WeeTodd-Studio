import Foundation
import InferenceContracts

/// Strict bridge from the saved headless H3 recipe to the first Swift T2VA
/// slice. Every setting this slice cannot execute fails before weights load.
public enum H3StudioRecipe {
  /// Admit FL2VA's timed keyframe contract before reading any image.
  /// The text recipe validator still owns every shared execution control.
  public static func compileFL2VA(data: Data,
    resolveImage: (String, Bool, Int, Int) throws -> H3StillReference) throws -> H3FL2VARequest {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var components = root["components"] as? [String: Any],
      components["task"] as? String == "fl2va",
      let vision = (components["vision_encoder"] as? String) ??
        (components["text_encoder"] as? String), vision.hasPrefix("/"),
      let conditioning = root["conditioning"] as? [String: Any],
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      conditioning["version"] as? Int == 1,
      conditioning["task"] as? String == "fflf",
      (conditioning["audio_policy"] as? String ?? "generated") == "generated",
      let inputs = conditioning["inputs"] as? [[String: Any]],
      (1...8).contains(inputs.count) else {
      throw H3CheckpointError.invalid("Swift H3 FL2VA needs one to eight timed images with generated audio.")
    }
    var paths: [String] = []
    _ = try ConditioningV1.inputs(conditioning, task: "fflf", audioPolicy: "generated", count: 1...8)
    var ids = Set<String>()
    var anchors: [H3PackedLayout.Anchor] = []
    for input in inputs {
      let role = input["role"] as? String
      let frame = input["frame_index"]
      let anchor: H3PackedLayout.Anchor?
      if role == "first", frame as? Int == 0 { anchor = .first }
      else if role == "last", frame as? String == "last" { anchor = .last }
      else if role == "last", let value = frame as? Int,
        (0...4095).contains(value) { anchor = .frame(value) }
      else if role == "keyframe", let value = frame as? Int,
        (0...4095).contains(value) { anchor = value == 0 ? .first : .frame(value) }
      else { anchor = nil }
      guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "strength", "sha256", "frame_index"]),
        let id = input["id"] as? String, !id.isEmpty, ids.insert(id).inserted,
        input["kind"] as? String == "image", let anchor,
        let path = input["path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
        let digest = input["sha256"] as? String, digest.count == 64,
        digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        (input["strength"] == nil || (input["strength"] as? NSNumber)
          .map({ CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1 }) == true)
      else {
        throw H3CheckpointError.invalid("Swift H3 FL2VA accepts timed image keyframes at full strength only.")
      }
      paths.append(path)
      anchors.append(anchor)
    }
    components.removeValue(forKey: "vision_encoder")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [],
      "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root))
    var priorFrame = -1
    for anchor in anchors {
      let frame: Int
      switch anchor {
      case .first: frame = 0
      case .last: frame = base.geometry.frames - 1
      case .frame(let value): frame = value
      }
      guard frame < base.geometry.frames, frame > priorFrame else {
        throw H3CheckpointError.invalid("Swift H3 FL2VA keyframes must have unique ascending positions inside the generated clip.")
      }
      priorFrame = frame
    }
    let images = try paths.enumerated().map { index, path in
      try resolveImage(path, index == 0, base.geometry.width, base.geometry.height)
    }
    return try H3FL2VARequest(base: base, vision: URL(fileURLWithPath: vision),
      images: images, anchors: anchors)
  }

  /// Reuse the strict T2VA execution-control admission after removing only the
  /// reference-specific fields. Media is resolved after every recipe control
  /// passes, so an unsupported input can never be silently discarded.
  public static func compileStillReferences(data: Data,
    resolveImage: (String) throws -> H3StillReference) throws -> H3Ref2VAStillRequest {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var components = root["components"] as? [String: Any],
      components["task"] as? String == "ref2va",
      components["allow_fl2va_weights_for_ref2va"] == nil ||
        components["allow_fl2va_weights_for_ref2va"] as? Bool == false,
      let vision = (components["vision_encoder"] as? String) ??
        (components["text_encoder"] as? String),
      vision.hasPrefix("/"),
      let conditioning = root["conditioning"] as? [String: Any],
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      conditioning["version"] as? Int == 1,
      conditioning["task"] as? String == "ref2va",
      (conditioning["audio_policy"] as? String ?? "generated") == "generated",
      let inputs = conditioning["inputs"] as? [[String: Any]],
      (1...9).contains(inputs.count) else {
      throw H3CheckpointError.invalid("Swift H3 still Ref2VA needs one to nine ordered image inputs and generated audio.")
    }
    var paths: [String] = []
    _ = try ConditioningV1.inputs(conditioning, task: "ref2va", audioPolicy: "generated", count: 1...9)
    var identities = Set<String>()
    for input in inputs {
      guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "strength", "sha256"]),
        let identity = input["id"] as? String, !identity.isEmpty,
        identities.insert(identity).inserted,
        input["kind"] as? String == "image",
        input["role"] as? String == "reference",
        let path = input["path"] as? String, path.hasPrefix("/"),
        !path.utf8.contains(0),
        let digest = input["sha256"] as? String, digest.count == 64,
        digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        (input["strength"] == nil || (input["strength"] as? NSNumber)
          .map({ CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1 }) == true)
      else {
        throw H3CheckpointError.invalid("Swift H3 still Ref2VA does not accept timed, weighted, or non-image references.")
      }
      paths.append(path)
    }
    components.removeValue(forKey: "vision_encoder")
    components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v",
      "inputs": [], "audio_policy": "generated"]
    let normalized = try JSONSerialization.data(withJSONObject: root)
    let base = try compile(data: normalized)
    let references = try paths.map(resolveImage)
    return try H3Ref2VAStillRequest(prompt: base.prompt,
      references: references, width: base.geometry.width,
      height: base.geometry.height,
      durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages,
      qwenVision: URL(fileURLWithPath: vision), tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      turboLoRA: base.turboLoRA,
      turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs)
  }

  /// Admit ordered still, video and standalone audio Ref2VA media.
  /// An audio-bearing movie contributes both visual and sound references.
  public static func compileMediaReferences(data: Data,
    resolveReference: (String, String, String) throws -> H3Ref2VAReference) throws
    -> H3Ref2VAStillRequest {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var components = root["components"] as? [String: Any],
      components["task"] as? String == "ref2va",
      components["allow_fl2va_weights_for_ref2va"] == nil ||
        components["allow_fl2va_weights_for_ref2va"] as? Bool == false,
      let vision = (components["vision_encoder"] as? String) ??
        (components["text_encoder"] as? String), vision.hasPrefix("/"),
      let conditioning = root["conditioning"] as? [String: Any],
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      conditioning["version"] as? Int == 1,
      conditioning["task"] as? String == "ref2va",
      (conditioning["audio_policy"] as? String ?? "generated") == "generated",
      let inputs = conditioning["inputs"] as? [[String: Any]],
      (1...12).contains(inputs.count) else {
      throw H3CheckpointError.invalid("Swift H3 Ref2VA needs ordered visual and optional audio references.")
    }
    _ = try ConditioningV1.inputs(conditioning, task: "ref2va",
      audioPolicy: "generated", count: 1...12)
    var paths: [(String, String, String)] = []
    var images = 0
    var videos = 0
    var audios = 0
    for input in inputs {
      let kind = input["kind"] as? String ?? ""
      if kind == "image" { images += 1 }
      if kind == "video" { videos += 1 }
      if kind == "audio" { audios += 1 }
      guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "strength", "sha256"]),
        ["image", "video", "audio"].contains(kind),
        input["role"] as? String == "reference",
        let path = input["path"] as? String, path.hasPrefix("/"),
        !path.utf8.contains(0),
        let digest = input["sha256"] as? String, digest.count == 64,
        digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        (input["strength"] == nil || (input["strength"] as? NSNumber)
          .map({ CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1 }) == true)
      else {
        throw H3CheckpointError.invalid("Swift H3 Ref2VA accepts full-strength image, video or audio references only.")
      }
      paths.append((path, kind, digest))
    }
    guard images + videos > 0, images <= 9, videos <= 3, audios <= 3 else {
      throw H3CheckpointError.invalid("Swift H3 Ref2VA needs a visual source and allows at most nine images, three videos and three audio sources.")
    }
    components.removeValue(forKey: "vision_encoder")
    components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [],
      "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root))
    let references = try paths.map { try resolveReference($0.0, $0.1, $0.2) }
    return try H3Ref2VAStillRequest(prompt: base.prompt,
      mediaReferences: references, width: base.geometry.width,
      height: base.geometry.height, durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages,
      qwenVision: URL(fileURLWithPath: vision), tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      turboLoRA: base.turboLoRA,
      turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs)
  }

  public static func compile(data: Data) throws -> H3T2VARequest {
    func emptyArray(_ object: [String: Any], _ key: String) -> Bool {
      guard let value = object[key] else { return true }
      guard let values = value as? [Any] else { return false }
      return values.isEmpty
    }
    func oneOf(_ object: [String: Any], _ key: String,
      default fallback: String, _ allowed: Set<String>) -> Bool {
      guard let value = object[key] else { return allowed.contains(fallback) }
      guard let string = value as? String else { return false }
      return allowed.contains(string)
    }
    func equals(_ object: [String: Any], _ key: String,
      default fallback: Bool, _ expected: Bool) -> Bool {
      guard let value = object[key] else { return fallback == expected }
      guard let boolean = value as? Bool else { return false }
      return boolean == expected
    }
    func zero(_ object: [String: Any], _ key: String) -> Bool {
      guard let value = object[key] else { return true }
      guard let number = value as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID() else { return false }
      return number.doubleValue == 0
    }
    func turboLoRAs(_ object: [String: Any]) -> [(URL, Float)]? {
      guard let entries = object["loras"] as? [[Any]],
        (1...4).contains(entries.count) else { return nil }
      var adapters: [(URL, Float)] = []
      for pair in entries {
        guard pair.count == 2, let path = pair[0] as? String,
          path.hasPrefix("/"), !path.utf8.contains(0),
          let number = pair[1] as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let strength = number.floatValue
        guard strength.isFinite, (0...2).contains(strength) else { return nil }
        adapters.append((URL(fileURLWithPath: path), strength))
      }
      guard Set(adapters.map { $0.0.standardizedFileURL.path }).count == adapters.count
      else { return nil }
      return adapters
    }
    guard data.count <= 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(root.keys).isSubset(of: ["format", "engine", "candidate", "components",
        "config", "prompt", "conditioning", "ffmpeg", "block_residency",
        "negative_prompt"]),
      root["format"] as? String == "weetodd-headless-v2",
      root["engine"] as? String == "h3",
      oneOf(root, "negative_prompt", default: "", [""]),
      oneOf(root, "block_residency", default: "checkpoint_default",
        ["checkpoint_default"]),
      let prompt = root["prompt"] as? String,
      let component = root["components"] as? [String: Any],
      Set(component.keys).isSubset(of: ["checkpoint", "transformer",
        "text_encoder", "processor", "tokenizer", "video_vae", "audio_vae",
        "task", "loras"]),
      oneOf(component, "task", default: "t2va", ["t2va"]),
      (component["loras"] == nil || emptyArray(component, "loras")
        || turboLoRAs(component) != nil),
      let transformer = component["transformer"] as? String,
      let qwen = component["text_encoder"] as? String,
      let tokenizer = component["tokenizer"] as? String,
      let video = component["video_vae"] as? String,
      let audio = component["audio_vae"] as? String,
      [transformer, qwen, tokenizer, video, audio].allSatisfy({ $0.hasPrefix("/") }),
      let conditioning = root["conditioning"] as? [String: Any],
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      conditioning["version"] as? Int == 1,
      conditioning["task"] as? String == "t2v",
      emptyArray(conditioning, "inputs"),
      oneOf(conditioning, "audio_policy", default: "generated", ["generated"]),
      let config = root["config"] as? [String: Any],
      Set(config.keys).isSubset(of: ["width", "height", "duration_seconds",
        "steps", "seed", "drop_adaln", "resolution_mode", "resolution_tier",
        "aspect_ratio", "memory_mode", "attention_chunk_size",
        "attention_head_chunk_size", "ffn_row_chunk_size",
        "projection_backend", "transformer_backend", "sampling_method",
        "inference_optimization", "paging_cache_gb"]),
      equals(config, "drop_adaln", default: true, true),
      oneOf(config, "resolution_mode", default: "custom", ["custom"]),
      oneOf(config, "resolution_tier", default: "custom", ["custom"]),
      oneOf(config, "aspect_ratio", default: "custom", ["custom"]),
      oneOf(config, "memory_mode", default: "normal", ["normal", "low_memory_bf16"]),
      oneOf(config, "attention_chunk_size", default: "automatic", ["automatic"]),
      oneOf(config, "attention_head_chunk_size", default: "automatic", ["automatic", "disabled"]),
      oneOf(config, "ffn_row_chunk_size", default: "automatic", ["automatic"]),
      oneOf(config, "projection_backend", default: "mlx", ["auto", "mlx"]),
      oneOf(config, "transformer_backend", default: "mlx", ["mlx"]),
      oneOf(config, "sampling_method", default: "euler", ["euler"]),
      oneOf(config, "inference_optimization", default: "off", ["off"]),
      zero(config, "paging_cache_gb"),
      let width = config["width"] as? Int,
      let height = config["height"] as? Int,
      let duration = config["duration_seconds"] as? Double,
      let steps = config["steps"] as? Int,
      let seed = config["seed"] as? Int, (0...Int(UInt32.max)).contains(seed) else {
      throw H3CheckpointError.invalid("Swift H3 currently admits only text-to-audiovisual Euler recipes with up to four distinct supported Turbo LoRAs and no unported controls.")
    }
    let adapters = turboLoRAs(component) ?? []
    let additional = try adapters.dropFirst().map {
      try H3LoRAAdapter(url: $0.0, strength: $0.1)
    }
    _ = try ConditioningV1.inputs(conditioning, task: "t2v", audioPolicy: "generated", count: 0...0)
    return try H3T2VARequest(prompt: prompt, width: width, height: height,
      durationSeconds: duration, seed: UInt64(seed), requestedSteps: steps,
      transformer: URL(fileURLWithPath: transformer),
      qwenPages: URL(fileURLWithPath: qwen),
      tokenizer: URL(fileURLWithPath: tokenizer),
      videoVAE: URL(fileURLWithPath: video),
      audioVAE: URL(fileURLWithPath: audio),
      turboLoRA: adapters.first?.0,
      turboLoRAStrength: adapters.first?.1 ?? 1,
      additionalLoRAs: additional)
  }
}
