import Foundation

public struct NativeModelScanResult {
  public let candidates: [String: [String]]
  public let warnings: [String]
}

/// Weight-free setup for the Swift audiovisual workers. This deliberately offers only
/// ordinary profiles; specialized adapters need their own validated contracts.
public enum NativeModelSetup {
  public static let engines: Set<String> = ["h3", "ltx25"]

  /// Inspect bounded headers and manifests in user-selected folders. This never
  /// reads tensor payloads, copies weights, or starts the optional Python runtime.
  /// The worker still validates the selected complete stack before generation.
  public static func scan(presetID: String, roots: [String]) throws -> NativeModelScanResult {
    guard let preset = catalog().first(where: { $0.id == presetID }), !roots.isEmpty else {
      throw StudioError.invalid("Choose a native preset and at least one model folder.")
    }
    let fm = FileManager.default
    var candidates = Dictionary(uniqueKeysWithValues: preset.components.map { ($0.key, [String]()) })
    var warnings: [String] = []
    var seen: Set<String> = []
    var pending = roots.reversed().map { URL(fileURLWithPath: $0) }
    var inspected = 0
    while let next = pending.popLast() {
      try Task.checkCancellation()
      inspected += 1
      if inspected > 20_000 {
        warnings.append("Scan entry limit reached. Choose a smaller model folder.")
        break
      }
      let url = next.standardizedFileURL.resolvingSymlinksInPath()
      guard seen.insert(url.path).inserted else { continue }
      var directory: ObjCBool = false
      guard fm.fileExists(atPath: url.path, isDirectory: &directory) else {
        if roots.contains(next.path) { warnings.append("Model folder is unavailable: \(next.path)") }
        continue
      }
      let kind = directory.boolValue ? "directory" : "file"
      if kind == "directory" || url.pathExtension == "safetensors" {
        for component in preset.components where (component.accepts ?? [component.kind]).contains(kind) {
          if (try? NativeModelInspector.matches(url, key: component.key,
            engine: preset.engine, task: preset.task)) == true {
            candidates[component.key, default: []].append(url.path)
          }
        }
      }
      guard directory.boolValue else { continue }
      let children = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles])) ?? []
      for child in children.sorted(by: { $0.path > $1.path }) {
        if [".git", ".cache", "pages"].contains(child.lastPathComponent) ||
          child.lastPathComponent.hasPrefix("._") { continue }
        pending.append(child)
      }
    }
    for component in preset.components {
      let matches = Array(Set(candidates[component.key, default: []])).sorted()
      candidates[component.key] = matches
      if matches.isEmpty { warnings.append("No compatible \(component.label) found. Import it directly if installed elsewhere.") }
      else if matches.count > 1 { warnings.append("Multiple candidates for \(component.label); choose the intended component.") }
    }
    warnings.append("Discovery checks headers and manifests. Worker preflight validates the selected stack.")
    return NativeModelScanResult(candidates: candidates, warnings: warnings)
  }

  public static func catalog() -> [ModelSetupPreset] {
    func component(_ key: String, _ label: String, _ accepts: [String]) -> ModelSetupComponent {
      ModelSetupComponent(key: key, label: label, kind: accepts[0], accepts: accepts)
    }
    let h3 = [
      component("checkpoint", "H3 task manifest", ["directory"]),
      component("transformer", "H3 transformer", ["directory", "file"]),
      component("text_encoder", "Qwen3-VL encoder", ["directory"]),
      component("processor", "Qwen3-VL processor", ["directory"]),
      component("tokenizer", "Qwen tokenizer", ["directory", "file"]),
      component("video_vae", "H3 video VAE", ["file", "directory"]),
      component("audio_vae", "H3 audio VAE", ["file", "directory"]),
    ]
    let ltx = [
      component("transformer_path", "LTX 2.5 distilled transformer", ["directory", "file"]),
      component("text_encoder_path", "LTX 2.5 Gemma 4 pack", ["directory", "file"]),
      component("video_vae_path", "LTX 2.5 video VAE", ["file"]),
      component("audio_vae_path", "LTX 2.5 audio VAE and vocoder", ["file"]),
      component("spatial_upscaler_path", "LTX spatial latent upscaler", ["file"]),
    ]
    return [("h3", "MiniMax H3", h3), ("ltx25", "LTX 2.5", ltx)].flatMap { engine, title, fields in
      (engine == "h3" ? [("text", "t2v", "Text to video"),
        ("image", "fflf", "Image to video"), ("reference", "ref2va", "Reference to video")]
        : [("text", "t2v", "Text to video"), ("image", "fflf", "Image to video")])
        .map { suffix, task, label in
          ModelSetupPreset(id: "swift-\(engine)-\(suffix)", name: "\(title) · \(label) · Swift",
            engine: engine, task: task,
            description: "Reuse installed components in place. Text-only setup runs Swift worker preflight now; image and reference tasks run it after clip media is attached.",
            components: fields)
        }
    }
  }

  public static func recipe(preset: ModelSetupPreset, selected: [String: String],
    memoryMode: ModelSetupMemoryMode) throws -> [String: Any] {
    guard catalog().contains(where: { $0.id == preset.id }),
      Set(selected.keys) == Set(preset.components.map(\.key)) else {
      throw StudioError.invalid("Choose one component for every native setup field.")
    }
    let manager = FileManager.default
    var components: [String: Any] = [:]
    for field in preset.components {
      guard let raw = selected[field.key], !raw.isEmpty, raw.hasPrefix("/"), !raw.utf8.contains(0) else {
        throw StudioError.invalid("Select an absolute path for \(field.label).")
      }
      let path = URL(fileURLWithPath: raw).standardizedFileURL.resolvingSymlinksInPath().path
      var directory: ObjCBool = false
      guard manager.fileExists(atPath: path, isDirectory: &directory),
        (field.accepts ?? [field.kind]).contains(directory.boolValue ? "directory" : "file") else {
        throw StudioError.invalid("\(field.label) is missing or has the wrong file type.")
      }
      components[field.key] = path
    }
    let lowMemory = memoryMode == .lowerMemory ||
      (memoryMode == .automatic && ProcessInfo.processInfo.physicalMemory <= 64 * 1_073_741_824)
    let config: [String: Any]
    if preset.engine == "h3" {
      components["task"] = ["t2v": "t2va", "fflf": "fl2va", "ref2va": "ref2va"][preset.task]!
      config = ["width": 384, "height": 256, "duration_seconds": 2.5,
        "steps": 20, "seed": 0, "drop_adaln": true, "resolution_mode": "custom",
        "memory_mode": lowMemory ? "low_memory_bf16" : "normal",
        "projection_backend": "mlx", "sampling_method": "euler"]
    } else {
      config = ["pipeline_mode": "distilled", "width": 768, "height": 512,
        "duration_seconds": 5.0, "frame_rate": 24.0, "seed": 0,
        "stage1_steps": 8, "stage2_steps": 3, "low_memory": true,
        "low_ram_streaming": false]
    }
    return ["format": "weetodd-headless-v2", "engine": preset.engine,
      "candidate": preset.id, "components": components, "config": config,
      "prompt": "A continuous scene with synchronized sound.",
      "conditioning": ["version": 1, "task": preset.task,
        "inputs": [], "audio_policy": "generated"]]
  }

  public static func stage(_ recipe: [String: Any], directory: String) throws -> String {
    let root = URL(fileURLWithPath: directory).standardizedFileURL
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let destination = root.appendingPathComponent("swift-setup-\(UUID().uuidString).json")
    let data = try JSONSerialization.data(withJSONObject: recipe, options: [.prettyPrinted, .sortedKeys])
    guard data.count <= 1024 * 1024 else { throw StudioError.invalid("Model recipe exceeds 1 MiB.") }
    try data.write(to: destination, options: .atomic)
    return destination.path
  }
}

private enum NativeModelInspector {
  private static func document(_ url: URL) throws -> [String: Any] {
    let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
    guard size > 0, size <= 2 * 1024 * 1024,
      let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
      throw StudioError.invalid("Invalid bounded model manifest.")
    }
    return value
  }

  private static func header(_ url: URL) throws -> [String: Any] {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
      throw StudioError.invalid("Truncated model header.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * $1.offset)) }
    guard length > 0, length <= 16 * 1024 * 1024,
      let bytes = try file.read(upToCount: Int(length)), bytes.count == Int(length),
      let value = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw StudioError.invalid("Invalid bounded SafeTensors header.")
    }
    return value
  }

  private static func embedded(_ value: Any?) -> [String: Any]? {
    if let object = value as? [String: Any] { return object }
    if let string = value as? String,
      let data = string.data(using: .utf8), data.count <= 2 * 1024 * 1024 {
      return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
    return nil
  }

  static func matches(_ url: URL, key: String, engine: String, task: String) throws -> Bool {
    let directory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
    if engine == "h3" {
      if key == "checkpoint" {
        guard directory, let info = try? document(url.appendingPathComponent("model_index.json"))["_minimax_h3"] as? [String: Any],
          let partition = info["partition"] as? String, let tasks = info["tasks"] as? [String] else { return false }
        let expected = task == "ref2va" ? "ref2va" : "fl2va"
        return partition == expected && tasks.contains(task == "t2v" ? "t2va" : task == "fflf" ? "fl2va" : task)
      }
      if directory {
        switch key {
        case "transformer":
          guard let config = try? document(url.appendingPathComponent("config.json")) else { return false }
          return config["latents_dim"] as? Int == 24 && config["audio_latents_dim"] as? Int == 32 &&
            config["text_dim"] as? Int == 5120 &&
            FileManager.default.fileExists(atPath: url.appendingPathComponent("paged_manifest.json").path)
        case "text_encoder":
          guard let config = try? document(url.appendingPathComponent("config.json")) else { return false }
          let text = embedded(config["text_config"]) ?? embedded(config["text_encoder"]) ?? config
          return (text["hidden_size"] as? Int ?? text["hidden"] as? Int) == 5120 &&
            (FileManager.default.fileExists(atPath: url.appendingPathComponent("paged_text_encoder_manifest.json").path) ||
              FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder.safetensors").path))
        case "processor":
          guard let config = (try? document(url.appendingPathComponent("processor_config.json"))) ??
            (try? document(url.appendingPathComponent("preprocessor_config.json"))) else { return false }
          return config["processor_class"] as? String == "Qwen3VLProcessor"
        case "tokenizer":
          guard let config = try? document(url.appendingPathComponent("tokenizer_config.json")) else { return false }
          return ["Qwen2Tokenizer", "Qwen2TokenizerFast"].contains(config["tokenizer_class"] as? String ?? "") &&
            FileManager.default.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path)
        default: return false
        }
      }
      guard ["video_vae", "audio_vae"].contains(key) else { return false }
      let tensors = try header(url)
      let metadata = tensors["__metadata__"] as? [String: String] ?? [:]
      return key == "video_vae"
        ? metadata["minimax_h3_video_vae"] != nil && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
        : metadata["minimax_h3_audio_vae"] != nil && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
    }
    if directory {
      guard ["transformer_path", "text_encoder_path"].contains(key),
        let manifest = try? document(url.appendingPathComponent("paged_manifest.json")),
        let fixed = manifest["fixed"] as? [String: Any], let file = fixed["file"] as? String,
        FileManager.default.fileExists(atPath: url.appendingPathComponent(file).path) else { return false }
      let metadata = manifest["metadata"] as? [String: Any] ?? [:]
      if key == "transformer_path" {
        let config = embedded(metadata["config"])
        return manifest["kind"] as? String == "transformer" &&
          (metadata["model_version"] as? String)?.hasPrefix("2.5") == true && config?["transformer"] is [String: Any]
      }
      let gemma = embedded(metadata["gemma_config"])
      return manifest["kind"] as? String == "gemma" && gemma?["model_type"] as? String == "gemma4_unified"
    }
    let tensors = try header(url)
    let metadata = tensors["__metadata__"] as? [String: String] ?? [:]
    let config = embedded(metadata["config"]) ?? [:]
    switch key {
    case "transformer_path":
      return metadata["model_version"]?.hasPrefix("2.5") == true && config["transformer"] is [String: Any] &&
        tensors.keys.contains(where: { $0.hasSuffix("patchify_proj.weight") })
    case "text_encoder_path":
      return embedded(metadata["gemma_config"])?["model_type"] as? String == "gemma4_unified" &&
        tensors.keys.contains("model.embed_tokens.weight")
    case "video_vae_path":
      return config["vae"] is [String: Any] && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
    case "audio_vae_path":
      return tensors.keys.contains(where: { $0.hasPrefix("audio_vae.decoder.") }) &&
        tensors.keys.contains(where: { $0.hasPrefix("vocoder.") })
    case "spatial_upscaler_path":
      return config["_class_name"] as? String == "LatentUpsampler" &&
        config["in_channels"] as? Int == 128 && config["dims"] as? Int == 3 &&
        config["spatial_upsample"] as? Bool == true && config["temporal_upsample"] as? Bool == false
    default: return false
    }
  }
}
