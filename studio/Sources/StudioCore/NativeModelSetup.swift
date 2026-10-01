import Foundation

/// Weight-free setup for the Swift audiovisual workers. This deliberately offers only
/// ordinary profiles; specialized adapters need their own validated contracts.
public enum NativeModelSetup {
  public static let engines: Set<String> = ["h3", "ltx25"]

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
