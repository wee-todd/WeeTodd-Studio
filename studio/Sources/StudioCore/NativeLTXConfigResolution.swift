import Foundation

/// Resolves installed bindings into the existing native setup contract. This
/// inspects bounded manifests/headers and never loads or converts tensor data.
enum NativeLTXConfigResolution {
  struct Result {
    var profile: String
    var recipe: [String: Any]
    var sourcePaths: [String]
  }

  static func resolve(clip: Clip, project: StudioProject,
    sources: [(path: String, recipe: [String: Any])]) throws -> Result {
    let selection = clip.generationSelection
    let binding = selection?.ltx25AdapterBinding
    if let binding {
      try LTX25GenerationConfig(name: "Selected adapter", selection: selection!).validate()
      let current = try NativeModelSetup.adapterBinding(at: URL(fileURLWithPath: binding.path), kind: binding.kind)
      guard current.supportsVoiceReferences == binding.supportsVoiceReferences else {
        throw StudioError.invalid("This adapter's reference capabilities changed. Select it again from the LoRA library.")
      }
    }
    let mode = selection?.ltx25Guidance?.mode
    let dev = mode != nil
    let dfr = selection?.ltx25DFR
    let automatic = selection?.ltx25AutomaticDuration != nil
    let presetID: String
    if let binding, binding.task != nil {
      presetID = binding.kind.presetID
    } else if dfr?.enabled == true {
      presetID = dfr!.temporalRounds == 0 ? "swift-ltx25-dfr-spatial" : "swift-ltx25-dfr-temporal-\(dfr!.temporalRounds)"
    } else if automatic {
      presetID = "swift-ltx25-auto-duration" + (mode.map { "-" + $0.rawValue } ?? "")
    } else { presetID = "swift-ltx25-" + (mode?.rawValue ?? "text") }
    guard let preset = NativeModelSetup.catalog().first(where: { $0.id == presetID }) else {
      throw StudioError.invalid("The selected LTX 2.5 config has no supported component contract.")
    }
    let transformerSource = sources.first { source in
      let components = source.recipe["components"] as? [String: Any] ?? [:]
      guard let path = (components["transformer_path"] ?? components["dev_transformer_path"]) as? String,
        path.hasPrefix("/") else { return false }
      return NativeModelSetup.matchesLTX25Transformer(URL(fileURLWithPath: path), dev: dev)
    }
    // Prefer shared components from the chosen transformer binding. Otherwise
    // reuse another registered stack's shared encoder/decoder paths in place.
    let ordered = transformerSource.map { first in
      [first] + sources.filter { $0.path != first.path }
    } ?? sources
    var linked: [String: String] = [:], used: Set<String> = [], missing: [String] = []
    for field in preset.components {
      if let binding, field.key == binding.kind.componentKey {
        linked[field.key] = binding.path; used.insert(binding.path); continue
      }
      let transformer = ["transformer_path", "dev_transformer_path"].contains(field.key)
      let candidate = (transformer ? transformerSource.map { [$0] } ?? [] : ordered).first { source in
        guard let path = component(field.key, source: source.recipe), path.hasPrefix("/"),
          FileManager.default.fileExists(atPath: path) else { return false }
        if field.key == "video_vae_path", selection?.ltx25DiffusionVAE != nil {
          return (try? NativeLTXDiffusionVAE.validate(at: URL(fileURLWithPath: path))) != nil
        }
        return NativeModelSetup.matchesLTX25Component(URL(fileURLWithPath: path), key: field.key)
      }
      if let candidate, let path = component(field.key, source: candidate.recipe) {
        linked[field.key] = path; used.insert(candidate.path)
      } else {
        missing.append(transformer ? "LTX 2.5 \(dev ? "unmerged Dev paged" : "distilled") transformer"
          : field.key == "text_encoder_path" ? "LTX 2.5 Gemma 4 text encoder"
          : field.key == "video_vae_path" && selection?.ltx25DiffusionVAE != nil ? "LTX 2.5 Diffusion VAE" : field.label)
      }
    }
    guard missing.isEmpty else {
      throw StudioError.invalid("Missing compatible components for \(selection?.ltx25ConfigName ?? (dev ? "Dev" : "Distilled")): \(missing.joined(separator: ", ")). Link or download these components in Model Setup; installed weights remain in place.")
    }
    var recipe = try NativeModelSetup.recipe(preset: preset, selected: linked, memoryMode: .automatic)
    var config = recipe["config"] as! [String: Any]
    config["frame_rate"] = clip.settings(in: project).fps
    recipe["config"] = config
    if let ffmpeg = ordered.compactMap({ $0.recipe["ffmpeg"] as? String }).first(where: { !$0.isEmpty }) {
      recipe["ffmpeg"] = ffmpeg
    }
    return Result(profile: transformerSource!.path, recipe: recipe, sourcePaths: used.sorted())
  }

  private static func component(_ key: String, source: [String: Any]) -> String? {
    let components = source["components"] as? [String: Any] ?? [:]
    if key == "dev_transformer_path" { return (components[key] ?? components["transformer_path"]) as? String }
    return (components[key] ?? (source["config"] as? [String: Any])?[key]) as? String
  }
}
