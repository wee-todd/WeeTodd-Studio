import Foundation

/// Captures an image draft into the same H3 recipe transport used by video.
/// Component/header admission remains in the shared worker, before inference.
public enum NativeH3ImagePreparation {
  public static func recipe(draft: DrawThingsImageDraft, requestID: UUID, seed: UInt32) throws -> Data {
    guard draft.executionProvider == .nativeH3 else { throw StudioError.invalid("Choose Local H3 first.") }
    if let issue = draft.managedCharacterPromptIssue ?? draft.characterSheetRequestIssue ?? draft.imageInputIssue {
      throw StudioError.invalid(issue)
    }
    guard !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      (1...100).contains(draft.steps), draft.guidance == 1, draft.strength == 1,
      draft.sampler == nil, draft.shift == nil, draft.negativePrompt.isEmpty,
      !draft.loras.contains(where: \.isEnabled) else {
      throw StudioError.invalid("H3 images use CFG 1 and the selected profile's sampler. Clear unsupported overrides and use the native H3 LoRA stack.")
    }
    guard (32...1920).contains(draft.width), (32...1920).contains(draft.height),
      draft.width.isMultiple(of: 32), draft.height.isMultiple(of: 32),
      draft.width * draft.height <= 1920 * 1088 else {
      throw StudioError.invalid("H3 images require multiples of 32, sides at most 1920 and a 1920 × 1088 pixel budget.")
    }
    guard draft.modelID.hasPrefix("/"), !draft.modelID.utf8.contains(0) else {
      throw StudioError.invalid("Select an installed H3 model profile.")
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: draft.modelID)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      let size = attributes[.size] as? NSNumber, (1...1024 * 1024).contains(size.intValue) else {
      throw StudioError.invalid("Select a regular H3 profile under 1 MiB.")
    }
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: draft.modelID))
    defer { try? handle.close() }
    let bytes = try handle.read(upToCount: 1024 * 1024 + 1) ?? Data()
    guard bytes.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
      root["format"] as? String == "weetodd-headless-v2", root["engine"] as? String == "h3",
      Set(root.keys).isSubset(of: ["format", "engine", "candidate", "components", "config", "prompt",
        "conditioning", "ffmpeg", "block_residency", "negative_prompt", "loras"]),
      var components = root["components"] as? [String: Any],
      let task = components["task"] as? String, ["t2va", "fl2va", "ref2va"].contains(task),
      var config = root["config"] as? [String: Any],
      (config["attention_policy"] as? String ?? "dense") == "dense",
      (config["transformer_backend"] as? String ?? "mlx") == "mlx",
      (components["allow_fl2va_weights_for_ref2va"] as? Bool ?? false) == false else {
      throw StudioError.invalid("Choose an ordinary H3 model profile without video-only controls, FastH3, VDN, or alternative attention.")
    }
    let settings = draft.h3Image ?? H3ImageSettings()
    if let tokenizer = components["tokenizer"] as? String {
      let file = URL(fileURLWithPath: tokenizer).appendingPathComponent("tokenizer.json")
      if FileManager.default.fileExists(atPath: file.path) {
        components["tokenizer"] = file.standardizedFileURL.resolvingSymlinksInPath().path
      }
    }
    guard components["loras"] == nil || components["loras"] is [[Any]] else {
      throw StudioError.invalid("The H3 profile LoRA stack must contain valid path/strength pairs.")
    }
    guard settings.loras.count <= 8, Set(settings.loras.map(\.fileKey)).count == settings.loras.count else {
      throw StudioError.invalid("Use at most eight distinct native H3 LoRAs.")
    }
    for member in settings.loras { try member.validate(for: .h3) }
    let attachments = settings.loras.map { member -> Attachment in
      var value = Attachment(assetID: member.asset.id, role: .lora)
      value.strength = member.strength; value.enabled = member.enabled; value.h3LoRA = member.h3Settings
      return value
    }
    let stack = try NativeH3LoRAComposition.compose(componentPairs: components["loras"] as? [[Any]] ?? [],
      rootStack: root["loras"], attachments: attachments, assets: settings.loras.map(\.asset),
      schedulePoints: draft.steps + 1, explicitEvaluations: draft.steps,
      samplingMethod: config["sampling_method"] as? String ?? "euler")
    if let descriptors = stack.descriptors { root["loras"] = descriptors; components["loras"] = [] }
    else { root.removeValue(forKey: "loras"); components["loras"] = stack.pairs }
    for input in ([draft.canvas].compactMap { $0 } + draft.moodboard) where input.enabled {
      guard input.strength.isFinite, (0...1).contains(input.strength) else {
        throw StudioError.invalid("Image reference strengths must be finite from 0–100%.")
      }
    }
    let active = draft.activeImageInputs
    guard active.isEmpty || task == "ref2va" else {
      throw StudioError.invalid("Choose an H3 Ref2VA model to use the canvas or mood-board images as references.")
    }
    let inputs = try active.map { value -> [String: Any] in
      guard value.input.strength == 1 else {
        throw StudioError.invalid("H3 image references use full strength. Set active reference strengths to 100%.")
      }
      let source = try value.input.canonical(role: "reference")
      return ["id": value.input.id.uuidString, "kind": "image", "role": "reference",
        "path": source["path"]!, "sha256": source["sha256"]!, "strength": 1.0]
    }
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": inputs.isEmpty ? "t2v" : "ref2va",
      "inputs": inputs, "audio_policy": "generated"]
    config["width"] = draft.width; config["height"] = draft.height
    config["steps"] = stack.schedulePoints; config["seed"] = seed
    config["duration_seconds"] = settings.decoderMode == .qualityFiveFrames ? 5.0 / 24 : 1.0 / 24
    root["config"] = config
    let prompt = draft.prompt
    root["prompt"] = prompt.contains("integrated_multimodal_description:") || prompt.contains("subject_definitions:")
      ? prompt : "integrated_multimodal_description: [Shot 1] \(prompt)\noverall_soundscape: None.\nnon_diegetic_music: None."
    var still: [String: Any] = ["version": 1, "request_id": requestID.uuidString,
      "mode": settings.decoderMode.rawValue, "frame_selection": settings.frameSelection.rawValue]
    if settings.decoderMode == .singleLatent {
      guard settings.imageVAEPath.hasPrefix("/"), !settings.imageVAEPath.utf8.contains(0) else {
        throw StudioError.invalid("Link or prepare the single-image H3 VAE.")
      }
      still["image_vae"] = settings.imageVAEPath
    }
    root["still_image"] = still
    return try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
  }
}
