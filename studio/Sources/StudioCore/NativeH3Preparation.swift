import CryptoKit
import Darwin
import Foundation

/// Weight-free preparation for the experimental Swift H3 text, FL2VA and Ref2VA slices.
/// The worker independently admits the final recipe and installed components.
public enum NativeH3Preparation {
  private static let rootKeys: Set<String> = ["format", "engine", "candidate", "components",
    "config", "prompt", "conditioning", "ffmpeg", "block_residency", "negative_prompt",
    "loras"]
  private static let componentKeys: Set<String> = ["checkpoint", "transformer", "text_encoder",
    "processor", "tokenizer", "video_vae", "audio_vae", "task", "loras",
    "vision_encoder", "allow_fl2va_weights_for_ref2va"]
  private static let configKeys: Set<String> = ["width", "height", "duration_seconds", "steps",
    "seed", "drop_adaln", "resolution_mode", "resolution_tier", "aspect_ratio", "memory_mode",
    "attention_chunk_size", "attention_head_chunk_size", "ffn_row_chunk_size",
    "projection_backend", "transformer_backend", "sampling_method",
    "inference_optimization", "paging_cache_gb"]

  private static func unsupported(_ detail: String) -> StudioError {
    .invalid("Swift H3 generation is experimental: \(detail).")
  }
  private static func keyframeIndex(_ attachment: Attachment, duration: Double) -> Int? {
    guard duration.isFinite, (2.5...15).contains(duration) else { return nil }
    var frames = Int((duration * 24).rounded(.toNearestOrEven))
    while frames % 17 != 5 { frames += 1 }
    switch attachment.role {
    case .first: return attachment.time == 0 ? 0 : nil
    // Studio displays the requested editorial interval, not H3's extra
    // alignment frames. Put its last image on the final visible frame.
    case .last: return attachment.time == 0
      ? min(frames - 1, Int(ceil(duration * 24)) - 1) : nil
    case .keyframe:
      guard attachment.time.isFinite, attachment.time >= 0,
        attachment.time <= duration else { return nil }
      let frame = Int((attachment.time * 24).rounded(.toNearestOrEven))
      let lastVisible = min(frames - 1, Int(ceil(duration * 24)) - 1)
      return (0...lastVisible).contains(frame) ? frame : nil
    default: return nil
    }
  }
  private static func data(_ value: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
  }
  private static func fingerprint(_ value: Any) throws -> String {
    "swift-json-v1:" + SHA256.hash(data: try data(value)).map { String(format: "%02x", $0) }.joined()
  }
  private static func canonical(_ path: String) throws -> String {
    let expanded = (path as NSString).expandingTildeInPath
    guard expanded.hasPrefix("/"), !expanded.utf8.contains(0) else {
      throw StudioError.invalid("Select an absolute local path.")
    }
    return URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath().path
  }
  private static func sourceSHA256(_ path: String,
    maxBytes: Int64 = 128 * 1024 * 1024) throws -> String {
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Cannot read an H3 reference.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      (1...maxBytes).contains(status.st_size) else {
      throw StudioError.invalid("H3 reference must be a bounded regular media file.")
    }
    var digest = SHA256()
    var read = 0
    while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
      read += chunk.count
      guard read <= Int(status.st_size) else {
        throw StudioError.invalid("H3 reference changed during preparation.")
      }
      digest.update(data: chunk)
    }
    guard read == Int(status.st_size) else {
      throw StudioError.invalid("H3 reference changed during preparation.")
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private static func read(_ path: String) throws -> Data {
    let fd = Darwin.open(try canonical(path), O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Cannot read the H3 profile. Relink it.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size <= 1024 * 1024 else {
      throw StudioError.invalid("The H3 profile must be a regular JSON file under 1 MiB.")
    }
    let bytes = try file.read(upToCount: 1024 * 1024 + 1) ?? Data()
    guard bytes.count <= 1024 * 1024 else {
      throw StudioError.invalid("The H3 profile grew beyond 1 MiB.")
    }
    return bytes
  }
  private static func profile(_ path: String) throws -> [String: Any] {
    guard let recipe = try JSONSerialization.jsonObject(with: read(path)) as? [String: Any],
      recipe["format"] as? String == "weetodd-headless-v2", recipe["engine"] as? String == "h3" else {
      throw StudioError.invalid("Select an H3 headless v2 profile.")
    }
    return recipe
  }
  private static func supportedTurboLoRA(_ value: Any?) -> Bool {
    guard let entries = value as? [Any], entries.count <= 1 else { return value == nil }
    guard let entry = entries.first else { return true }
    guard let pair = entry as? [Any], pair.count == 2,
      let path = pair[0] as? String, path.hasPrefix("/"),
      let number = pair[1] as? NSNumber,
      CFGetTypeID(number) != CFBooleanGetTypeID() else { return false }
    let strength = number.doubleValue
    return strength.isFinite && (0...2).contains(strength)
  }
  private static func emptyProfileLoRA(_ value: Any?) -> Bool {
    value == nil || (value as? [Any])?.isEmpty == true
  }
  private static func rootTurboLoRA(_ value: Any?) -> [Any]? {
    guard let object = value as? [String: Any],
      Set(object.keys) == Set(["adapters"]),
      let adapters = object["adapters"] as? [[String: Any]],
      adapters.count == 1,
      Set(adapters[0].keys).isSubset(of: ["path", "strength", "profile", "qkv_layout"]),
      let path = adapters[0]["path"] as? String, path.hasPrefix("/"),
      let strength = adapters[0]["strength"] as? NSNumber,
      CFGetTypeID(strength) != CFBooleanGetTypeID(),
      strength.doubleValue.isFinite, (0...2).contains(strength.doubleValue),
      adapters[0]["profile"] as? String == "turbo",
      adapters[0]["qkv_layout"] as? String == "contiguous_qkv" else { return nil }
    return [path, strength.doubleValue]
  }
  private static func supported(_ recipe: [String: Any]) -> Bool {
    guard Set(recipe.keys).isSubset(of: rootKeys),
      (recipe["negative_prompt"] as? String ?? "").isEmpty,
      (recipe["block_residency"] as? String ?? "checkpoint_default") == "checkpoint_default",
      let components = recipe["components"] as? [String: Any],
      Set(components.keys).isSubset(of: componentKeys),
      let modelTask = components["task"] as? String,
      ["t2va", "fl2va", "ref2va"].contains(modelTask),
      (modelTask != "ref2va" || (components["vision_encoder"] == nil ||
        (components["vision_encoder"] as? String)?.hasPrefix("/") == true)
        && (components["allow_fl2va_weights_for_ref2va"] == nil ||
          components["allow_fl2va_weights_for_ref2va"] as? Bool == false)),
      (modelTask != "t2va" || (components["vision_encoder"] == nil &&
        components["allow_fl2va_weights_for_ref2va"] == nil)),
      (modelTask != "fl2va" ||
        (components["vision_encoder"] == nil ||
          (components["vision_encoder"] as? String)?.hasPrefix("/") == true)
        && components["allow_fl2va_weights_for_ref2va"] == nil),
      supportedTurboLoRA(components["loras"]),
      (recipe["loras"] == nil || (emptyProfileLoRA(components["loras"]) &&
        rootTurboLoRA(recipe["loras"]) != nil)),
      let config = recipe["config"] as? [String: Any],
      Set(config.keys).isSubset(of: configKeys),
      (config["drop_adaln"] as? Bool ?? true),
      (config["sampling_method"] as? String ?? "euler") == "euler",
      ["auto", "mlx"].contains(config["projection_backend"] as? String ?? "mlx"),
      (config["transformer_backend"] as? String ?? "mlx") == "mlx",
      (config["resolution_mode"] as? String ?? "custom") == "custom",
      (config["resolution_tier"] as? String ?? "custom") == "custom",
      (config["aspect_ratio"] as? String ?? "custom") == "custom",
      ["normal", "low_memory_bf16"].contains(config["memory_mode"] as? String ?? "normal"),
      (config["attention_chunk_size"] as? String ?? "automatic") == "automatic",
      ["automatic", "disabled"].contains(config["attention_head_chunk_size"] as? String ?? "automatic"),
      (config["ffn_row_chunk_size"] as? String ?? "automatic") == "automatic",
      (config["inference_optimization"] as? String ?? "off") == "off",
      (config["paging_cache_gb"] as? Double ?? 0) == 0,
      let conditioning = recipe["conditioning"] as? [String: Any],
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      conditioning["version"] as? Int == 1,
      conditioning["task"] as? String ==
        (modelTask == "ref2va" ? "ref2va" : modelTask == "fl2va" ? "fflf" : "t2v"),
      (conditioning["inputs"] as? [Any] ?? []).isEmpty,
      (conditioning["audio_policy"] as? String ?? "generated") == "generated" else { return false }
    return ["transformer", "text_encoder", "tokenizer", "video_vae", "audio_vae"]
      .allSatisfy { key in (components[key] as? String)?.hasPrefix("/") == true }
  }
  private static func descriptor(_ recipe: [String: Any]) -> [String: Any] {
    let config = recipe["config"] as? [String: Any] ?? [:]
    let modelTask = (recipe["components"] as? [String: Any])?["task"] as? String
    let task = modelTask == "ref2va" ? "ref2va" : modelTask == "fl2va" ? "fflf" : "t2v"
    return ["supportedTasks": [task], "controls": [
      "evaluations": max(0, (config["steps"] as? Int ?? 20) - 1),
      "stepsEditable": true, "refinementStepsEditable": false,
      "cfgEditable": false, "shiftEditable": false,
      "stepsExplanation": "H3 uses one fewer model evaluation than sigma grid points.",
      "cfgExplanation": "Guidance is distilled into H3.",
      "shiftExplanation": "Swift H3 does not expose a Shift override."],
      "presets": [["id": "custom", "name": "Custom", "description": "Use the admitted profile settings."]]]
  }
  public static func catalog(directory: String) throws -> [[String: Any]] {
    guard !directory.isEmpty else { return [] }
    let root = URL(fileURLWithPath: try canonical(directory))
    guard FileManager.default.fileExists(atPath: root.path) else { return [] }
    let urls = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard urls.count <= 1000 else { throw StudioError.invalid("Limit the model profile directory to 1,000 recipes.") }
    return try urls.compactMap { url in
      try Task.checkCancellation()
      guard let recipe = try? profile(url.path), supported(recipe) else { return nil }
      return ["id": try canonical(url.path),
        "name": url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " "),
        "engine": "h3", "task": (recipe["components"] as? [String: Any])?["task"] as? String == "ref2va"
          ? "ref2va" : (recipe["components"] as? [String: Any])?["task"] as? String == "fl2va"
            ? "fflf" : "t2v", "generation": descriptor(recipe)]
    }
  }
  private static func resolve(_ request: [String: Any]) throws -> (StudioProject, Clip, [String: Any], String, [String: Any]) {
    guard let projectValue = request["project"], let runtime = request["runtime"] as? [String: Any],
      let clipID = request["clipID"] as? String else {
      throw StudioError.invalid("Missing Studio H3 preparation request.")
    }
    let project = try JSONDecoder().decode(StudioProject.self, from: data(projectValue))
    guard let clip = project.clips.first(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      clip.engine == .h3 else { throw StudioError.invalid("Select an H3 clip.") }
    guard clip.continuityMode == "independent", !project.isContinuousSceneMember(clip),
      clip.audioDriverSelection == nil, clip.musicSource == nil,
      clip.extensionDirection.isEmpty, clip.extensionSource.isEmpty else {
      throw unsupported("continuity, music drivers and extension are not ported")
    }
    let task = clip.inferredTask
    guard ["t2v", "t2va", "i2v", "fflf", "ref2va"].contains(task) else {
      throw unsupported("task \(clip.inferredTask) is not ported")
    }
    let supportedRoles: Set<MediaRole> = task == "ref2va" ? [.reference, .lora]
      : ["i2v", "fflf"].contains(task) ? [.first, .last, .keyframe, .lora] : [.lora]
    let enabledFrames = clip.attachments.filter {
      $0.isEnabled && [.first, .last, .keyframe].contains($0.role)
    }
    let frameIndices = enabledFrames.compactMap { keyframeIndex($0, duration: clip.duration) }
    guard clip.attachments.allSatisfy({ supportedRoles.contains($0.role) }),
      clip.attachments.filter({ $0.role == .lora && $0.isEnabled }).count <= 1,
      task != "ref2va" || (1...12).contains(clip.attachments.filter({ $0.role == .reference && $0.isEnabled }).count),
      !["i2v", "fflf"].contains(task) ||
        ((1...8).contains(frameIndices.count) && frameIndices.count == enabledFrames.count &&
          Set(frameIndices).count == frameIndices.count &&
          (task != "i2v" || (frameIndices.count == 1 && frameIndices[0] == 0))) else {
      throw unsupported("this task needs one to eight unique timed images or one to twelve references and at most one Turbo LoRA")
    }
    let selection = clip.generationSelection
    guard selection?.refinementSteps == nil, selection?.cfg == nil, selection?.shift == nil,
      selection?.memoryPolicy == nil || selection?.memoryPolicy == "recipe",
      selection?.projectionBackend == nil || ["auto", "mlx"].contains(selection!.projectionBackend!),
      selection?.transformerBackend == nil || selection?.transformerBackend == "mlx",
      clip.h3PagingCacheGB == nil || clip.h3PagingCacheGB == 0,
      clip.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw unsupported("guidance, memory, backend or negative-prompt overrides cannot be executed")
    }
    let profiles = try catalog(directory: runtime["profilesDirectory"] as? String ?? "")
    let catalogTask = task == "ref2va" ? "ref2va" : ["i2v", "fflf"].contains(task) ? "fflf" : "t2v"
    guard let chosen = profiles.first(where: {
      $0["task"] as? String == catalogTask &&
        (clip.profileID == "auto" || $0["id"] as? String == clip.profileID)
    }),
      let path = chosen["id"] as? String else {
      throw unsupported("no compatible \(catalogTask) profile is installed")
    }
    let recipe = try profile(path)
    guard supported(recipe) else { throw unsupported("the selected profile changed or contains unported settings") }
    return (project, clip, runtime, path, recipe)
  }
  public static func compose(request: [String: Any]) throws -> [String: Any] {
    let (project, clip, runtime, path, original) = try resolve(request)
    let globalAssets = try JSONDecoder().decode([MediaAsset].self,
      from: data(request["globalAssets"] ?? []))
    let availableAssets = project.assets + globalAssets
    var recipe = original
    let text = clip.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { throw StudioError.invalid("Write a prompt before preparing the render.") }
    let prompt: String
    if text.hasPrefix("integrated_multimodal_description:") || text.contains("[video continuation") {
      prompt = text
    } else if ["integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"]
      .allSatisfy({ text.contains($0) }) {
      prompt = text
    } else {
      prompt = "integrated_multimodal_description: [Shot 1] \(text)\n\noverall_soundscape: \(clip.soundscape)\n\nnon_diegetic_music: \(clip.music)"
    }
    guard clip.duration.isFinite, (2.5...15).contains(clip.duration),
      (0...Int(UInt32.max)).contains(clip.seed),
      clip.generationWidth > 0, clip.generationHeight > 0,
      clip.generationWidth % 32 == 0, clip.generationHeight % 32 == 0,
      clip.generationWidth <= 4096, clip.generationHeight <= 4096 else {
      throw unsupported("duration, dimensions or seed are outside the H3 contract")
    }
    var config = recipe["config"] as! [String: Any]
    let steps: Int
    if let evaluations = clip.generationSelection?.steps {
      guard (1...99).contains(evaluations) else {
        throw unsupported("model evaluations must be between 1 and 99")
      }
      steps = evaluations + 1
    } else {
      steps = config["steps"] as? Int ?? 20
    }
    guard (2...100).contains(steps) else { throw unsupported("sigma grid points must be between 2 and 100") }
    config["width"] = clip.generationWidth; config["height"] = clip.generationHeight
    config["duration_seconds"] = clip.duration; config["seed"] = clip.seed
    config["steps"] = steps
    if let backend = clip.generationSelection?.projectionBackend { config["projection_backend"] = backend }
    recipe["config"] = config; recipe["prompt"] = prompt
    var components = recipe["components"] as! [String: Any]
    var loras = components["loras"] as? [[Any]] ?? []
    if let legacyStack = recipe.removeValue(forKey: "loras") {
      guard loras.isEmpty, let adapter = rootTurboLoRA(legacyStack) else {
        throw unsupported("the profile adapter cannot be mapped to the Swift Turbo reader")
      }
      loras.append(adapter)
    }
    if let tokenizer = components["tokenizer"] as? String {
      let tokenFile = URL(fileURLWithPath: tokenizer).appendingPathComponent("tokenizer.json")
      if FileManager.default.fileExists(atPath: tokenFile.path) {
        components["tokenizer"] = try canonical(tokenFile.path)
      }
    }
    var referenceInputs: [[String: Any]] = []
    let endpointAttachments = clip.attachments.filter {
      $0.isEnabled && Set<MediaRole>([.first, .last, .keyframe]).contains($0.role)
    }.sorted {
      keyframeIndex($0, duration: clip.duration)! < keyframeIndex($1, duration: clip.duration)!
    }
    let orderedAttachments = ["i2v", "fflf"].contains(clip.inferredTask)
      ? endpointAttachments + clip.attachments.filter { $0.role == .lora }
      : clip.attachments
    for attachment in orderedAttachments where attachment.isEnabled {
      guard let asset = availableAssets.last(where: { $0.id == attachment.assetID }) else {
        throw StudioError.invalid("Relink a missing H3 attachment.")
      }
      if Set<MediaRole>([.reference, .first, .last, .keyframe]).contains(attachment.role) {
        let isVideoReference = attachment.role == .reference &&
          (asset.kind == .video || asset.kind == .sequence)
        let isAudioReference = attachment.role == .reference && asset.kind == .audio
        guard (asset.kind == .image || isVideoReference || isAudioReference),
          attachment.strength == 1,
          (attachment.role == .keyframe || attachment.time == 0),
          attachment.referenceRole == nil,
          attachment.referencePriority == nil,
          attachment.referenceFrames == nil,
          attachment.referenceSizePolicy == nil,
          attachment.attentionStrength == nil else {
          throw unsupported("H3 image, video and audio references require full strength and no timing or specialized controls")
        }
        let imagePath = try canonical(asset.path)
        guard FileManager.default.isReadableFile(atPath: imagePath) else {
          throw StudioError.invalid("Relink the H3 reference: \(asset.name)")
        }
        var input: [String: Any] = ["id": attachment.id.uuidString,
          "kind": isVideoReference ? "video" : isAudioReference ? "audio" : "image",
          "role": attachment.role.rawValue, "path": imagePath,
          "strength": 1.0,
          "sha256": try sourceSHA256(imagePath,
            maxBytes: isVideoReference ? 4 * 1024 * 1024 * 1024
              : isAudioReference ? 1024 * 1024 * 1024 : 128 * 1024 * 1024)]
        if attachment.role == .first { input["frame_index"] = 0 }
        if attachment.role == .last {
          input["frame_index"] = keyframeIndex(attachment, duration: clip.duration)!
        }
        if attachment.role == .keyframe {
          input["frame_index"] = keyframeIndex(attachment, duration: clip.duration)!
        }
        referenceInputs.append(input)
        continue
      }
      guard loras.isEmpty else {
        throw unsupported("select only one Turbo LoRA, including the profile adapter")
      }
      try LoRAMember(asset: asset, strength: attachment.strength).validate(for: .h3)
      guard asset.loraAdalnInputGrid == nil,
        asset.loraLayout == nil || asset.loraLayout == "contiguous_qkv",
        asset.loraProfile == nil || asset.loraProfile == "turbo" else {
        throw unsupported("this H3 LoRA requires a different layout or adapter profile")
      }
      let adapterPath = try canonical(asset.path)
      guard FileManager.default.isReadableFile(atPath: adapterPath) else {
        throw StudioError.invalid("Relink the H3 Turbo LoRA: \(asset.name)")
      }
      loras.append([adapterPath, attachment.strength])
    }
    if !loras.isEmpty { components["loras"] = loras }
    recipe["components"] = components
    if clip.inferredTask == "ref2va" {
      let imageCount = referenceInputs.filter { $0["kind"] as? String == "image" }.count
      let videoCount = referenceInputs.filter { $0["kind"] as? String == "video" }.count
      let audioCount = referenceInputs.count - imageCount - videoCount
      guard (1...12).contains(referenceInputs.count), imageCount + videoCount > 0,
        imageCount <= 9, videoCount <= 3, audioCount <= 3 else {
        throw unsupported("Ref2VA needs a visual source and allows at most nine images, three videos and three audio references")
      }
      recipe["conditioning"] = ["version": 1, "task": "ref2va",
        "inputs": referenceInputs, "audio_policy": "generated"]
    } else if ["i2v", "fflf"].contains(clip.inferredTask) {
      recipe["conditioning"] = ["version": 1, "task": "fflf",
        "inputs": referenceInputs, "audio_policy": "generated"]
    }
    let configuredFFmpeg = runtime["ffmpegPath"] as? String ?? ""
    let ffmpeg = configuredFFmpeg.isEmpty ? [recipe["ffmpeg"] as? String ?? "",
      "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].first {
        !$0.isEmpty && FileManager.default.isExecutableFile(atPath: $0)
      } ?? "" : configuredFFmpeg
    guard !ffmpeg.isEmpty, FileManager.default.isExecutableFile(atPath: ffmpeg) else {
      throw StudioError.invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    recipe["ffmpeg"] = try canonical(ffmpeg)
    let report: [String: Any] = ["profile": URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
      "generation": descriptor(recipe), "resolvedFingerprint": try fingerprint(recipe),
      "selectionFingerprint": try fingerprint(original),
      "task": clip.inferredTask == "ref2va" ? "ref2va"
        : ["i2v", "fflf"].contains(clip.inferredTask) ? "fflf" : "t2v", "nativeFPS": 24,
      "nativePreparation": "swift", "productionQualified": false,
      "movieSettings": try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip.settings(in: project))),
      "conditioning": ["inputs": referenceInputs.count]]
    return ["recipe": recipe, "report": report]
  }
  public static func describe(request: [String: Any]) throws -> [String: Any] {
    let (_, _, _, path, original) = try resolve(request)
    var errors: [String] = [], resolved = ""
    var content = original
    do {
      let result = try compose(request: request)
      content = result["recipe"] as! [String: Any]
      resolved = (result["report"] as? [String: Any])?["resolvedFingerprint"] as? String ?? ""
    } catch { errors.append(error.localizedDescription) }
    let components = content["components"] as? [String: Any] ?? [:]
    var sources = [path] + components.values.compactMap { $0 as? String }.filter { $0.hasPrefix("/") }
    let inputs = (content["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]] ?? []
    sources += inputs.compactMap { $0["path"] as? String }
    for pair in components["loras"] as? [[Any]] ?? [] {
      if let adapterPath = pair.first as? String { sources.append(adapterPath) }
    }
    for dependency in sources {
      for name in ["paged_manifest.json", "model_identity.json", "conversion_provenance.json"] {
        let file = URL(fileURLWithPath: dependency).appendingPathComponent(name).path
        if FileManager.default.fileExists(atPath: file) { sources.append(file) }
      }
    }
    return ["profileID": path, "generation": descriptor(content), "fingerprint": resolved,
      "selectionFingerprint": try fingerprint(original), "sourcePaths": Array(Set(sources)).sorted(),
      "warnings": ["Swift H3 generation is experimental and not production qualified."],
      "readinessErrors": errors]
  }
  public static func prepare(request: [String: Any], destination: URL) throws -> [String: Any] {
    let result = try compose(request: request)
    let content = result["recipe"] as! [String: Any]
    let parent = destination.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".h3-prepare-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: staging) }
    try data(content).write(to: staging.appendingPathComponent("recipe.json"), options: .withoutOverwriting)
    try data(request).write(to: staging.appendingPathComponent("editor-request.json"), options: .withoutOverwriting)
    try Task.checkCancellation()
    try FileManager.default.moveItem(at: staging, to: destination)
    return ["recipePath": destination.appendingPathComponent("recipe.json").path,
      "prompt": content["prompt"]!, "report": result["report"]!]
  }
}
