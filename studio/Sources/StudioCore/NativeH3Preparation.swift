import CryptoKit
import AVFoundation
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
    "vision_encoder", "allow_fl2va_weights_for_ref2va", "fun_controlnet"]
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
  private static func supportedLoRA(_ value: Any?) -> Bool {
    guard let entries = value as? [[Any]], entries.count <= 4 else { return value == nil }
    var paths = Set<String>()
    for pair in entries {
      guard pair.count == 2,
        let path = pair[0] as? String, path.hasPrefix("/"),
        !path.utf8.contains(0),
        let number = pair[1] as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue.isFinite,
        (0...2).contains(number.doubleValue),
        paths.insert(URL(fileURLWithPath: path).standardizedFileURL.path).inserted
      else { return false }
    }
    return true
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
      (components["fun_controlnet"] == nil || (modelTask == "t2va" &&
        (components["fun_controlnet"] as? String)?.hasPrefix("/") == true &&
        emptyProfileLoRA(components["loras"]) && recipe["loras"] == nil)),
      supportedLoRA(components["loras"]),
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
        (components["fun_controlnet"] != nil ? "control" :
          modelTask == "ref2va" ? "ref2va" : modelTask == "fl2va" ? "fflf" : "t2v"),
      (conditioning["inputs"] as? [Any] ?? []).isEmpty,
      (conditioning["audio_policy"] as? String ?? "generated") == "generated" else { return false }
    return ["transformer", "text_encoder", "tokenizer", "video_vae", "audio_vae"]
      .allSatisfy { key in (components[key] as? String)?.hasPrefix("/") == true }
  }
  private static func descriptor(_ recipe: [String: Any]) -> [String: Any] {
    let config = recipe["config"] as? [String: Any] ?? [:]
    let modelTask = (recipe["components"] as? [String: Any])?["task"] as? String
    let task = (recipe["components"] as? [String: Any])?["fun_controlnet"] != nil ? "control"
      : modelTask == "ref2va" ? "ref2va" : modelTask == "fl2va" ? "fflf" : "t2v"
    return ["supportedTasks": task == "ref2va" ? ["ref2va", "a2v", "extension"] : [task], "controls": [
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
        "engine": "h3", "task": (recipe["components"] as? [String: Any])?["fun_controlnet"] != nil ? "control"
          : (recipe["components"] as? [String: Any])?["task"] as? String == "ref2va"
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
    let task = clip.inferredTask
    guard ["independent","motion"].contains(clip.continuityMode), !project.isContinuousSceneMember(clip),
      (clip.audioDriverSelection == nil || task == "a2v"), clip.musicSource == nil,
      clip.extensionDirection.isEmpty, clip.extensionSource.isEmpty else {
      throw unsupported("continuity, planned music and extension are not ported")
    }
    guard ["t2v", "t2va", "i2v", "fflf", "ref2va", "a2v", "control"].contains(task) else {
      throw unsupported("task \(clip.inferredTask) is not ported")
    }
    let supportedRoles: Set<MediaRole> = task == "control" ? [.control]
      : task == "ref2va" ? [.reference, .lora]
      : task == "a2v" ? [.audioDriver, .first, .lora]
      : ["i2v", "fflf"].contains(task) ? [.first, .last, .keyframe, .lora] : [.lora]
    let enabledFrames = clip.attachments.filter {
      $0.isEnabled && [.first, .last, .keyframe].contains($0.role)
    }
    let frameIndices = enabledFrames.compactMap { keyframeIndex($0, duration: clip.duration) }
    let audioDrivers = clip.attachments.filter { $0.role == .audioDriver && $0.isEnabled }
    let openingImages = clip.attachments.filter { $0.role == .first && $0.isEnabled }
    guard clip.attachments.allSatisfy({ supportedRoles.contains($0.role) }),
      (task != "a2v" || (audioDrivers.count == 1 && openingImages.count <= 1)),
      (task != "control" || (clip.continuityMode == "independent" &&
        clip.attachments.filter({ $0.role == .control && $0.isEnabled }).count == 1)),
      clip.attachments.filter({ $0.role == .lora && $0.isEnabled }).count <= 4,
      task != "ref2va" || (1...12).contains(clip.attachments.filter({ $0.role == .reference && $0.isEnabled }).count),
      !["i2v", "fflf"].contains(task) ||
        ((1...8).contains(frameIndices.count) && frameIndices.count == enabledFrames.count &&
          Set(frameIndices).count == frameIndices.count &&
          (task != "i2v" || (frameIndices.count == 1 && frameIndices[0] == 0))) else {
      throw unsupported("this task needs one to eight unique timed images or one to twelve references and at most four Turbo LoRAs")
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
    let catalogTask = task == "control" ? "control" : ["ref2va", "a2v"].contains(task) ? "ref2va"
      : ["i2v", "fflf"].contains(task) ? "fflf" : "t2v"
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
  private static func frameRequest(_ request:[String:Any],imagePath:String?=nil) throws
    -> (request:[String:Any],source:NativeLTXFrameSource)? {
    guard let raw=request["project"],let clipID=request["clipID"] as? String else { return nil }
    var project=try JSONDecoder().decode(StudioProject.self,from:data(raw))
    guard let index=project.clips.firstIndex(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      project.clips[index].continuityMode == "frame" else { return nil }
    let clip=project.clips[index]
    guard clip.engine == .h3,!project.isContinuousSceneMember(clip),
      ["t2v","i2v","fflf","a2v"].contains(clip.inferredTask),
      !project.shouldSaveContinuityContext(for:clip) else {
      throw unsupported("frame continuity cannot combine MSR, a scene, or saved motion context")
    }
    let source=try NativeLTXFrameSource(project:project,clip:clip)
    var asset=MediaAsset(name:"Previous visible frame",kind:.image,path:imagePath ?? source.url.path)
    asset.id=source.clipID
    project.assets.removeAll { $0.id == asset.id };project.assets.append(asset)
    project.clips[index].attachments.removeAll { $0.role == .first }
    var opening=Attachment(assetID:asset.id,role:.first);opening.id=source.clipID
    project.clips[index].attachments.insert(opening,at:0)
    project.clips[index].continuity=nil
    if clip.inferredTask != "a2v" {
      project.clips[index].generationSelection=clip.generationSelection ?? GenerationSelection(task:"i2v")
      project.clips[index].generationSelection!.task=clip.attachments.contains { $0.isEnabled && [.last,.keyframe].contains($0.role) } ? "fflf" : "i2v"
    }
    var normalized=request;normalized["project"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(project))
    return (normalized,source)
  }
  private static func extensionRequest(_ request:[String:Any],moviePath:String?=nil) throws
    -> (request:[String:Any],source:NativeLTXMovieSource,clip:Clip)? {
    guard let raw=request["project"],let clipID=request["clipID"] as? String else { return nil }
    var project=try JSONDecoder().decode(StudioProject.self,from:data(raw))
    guard let index=project.clips.firstIndex(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      !project.clips[index].extensionDirection.isEmpty else { return nil }
    let clip=project.clips[index]
    guard clip.engine == .h3,clip.extensionDirection == "after",clip.continuityMode == "independent",
      !project.isContinuousSceneMember(clip),!project.shouldSaveContinuityContext(for:clip),
      clip.audioDriverSelection == nil,clip.musicSource == nil,
      clip.attachments.allSatisfy({ $0.role == .lora }),clip.duration.isFinite,(4...15).contains(clip.duration) else {
      throw unsupported("external extension needs a 4–15 second after-clip with no extra media or saved latent context")
    }
    let source=try NativeLTXMovieSource(project:project,clip:clip)
    let prompt=try extensionPrompt(clip:clip)
    var asset=MediaAsset(name:"Visible source tail",kind:.video,path:moviePath ?? source.url.path);asset.id=clip.id
    project.assets.removeAll { $0.id == asset.id };project.assets.append(asset)
    var reference=Attachment(assetID:asset.id,role:.reference);reference.id=clip.id
    project.clips[index].attachments.append(reference)
    project.clips[index].extensionDirection="";project.clips[index].extensionSource="";project.clips[index].extensionClipID=nil
    project.clips[index].prompt=prompt
    project.clips[index].generationSelection=clip.generationSelection ?? GenerationSelection(task:"ref2va")
    project.clips[index].generationSelection!.task="ref2va"
    var normalized=request;normalized["project"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(project))
    return (normalized,source,clip)
  }
  static func extensionPrompt(clip:Clip) throws -> String {
    let text=clip.prompt.trimmingCharacters(in:.whitespacesAndNewlines)
    guard !text.isEmpty else { throw StudioError.invalid("Write the continuation action before preparing the extension.") }
    let markers=["subject_definitions:","summary:","[video continuation","retention_analysis:",
      "detailed_description:","overall_soundscape:","non_diegetic_music:","<Video 1>","<Picture 1>"]
    if markers.allSatisfy(text.contains) { return text }
    var action=text,soundscape=clip.soundscape,music=clip.music
    if text.hasPrefix("integrated_multimodal_description:") {
      let fields=["integrated_multimodal_description:","overall_soundscape:","non_diegetic_music:"]
      let ranges=fields.compactMap { text.range(of:$0) }
      guard ranges.count == 3,ranges[0].upperBound <= ranges[1].lowerBound,
        ranges[1].upperBound <= ranges[2].lowerBound else {
        throw StudioError.invalid("The H3 prompt needs complete, ordered action, soundscape and music sections.")
      }
      action=String(text[ranges[0].upperBound..<ranges[1].lowerBound]).trimmingCharacters(in:.whitespacesAndNewlines)
      soundscape=String(text[ranges[1].upperBound..<ranges[2].lowerBound]).trimmingCharacters(in:.whitespacesAndNewlines)
      music=String(text[ranges[2].upperBound...]).trimmingCharacters(in:.whitespacesAndNewlines)
      guard !action.isEmpty,!soundscape.isEmpty,!music.isEmpty else {
        throw StudioError.invalid("The H3 prompt needs nonempty action, soundscape and music sections.")
      }
    } else if ["subject_definitions:","retention_analysis:","[video continuation"].contains(where:text.contains) {
      throw StudioError.invalid("The continuation prompt is incomplete. Supply all six H3 reference sections or write the action in plain text.")
    }
    return """
      subject_definitions:
      <Video 1> is the source scene's visible audiovisual tail. <Picture 1> is its terminal reference frame and the opening seam guide.

      summary:
      [video continuation + keyframe completion] Continue the scene from <Picture 1>. \(action)

      retention_analysis:
      <Video 1>: fully_preserved - retain subject identity, scene, camera, lighting and sound continuity.
      <Picture 1>: fully_preserved - use this as the opening frame, then carry out the requested action.

      detailed_description:
      \(action.hasPrefix("[Shot ") ? action : "[Shot 1] " + action)

      overall_soundscape:
      \(soundscape)

      non_diegetic_music:
      \(music)
      """
  }
  private static func extensionResult(_ result:[String:Any],dependency:[String:Any]) throws -> [String:Any] {
    var recipe=result["recipe"] as! [String:Any],conditioning=recipe["conditioning"] as! [String:Any]
    conditioning["task"]="extension";recipe["conditioning"]=conditioning
    var report=result["report"] as! [String:Any]
    report["task"]="extension";report["continuity"]=dependency
    report["resolvedFingerprint"]=try fingerprint(["recipe":recipe,"continuity":dependency])
    return ["recipe":recipe,"report":report]
  }
  public static func compose(request: [String: Any]) throws -> [String: Any] {
    guard try frameRequest(request) == nil,try extensionRequest(request) == nil else {
      throw StudioError.invalid("Frame continuity and extension require native media preparation before composing a runnable recipe.")
    }
    return try compose(request:request,frameInput:nil,continuity:nil)
  }
  private static func compose(request:[String:Any],frameInput:(path:String,sha256:String)?,
    continuity:[String:Any]?,videoInput:(path:String,sha256:String)?=nil) throws -> [String:Any] {
    let (project, clip, runtime, path, original) = try resolve(request)
    let globalAssets = try JSONDecoder().decode([MediaAsset].self,
      from: data(request["globalAssets"] ?? []))
    let availableAssets = project.assets + globalAssets
    var recipe = original
    let motion=try NativeH3MotionPlan(project:project,clip:clip)
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
    guard clip.duration.isFinite,
      (2.5...(motion?.contract["save_context"] as? Bool == true && clip.continuityMode != "motion" ? 362.0/24 : 15)).contains(clip.duration),
      (0...Int(UInt32.max)).contains(clip.seed),
      clip.generationWidth > 0, clip.generationHeight > 0,
      clip.generationWidth % 32 == 0, clip.generationHeight % 32 == 0,
      clip.generationWidth <= 4096, clip.generationHeight <= 4096 else {
      throw unsupported("duration, dimensions or seed are outside the H3 contract")
    }
    var config = recipe["config"] as! [String: Any]
    var steps: Int
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
    config["duration_seconds"] = motion?.duration ?? clip.duration; config["seed"] = clip.seed
    config["steps"] = steps
    if let backend = clip.generationSelection?.projectionBackend { config["projection_backend"] = backend }
    recipe["config"] = config; recipe["prompt"] = prompt
    var components = recipe["components"] as! [String: Any]
    var loras = components["loras"] as? [[Any]] ?? []
    var selectedLoRAProfiles: [String: String] = [:]
    if let legacyStack = recipe.removeValue(forKey: "loras") {
      guard loras.isEmpty, let adapter = rootTurboLoRA(legacyStack) else {
        throw unsupported("the profile adapter cannot be mapped to the Swift Turbo reader")
      }
      loras.append(adapter)
      selectedLoRAProfiles[try canonical(adapter[0] as! String)] = "turbo"
    }
    if let tokenizer = components["tokenizer"] as? String {
      let tokenFile = URL(fileURLWithPath: tokenizer).appendingPathComponent("tokenizer.json")
      if FileManager.default.fileExists(atPath: tokenFile.path) {
        components["tokenizer"] = try canonical(tokenFile.path)
      }
    }
    if clip.inferredTask == "control" {
      guard motion == nil, loras.isEmpty,
        clip.generationWidth <= 2048, clip.generationHeight <= 2048,
        clip.generationWidth * clip.generationHeight <= 768 * 1344,
        let control = components["fun_controlnet"] as? String,
        let transformer = components["transformer"] as? String else {
        throw unsupported("Fun control requires a full-width adapter, dense sampling without LoRAs, and a bounded output canvas")
      }
      let controlPath = try canonical(control)
      try NativeH3FunControlMetadata.validate(control: controlPath, transformer: canonical(transformer))
      components["fun_controlnet"] = controlPath
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
      if attachment.role == .control {
        guard asset.kind == .video, attachment.time == 0,
          attachment.strength.isFinite, (0...1).contains(attachment.strength),
          ["canny_edges", "depth_map", "hed_edges", "mlsd_lines", "pose_skeleton"].contains(attachment.controlType),
          attachment.referenceRole == nil, attachment.referencePriority == nil,
          attachment.referenceFrames == nil, attachment.referenceSizePolicy == nil,
          attachment.attentionStrength == nil, attachment.audioSourceStart == nil,
          attachment.audioSourceDuration == nil else {
          throw unsupported("Fun control accepts one preprocessed Canny, depth, HED, MLSD or pose video at strength 0–1")
        }
        let controlPath = try canonical(asset.path)
        referenceInputs.append(["id": attachment.id.uuidString, "kind": "video", "role": "control",
          "path": controlPath, "sha256": try sourceSHA256(controlPath, maxBytes: 4 * 1024 * 1024 * 1024),
          "strength": attachment.strength, "control_type": attachment.controlType])
        continue
      }
      if Set<MediaRole>([.reference, .first, .last, .keyframe, .audioDriver]).contains(attachment.role) {
        let isVideoReference = attachment.role == .reference &&
          (asset.kind == .video || asset.kind == .sequence)
        let isAudioReference = (attachment.role == .reference || attachment.role == .audioDriver)
          && asset.kind == .audio
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
        let frozen=attachment.role == .first ? frameInput : isVideoReference && videoInput?.path == imagePath ? videoInput : nil
        guard frozen != nil || FileManager.default.isReadableFile(atPath: imagePath) else {
          throw StudioError.invalid("Relink the H3 reference: \(asset.name)")
        }
        var input: [String: Any] = ["id": attachment.id.uuidString,
          "kind": isVideoReference ? "video" : isAudioReference ? "audio" : "image",
          "role": attachment.role.rawValue, "path": imagePath,
          "strength": 1.0,
          "sha256": frozen != nil ? frozen!.sha256 : try sourceSHA256(imagePath,
            maxBytes: isVideoReference ? 4 * 1024 * 1024 * 1024
              : isAudioReference ? 1024 * 1024 * 1024 : 128 * 1024 * 1024)]
        if attachment.role == .audioDriver {
          input["role"] = "audio_driver"
          let preparedMix = clip.audioDriverSelection != nil
          guard !preparedMix || (clip.audioDriverMixKey?.isEmpty == false &&
            asset.scope == .clip && asset.owner == clip.id &&
            attachment.audioSourceStart == nil &&
            attachment.audioSourceDuration == nil &&
            asset.duration + 0.01 >= clip.duration) else {
            throw unsupported("prepare the current timeline audio mix before H3 A2V")
          }
          let start = preparedMix ? 0 : attachment.audioSourceStart
          let length = preparedMix ? clip.duration : attachment.audioSourceDuration
          guard let start, start.isFinite, (0...86400).contains(start),
            let length, length.isFinite,
            length >= clip.duration - 0.001, length <= 15,
            asset.duration <= 0 || start + length <= asset.duration + 0.01 else {
            throw unsupported("H3 A2V needs a source interval covering the visible clip")
          }
          input["source_start_seconds"] = start
          input["source_duration_seconds"] = length
        }
        if attachment.role == .first {
          input["frame_index"] = 0
          if clip.inferredTask == "a2v" { input["role"] = "keyframe" }
        }
        if attachment.role == .last {
          input["frame_index"] = keyframeIndex(attachment, duration: clip.duration)!
        }
        if attachment.role == .keyframe {
          input["frame_index"] = keyframeIndex(attachment, duration: clip.duration)!
        }
        referenceInputs.append(input)
        continue
      }
      guard loras.count < 4 else {
        throw unsupported("select at most four H3 LoRAs, including profile adapters")
      }
      try LoRAMember(asset: asset, strength: attachment.strength).validate(for: .h3)
      guard asset.loraAdalnInputGrid == nil,
        asset.loraLayout == nil || asset.loraLayout == "contiguous_qkv",
        asset.loraProfile == nil || ["standard", "turbo"].contains(asset.loraProfile!) else {
        throw unsupported("this H3 LoRA requires a different layout or adapter profile")
      }
      let adapterPath = try canonical(asset.path)
      guard FileManager.default.isReadableFile(atPath: adapterPath) else {
        throw StudioError.invalid("Relink the H3 LoRA: \(asset.name)")
      }
      let normalizedPath = URL(fileURLWithPath: adapterPath).standardizedFileURL.path
      guard !loras.contains(where: {
        guard let path = $0.first as? String else { return false }
        return URL(fileURLWithPath: path).standardizedFileURL.path == normalizedPath
      }) else {
        throw unsupported("the same H3 LoRA is attached more than once")
      }
      loras.append([adapterPath, attachment.strength])
      selectedLoRAProfiles[adapterPath] = asset.loraProfile
    }
    if !loras.isEmpty {
      var paths = Set<String>()
      var hasTurbo = false
      for index in loras.indices {
        let path = try canonical(loras[index][0] as! String)
        guard paths.insert(path).inserted else {
          throw unsupported("the same H3 LoRA is attached more than once")
        }
        loras[index][0] = path
        let selectedProfile = selectedLoRAProfiles[path]
        let metadata = try NativeLoRAInspection.inspect(URL(fileURLWithPath: path),
          modelHint: .h3, selectedH3Profile: selectedProfile)
        hasTurbo = hasTurbo || selectedProfile == "turbo" || metadata["loraProfile"] as? String == "turbo"
      }
      // Studio Steps are evaluations; the native recipe stores one extra sigma point.
      // An ordinary profile default is not an explicit user override of Turbo's schedule.
      if hasTurbo, clip.generationSelection?.steps == nil { steps = 5 }
      for pair in loras {
        let path = pair[0] as! String
        let metadata = try NativeLoRAInspection.validateH3Sampling(path: path,
          selectedProfile: selectedLoRAProfiles[path], schedulePoints: steps)
        guard metadata["loraModel"] as? String == "h3",
          metadata["loraRequiresAdalnGrid"] as? Bool != true,
          metadata["loraLayout"] == nil || metadata["loraLayout"] as? String == "contiguous_qkv" else {
          throw unsupported("this H3 LoRA requires a different training model, layout or AdaLN grid")
        }
      }
      config["steps"] = steps; recipe["config"] = config
      components["loras"] = loras
    }
    recipe["components"] = components
    if let motion { recipe["continuation"]=motion.contract }
    if clip.inferredTask == "control" {
      recipe["conditioning"] = ["version": 1, "task": "control", "inputs": referenceInputs, "audio_policy": "generated"]
    } else if clip.inferredTask == "a2v" {
      recipe["conditioning"] = ["version": 1, "task": "a2v",
        "inputs": referenceInputs, "audio_policy": "generated"]
    } else if clip.inferredTask == "ref2va" {
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
    var report: [String: Any] = ["profile": URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
      "generation": descriptor(recipe), "resolvedFingerprint": try fingerprint(continuity == nil ? recipe : ["recipe":recipe,"continuity":continuity!]),
      "selectionFingerprint": try fingerprint(original),
      "task": ["ref2va", "a2v", "control"].contains(clip.inferredTask) ? clip.inferredTask
        : ["i2v", "fflf"].contains(clip.inferredTask) ? "fflf" : "t2v", "nativeFPS": 24,
      "nativePreparation": "swift", "productionQualified": false,
      "movieSettings": try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip.settings(in: project))),
      "conditioning": ["inputs": referenceInputs.count]]
    if let continuity { report["continuity"]=continuity }
    if let motion {
      report["continuity"]=motion.dependency;report["warnings"]=motion.warnings
      report["resolvedFingerprint"]=try fingerprint(["recipe":recipe,"continuity":motion.dependency])
    }
    return ["recipe": recipe, "report": report]
  }
  public static func describe(request: [String: Any]) throws -> [String: Any] {
    let frame=try frameRequest(request)
    let extensionSource=try extensionRequest(request)
    let normalized=extensionSource?.request ?? frame?.request ?? request
    let (_, _, _, path, original) = try resolve(normalized)
    var errors: [String] = [], resolved = "",warnings=["Swift H3 generation is experimental and not production qualified."]
    var content = original
    do {
      var dependency=frame?.source.report
      if dependency != nil { dependency!["engine"]="h3" }
      var result = try compose(request:normalized,
        frameInput:frame.map { ($0.source.url.path,String(repeating:"0",count:64)) },continuity:dependency,
        videoInput:extensionSource.map { ($0.source.url.path,String(repeating:"0",count:64)) })
      if let extensionSource {
        var source=extensionSource.source.report;source["engine"]="h3";source["mode"]="externalExtension"
        result=try extensionResult(result,dependency:source)
      }
      content = result["recipe"] as! [String: Any]
      resolved = (result["report"] as? [String: Any])?["resolvedFingerprint"] as? String ?? ""
      warnings += (result["report"] as? [String:Any])?["warnings"] as? [String] ?? []
    } catch { errors.append(error.localizedDescription) }
    let components = content["components"] as? [String: Any] ?? [:]
    var sources = [path] + components.values.compactMap { $0 as? String }.filter { $0.hasPrefix("/") }
    let inputs = (content["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]] ?? []
    sources += inputs.compactMap { $0["path"] as? String }
    if let context=(content["continuation"] as? [String:Any])?["source_context"] as? String {
      sources += [context,URL(fileURLWithPath:context).deletingLastPathComponent().appendingPathComponent("latents.f32").path]
    }
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
      "warnings": warnings,
      "readinessErrors": errors]
  }
  public static func prepareWithMedia(request:[String:Any],destination:URL) async throws -> [String:Any] {
    if let source=try extensionRequest(request) {
      let provisional=try compose(request:source.request,frameInput:nil,continuity:nil,
        videoInput:(source.source.url.path,String(repeating:"0",count:64)))
      let parent=destination.deletingLastPathComponent()
      try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
      let staging=parent.appendingPathComponent(".h3-prepare-"+UUID().uuidString)
      try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:false)
      defer { try? FileManager.default.removeItem(at:staging) }
      let mediaDuration=try await AVURLAsset(url:source.source.url).load(.duration).seconds
      let visible=source.source.duration ?? mediaDuration
      guard visible.isFinite,visible>=5.0/24 else { throw unsupported("extension source needs at least five visible frames") }
      let available=min(175,Int(floor(min(visible,175.0/24)*24+1e-7)))
      let contextFrames=5+17*((available-5)/17)
      let scale=min(256.0/Double(source.clip.generationWidth),256.0/Double(source.clip.generationHeight))
      let width=max(64,Int(Double(source.clip.generationWidth)*scale/32)*32)
      let height=max(64,Int(Double(source.clip.generationHeight)*scale/32)*32)
      let content=provisional["recipe"] as! [String:Any],name="source-tail.mp4"
      let digest=try await source.source.extract(to:staging.appendingPathComponent(name),
        ffmpeg:URL(fileURLWithPath:content["ffmpeg"] as! String),fps:24,width:width,height:height,contextFrames:contextFrames)
      let final=destination.appendingPathComponent(name).path
      let normalized=try extensionRequest(request,moviePath:final)!
      var dependency=source.source.report;dependency["engine"]="h3";dependency["mode"]="externalExtension"
      dependency["preparedSHA256"]=digest;dependency["contextFrames"]=contextFrames
      dependency["referenceWidth"]=width;dependency["referenceHeight"]=height
      let result=try extensionResult(compose(request:normalized.request,frameInput:nil,continuity:nil,
        videoInput:(final,digest)),dependency:dependency)
      let recipe=result["recipe"] as! [String:Any]
      try data(recipe).write(to:staging.appendingPathComponent("recipe.json"),options:.withoutOverwriting)
      try data(request).write(to:staging.appendingPathComponent("editor-request.json"),options:.withoutOverwriting)
      try data(dependency).write(to:staging.appendingPathComponent("continuity.json"),options:.withoutOverwriting)
      try Task.checkCancellation();try source.source.verify()
      try FileManager.default.moveItem(at:staging,to:destination)
      return ["recipePath":destination.appendingPathComponent("recipe.json").path,"prompt":recipe["prompt"]!,"report":result["report"]!]
    }
    guard let frame=try frameRequest(request) else { return try prepare(request:request,destination:destination) }
    var dependency=frame.source.report;dependency["engine"]="h3"
    _ = try compose(request:frame.request,frameInput:(frame.source.url.path,String(repeating:"0",count:64)),continuity:dependency)
    let parent=destination.deletingLastPathComponent()
    try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
    let staging=parent.appendingPathComponent(".h3-prepare-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:staging) }
    let name="previous-frame.png",image=staging.appendingPathComponent(name)
    let time=try await frame.source.extract(to:image)
    let digest=try sourceSHA256(image.path,maxBytes:128*1024*1024)
    dependency["sourceFrameTime"]=time
    let final=destination.appendingPathComponent(name).path
    let normalized=try frameRequest(request,imagePath:final)!
    let result=try compose(request:normalized.request,frameInput:(final,digest),continuity:dependency)
    let content=result["recipe"] as! [String:Any]
    try data(content).write(to:staging.appendingPathComponent("recipe.json"),options:.withoutOverwriting)
    try data(request).write(to:staging.appendingPathComponent("editor-request.json"),options:.withoutOverwriting)
    try data(dependency).write(to:staging.appendingPathComponent("continuity.json"),options:.withoutOverwriting)
    try Task.checkCancellation();try frame.source.verify()
    try FileManager.default.moveItem(at:staging,to:destination)
    return ["recipePath":destination.appendingPathComponent("recipe.json").path,
      "prompt":content["prompt"]!,"report":result["report"]!]
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
