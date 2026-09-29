import CryptoKit
import Darwin
import Foundation

/// Weight-free Studio preparation. The shared native worker performs component admission
/// before Studio accepts this snapshot; this service never owns inference or loads tensors.
public enum NativeLTXPreparation {
  private static func unsupported(_ detail: String) -> StudioError {
    .invalid("Swift LTX preparation: \(detail). Select the Python renderer explicitly for advanced workflows.")
  }
  private static func object<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
  }
  private static func data(_ value: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
  }
  private static func fingerprint(_ value: Any) throws -> String {
    // Distinct serialization namespace: never misrepresent this as Python's JSON digest.
    "swift-json-v1:" + SHA256.hash(data: try data(value)).map { String(format: "%02x", $0) }.joined()
  }
  private static func canonical(_ path: String) throws -> String {
    let expanded = (path as NSString).expandingTildeInPath
    guard expanded.hasPrefix("/"), !expanded.utf8.contains(0) else {
      throw StudioError.invalid("Select an absolute local file path: \(path)")
    }
    return URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath().path
  }
  /// Bounded regular-file reads also reject FIFOs and devices before reading.
  private static func read(_ path: String, limit: Int) throws -> Data {
    let fd = Darwin.open(try canonical(path), O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Cannot read \(path). Relink the file.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG, status.st_size <= limit else {
      throw StudioError.invalid("Expected a regular JSON file no larger than \(limit) bytes.")
    }
    let bytes = try file.read(upToCount: limit + 1) ?? Data()
    guard bytes.count <= limit else { throw StudioError.invalid("JSON file grew beyond its inspection limit.") }
    return bytes
  }
  private static func recipe(_ path: String) throws -> [String: Any] {
    guard let result = try JSONSerialization.jsonObject(with: read(path, limit: 1024 * 1024)) as? [String: Any],
      result["format"] as? String == "weetodd-headless-v2", result["engine"] as? String == "ltx25",
      result["config"] is [String: Any], result["components"] is [String: Any] else {
      throw StudioError.invalid("Import a valid LTX 2.5 headless v2 recipe.")
    }
    return result
  }
  private static func descriptor(_ recipe: [String: Any]) -> [String: Any] {
    let config = recipe["config"] as? [String: Any] ?? [:]
    let components = recipe["components"] as? [String: Any] ?? [:]
    let task = (recipe["conditioning"] as? [String: Any])?["task"] as? String ?? "t2v"
    let ordinary = (config["pipeline_mode"] as? String ?? "distilled") == "distilled"
      && (config["duration_mode"] as? String ?? "manual") == "manual"
      && config["stage1_steps"] as? Int == 8 && config["stage2_steps"] as? Int == 3
      && (config["stage1_sampler"] as? String ?? "euler_ancestral") == "euler_ancestral"
      && (config["dfr_enabled"] as? Bool ?? false) == false
      && (components["ic_loras"] as? [Any] ?? []).isEmpty
      && (components["msr_lora_path"] as? String ?? "").isEmpty
      && (components["distilled_lora_path"] as? String ?? "").isEmpty
      && (components["duration_head_path"] as? String ?? "").isEmpty
      && config["ic_lora_single_stage"] as? Bool != true && !["control", "ref2va"].contains(task)
    return ["supportedTasks": ordinary ? ["t2v", "i2v", "fflf", "a2v"] : [],
      "controls": ["evaluations": config["stage1_steps"] ?? 8, "refinementSteps": config["stage2_steps"] ?? 3,
        "cfg": config["video_cfg_scale"] ?? 1, "stepsEditable": false, "refinementStepsEditable": false,
        "cfgEditable": false, "shiftEditable": false,
        "stepsExplanation": "Swift distilled sampling uses the qualified 8 + 3 schedule.",
        "cfgExplanation": "Distilled guidance is fixed.",
        "shiftExplanation": "This native adapter does not expose a Shift override."],
      "presets": [
        ["id": "custom", "name": "Custom", "description": "Preserve imported recipe settings."],
        ["id": "balanced", "name": "Balanced", "description": "Preserve validated recipe sampling settings."],
        ["id": "speed", "name": "Speed", "description": "Keep the qualified recipe sampling settings."],
        ["id": "lowMemory", "name": "Low memory", "description": "Swift uses staged loading and bounded block residency."]]]
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
      guard let content = try? recipe(url.path) else { return nil }
      let generation = descriptor(content)
      guard !((generation["supportedTasks"] as? [String]) ?? []).isEmpty else { return nil }
      return ["id": try canonical(url.path), "name": url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_", with: " "),
        "engine": "ltx25", "task": (content["conditioning"] as? [String: Any])?["task"] ?? "t2v",
        "generation": generation]
    }
  }
  private struct Context {
    let project: StudioProject
    let clip: Clip
    let assets: [MediaAsset]
    let runtime: [String: Any]
    let profile: String
    var recipe: [String: Any]
    let task: String
    let warnings: [String]
    let frameSource: NativeLTXFrameSource?
  }
  private static func resolve(_ request: [String: Any]) throws -> Context {
    guard let projectValue = request["project"], let runtime = request["runtime"] as? [String: Any],
      let clipID = request["clipID"] as? String else { throw StudioError.invalid("Missing Studio preparation request.") }
    let project = try JSONDecoder().decode(StudioProject.self, from: data(projectValue))
    guard let clip = project.clips.first(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      clip.engine == .ltx25 else { throw StudioError.invalid("Select an LTX 2.5 clip.") }
    guard ["independent", "frame"].contains(clip.continuityMode),
      !project.isContinuousSceneMember(clip), clip.audioDriverSelection == nil, clip.musicSource == nil,
      clip.extensionDirection.isEmpty else { throw unsupported("continuity, continuous scenes, music drivers and extension are not yet ported") }
    let selection = clip.generationSelection
    guard selection?.steps == nil, selection?.refinementSteps == nil, selection?.cfg == nil,
      selection?.shift == nil, selection?.projectionBackend == nil, selection?.transformerBackend == nil else {
      throw unsupported("sampling and backend overrides cannot change the qualified distilled schedule")
    }
    guard selection?.memoryPolicy == nil || selection?.memoryPolicy == "recipe" else {
      throw unsupported("explicit memory-policy overrides are not supported; Swift uses staged loading")
    }
    guard clip.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw unsupported("distilled 8 + 3 sampling does not evaluate negative prompts; clear the negative prompt")
    }
    let frameSource = try clip.continuityMode == "frame" ? NativeLTXFrameSource(project: project, clip: clip) : nil
    let task = frameSource != nil
      ? (clip.attachments.contains { [.last, .keyframe].contains($0.role) } ? "fflf" : "i2v")
      : selection?.task ?? clip.inferredTask
    guard ["t2v", "i2v", "fflf", "a2v"].contains(task) else { throw unsupported("task \(task) is not yet ported") }
    let profiles = try catalog(directory: runtime["profilesDirectory"] as? String ?? "")
    var candidates = profiles.filter {
      (clip.profileID == "auto" || $0["id"] as? String == clip.profileID)
        && (($0["generation"] as? [String: Any])?["supportedTasks"] as? [String] ?? []).contains(task)
    }
    if selection != nil, selection?.preset != .custom, clip.profileID == "auto" {
      let nativeTask = task == "i2v" ? "fflf" : task
      candidates = candidates.filter { $0["task"] as? String == nativeTask }
        + candidates.filter { $0["task"] as? String != nativeTask }
    }
    guard let chosen = candidates.first, let profile = chosen["id"] as? String else {
      throw StudioError.invalid("No compatible Swift LTX recipe is available. Import a distilled recipe or select Automatic.")
    }
    let assets = try JSONDecoder().decode([MediaAsset].self, from: data(request["globalAssets"] ?? []))
    let selectedRecipe = try recipe(profile)
    var warnings = selection?.preset == .speed
      ? ["Speed preserves the qualified 8 + 3 sampling schedule."] : []
    if let inherited = (selectedRecipe["config"] as? [String: Any])?["negative_prompt"] as? String,
      !inherited.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      warnings.append("The selected profile's negative prompt is not evaluated by distilled Swift sampling and is omitted.")
    }
    return Context(project: project, clip: clip, assets: project.assets + assets, runtime: runtime,
      profile: profile, recipe: selectedRecipe, task: task, warnings: warnings, frameSource: frameSource)
  }
  public static func describe(request: [String: Any]) throws -> [String: Any] {
    let context = try resolve(request)
    var errors: [String] = [], resolved = ""
    var content = context.recipe
    do {
      // Only the digest/controls escape description. The movie path identifies
      // the planned dependency; a runnable recipe requires extracted pixels.
      let composed = try compose(context, firstFramePath: context.frameSource?.url.path)
      content = composed["recipe"] as! [String: Any]
      resolved = (composed["report"] as? [String: Any])?["resolvedFingerprint"] as? String ?? ""
    } catch { errors.append(error.localizedDescription) }
    func paths(_ value: Any) -> [String] {
      if let d = value as? [String: Any] { return d.values.flatMap(paths) }
      if let a = value as? [Any] { return a.flatMap(paths) }
      if let s = value as? String, s.hasPrefix("/") || s.hasPrefix("~"), let p = try? canonical(s) { return [p] }
      return []
    }
    var dependencies = paths(content["components"] ?? [:]) + paths(content["conditioning"] ?? [:])
    for dependency in dependencies {
      for name in ["paged_manifest.json", "model_identity.json", "conversion_provenance.json"] {
        let path = URL(fileURLWithPath: dependency).appendingPathComponent(name).path
        if FileManager.default.fileExists(atPath: path) { dependencies.append(path) }
      }
    }
    return ["profileID": context.profile, "generation": descriptor(content), "fingerprint": resolved,
      "selectionFingerprint": try fingerprint(context.recipe), "sourcePaths": Array(Set([context.profile] + dependencies)).sorted(),
      "warnings": context.warnings, "readinessErrors": errors]
  }
  public static func compose(request: [String: Any]) throws -> [String: Any] { try compose(resolve(request)) }
  private static func compose(_ context: Context, firstFramePath: String? = nil) throws -> [String: Any] {
    try Task.checkCancellation()
    guard context.frameSource == nil || firstFramePath != nil else {
      throw StudioError.invalid("Match previous frame requires native media preparation before composing a runnable recipe.")
    }
    let clip = context.clip
    let prompt = clip.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else { throw StudioError.invalid("Write a prompt before preparing the render.") }
    var content = context.recipe
    // Stored profile media/context never become hidden clip dependencies.
    content.removeValue(forKey: "continuation"); content.removeValue(forKey: "reference_images")
    var config = content["config"] as! [String: Any]
    guard (config["duration_mode"] as? String ?? "manual") == "manual",
      let fps = config["frame_rate"] as? Double, fps.isFinite, (1...120).contains(fps),
      clip.duration.isFinite, clip.duration > 0, clip.duration <= 20,
      (0...Int(UInt32.max)).contains(clip.seed), (64...4096).contains(clip.generationWidth),
      (64...4096).contains(clip.generationHeight), clip.generationWidth % 64 == 0, clip.generationHeight % 64 == 0 else {
      throw unsupported("invalid duration, frame rate, dimensions, seed or automatic duration")
    }
    let duration = ceil(clip.duration * fps / 8 - 1e-9) * 8 / fps
    guard duration <= 20 else { throw unsupported("rounded generation duration exceeds 20 seconds") }
    config["width"] = clip.generationWidth; config["height"] = clip.generationHeight
    config["seed"] = clip.seed; config["duration_seconds"] = duration
    config["negative_prompt"] = ""
    var components = content["components"] as! [String: Any]
    if let original = components["loras"], !(original is [[Any]]) {
      throw StudioError.invalid("Profile LoRAs must be an array of path/strength pairs.")
    }
    var loras = components["loras"] as? [[Any]] ?? []
    var seen = Set<String>()
    for pair in loras {
      guard pair.count == 2, let path = pair[0] as? String, seen.insert(try canonical(path)).inserted else {
        throw StudioError.invalid("Invalid or duplicate profile LoRA.")
      }
    }
    var inputs: [[String: Any]] = []
    let attachments = clip.attachments.filter { ($0.role != .lora || $0.isEnabled) && !(context.frameSource != nil && $0.role == .first) }
    var roles = Set(attachments.filter { $0.role != .lora }.map(\.role))
    if let firstFramePath {
      roles.insert(.first)
      inputs.append(["id": "continuity-" + clip.id.uuidString, "kind": "image", "role": "keyframe",
        "path": firstFramePath, "strength": 1.0, "frame_index": 0])
    }
    let allowed: Set<MediaRole> = context.task == "t2v" ? [] : context.task == "i2v" ? [.first]
      : context.task == "a2v" ? [.audioDriver, .first] : [.first, .last, .keyframe]
    guard roles.isSubset(of: allowed) else { throw unsupported("attached media conflict with \(context.task); no inputs were discarded") }
    if context.task == "i2v", !roles.contains(.first) { throw StudioError.invalid("Image to video requires a First frame image.") }
    if context.task == "fflf", clip.generationSelection != nil, !roles.isSuperset(of: [.first, .last]) {
      throw StudioError.invalid("First and last frames requires First frame and Last frame images.")
    }
    if context.task == "a2v", attachments.filter({ $0.role == .audioDriver }).count != 1 {
      throw StudioError.invalid("Audio to video requires exactly one source audio attachment.")
    }
    if context.task == "a2v", attachments.filter({ $0.role == .first }).count > 1 {
      throw StudioError.invalid("Audio to video accepts at most one opening-frame image.")
    }
    for attachment in attachments {
      try Task.checkCancellation()
      guard let asset = context.assets.last(where: { $0.id == attachment.assetID }) else {
        throw StudioError.invalid("A clip attachment is missing from its asset store.")
      }
      let path = try canonical(asset.path)
      guard FileManager.default.isReadableFile(atPath: path) else { throw StudioError.invalid("Relink missing media: \(asset.name)") }
      if attachment.role == .lora {
        guard asset.loraProfile == nil, asset.loraLayout == nil, asset.loraAdalnInputGrid == nil else {
          throw StudioError.invalid("H3 LoRA profile, layout and AdaLN grid require the H3 engine.")
        }
        try LoRAMember(asset: asset, strength: attachment.strength).validate(for: .ltx25)
        guard attachment.strength > 0, seen.insert(path).inserted else {
          throw StudioError.invalid("Disable zero-strength LoRAs and remove duplicate files before rendering.")
        }
        try NativeLTXLoRAMetadata.validate(path: path, model: asset.loraModel!.rawValue)
        loras.append([path, attachment.strength]); continue
      }
      if attachment.role == .audioDriver {
        guard asset.kind == .audio, attachment.strength == 1,
          let start = attachment.audioSourceStart, start.isFinite, (0...86400).contains(start),
          let sourceDuration = attachment.audioSourceDuration, sourceDuration.isFinite,
          (0.001...86400).contains(sourceDuration),
          asset.duration <= 0 || start + sourceDuration <= asset.duration + 0.01 else {
          throw StudioError.invalid("Choose one audio source with strength 1 and an explicit valid source in-point and duration.")
        }
        inputs.append(["id": attachment.id.uuidString, "kind": "audio", "role": "audio_driver",
          "path": path, "strength": 1, "source_start_seconds": start,
          "source_duration_seconds": sourceDuration]); continue
      }
      guard asset.kind == .image, attachment.strength.isFinite, (0...1).contains(attachment.strength) else {
        throw StudioError.invalid("Endpoint references must be images with strength from 0 to 1.")
      }
      guard attachment.time.isFinite, attachment.time >= 0, attachment.time <= duration else {
        throw StudioError.invalid("Endpoint time must lie within the generated clip.")
      }
      let frame: Any = attachment.role == .last ? "last" : attachment.role == .first ? 0 : Int((attachment.time * fps).rounded(.toNearestOrEven))
      inputs.append(["id": attachment.id.uuidString, "kind": "image", "role": "keyframe", "path": path,
        "strength": attachment.strength, "frame_index": frame])
    }
    let frames = Int((duration * fps / 8).rounded(.toNearestOrEven)) * 8 + 1
    if context.task != "a2v" {
      let indices = inputs.map { $0["frame_index"] as? String == "last" ? frames - 1 : $0["frame_index"] as! Int }
      guard Set(indices).count == indices.count, indices.allSatisfy({ $0 == 0 || $0 == frames - 1 }),
        inputs.count <= 2, context.task == "t2v" || indices.contains(0) else {
        throw unsupported("only unique first and last endpoints are supported")
      }
    }
    if !loras.isEmpty { components["loras"] = loras }
    var contract = content["conditioning"] as? [String: Any] ?? [:]
    let task = context.task == "i2v" ? "fflf" : context.task
    for key in ["version", "task", "inputs", "extension"] { contract.removeValue(forKey: key) }
    if (content["conditioning"] as? [String: Any])?["task"] as? String != task { contract.removeValue(forKey: "audio_policy") }
    contract["version"] = 1; contract["task"] = task; contract["inputs"] = inputs
    content["conditioning"] = contract; content["config"] = config; content["components"] = components
    content["prompt"] = prompt
    let configured = context.runtime["ffmpegPath"] as? String ?? ""
    let ffmpeg = configured.isEmpty ? [content["ffmpeg"] as? String ?? "", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].first {
      !$0.isEmpty && FileManager.default.isExecutableFile(atPath: $0)
    } ?? "" : configured
    guard !ffmpeg.isEmpty, FileManager.default.isExecutableFile(atPath: ffmpeg) else {
      throw StudioError.invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    content["ffmpeg"] = try canonical(ffmpeg)
    // FFprobe is unnecessary: generated media inspection is native AVFoundation.
    var identity: [String: Any] = ["recipe": content]
    if let source = context.frameSource { identity["continuity"] = source.report }
    var report: [String: Any] = ["profile": URL(fileURLWithPath: context.profile).deletingPathExtension().lastPathComponent,
      "generation": descriptor(content), "resolvedFingerprint": try fingerprint(context.frameSource == nil ? content : identity),
      "selectionFingerprint": try fingerprint(context.recipe), "warnings": context.warnings, "task": task,
      "nativeFPS": fps, "preserveEditorialDuration": true, "movieSettings": try object(clip.settings(in: context.project)),
      "nativePreparation": "swift", "conditioning": ["frames": frames, "inputs": inputs.count]]
    if let source = context.frameSource { report["continuity"] = source.report }
    return ["recipe": content, "report": report]
  }
  /// Native media extraction runs off the UI thread. Publish the image, recipe
  /// and original editor request together; never rewrite stored attachments.
  public static func prepareWithMedia(request: [String: Any], destination: URL) async throws -> [String: Any] {
    let context = try resolve(request)
    guard let source = context.frameSource else { return try prepare(request: request, destination: destination) }
    // Validate prompt, sampling and attachment contracts before decoding media.
    _ = try compose(context, firstFramePath: source.url.path)
    let parent = destination.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".prepare-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: staging) }
    let name = "previous-frame.png"
    let time = try await source.extract(to: staging.appendingPathComponent(name))
    let result = try compose(context, firstFramePath: destination.appendingPathComponent(name).path)
    let content = result["recipe"] as! [String: Any]
    var report = result["report"] as! [String: Any], continuity = source.report
    continuity["sourceFrameTime"] = time; report["continuity"] = continuity
    try data(content).write(to: staging.appendingPathComponent("recipe.json"), options: .withoutOverwriting)
    try data(request).write(to: staging.appendingPathComponent("editor-request.json"), options: .withoutOverwriting)
    try data(continuity).write(to: staging.appendingPathComponent("continuity.json"), options: .withoutOverwriting)
    try Task.checkCancellation(); try source.verify()
    try FileManager.default.moveItem(at: staging, to: destination)
    return ["recipePath": destination.appendingPathComponent("recipe.json").path, "prompt": content["prompt"]!, "report": report]
  }
  public static func prepare(request: [String: Any], destination: URL) throws -> [String: Any] {
    let result = try compose(request: request), content = result["recipe"] as! [String: Any]
    let parent = destination.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".prepare-" + UUID().uuidString)
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
