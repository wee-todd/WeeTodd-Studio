import CryptoKit
import CoreFoundation
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
  private static func verifyPlannedAudio(_ source: MusicShotSource) throws -> String {
    let path = try canonical(source.path)
    guard source.sha256.count == 64,
      source.sha256.allSatisfy(\.isHexDigit),
      source.start.isFinite, source.start >= 0,
      source.duration.isFinite, source.duration > 0,
      ["a2v", "t2v"].contains(source.task) else {
      throw StudioError.invalid("The planned song source has invalid timing, task or checksum.")
    }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Relink the planned song source.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      (1...4 * 1024 * 1024 * 1024).contains(status.st_size) else {
      throw StudioError.invalid("The planned song source must be a regular file under 4 GB.")
    }
    var digest = SHA256(), readBytes: Int64 = 0
    while let block = try handle.read(upToCount: 4 * 1024 * 1024), !block.isEmpty {
      try Task.checkCancellation()
      readBytes += Int64(block.count)
      guard readBytes <= status.st_size else { throw StudioError.invalid("The planned song changed while reading.") }
      digest.update(data: block)
    }
    guard readBytes == status.st_size,
      digest.finalize().map({ String(format: "%02x", $0) }).joined().caseInsensitiveCompare(source.sha256) == .orderedSame else {
      throw StudioError.invalid("The planned song changed. Reanalyze or relink the source before rendering.")
    }
    return path
  }
  private static func recipe(_ path: String) throws -> [String: Any] {
    guard let result = try JSONSerialization.jsonObject(with: read(path, limit: 1024 * 1024)) as? [String: Any],
      result["format"] as? String == "weetodd-headless-v2", result["engine"] as? String == "ltx25",
      result["config"] is [String: Any], result["components"] is [String: Any] else {
      throw StudioError.invalid("Import a valid LTX 2.5 headless v2 recipe.")
    }
    return result
  }
  private static func specialization(_ recipe:[String:Any]) -> String? {
    if let condition=recipe["conditioning"] as? [String:Any],
      condition["task"] as? String == "control",let family=condition["control_family"] as? String,
      ["motion_track","crossview_warp","crossview_ingredients"].contains(family),
      let config=recipe["config"] as? [String:Any],config["stage1_steps"] as? Int == 8,
      config["stage2_steps"] as? Int == 3,config["ic_lora_single_stage"] as? Bool != true,
      (config["dfr_enabled"] as? Bool ?? false) == false,
      let components=recipe["components"] as? [String:Any],
      (components["msr_lora_path"] as? String ?? "").isEmpty,
      let adapters=components["ic_loras"] as? [[Any]],
      adapters.count == (family == "crossview_ingredients" ? 2 : 1),
      adapters.allSatisfy({ $0.count == 2 && ($0[0] as? String)?.hasPrefix("/") == true
        && ($0[1] as? Double).map { $0.isFinite && $0>0 && $0<=3 } == true }) { return family }
    guard let config=recipe["config"] as? [String:Any],
      let components=recipe["components"] as? [String:Any],
      (config["pipeline_mode"] as? String ?? "distilled") == "distilled",
      config["stage1_steps"] as? Int == 8,config["stage2_steps"] as? Int == 3,
      (config["dfr_enabled"] as? Bool ?? false) == false,
      (components["loras"] as? [Any] ?? []).isEmpty,
      let adapters=components["ic_loras"] as? [[Any]],adapters.count == 1,adapters[0].count == 2,
      let path=adapters[0][0] as? String,path.hasPrefix("/"),
      let strength=adapters[0][1] as? Double,strength.isFinite,strength>0,strength<=3 else { return nil }
    let task=(recipe["conditioning"] as? [String:Any])?["task"] as? String
    if task == "control",config["ic_lora_single_stage"] as? Bool != true,
      strength<=2,(components["msr_lora_path"] as? String ?? "").isEmpty { return "union" }
    guard config["ic_lora_single_stage"] as? Bool == true else { return nil }
    if task == "ref2va",components["msr_lora_path"] as? String == path,
      ((components["msr_lora_strength"] as? Double) ?? 1) == strength { return "msr" }
    if task == "control",(components["msr_lora_path"] as? String ?? "").isEmpty { return "ingredients" }
    return nil
  }
  private static func sourceSHA256(_ path:String,maxBytes:Int64=128*1024*1024) throws -> String {
    let fd=Darwin.open(path,O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd>=0 else { throw StudioError.invalid("Cannot read the reference media.") }
    let handle=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? handle.close() }
    var status=stat()
    guard fstat(fd,&status)==0,status.st_mode & S_IFMT == S_IFREG,
      (1...maxBytes).contains(status.st_size) else {
      throw StudioError.invalid("Reference media must be regular files under \(maxBytes/1024/1024) MiB.")
    }
    var digest=SHA256(),count=0
    while let bytes=try handle.read(upToCount:1024*1024),!bytes.isEmpty {
      try Task.checkCancellation();count+=bytes.count
      guard count<=status.st_size else { throw StudioError.invalid("Reference media changed during preparation.") }
      digest.update(data:bytes)
    }
    var after=stat(),current=stat()
    guard count==status.st_size,fstat(fd,&after)==0,lstat(path,&current)==0,
      after.st_size==status.st_size,after.st_mtimespec.tv_sec==status.st_mtimespec.tv_sec,
      after.st_mtimespec.tv_nsec==status.st_mtimespec.tv_nsec,
      after.st_ctimespec.tv_sec==status.st_ctimespec.tv_sec,after.st_ctimespec.tv_nsec==status.st_ctimespec.tv_nsec,
      current.st_dev==status.st_dev,current.st_ino==status.st_ino,
      current.st_size==after.st_size,current.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,
      current.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
      current.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,current.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec else {
      throw StudioError.invalid("Reference media changed during preparation.")
    }
    return digest.finalize().map { String(format:"%02x",$0) }.joined()
  }
  private static func descriptor(_ recipe: [String: Any]) -> [String: Any] {
    let config = recipe["config"] as? [String: Any] ?? [:]
    let components = recipe["components"] as? [String: Any] ?? [:]
    let task = (recipe["conditioning"] as? [String: Any])?["task"] as? String ?? "t2v"
    let specialized=specialization(recipe)
    let authoredIngredients=specialized == "ingredients" && config["single_stage_sampler"] as? String == "euler_ancestral_cfg_pp"
    let guided = guidedProfile(recipe)
    let durationMode = config["duration_mode"] as? String ?? "manual"
    let headPath = components["duration_head_path"] as? String ?? ""
    let headSHA = headPath.hasPrefix("/") ? try? NativeLTXAutomaticDuration.validateHead(at:URL(fileURLWithPath:headPath)) : nil
    let ordinary = (config["pipeline_mode"] as? String ?? "distilled") == "distilled"
      && ["manual","automatic"].contains(durationMode)
      && config["stage1_steps"] as? Int == 8 && config["stage2_steps"] as? Int == 3
      && (config["stage1_sampler"] as? String ?? "euler_ancestral") == "euler_ancestral"
      && (components["ic_loras"] as? [Any] ?? []).isEmpty
      && (components["msr_lora_path"] as? String ?? "").isEmpty
      && (components["distilled_lora_path"] as? String ?? "").isEmpty
      && config["ic_lora_single_stage"] as? Bool != true && !["control", "ref2va"].contains(task)
    let singleStage=(config["ic_lora_single_stage"] as? Bool ?? false) && specialized == nil && !guided
    let singleStageMethod=LTX25SingleStageMethod(rawValue:config["stage1_sampler"] as? String ?? "")
    let singleStageSchedule=LTX25NegativeSchedule(rawValue:config["cfg_pp_schedule"] as? String ?? "full")
    let singleStageValid=singleStage && singleStageMethod != nil && singleStageSchedule != nil && config["stage1_steps"] as? Int == 8 && config["stage2_steps"] as? Int == 0
    let dfrEnabled=config["dfr_enabled"] as? Bool ?? false
    let adapter=config["dfr_detailing_lora_path"] as? String ?? ""
    let strength=(config["dfr_detailing_lora_strength"] as? NSNumber)?.doubleValue ?? 0
    let rounds=config["dfr_temporal_rounds"] as? Int ?? 0
    let temporal=config["dfr_temporal_upsampler_path"] as? String ?? ""
    let fps=(config["frame_rate"] as? NSNumber)?.doubleValue ?? 0
    let automaticAvailable = specialized == nil && !dfrEnabled && headSHA != nil && (ordinary || guided)
    let dfrValid=ordinary && dfrEnabled && adapter.hasPrefix("/") && strength.isFinite && (0...3).contains(strength) && strength > 0
      && (0...2).contains(rounds) && (rounds == 0 ? temporal.isEmpty : temporal.hasPrefix("/") && fps*Double(1 << rounds) <= 120)
      && (config["dfr_prebaked_transformer_path"] as? String ?? "").isEmpty
      && (config["generated_keyframes"] as? Int ?? 0) == 0
      && (components["loras"] as? [Any] ?? []).isEmpty
    return ["pipelineMode": config["pipeline_mode"] as? String ?? "distilled",
      "durationMode":durationMode,"automaticDurationAvailable":automaticAvailable,
      "automaticDuration": ["minimum_seconds":config["auto_duration_min_seconds"] ?? 1.0,
        "maximum_seconds":config["auto_duration_max_seconds"] ?? 20.0],
      "guidance": config.filter { ["audio_cfg_scale", "stg_scale", "video_rescale_scale", "audio_rescale_scale", "modality_scale", "stg_blocks", "stage1_sigmas"].contains($0.key) },
      "experimentalGuidance": guided, "dfrEnabled": dfrValid,"referenceFamily": specialized ?? "ordinary",
      "singleStageAvailable":(ordinary || singleStageValid) && !dfrEnabled && specialized == nil,
      "singleStageEnabled":singleStageValid,
      "ordinaryKeyframesAvailable":(ordinary || guided || singleStageValid) && !dfrEnabled && specialized == nil,
      "supportedTasks": singleStageValid ? (singleStageMethod == .cfgpp ? ["t2v","i2v","fflf"]:["t2v","i2v","fflf","a2v"]) : durationMode == "automatic" ? automaticAvailable ? ["t2v","i2v","fflf"] : [] : guided ? ["t2v", "i2v", "fflf", "a2v"] : specialized != nil ? [specialized == "msr" ? "ref2va" : "control"] : dfrValid ? ["t2v", "i2v", "fflf"] : ordinary && !dfrEnabled
      ? ["t2v", "i2v", "fflf", "a2v", "extension"] : [],
      "controls": ["evaluations": singleStageValid ? (singleStageMethod == .cfgpp ? singleStageSchedule!.evaluationCount : 8) : authoredIngredients ? 16 : (config["stage1_steps"] as? Int ?? 8), "refinementSteps": ["msr","ingredients"].contains(specialized ?? "") ? 0 : config["stage2_steps"] ?? 3,
        "cfg": config["video_cfg_scale"] ?? 1, "stepsEditable": guided, "refinementStepsEditable": false,
        "cfgEditable": guided, "shiftEditable": false,
        "stepsExplanation": singleStageValid ? "Eight full-resolution updates with no spatial upscale or refinement stage. CFG++ negative passes follow the selected schedule; terminal negative prediction is omitted." : guided ? "Guided steps are scheduler updates; guidance can require multiple transformer predictions per update. Refinement uses three deterministic steps. Experimental and not quality qualified." : authoredIngredients ? "Ingredients CFG++ uses eight steps and sixteen serial transformer evaluations." : ["msr","ingredients"].contains(specialized ?? "") ? "Swift reference sampling uses eight full-resolution evaluations." : "Swift distilled sampling uses the qualified 8 + 3 schedule.",
        "cfgExplanation": guided ? "Video CFG; audio CFG is configured separately." : "Distilled guidance is fixed.",
        "shiftExplanation": "This native adapter does not expose a Shift override."],
      "presets": [
        ["id": "custom", "name": "Custom", "description": "Preserve imported recipe settings."],
        ["id": "balanced", "name": "Balanced", "description": "Preserve validated recipe sampling settings."],
        ["id": "speed", "name": "Speed", "description": "Keep the qualified recipe sampling settings."],
        ["id": "lowMemory", "name": "Low memory", "description": "Swift uses staged loading and bounded block residency."]]]
  }
  private static func guidedProfile(_ recipe: [String: Any]) -> Bool {
    guard let config = recipe["config"] as? [String: Any],
      let mode = LTX25GuidanceMode(rawValue: config["pipeline_mode"] as? String ?? ""),
      let components = recipe["components"] as? [String: Any],
      (components["distilled_lora_path"] as? String)?.hasPrefix("/") == true,
      (components["ic_loras"] as? [Any] ?? []).isEmpty,
      (components["msr_lora_path"] as? String ?? "").isEmpty,
      ["manual","automatic"].contains(config["duration_mode"] as? String ?? "manual"),
      (config["stage1_sampler"] as? String ?? (mode == .guided ? "euler_guided" : "res_2s_guided")) == (mode == .guided ? "euler_guided" : "res_2s_guided"),
      config["stage2_steps"] as? Int == 3,
      (config["stage2_sampler"] as? String ?? "euler") == "euler",
      (config["dfr_enabled"] as? Bool ?? false) == false,
      (config["ic_lora_single_stage"] as? Bool ?? false) == false,
      !["ref2va", "control", "extension"].contains((recipe["conditioning"] as? [String: Any])?["task"] as? String ?? "t2v") else { return false }
    return true
  }

  private static func guidedConfig(_ original: [String: Any], selection: GenerationSelection,
    negativePrompt: String) throws -> [String: Any] {
    guard let settings = selection.ltx25Guidance, settings.experimentalEnabled else {
      throw unsupported("guided sampling requires explicit experimental opt-in")
    }
    var config = original
    if let raw = original["stage1_steps"] {
      guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
        (1...64).contains(number.doubleValue) else { throw unsupported("guided stage-one updates must be an integer from 1 to 64") }
    }
    let mode = settings.mode, steps = selection.steps ?? (original["stage1_steps"] as? Int ?? (mode == .guided ? 30 : 15))
    guard (1...64).contains(steps), selection.refinementSteps == nil || selection.refinementSteps == 3 else {
      throw unsupported("guided stage one requires 1–64 updates and refinement remains three steps")
    }
    config["pipeline_mode"] = mode.rawValue
    config["stage1_steps"] = steps; config["stage2_steps"] = 3
    config["stage1_sampler"] = mode == .guided ? "euler_guided" : "res_2s_guided"
    config["stage2_sampler"] = "euler"
    let values: [(String, Double?, Double, ClosedRange<Double>)] = [
      ("video_cfg_scale", selection.cfg, 3, 0...100),
      ("audio_cfg_scale", settings.audioCFG, 7, 0...100),
      ("stg_scale", settings.stgScale, mode == .guided ? 1 : 0, 0...100),
      ("video_rescale_scale", settings.videoRescale, mode == .guided ? 0.7 : 0.45, 0...1),
      ("audio_rescale_scale", settings.audioRescale, mode == .guided ? 0.7 : 1, 0...1),
      ("modality_scale", settings.modalityScale, 3, 0...100)]
    for (key, override, fallback, bounds) in values {
      if let value = original[key], !(value is NSNumber) { throw unsupported("\(key) must be numeric") }
      let inherited = original[key] as? NSNumber
      guard inherited == nil || CFGetTypeID(inherited!) != CFBooleanGetTypeID() else { throw unsupported("\(key) must be numeric") }
      let value = override ?? inherited?.doubleValue ?? fallback
      guard value.isFinite, bounds.contains(value) else { throw unsupported("\(key) is outside its supported range") }
      config[key] = value
    }
    if let raw = original["stg_blocks"] {
      guard let values = raw as? [NSNumber], values.allSatisfy({
        CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue.isFinite && $0.doubleValue.rounded() == $0.doubleValue
      }) else { throw unsupported("STG blocks must be integer indices") }
    }
    let blocks = settings.stgBlocks ?? (original["stg_blocks"] as? [Int] ?? (mode == .guided ? [28] : []))
    guard blocks.count <= 48, Set(blocks).count == blocks.count, blocks.allSatisfy({ (0..<48).contains($0) }) else {
      throw unsupported("STG blocks must be unique indices from 0 to 47")
    }
    config["stg_blocks"] = blocks
    if let raw = original["stage1_sigmas"], !(raw is NSNull) {
      guard let values = raw as? [NSNumber], values.allSatisfy({ CFGetTypeID($0) != CFBooleanGetTypeID() }) else {
        throw unsupported("custom sigmas must be a numeric array or null")
      }
    }
    if settings.sigmas?.isEmpty == true { config["stage1_sigmas"] = NSNull() }
    else if let sigmas = settings.sigmas ?? (original["stage1_sigmas"] as? [Double]) {
      guard sigmas.count == steps + 1, sigmas.count <= 65,
        sigmas.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
        sigmas[0] > 0, sigmas.last == 0,
        zip(sigmas, sigmas.dropFirst()).allSatisfy({ $0 > $1 }) else {
        throw unsupported("custom sigmas need one more point than updates, descending from (0,1] to zero")
      }
      config["stage1_sigmas"] = sigmas
    }
    config["negative_prompt"] = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    return config
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
    let movieSource: NativeLTXMovieSource?
  }
  private static func sceneRequest(_ request: [String: Any]) throws -> (StudioProject, [Clip])? {
    guard let value = request["project"], let clipID = request["clipID"] as? String else { return nil }
    let project = try JSONDecoder().decode(StudioProject.self, from: data(value))
    guard let clip = project.clips.first(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      project.isContinuousSceneMember(clip) else { return nil }
    return (project, try project.continuousSceneMembers(for: clip))
  }
  private static func composeScene(_ request: [String: Any], project: StudioProject,
    members: [Clip]) throws -> [String: Any] {
    let versionTwo = NativeLTXSceneImageInputs.requiresVersionTwo(members)
    guard members.allSatisfy({ $0.generationSelection?.ltx25Keyframes.map {
      $0.experimentalEnabled && $0.generatedCount == 0
    } ?? true }) else {
      throw unsupported("continuous scenes accept timed images, not generated keyframe slots")
    }
    guard members.allSatisfy({ $0.generationSelection?.ltx25AutomaticDuration == nil }) else {
      throw unsupported("automatic duration cannot change continuous-scene shot intervals; use manual timing")
    }
    guard members.allSatisfy({ $0.generationSelection?.ltx25Guidance == nil }) else {
      throw unsupported("guided continuous-scene sampling is not admitted")
    }
    guard (2...6).contains(members.count) else { throw unsupported("a scene needs two to six shots") }
    var independent = versionTwo ? try NativeLTXSceneImageInputs.baseProject(project,members:members) : project
    let memberIDs = Set(members.map(\.id))
    for index in independent.clips.indices where memberIDs.contains(independent.clips[index].id) {
      independent.clips[index].continuity = ClipContinuity(mode: "independent")
    }
    var recipes: [[String: Any]] = [], reports: [[String: Any]] = []
    for member in members {
      guard member.extensionDirection.isEmpty, member.extensionSource.isEmpty,
        member.audioDriverSelection == nil,
        member.attachments.allSatisfy({ (versionTwo ? [.lora,.first,.last,.keyframe,.audioDriver] : [.lora,.first,.audioDriver]).contains($0.role) }) else {
        throw unsupported("Swift continuous scenes accept timed images, one continuous source-audio driver and ordinary LoRAs")
      }
      var one = request
      one["project"] = try object(independent)
      one["clipID"] = member.id.uuidString
      let composed = try compose(resolve(one))
      let recipe = composed["recipe"] as! [String: Any]
      guard ((recipe["config"] as? [String: Any])?["dfr_enabled"] as? Bool ?? false) == false,
        ((recipe["config"] as? [String: Any])?["pipeline_mode"] as? String ?? "distilled") == "distilled" else {
        throw unsupported("DFR and guided sampling do not support continuous scenes")
      }
      let task = (recipe["conditioning"] as? [String: Any])?["task"] as? String
      let hasImage = !versionTwo && member.attachments.contains { $0.role == .first }
      let hasAudio = member.attachments.contains { $0.role == .audioDriver }
      guard task == (hasAudio ? "a2v" : hasImage ? "fflf" : "t2v"),
        (((recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])?.count ?? 0) ==
          (hasImage ? 1 : 0) + (hasAudio ? 1 : 0) else {
        throw unsupported("each Swift scene shot accepts one first-frame image and one source-audio interval")
      }
      recipes.append(recipe); reports.append(composed["report"] as! [String: Any])
    }
    func effective(_ value: [String: Any]) throws -> Data {
      var common = value
      common.removeValue(forKey: "prompt")
      common.removeValue(forKey: "conditioning")
      var config = common["config"] as! [String: Any]
      config.removeValue(forKey: "seed")
      config.removeValue(forKey: "duration_seconds")
      common["config"] = config
      return try data(common)
    }
    let expected = try effective(recipes[0])
    guard try recipes.dropFirst().allSatisfy({ try effective($0) == expected }) else {
      throw unsupported("all scene shots need identical effective components, LoRAs and sampling settings")
    }
    let config = recipes[0]["config"] as! [String: Any]
    let fps = config["frame_rate"] as! Double
    var boundaries = [0], cumulative = 0.0
    for member in members {
      cumulative += member.duration
      boundaries.append(Int((cumulative * fps / 8).rounded(.toNearestOrEven)) * 8)
    }
    let lengths = zip(boundaries.dropLast(), boundaries.dropFirst()).map { $1 - $0 }
    guard cumulative <= 30, lengths.first! >= 32,
      lengths.dropFirst().allSatisfy({ $0 > 0 }) else {
      throw unsupported("scene durations must resolve to positive eight-frame shot ranges, with at least 32 frames in the first shot, within 30 seconds")
    }
    let drivers: [[String: Any]?] = recipes.map { recipe in
      ((recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])?
        .first(where: { $0["role"] as? String == "audio_driver" })
    }
    guard drivers.allSatisfy({ $0 == nil }) || drivers.allSatisfy({ $0 != nil }) else {
      throw unsupported("audio-driven scenes need one consecutive interval of the same source on every shot")
    }
    if let firstDriver = drivers.first!,
      let path = firstDriver["path"] as? String,
      let start = firstDriver["source_start_seconds"] as? Double {
      var elapsed = 0.0
      for (index, member) in members.enumerated() {
        guard let driver = drivers[index],
          driver["path"] as? String == path,
          let selectedStart = driver["source_start_seconds"] as? Double,
          let selectedDuration = driver["source_duration_seconds"] as? Double,
          abs(selectedStart - (start + elapsed)) <= 1e-4,
          selectedDuration + 1e-4 >= member.duration,
          abs(Double(lengths[index]) / fps - member.duration) <= 1e-4 else {
          throw unsupported("audio-driven scene shots need grid-aligned durations and consecutive intervals of the same source")
        }
        elapsed += member.duration
      }
    }
    var content = recipes[0]
    var sceneConfig = config
    sceneConfig["duration_seconds"] = Double(boundaries.last!) / fps
    content["config"] = sceneConfig
    if drivers.first! != nil {
      var conditioning = content["conditioning"] as! [String: Any]
      var inputs = conditioning["inputs"] as! [[String: Any]]
      guard let index = inputs.firstIndex(where: { $0["role"] as? String == "audio_driver" }) else {
        throw unsupported("the scene audio driver disappeared during recipe resolution")
      }
      inputs[index]["source_duration_seconds"] = Double(boundaries.last!) / fps
      conditioning["inputs"] = inputs
      content["conditioning"] = conditioning
    }
    if versionTwo {
      let globalAssets = try JSONDecoder().decode([MediaAsset].self,from:data(request["globalAssets"] ?? []))
      let images = try NativeLTXSceneImageInputs.make(members:members,assets:project.assets+globalAssets,
        segmentStarts:Array(boundaries.dropLast()),segmentFrames:lengths,fps:fps)
      var conditioning = content["conditioning"] as! [String:Any]
      let audio = (conditioning["inputs"] as! [[String:Any]]).filter { $0["role"] as? String == "audio_driver" }
      conditioning["inputs"] = images+audio
      conditioning["task"] = audio.isEmpty ? (images.isEmpty ? "t2v":"fflf") : "a2v"
      content["conditioning"] = conditioning
    }
    let sharedSound = members[0].soundscape.trimmingCharacters(in: .whitespacesAndNewlines)
    let sharedMusic = members[0].music.trimmingCharacters(in: .whitespacesAndNewlines)
    let segments: [[String: Any]] = zip(members, recipes).enumerated().map { index, pair in
      let (member, recipe) = pair
      var prompt = recipe["prompt"] as! String
      if !sharedSound.isEmpty { prompt += "\nSound: " + sharedSound }
      if !sharedMusic.isEmpty && sharedMusic != "N/A" { prompt += "\nMusic: " + sharedMusic }
      var segment: [String: Any] = ["clip_id": member.id.uuidString, "prompt": prompt,
        "duration_seconds": member.duration,
        "seed": (recipe["config"] as! [String: Any])["seed"]!]
      if !versionTwo,index > 0,
        let image = ((recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])?
          .first(where: { $0["role"] as? String == "keyframe" }) {
        segment["image_input"] = image
      }
      return segment
    }
    let decodeMode = members[0].continuity?.sceneDecodeMode ?? "single"
    guard ["single", "windowed"].contains(decodeMode) else {
      throw unsupported("choose Full decode or Bounded decode for the Swift scene")
    }
    let boundaryPolicy = members[0].continuity?.boundaryImagePolicy ?? "balanced"
    guard ["balanced", "strict"].contains(boundaryPolicy) else {
      throw unsupported("choose Balanced or Strict image boundaries for the Swift scene")
    }
    var scene: [String: Any] = ["version": versionTwo ? 2:1, "segments": segments,
      "overlap_frames": 25, "boundary_image_policy": boundaryPolicy,
      "soundscape": "", "music": ""]
    if decodeMode == "windowed" {
      scene["decode_mode"] = "windowed"
      scene["decode_window_frames"] = 361
    }
    content["scene"] = scene
    content["prompt"] = segments.enumerated().map { "Shot \($0.offset + 1): \($0.element["prompt"]!)" }.joined(separator: "\n\n")
    let ranges: [[String: Any]] = zip(members, zip(boundaries.dropLast(), lengths)).map { member, range in
      ["clip_id": member.id.uuidString, "source_in": Double(range.0) / fps,
        "duration": Double(range.1) / fps]
    }
    var report = reports[0]
    report["scene"] = ["version": versionTwo ? 2:1, "members": ranges,
      "frame_rate": fps, "publication_mode": decodeMode == "windowed"
        ? "windowed_decode_native_latent_chain" : "single_decode_native_latent_chain"] as [String: Any]
    report["scenePlan"] = ["requested_durations": members.map(\.duration),
      "segment_frame_counts": lengths, "total_frames": boundaries.last! + 1]
    report["task"] = "scene"
    report["conditioning"] = ["frames": boundaries.last! + 1,
      "inputs": versionTwo ? (((content["conditioning"] as? [String:Any])?["inputs"] as? [Any])?.count ?? 0)
        : recipes.reduce(0) { $0 + (((($1["conditioning"] as? [String: Any])?["inputs"] as? [Any])?.count) ?? 0) }]
    report["resolvedFingerprint"] = try fingerprint(content)
    report["warnings"] = reports.flatMap { $0["warnings"] as? [String] ?? [] } +
      (versionTwo ? ["Scene interior and terminal image anchors are experimental; visual quality is not qualified."] : [])
    if versionTwo { report["productionQualified"] = false }
    return ["recipe": content, "report": report]
  }
  private static func resolve(_ request: [String: Any]) throws -> Context {
    guard let projectValue = request["project"], let runtime = request["runtime"] as? [String: Any],
      let clipID = request["clipID"] as? String else { throw StudioError.invalid("Missing Studio preparation request.") }
    let project = try JSONDecoder().decode(StudioProject.self, from: data(projectValue))
    guard let clip = project.clips.first(where: { $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }),
      clip.engine == .ltx25 else { throw StudioError.invalid("Select an LTX 2.5 clip.") }
    if clip.audioDriverSelection != nil && clip.audioDriverMixKey?.isEmpty != false {
      throw StudioError.invalid("Prepare and audition the current timeline audio mix before native A2V generation.")
    }
    guard ["independent", "frame", "motion"].contains(clip.continuityMode),
      !project.isContinuousSceneMember(clip),
      (clip.extensionDirection.isEmpty || clip.extensionDirection == "after"),
      !(clip.continuityMode != "independent" && !clip.extensionDirection.isEmpty) else {
      throw unsupported("this continuity, scene, music driver or extension direction is not yet ported")
    }
    let selection = clip.generationSelection
    let guided = selection?.ltx25Guidance
    let automatic = selection?.ltx25AutomaticDuration
    let keyframes=selection?.ltx25Keyframes
    let singleStage=selection?.ltx25SingleStage
    if let singleStage {
      guard singleStage.experimentalEnabled,guided == nil,automatic == nil,
        clip.continuityMode == "independent",clip.extensionDirection.isEmpty,
        singleStage.method == .cfgpp || singleStage.negativeSchedule == .full else {
        throw unsupported("single-stage sampling needs explicit opt-in, an independent distilled shot, and a matching negative schedule")
      }
    }
    if let keyframes {
      guard keyframes.experimentalEnabled,(0...8).contains(keyframes.generatedCount),
        clip.continuityMode == "independent",clip.extensionDirection.isEmpty else {
        throw unsupported("ordinary keyframes require explicit experimental opt-in, 0–8 generated slots and an independent shot")
      }
    }
    guard (guided != nil || (selection?.steps == nil && selection?.refinementSteps == nil && selection?.cfg == nil)),
      selection?.h3SamplingMethod == nil,selection?.h3Reference == nil,selection?.h3Joint == nil,selection?.h3MotionFidelity == nil,
      clip.attachments.allSatisfy({ $0.h3LoRA == nil && $0.h3ReferencePlacement == nil }),selection?.shift == nil,
      selection?.projectionBackend == nil, selection?.transformerBackend == nil else {
      throw unsupported("sampling and backend overrides cannot change the qualified distilled schedule")
    }
    guard selection?.memoryPolicy == nil || selection?.memoryPolicy == "recipe" else {
      throw unsupported("explicit memory-policy overrides are not supported; Swift uses staged loading")
    }
    guard guided != nil || singleStage?.method == .cfgpp || clip.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw unsupported("distilled 8 + 3 sampling does not evaluate negative prompts; clear the negative prompt")
    }
    let frameSource = try clip.continuityMode == "frame" ? NativeLTXFrameSource(project: project, clip: clip) : nil
    let movieSource = try (clip.continuityMode == "motion" || clip.extensionDirection == "after")
      ? NativeLTXMovieSource(project: project, clip: clip) : nil
    if movieSource != nil, let selectedTask = selection?.task,
      !["t2v", "extension"].contains(selectedTask) {
      throw unsupported("motion extension cannot combine with selected task \(selectedTask)")
    }
    let task = movieSource != nil ? "extension" : frameSource != nil
      ? (clip.attachments.contains { [.last, .keyframe].contains($0.role) } ? "fflf" : "i2v")
      : selection?.task ?? clip.inferredTask
    guard ["t2v", "i2v", "fflf", "a2v", "extension","ref2va","control"].contains(task) else { throw unsupported("task \(task) is not yet ported") }
    if automatic != nil {
      guard automatic?.experimentalEnabled == true, ["t2v","i2v","fflf"].contains(task),
        movieSource == nil, !project.isContinuousSceneMember(clip) else {
        throw unsupported("automatic duration requires explicit experimental opt-in and an ordinary T2V, I2V or first/last shot; audio-driven, scene, extension and specialized controls require manual timing")
      }
    }
    guard clip.audioDriverSelection == nil || task == "a2v" else {
      throw unsupported("a prepared timeline audio driver requires the A2V task")
    }
    let profiles = try catalog(directory: runtime["profilesDirectory"] as? String ?? "")
    var candidates = profiles.filter {
      (clip.profileID == "auto" || $0["id"] as? String == clip.profileID)
        && (($0["generation"] as? [String: Any])?["supportedTasks"] as? [String] ?? []).contains(task)
        && (($0["generation"] as? [String: Any])?["pipelineMode"] as? String ?? "distilled") == (guided?.mode.rawValue ?? "distilled")
        && (automatic == nil
          ? (($0["generation"] as? [String:Any])?["durationMode"] as? String ?? "manual") == "manual"
          : (($0["generation"] as? [String:Any])?["automaticDurationAvailable"] as? Bool == true))
    }
    if task == "control",clip.profileID == "auto" {
      let controls=clip.attachments.filter { $0.role == .control }.map(\.controlType)
      if !controls.isEmpty {
        let family=controls == ["motion_track"] ? "motion_track"
          : controls.filter { $0 == "crossview_warp" }.count == 2
            ? (controls.contains("ingredients_reference_sheet") ? "crossview_ingredients" : "crossview_warp")
          : controls.allSatisfy { $0 == "ingredients_reference_sheet" } ? "ingredients"
          : controls.allSatisfy { ["canny_edges","depth_map","pose_skeleton"].contains($0) } ? "union" : "unsupported"
        candidates=candidates.filter { ($0["generation"] as? [String:Any])?["referenceFamily"] as? String == family }
      }
    }
    if selection != nil, selection?.preset != .custom, clip.profileID == "auto" {
      let nativeTask = task == "i2v" ? "fflf" : task
      candidates = candidates.filter { $0["task"] as? String == nativeTask }
        + candidates.filter { $0["task"] as? String != nativeTask }
    }
    if clip.profileID == "auto" {
      candidates = candidates.filter {
        (($0["generation"] as? [String: Any])?["dfrEnabled"] as? Bool ?? false) == false
      } + candidates.filter {
        (($0["generation"] as? [String: Any])?["dfrEnabled"] as? Bool ?? false) == true
      }
    }
    guard let chosen = candidates.first, let profile = chosen["id"] as? String else {
      throw StudioError.invalid(automatic != nil
        ? "No compatible automatic-duration profile is available. Link a valid LTX 2.5 duration head in native model setup and enable experimental automatic timing for an ordinary shot."
        : guided == nil
        ? "No compatible Swift LTX recipe is available. Import a distilled recipe or select Automatic."
        : "No compatible Swift Dev guided recipe is available. Use model setup to link an unmerged Dev paged pack and the distilled refinement adapter, then select Automatic.")
    }
    let assets = try JSONDecoder().decode([MediaAsset].self, from: data(request["globalAssets"] ?? []))
    let selectedRecipe = try recipe(profile)
    let selectedConfig=selectedRecipe["config"] as! [String:Any]
    if let declared=selectedConfig["generated_keyframes"] {
      guard let count=declared as? NSNumber,CFGetTypeID(count) != CFBooleanGetTypeID(),
        count.doubleValue.isFinite,count.doubleValue.rounded()==count.doubleValue,
        (0...8).contains(count.intValue),keyframes != nil || count.intValue == 0 else {
        throw unsupported("profile generated-keyframe slots require explicit experimental keyframe controls")
      }
    }
    if singleStage != nil {
      guard specialization(selectedRecipe) == nil, !guidedProfile(selectedRecipe),
        (selectedConfig["dfr_enabled"] as? Bool ?? false) == false,
        ["t2v","i2v","fflf","a2v"].contains(task),singleStage?.method != .cfgpp || task != "a2v" else {
        throw unsupported("single-stage sampling requires an ordinary distilled profile; CFG++ generates audio rather than freezing an A2V driver")
      }
    }
    if keyframes != nil {
      guard specialization(selectedRecipe) == nil,(selectedConfig["dfr_enabled"] as? Bool ?? false) == false,
        ["t2v","i2v","fflf","a2v"].contains(task) else {
        throw unsupported("ordinary keyframes do not combine with specialized, DFR, extension or scene recipes")
      }
    }
    if automatic != nil, specialization(selectedRecipe) != nil {
      throw unsupported("specialized reference/control profiles require manual duration")
    }
    if let guided {
      guard guided.experimentalEnabled, guidedProfile(selectedRecipe),
        frameSource == nil, movieSource == nil, !project.isContinuousSceneMember(clip) else {
        throw unsupported("guided sampling requires explicit experimental opt-in and an ordinary Dev profile")
      }
      _ = try guidedConfig(selectedRecipe["config"] as! [String: Any], selection: selection!, negativePrompt: clip.negativePrompt)
    }
    var warnings = selection?.preset == .speed
      ? [specialization(selectedRecipe) == nil || specialization(selectedRecipe) == "union" ? "Speed preserves the qualified 8 + 3 sampling schedule." : "Speed preserves eight full-resolution reference evaluations."] : []
    if guided != nil { warnings = ["Swift Dev guided sampling is experimental. Visual quality and performance are not qualified."] }
    if automatic != nil { warnings.append("Automatic duration is experimental; the accepted take uses the predicted video interval. Audio-driven and continuous scenes require manual duration.") }
    if keyframes != nil { warnings.append("Ordinary timed images and generated keyframes are experimental. Generated slots apply to stage one only; quality is not qualified.") }
    if singleStage != nil { warnings = ["Full-resolution single-stage sampling is experimental until real-model qualification. The selected negative schedule controls actual transformer evaluations."] }
    if guided == nil,singleStage == nil, let inherited = (selectedRecipe["config"] as? [String: Any])?["negative_prompt"] as? String,
      !inherited.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      warnings.append("The selected profile's negative prompt is not evaluated by distilled Swift sampling and is omitted.")
    }
    return Context(project: project, clip: clip, assets: project.assets + assets, runtime: runtime,
      profile: profile, recipe: selectedRecipe, task: task, warnings: warnings,
      frameSource: frameSource, movieSource: movieSource)
  }
  /// Resolve model paths only after admitting the actual movie editor intent.
  /// The independent probe is never composed or rendered; its schedule and duration
  /// do not become movie settings. The dedicated movie request owns all execution.
  public static func componentsForMovie(request:[String:Any],settings:LTX25MovieUpscaleSettings)
    throws -> (components:[String:String],profileID:String,selectionFingerprint:String,ffmpeg:URL) {
    guard let value=request["project"],let clipID=request["clipID"] as? String else {
      throw StudioError.invalid("Missing Studio movie preparation request.")
    }
    var project=try JSONDecoder().decode(StudioProject.self,from:data(value))
    guard let index=project.clips.firstIndex(where:{ $0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame }) else {
      throw StudioError.invalid("Select the movie upscale clip.")
    }
    let original=project.clips[index]
    try NativeLTXMoviePreparation.validateEditor(clip:original,settings:settings,
      isContinuousScene:project.isContinuousSceneMember(original))
    guard original.generationSelection?.ltx25MovieUpscale == settings,
      original.generationSelection?.task == "video_upscale" else {
      throw StudioError.invalid("Movie settings must match the frozen editor task.")
    }
    project.clips[index].generationSelection=GenerationSelection(task:"t2v",preset:.custom)
    project.clips[index].attachments=[]
    var probe=request;probe["project"]=try object(project)
    let context=try resolve(probe)
    guard specialization(context.recipe)==nil,!guidedProfile(context.recipe) else {
      throw StudioError.invalid("Source movie refinement needs ordinary native distilled model components.")
    }
    let movieProfileControls=NativeLTXDiffusionVAE.profileControls(context.recipe["config"] as? [String:Any] ?? [:])
    guard movieProfileControls.isEmpty else { throw StudioError.invalid("Movie upscaling uses its explicit advanced decoder controls; select a profile without custom Diffusion VAE controls.") }
    var modelComponents=context.recipe["components"] as! [String:Any]
    guard (modelComponents["loras"] as? [Any] ?? []).isEmpty,
      (modelComponents["stage_two_loras"] as? [Any] ?? []).isEmpty else {
      throw StudioError.invalid("Movie upscaling cannot ignore adapters embedded in the selected profile.")
    }
    if let detail=(context.recipe["config"] as? [String:Any])?["dfr_detailing_lora_path"] as? String {
      modelComponents["dfr_detailing_lora_path"]=detail
    }
    let paths=try NativeLTXMoviePreparation.components(from:modelComponents,
      mode:settings.mode,pixelSpatialAdapterPath:settings.pixelSpatialAdapterPath)
    let configured=context.runtime["ffmpegPath"] as? String ?? ""
    let executable=configured.isEmpty ? [context.recipe["ffmpeg"] as? String ?? "","/opt/homebrew/bin/ffmpeg","/usr/local/bin/ffmpeg"].first {
      !$0.isEmpty && FileManager.default.isExecutableFile(atPath:$0)
    } ?? "" : configured
    guard FileManager.default.isExecutableFile(atPath:executable) else {
      throw StudioError.invalid("Select executable FFmpeg for native movie preparation.")
    }
    return (paths,context.profile,try fingerprint(context.recipe),URL(fileURLWithPath:try canonical(executable)))
  }

  public static func describe(request: [String: Any]) throws -> [String: Any] {
    if try NativeLTXMovieEditorPreparation.matches(request:request) { return try NativeLTXMovieEditorPreparation.describe(request:request) }
    try NativeMovieIntervalAdmission.rejectInOrdinaryRequest(request)
    if let (project, members) = try sceneRequest(request) {
      var independent = NativeLTXSceneImageInputs.requiresVersionTwo(members)
        ? try NativeLTXSceneImageInputs.baseProject(project,members:members) : project
      for index in independent.clips.indices where members.contains(where: { $0.id == independent.clips[index].id }) {
        independent.clips[index].continuity = ClipContinuity(mode: "independent")
      }
      var leader = request
      leader["project"] = try object(independent)
      leader["clipID"] = members[0].id.uuidString
      var result = try describe(request: leader)
      do {
        let composed = try composeScene(request, project: project, members: members)
        result["fingerprint"] = (composed["report"] as? [String: Any])?["resolvedFingerprint"]
        let scene = (composed["recipe"] as? [String: Any])?["scene"] as? [String: Any]
        let segments = scene?["segments"] as? [[String: Any]] ?? []
        let laterImages = try segments.compactMap { ($0["image_input"] as? [String: Any])?["path"] as? String }
          .map(canonical)
        let images = (((composed["recipe"] as? [String:Any])?["conditioning"] as? [String:Any])?["inputs"] as? [[String:Any]] ?? [])
          .filter { $0["kind"] as? String == "image" }
        let rootImages = try images.compactMap { $0["path"] as? String }.map(canonical)
        let plannedSound = try members.compactMap { $0.musicSource?.path }.map(canonical)
        result["sourcePaths"] = Array(Set((result["sourcePaths"] as? [String] ?? []) +
          laterImages + rootImages + plannedSound)).sorted()
      } catch {
        result["readinessErrors"] = (result["readinessErrors"] as? [String] ?? []) + [error.localizedDescription]
      }
      return result
    }
    let context = try resolve(request)
    var errors: [String] = [], resolved = ""
    var content = context.recipe
    do {
      // Only the digest/controls escape description. The movie path identifies
      // the planned dependency; a runnable recipe requires extracted pixels.
      let composed = try compose(context, firstFramePath: context.frameSource?.url.path,
        moviePath: context.movieSource?.url.path,
        movieSHA256: context.movieSource == nil ? nil : String(repeating: "0", count: 64))
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
    if let source = context.clip.musicSource { dependencies.append(try canonical(source.path)) }
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
  public static func compose(request: [String: Any]) throws -> [String: Any] {
    try NativeMovieIntervalAdmission.rejectInOrdinaryRequest(request)
    if let (project, members) = try sceneRequest(request) {
      return try composeScene(request, project: project, members: members)
    }
    return try compose(resolve(request))
  }
  private static func compose(_ context: Context, firstFramePath: String? = nil,
    moviePath: String? = nil, movieSHA256: String? = nil) throws -> [String: Any] {
    try Task.checkCancellation()
    guard context.frameSource == nil || firstFramePath != nil else {
      throw StudioError.invalid("Match previous frame requires native media preparation before composing a runnable recipe.")
    }
    guard context.movieSource == nil || (moviePath != nil && movieSHA256 != nil) else {
      throw StudioError.invalid("Motion extension requires native media preparation before composing a runnable recipe.")
    }
    let clip = context.clip
    let prompt = clip.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else { throw StudioError.invalid("Write a prompt before preparing the render.") }
    var content = context.recipe
    let specialized=specialization(content)
    if ["union","motion_track"].contains(specialized ?? ""),(clip.generationWidth%128 != 0 || clip.generationHeight%128 != 0) {
      throw StudioError.invalid("Union Control requires final dimensions divisible by 128 for its quarter-canvas guide.")
    }
    // Stored profile media/context never become hidden clip dependencies.
    content.removeValue(forKey: "continuation"); content.removeValue(forKey: "reference_images")
    var config = content["config"] as! [String: Any]
    let keyframes=clip.generationSelection?.ltx25Keyframes
    let singleStage=clip.generationSelection?.ltx25SingleStage
    let dimensionGrid=singleStage == nil ? 64:32
    let automatic = clip.generationSelection?.ltx25AutomaticDuration
    guard let fps = config["frame_rate"] as? Double, fps.isFinite, (1...120).contains(fps),
      clip.duration.isFinite, clip.duration > 0, automatic != nil || clip.duration <= 20,
      (0...Int(UInt32.max)).contains(clip.seed), (64...4096).contains(clip.generationWidth),
      (64...4096).contains(clip.generationHeight), clip.generationWidth % dimensionGrid == 0, clip.generationHeight % dimensionGrid == 0 else {
      throw unsupported("invalid duration, frame rate, dimensions, seed or automatic duration")
    }
    var automaticPolicy: [String:Any]?
    if let automatic {
      guard automatic.experimentalEnabled, specialized == nil,
        context.movieSource == nil, !context.project.isContinuousSceneMember(clip),
        ["t2v","i2v","fflf"].contains(context.task),
        let head = (content["components"] as? [String:Any])?["duration_head_path"] as? String else {
        throw unsupported("automatic duration needs an ordinary shot and a compatible duration head")
      }
      let headPath = try canonical(head), sha = try NativeLTXAutomaticDuration.validateHead(at:URL(fileURLWithPath:headPath))
      if let pin = (content["components"] as? [String:Any])?["duration_head_header_sha256"] {
        guard pin as? String == sha else { throw StudioError.invalid("The duration-head header changed. Relink its native model profile before preparing automatic timing.") }
      }
      automaticPolicy = ["head_checkpoint_path":headPath,"head_header_sha256":sha,
        "minimum_seconds":automatic.minimumSeconds,"maximum_seconds":automatic.maximumSeconds]
      config["duration_mode"] = "automatic"
      config["auto_duration_min_seconds"] = automatic.minimumSeconds
      config["auto_duration_max_seconds"] = automatic.maximumSeconds
    }
    let additionalFrames = try automatic.map {
      try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:$0.minimumSeconds,maximumSeconds:$0.maximumSeconds,fps:fps)-1
    } ?? Int(ceil(clip.duration * fps / 8 - 1e-9)) * 8
    let contextFrames = context.movieSource == nil ? 0 : clip.continuityMode == "motion" ? 49 : 25
    let frames = contextFrames + additionalFrames + (context.movieSource == nil ? 1 : 0)
    if ["ingredients","crossview_ingredients"].contains(specialized ?? ""),frames<121 {
      throw StudioError.invalid("Ingredients needs at least 121 frames (five seconds at 24 fps).")
    }
    let duration = Double(additionalFrames) / fps
    guard Double(frames - 1) / fps <= (automatic == nil ? 20 : 30) else {
      throw unsupported("rounded generation plus source context exceeds 20 seconds")
    }
    config["width"] = clip.generationWidth; config["height"] = clip.generationHeight
    config["seed"] = clip.seed; config["duration_seconds"] = duration
    if let keyframes { config["generated_keyframes"]=keyframes.generatedCount }
    if let selection = clip.generationSelection, selection.ltx25Guidance != nil {
      config = try guidedConfig(config, selection: selection, negativePrompt: clip.negativePrompt)
    } else { config["negative_prompt"] = "" }
    if let singleStage {
      config["ic_lora_single_stage"]=true;config["stage1_steps"]=8;config["stage2_steps"]=0
      config["stage1_sampler"]=singleStage.method.rawValue;config["stage2_sampler"]="euler"
      config["stage1_eta"]=singleStage.method == .euler ? 0:1;config["stage1_s_noise"]=1
      config["ancestral_seed_offset"]=10000;config["cfg_pp_batched"]=false
      config["cfg_pp_schedule"]=singleStage.negativeSchedule.rawValue
      config["negative_prompt"]=singleStage.method == .cfgpp ? clip.negativePrompt:""
    }
    var components = content["components"] as! [String: Any]
    if let diffusion=clip.generationSelection?.ltx25DiffusionVAE {
      guard let checkpoint=components["video_vae_path"] as? String else { throw StudioError.invalid("Diffusion VAE requires its selected native checkpoint.") }
      config=try NativeLTXDiffusionVAE.apply(diffusion,checkpoint:URL(fileURLWithPath:checkpoint),config:config)
    }
    if let automaticPolicy {
      components["duration_head_path"] = automaticPolicy["head_checkpoint_path"]
      components["duration_head_header_sha256"] = automaticPolicy["head_header_sha256"]
    }
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
    let allowed: Set<MediaRole> = ["t2v", "extension"].contains(context.task) ? [] : context.task == "i2v" ? (keyframes == nil ? [.first] : [.first,.last,.keyframe])
      : context.task == "a2v" ? (keyframes == nil ? [.audioDriver,.first] : [.audioDriver,.first,.last,.keyframe]) : context.task == "ref2va" ? [.reference]
      : context.task == "control" ? [.control] : [.first, .last, .keyframe]
    guard roles.isSubset(of: allowed) else { throw unsupported("attached media conflict with \(context.task); no inputs were discarded") }
    if context.task == "i2v", !roles.contains(.first),keyframes == nil || roles.isDisjoint(with:[.last,.keyframe]) { throw StudioError.invalid("Image to video requires an image attachment.") }
    if context.task == "fflf", keyframes == nil,clip.generationSelection != nil, !roles.isSuperset(of: [.first, .last]) {
      throw StudioError.invalid("First and last frames requires First frame and Last frame images.")
    }
    if context.task == "a2v", attachments.filter({ $0.role == .audioDriver }).count != 1 {
      throw StudioError.invalid("Audio to video requires exactly one source audio attachment.")
    }
    if context.task == "a2v", keyframes == nil,attachments.filter({ $0.role == .first }).count > 1 {
      throw StudioError.invalid("Audio to video accepts at most one opening-frame image.")
    }
    let msrImages=attachments.filter { item in
      item.role == .reference && context.assets.last(where:{$0.id == item.assetID})?.kind == .image
    }
    let msrOrdered=msrImages.filter { ($0.referenceRole ?? "subject") != "background" } + msrImages.filter { $0.referenceRole == "background" }
    var voiceSlots=Set<Int>()
    guard specialized == "msr" || !attachments.contains(where:{$0.msrAudioReferenceID != nil}) else {
      throw unsupported("MSR voice bindings require an MSR V2 profile")
    }
    if specialized == "msr",!(1...5).contains(msrImages.count) {
      throw StudioError.invalid("MSR needs one to five described still images.")
    }
    if ["ingredients","union","motion_track"].contains(specialized ?? ""),attachments.filter({ $0.role != .lora }).count != 1 {
      throw StudioError.invalid("This control profile needs exactly one guide attachment.")
    }
    if let family=specialized,family.hasPrefix("crossview"),
      attachments.filter({ $0.role != .lora }).count != (family == "crossview_ingredients" ? 3 : 2) {
      throw StudioError.invalid("CrossView needs a warp movie, its original source movie, and a sheet when using Ingredients.")
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
      if specialized == "msr",attachment.role == .reference,asset.kind == .audio {
        guard let bound=attachment.msrAudioReferenceID,let index=msrOrdered.firstIndex(where:{$0.id == bound}),index<2,
          msrOrdered[index].referenceRole != "background",voiceSlots.insert(index+1).inserted,
          attachment.strength == 1,attachment.time == 0,
          attachment.referenceRole == nil,attachment.referencePriority == nil,attachment.referenceFrames == nil,
          attachment.referenceSizePolicy == nil,attachment.attentionStrength == nil,
          let start=attachment.audioSourceStart,start.isFinite,(0...86400).contains(start),
          let duration=attachment.audioSourceDuration,duration.isFinite,(0.033...86400).contains(duration),
          asset.duration>0,start+duration<=asset.duration+0.01 else {
          throw StudioError.invalid("MSR V2 voices need a unique binding to character image 1 or 2, strength 1 and a valid explicit source interval.")
        }
        inputs.append(["id":attachment.id.uuidString,"kind":"audio","role":"reference","path":path,
          "sha256":try sourceSHA256(path,maxBytes:4*1024*1024*1024),"strength":1,"image_slot":index+1,
          "source_start_seconds":start,"source_duration_seconds":duration]);continue
      }
      if attachment.role == .audioDriver {
        let preparedTimelineMix = clip.audioDriverSelection != nil
        guard !preparedTimelineMix || (clip.audioDriverMixKey?.isEmpty == false &&
          asset.scope == .clip && asset.owner == clip.id &&
          attachment.audioSourceStart == nil && attachment.audioSourceDuration == nil &&
          asset.duration + 0.01 >= duration) else {
          throw StudioError.invalid("Prepare the current timeline audio mix before native A2V generation.")
        }
        let start = preparedTimelineMix ? 0 : attachment.audioSourceStart
        let sourceDuration = preparedTimelineMix ? duration : attachment.audioSourceDuration
        guard asset.kind == .audio, attachment.strength == 1,
          let start, start.isFinite, (0...86400).contains(start),
          let sourceDuration, sourceDuration.isFinite,
          (0.001...86400).contains(sourceDuration),
          asset.duration <= 0 || start + sourceDuration <= asset.duration + 0.01 else {
          throw StudioError.invalid("Choose one audio source with strength 1 and an explicit valid source in-point and duration.")
        }
        inputs.append(["id": attachment.id.uuidString, "kind": "audio", "role": "audio_driver",
          "path": path, "strength": 1, "source_start_seconds": start,
          "source_duration_seconds": sourceDuration]); continue
      }
      if attachment.role == .reference || attachment.role == .control {
        if ["motion_track","crossview_warp","crossview_ingredients"].contains(specialized ?? ""),
          (attachment.attentionStrength ?? 1) != 1 ||
            (attachment.referenceSizePolicy ?? "sol_auto") != "sol_auto" {
          throw StudioError.invalid("This IC control does not expose per-reference attention or size overrides.")
        }
        if ["motion_track","crossview_warp","crossview_ingredients"].contains(specialized ?? ""),
          attachment.controlType != "ingredients_reference_sheet" {
          let crossview=specialized != "motion_track",role=attachment.referenceRole ?? ""
          guard asset.kind == .video,attachment.role == .control,attachment.time == 0,
            attachment.controlType == (crossview ? "crossview_warp" : "motion_track"),
            !crossview || ["warp","source"].contains(role),
            attachment.strength.isFinite,(0...1).contains(attachment.strength) else {
            throw StudioError.invalid("MotionTrack needs a track movie; CrossView needs explicitly labeled warp/source movies at time zero.")
          }
          var input:[String:Any]=["id":attachment.id.uuidString,"kind":"video","role":"control","path":path,
            "control_type":attachment.controlType,"strength":attachment.strength,
            "sha256":try sourceSHA256(path,maxBytes:4*1024*1024*1024)]
          if crossview { input["reference_role"]=role }
          inputs.append(input);continue
        }
        if specialized == "union" {
          guard asset.kind == .video,attachment.role == .control,
            ["canny_edges","depth_map","pose_skeleton"].contains(attachment.controlType),
            attachment.strength.isFinite,(0...1).contains(attachment.strength),attachment.time == 0 else {
            throw StudioError.invalid("Union needs one preprocessed Canny, depth or pose movie at time zero, with strength from 0 to 1.")
          }
          inputs.append(["id":attachment.id.uuidString,"kind":"video","role":"control","path":path,
            "control_type":attachment.controlType,"strength":attachment.strength,
            "sha256":try sourceSHA256(path,maxBytes:4*1024*1024*1024)]);continue
        }
        guard asset.kind == .image,attachment.strength.isFinite,(0...1).contains(attachment.strength),
          !attachment.description.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
          throw StudioError.invalid("Describe each reference image and use strength from 0 to 1.")
        }
        var input:[String:Any]=["id":attachment.id.uuidString,"kind":"image","path":path,
          "sha256":try sourceSHA256(path),"strength":attachment.strength,"description":attachment.description]
        if attachment.role == .reference {
          let role=attachment.referenceRole ?? "subject",priority=attachment.referencePriority ?? "auto"
          let referenceFrames=attachment.referenceFrames ?? "auto",size=attachment.referenceSizePolicy ?? "sol_auto"
          let attention=attachment.attentionStrength ?? 1
          guard ["subject","object","clothing","background"].contains(role),
            ["auto","primary","supporting","background"].contains(priority),
            ["auto","25","33"].contains(referenceFrames),["sol_auto","quality","balanced","speed"].contains(size),
            attention.isFinite,(0...1).contains(attention) else { throw StudioError.invalid("Invalid MSR reference controls.") }
          input["role"]="reference";input["reference_role"]=role;input["reference_priority"]=priority
          input["reference_frames"]=referenceFrames;input["reference_size_policy"]=size;input["attention_strength"]=attention
        } else {
          guard attachment.controlType == "ingredients_reference_sheet" else {
            throw unsupported("this profile requires an Ingredients sheet")
          }
          input["role"]="control";input["control_type"]=attachment.controlType
        }
        inputs.append(input);continue
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
    if let family=specialized,family.hasPrefix("crossview") {
      guard inputs.filter({ $0["reference_role"] as? String == "warp" }).count == 1,
        inputs.filter({ $0["reference_role"] as? String == "source" }).count == 1,
        inputs.filter({ $0["control_type"] as? String == "ingredients_reference_sheet" }).count == (family == "crossview_ingredients" ? 1 : 0) else {
        throw StudioError.invalid("CrossView needs one warp and one original source movie, with a single sheet only for the combined profile.")
      }
      inputs=inputs.filter { $0["reference_role"] as? String == "warp" }
        + inputs.filter { $0["reference_role"] as? String == "source" }
        + inputs.filter { $0["control_type"] as? String == "ingredients_reference_sheet" }
    }
    if inputs.filter({ $0["reference_role"] as? String == "background" }).count>1 {
      throw StudioError.invalid("MSR accepts at most one background image.")
    }
    if let keyframes {
      let images=inputs.filter { $0["kind"] as? String == "image" }
      let indices=images.map { $0["frame_index"] as? String == "last" ? frames-1 : $0["frame_index"] as! Int }
      guard images.count<=8,Set(indices).count==indices.count,
        indices.allSatisfy({ $0>=0 && $0<frames }),
        context.task != "fflf" || !images.isEmpty,
        keyframes.generatedCount == 0 || frames>=keyframes.generatedCount+2 else {
        throw unsupported("ordinary keyframes require up to eight unique rounded input frames inside the model interval")
      }
      let plane=(clip.generationWidth/32)*(clip.generationHeight/32)
      let rows=(additionalFrames/8+1+images.count-(indices.contains(0) ? 1 : 0)+keyframes.generatedCount)*plane
      guard rows<=131072 else { throw unsupported("ordinary keyframe tokens exceed native video admission") }
    } else if !["a2v","extension","ref2va","control"].contains(context.task) {
      let indices = inputs.map { $0["frame_index"] as? String == "last" ? frames - 1 : $0["frame_index"] as! Int }
      guard Set(indices).count == indices.count, indices.allSatisfy({ $0 == 0 || $0 == frames - 1 }),
        inputs.count <= 2, context.task == "t2v" || indices.contains(0) else {
        throw unsupported("only unique first and last endpoints are supported")
      }
    }
    if let source = clip.musicSource {
      let sourcePath = try verifyPlannedAudio(source)
      guard source.task == (context.task == "a2v" ? "a2v" : "t2v"),
        source.duration <= clip.duration + 1e-4,
        clip.duration - source.duration < 1 / fps + 1e-4 else {
        throw StudioError.invalid("The planned song task or duration differs from this clip.")
      }
      if source.task == "a2v" {
        guard let driver = inputs.first(where: { $0["role"] as? String == "audio_driver" }),
          driver["path"] as? String == sourcePath,
          let start = driver["source_start_seconds"] as? Double,
          let duration = driver["source_duration_seconds"] as? Double,
          abs(start - source.start) < 1e-4,
          abs(duration - source.duration) < 1e-4 else {
          throw StudioError.invalid("The planned song and A2V attachment must select the same source interval.")
        }
      }
    }
    if !loras.isEmpty { components["loras"] = loras }
    var contract = content["conditioning"] as? [String: Any] ?? [:]
    let task = context.task == "i2v" ? "fflf" : context.task
    for key in ["version", "task", "inputs", "extension", "publication_audio"] { contract.removeValue(forKey: key) }
    if (content["conditioning"] as? [String: Any])?["task"] as? String != task { contract.removeValue(forKey: "audio_policy") }
    contract["version"] = 1; contract["task"] = task; contract["inputs"] = inputs
    if let family=specialized,["motion_track","crossview_warp","crossview_ingredients"].contains(family) {
      contract["control_family"]=family;contract["audio_policy"]=family == "motion_track" ? "generated" : "source"
    }
    if task == "extension" {
      guard let moviePath, let movieSHA256, context.movieSource != nil else {
        throw StudioError.invalid("Select and prepare an LTX extension source movie.")
      }
      contract["inputs"] = [["id": "extension-source", "kind": "video", "role": "reference",
        "path": try canonical(moviePath), "sha256": movieSHA256]]
      contract["extension"] = ["direction": "after", "context_frames": contextFrames,
        "additional_frames": additionalFrames]
      contract["audio_policy"] = "source_reencoded_and_generated_extension"
    }
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
    if let source = context.movieSource { identity["continuity"] = source.report }
    var report: [String: Any] = ["profile": URL(fileURLWithPath: context.profile).deletingPathExtension().lastPathComponent,
      "generation": descriptor(content), "resolvedFingerprint": try fingerprint(context.frameSource == nil ? content : identity),
      "selectionFingerprint": try fingerprint(context.recipe), "warnings": context.warnings, "task": task,
      "nativeFPS": fps, "preserveEditorialDuration": automatic == nil, "durationMode":automatic == nil ? "manual" : "automatic",
      "movieSettings": try object(clip.settings(in: context.project)),
      "nativePreparation": "swift", "conditioning": ["frames": frames,
        "inputs": task == "extension" ? 1 : inputs.count]]
    if clip.generationSelection?.ltx25Guidance != nil || clip.generationSelection?.ltx25SingleStage != nil { report["productionQualified"] = false }
    if let automaticPolicy { report["automaticDuration"] = automaticPolicy; report["productionQualified"] = false }
    if let keyframes {
      report["ordinaryKeyframes"]=["generatedCount":keyframes.generatedCount,
        "timedImageCount":inputs.filter { $0["kind"] as? String == "image" }.count,"generatedStages":[1]]
      report["productionQualified"]=false
    }
    if let source = context.frameSource { report["continuity"] = source.report }
    if let source = context.movieSource { report["continuity"] = source.report }
    return ["recipe": content, "report": report]
  }
  /// Native media extraction runs off the UI thread. Publish the image, recipe
  /// and original editor request together; never rewrite stored attachments.
  public static func prepareWithMedia(request: [String: Any], destination: URL) async throws -> [String: Any] {
    if try NativeLTXMovieEditorPreparation.matches(request:request) { return try await NativeLTXMovieEditorPreparation.prepare(request:request,destination:destination) }
    try NativeMovieIntervalAdmission.rejectInOrdinaryRequest(request)
    if try sceneRequest(request) != nil { return try prepare(request: request, destination: destination) }
    let context = try resolve(request)
    if let family=specialization(context.recipe),["union","motion_track","crossview_warp","crossview_ingredients"].contains(family) {
      var composed=try compose(context),content=composed["recipe"] as! [String:Any]
      var conditioning=content["conditioning"] as! [String:Any]
      var inputs=conditioning["inputs"] as! [[String:Any]]
      let originalSources:[(path:String,sha256:String,maxBytes:Int64)]=inputs.map { input in
        let sourcePath=input["path"] as! String,sourceDigest=input["sha256"] as! String
        let maxBytes:Int64=(input["kind"] as? String == "image") ? 134_217_728 : 4_294_967_296
        return (path:sourcePath,sha256:sourceDigest,maxBytes:maxBytes)
      }
      let config=content["config"] as! [String:Any],fps=config["frame_rate"] as! Double
      let frames=Int(((config["duration_seconds"] as! Double)*fps/8).rounded())*8+1
      let parent=destination.deletingLastPathComponent()
      try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
      let staging=parent.appendingPathComponent(".prepare-"+UUID().uuidString)
      try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:false)
      defer { try? FileManager.default.removeItem(at:staging) }
      let downscale=["union","motion_track"].contains(family) ? 2 : 1
      var guideReports:[[String:Any]]=[]
      for index in inputs.indices {
        let original=inputs[index]["path"] as! String,name=family == "union" ? "union-guide.rgb24" : "control-guide-\(index).rgb24"
        let digest:String
        if inputs[index]["kind"] as? String == "image" {
          if family == "crossview_ingredients",let description=inputs[index]["description"] as? String,
            let prompt=content["prompt"] as? String,
            !prompt.hasPrefix("Reference sheet:") && !prompt.hasPrefix("### Reference Sheet Description") {
            content["prompt"]="Reference sheet: "+description+"\n\nGenerated video: "+prompt
          }
          digest=try NativeLTXControlGuide.prepareSheet(source:URL(fileURLWithPath:original),
            destination:staging.appendingPathComponent(name),width:(config["width"] as! Int)/2,
            height:(config["height"] as! Int)/2,frames:frames)
          guard try sourceSHA256(original)==originalSources[index].sha256 else {
            throw StudioError.invalid("The Ingredients source changed after composition. Prepare the clip again.")
          }
          inputs[index].removeValue(forKey:"description")
          inputs[index]["kind"]="video";inputs[index]["reference_role"]="ingredients"
        } else {
          digest=try await NativeLTXControlGuide.prepare(source:URL(fileURLWithPath:original),
            destination:staging.appendingPathComponent(name),width:config["width"] as! Int,height:config["height"] as! Int,
            frames:frames,fps:fps,editorialDuration:context.clip.duration,referenceDownscale:downscale)
        }
        if inputs[index]["reference_role"] as? String == "source" {
          let audioName="crossview-source.wav"
          let audioDigest=try await NativeLTXControlGuide.prepareAudio(source:URL(fileURLWithPath:original),
            destination:staging.appendingPathComponent(audioName),duration:context.clip.duration)
          conditioning["publication_audio"]=["path":destination.appendingPathComponent(audioName).path,
            "sha256":audioDigest,"source_start_seconds":0,"source_duration_seconds":context.clip.duration]
        }
        inputs[index]["path"]=destination.appendingPathComponent(name).path;inputs[index]["sha256"]=digest;inputs[index]["format"]="rgb24"
        guideReports.append(["source":original,"sha256":digest,"frames":frames,
          "source_sha256":originalSources[index].sha256,
          "resize_policy":inputs[index]["reference_role"] as? String == "ingredients" ? "fit_letterbox" : "cover_center_crop",
          "width":(config["width"] as! Int)/(2*downscale),"height":(config["height"] as! Int)/(2*downscale),
          "nativePreparation":"swift","role":inputs[index]["reference_role"] ?? "control"])
      }
      for source in originalSources {
        guard try sourceSHA256(source.path,maxBytes:source.maxBytes)==source.sha256 else {
          throw StudioError.invalid("A control source changed after composition. Prepare the clip again.")
        }
      }
      conditioning["inputs"]=inputs;content["conditioning"]=conditioning
      var report=composed["report"] as! [String:Any]
      report["resolvedFingerprint"]=try fingerprint(content)
      report["controlGuides"]=guideReports
      if family == "union" { report["controlGuide"]=guideReports[0] }
      try data(content).write(to:staging.appendingPathComponent("recipe.json"),options:.withoutOverwriting)
      try data(request).write(to:staging.appendingPathComponent("editor-request.json"),options:.withoutOverwriting)
      try Task.checkCancellation();try FileManager.default.moveItem(at:staging,to:destination)
      composed["report"]=report
      return ["recipePath":destination.appendingPathComponent("recipe.json").path,"prompt":content["prompt"]!,"report":report]
    }
    if let source = context.movieSource {
      let provisional = try compose(context, moviePath: source.url.path,
        movieSHA256: String(repeating: "0", count: 64))
      let content = provisional["recipe"] as! [String: Any]
      let config = content["config"] as! [String: Any]
      let conditioning = content["conditioning"] as! [String: Any]
      let extensionWindow = conditioning["extension"] as! [String: Any]
      let contextFrames = extensionWindow["context_frames"] as! Int
      guard let fps = config["frame_rate"] as? Double,
        let width = config["width"] as? Int, let height = config["height"] as? Int,
        let ffmpeg = content["ffmpeg"] as? String else {
        throw StudioError.invalid("LTX extension source settings are incomplete.")
      }
      let parent = destination.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
      let staging = parent.appendingPathComponent(".prepare-" + UUID().uuidString)
      try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
      defer { try? FileManager.default.removeItem(at: staging) }
      let name = "source-tail.mp4"
      let digest = try await source.extract(to: staging.appendingPathComponent(name),
        ffmpeg: URL(fileURLWithPath: ffmpeg), fps: fps, width: width, height: height,
        contextFrames: contextFrames)
      let composed = try compose(context, moviePath: destination.appendingPathComponent(name).path,
        movieSHA256: digest)
      let recipe = composed["recipe"] as! [String: Any]
      var report = composed["report"] as! [String: Any]
      var continuity = source.report
      continuity["preparedSHA256"] = digest
      continuity["contextFrames"] = contextFrames
      report["continuity"] = continuity
      try data(recipe).write(to: staging.appendingPathComponent("recipe.json"), options: .withoutOverwriting)
      try data(request).write(to: staging.appendingPathComponent("editor-request.json"), options: .withoutOverwriting)
      try data(continuity).write(to: staging.appendingPathComponent("continuity.json"), options: .withoutOverwriting)
      try Task.checkCancellation(); try source.verify()
      try FileManager.default.moveItem(at: staging, to: destination)
      return ["recipePath": destination.appendingPathComponent("recipe.json").path,
        "prompt": recipe["prompt"]!, "report": report]
    }
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
