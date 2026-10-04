import Foundation
import InferenceContracts

/// Strict bridge from the saved headless H3 recipe to the first Swift T2VA
/// slice. Every setting this slice cannot execute fails before weights load.
public enum H3StudioRecipe {
  /// The released external extension is Ref2VA: complete source movie/audio
  /// followed by its last frame as a target-frame-zero seam guide.
  public static func compileExtension(data: Data,
    resolveVideo: (String, String) throws -> (H3VideoReference, H3StillReference)) throws
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
      conditioning["task"] as? String == "extension",
      (conditioning["audio_policy"] as? String ?? "generated") == "generated",
      let inputs = conditioning["inputs"] as? [[String: Any]], inputs.count == 1,
      let input = inputs.first,
      Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "strength", "sha256"]),
      let sourceID = input["id"] as? String, !sourceID.isEmpty,
      input["kind"] as? String == "video", input["role"] as? String == "reference",
      let path = input["path"] as? String, path.hasPrefix("/"),
      !path.utf8.contains(0),
      let digest = input["sha256"] as? String, digest.count == 64,
      digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      (input["strength"] as? NSNumber).map({
        CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1
      }) == true,
      let prompt = root["prompt"] as? String,
      ["subject_definitions:", "summary:", "[video continuation",
        "retention_analysis:", "detailed_description:", "overall_soundscape:",
        "non_diegetic_music:", "<Video 1>", "<Picture 1>"].allSatisfy(prompt.contains),
      let config = root["config"] as? [String: Any],
      let duration = config["duration_seconds"] as? Double,
      (4...15).contains(duration) else {
      throw H3CheckpointError.invalid("Swift H3 external extension needs one full-strength audiovisual source, a 4–15 second Ref2VA window and the continuation prompt structure.")
    }
    let noise = try H3ReferenceNoiseControls.parse(config)
    removeReferenceNoise(&root)
    components.removeValue(forKey: "vision_encoder")
    components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [],
      "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root))
    let (source, last) = try resolveVideo(path, digest)
    guard source.audio != nil, source.frameCount >= 5 else {
      throw H3CheckpointError.invalid("H3 external extension requires a movie with soundtrack.")
    }
    try H3StillReferenceMedia.validateCanvas(width: last.width, height: last.height)
    guard last.rgb8.count == last.width * last.height * 3 else {
      throw H3CheckpointError.invalid("H3 extension seam needs complete bounded RGB8 pixels.")
    }
    return try H3Ref2VAStillRequest(prompt: base.prompt,
      mediaReferences: [.video(source), .timedImage(last, frame: 0)],
      width: base.geometry.width, height: base.geometry.height,
      durationSeconds: base.durationSeconds, seed: base.seed,
      requestedSteps: base.requestedSteps, transformer: base.transformer,
      qwenPages: base.qwenPages, qwenVision: URL(fileURLWithPath: vision),
      tokenizer: base.tokenizer, videoVAE: base.videoVAE,
      audioVAE: base.audioVAE, turboLoRA: base.turboLoRA,
      turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs, loRAAdapters: base.loRAAdapters,
      videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      samplingMethod: base.samplingMethod, referenceNoise: noise)
  }

  /// Admit FL2VA's timed keyframe contract before reading any image.
  /// The text recipe validator still owns every shared execution control.
  public static func compileFL2VA(data: Data, canvasAdmission: H3CanvasAdmission = .ordinary,
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
    let noise = try H3ReferenceNoiseControls.parse(root["config"] as? [String: Any] ?? [:])
    removeReferenceNoise(&root)
    components.removeValue(forKey: "vision_encoder")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [],
      "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root), canvasAdmission: canvasAdmission)
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
      images: images, anchors: anchors, referenceNoise: noise)
  }

  /// Reuse the strict T2VA execution-control admission after removing only the
  /// reference-specific fields. Media is resolved after every recipe control
  /// passes, so an unsupported input can never be silently discarded.
  public static func compileStillReferences(data: Data,
    resolveImage: (String) throws -> H3StillReference) throws -> H3Ref2VAStillRequest {
    guard data.count <= 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let conditioning = root["conditioning"] as? [String: Any],
      let inputs = conditioning["inputs"] as? [[String: Any]],
      (1...9).contains(inputs.count), inputs.allSatisfy({ $0["kind"] as? String == "image" }) else {
      throw H3CheckpointError.invalid("The still-only compiler needs one to nine image references.")
    }
    return try compileMediaReferences(data: data) { path, kind, _ in
      guard kind == "image" else { throw H3CheckpointError.invalid("The still-only resolver needs image references.") }
      return .image(try resolveImage(path))
    }
  }

  /// Admit ordered still, video and standalone audio Ref2VA media.
  /// An audio-bearing movie contributes both visual and sound references.
  public static func compileMediaReferences(data: Data,
    resolveReference: (String, String, String) throws -> H3Ref2VAReference) throws
    -> H3Ref2VAStillRequest {
    try compileMediaReferences(data: data) { path, kind, digest, _, controls in
      guard controls == nil else {
        throw H3CheckpointError.invalid("Reference preparation controls require a policy-aware media resolver.")
      }
      return try resolveReference(path, kind, digest)
    }
  }

  public static func compileMediaReferences(data: Data, canvasAdmission: H3CanvasAdmission = .ordinary,
    resolveReference: (String, String, String, H3Geometry, H3ReferencePreparationControls?) throws -> H3Ref2VAReference) throws
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
    var paths: [(path: String, kind: String, digest: String, placement: Any?, sidecar: (String, String)?, controls: H3ReferencePreparationControls?)] = []
    var images = 0, videos = 0, audios = 0
    for input in inputs {
      let kind = input["kind"] as? String ?? ""
      if kind == "image" { images += 1 }
      if kind == "video" { videos += 1 }
      if kind == "audio" { audios += 1 }
      guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "strength", "sha256",
          "frame_index", "soundtrack_path", "soundtrack_sha256", "image_pixel_budget_percent", "size_policy", "temporal_density"]),
        ["image", "video", "audio"].contains(kind), input["role"] as? String == "reference",
        let path = input["path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
        let digest = input["sha256"] as? String, validMediaDigest(digest),
        input["strength"] == nil || (input["strength"] as? NSNumber).map({
          CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1 }) == true else {
        throw H3CheckpointError.invalid("H3 Ref2VA needs full-strength hashed media and supported placement.")
      }
      var sidecar: (String, String)?
      if input["soundtrack_path"] != nil || input["soundtrack_sha256"] != nil {
        guard kind == "video", let path = input["soundtrack_path"] as? String,
          path.hasPrefix("/"), !path.utf8.contains(0),
          let digest = input["soundtrack_sha256"] as? String, validMediaDigest(digest) else {
          throw H3CheckpointError.invalid("A movie soundtrack sidecar needs its own absolute path and SHA256.")
        }
        sidecar = (path, digest); audios += 1
      }
      paths.append((path, kind, digest, input["frame_index"], sidecar,
        try H3ReferencePreparationControls.parse(input, kind: kind)))
    }
    guard images <= 9, videos <= 3, audios <= 3,
      images + videos > 0 || paths.allSatisfy({ $0.placement != nil }) else {
      throw H3CheckpointError.invalid("H3 Ref2VA needs visual media or timed audio, at most nine images, three movies and three standalone audio/sidecar sources.")
    }
    let noise = try H3ReferenceNoiseControls.parse(root["config"] as? [String: Any] ?? [:])
    removeReferenceNoise(&root)
    components.removeValue(forKey: "vision_encoder")
    components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [],
      "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root), canvasAdmission: canvasAdmission)
    // Validate every placement before resolving any media, including the last item.
    let frames = try paths.map { try H3ReferencePlacement.frame($0.placement, frames: base.geometry.frames) }
    let references = try zip(paths, frames).map { item, frame in
      var reference = try resolveReference(item.path, item.kind, item.digest, base.geometry, item.controls)
      switch (item.kind, reference) {
      case ("image", .image), ("video", .video), ("audio", .audio): break
      default: throw H3CheckpointError.invalid("A reference resolved to the wrong media kind.")
      }
      if let controls = item.controls {
        switch reference {
        case .image(let image):
          guard image.pixelBudgetPercent == controls.imagePixelBudgetPercent else {
            throw H3CheckpointError.invalid("The image resolver did not apply its declared reference pixel budget.")
          }
        case .video(let video):
          guard video.controls == controls else {
            throw H3CheckpointError.invalid("The movie resolver did not apply its declared reference policy.")
          }
        default: throw H3CheckpointError.invalid("Reference media policy resolved to an unsupported kind.")
        }
      }
      if let sidecar = item.sidecar {
        let resolved = try resolveReference(sidecar.0, "audio", sidecar.1, base.geometry, nil)
        guard case .audio(let audio) = resolved, case .video(let video) = reference else {
          throw H3CheckpointError.invalid("Movie soundtrack sidecar resolved to another media kind.")
        }
        reference = .video(H3VideoReference(rgb8: video.rgb8, frameCount: video.frameCount,
          width: video.width, height: video.height, audio: audio,
          controls: video.controls, temporalDecision: video.temporalDecision))
      }
      return try H3ReferencePlacement.placing(reference, frame: frame)
    }
    return try H3Ref2VAStillRequest(prompt: base.prompt,
      mediaReferences: references, width: base.geometry.width,
      height: base.geometry.height, durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages,
      qwenVision: URL(fileURLWithPath: vision), tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      turboLoRA: base.turboLoRA,
      turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs, loRAAdapters: base.loRAAdapters,
      videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      samplingMethod: base.samplingMethod, referenceNoise: noise, canvasAdmission: canvasAdmission)
  }

  /// An A2V driver is placed at frame zero of the target packed timeline.
  /// The source waveform conditions generated sound and motion; it is never
  /// copied into the output movie. Timed still anchors use their explicit generated-frame origin. Check the entire contract before resolving media.
  public static func compileA2V(data: Data,
    driverTargetFrame: Int = 0, visibleDurationSeconds: Double? = nil,
    resolveReference: (String, String, Double, Double) throws -> H3Ref2VAReference)
    throws -> H3Ref2VAStillRequest {
    try compileA2V(data: data, driverTargetFrame: driverTargetFrame,
      visibleDurationSeconds: visibleDurationSeconds) { path, digest, start, duration, _, controls in
      guard controls == nil else {
        throw H3CheckpointError.invalid("A2V image pixel budgets require a policy-aware media resolver.")
      }
      return try resolveReference(path, digest, start, duration)
    }
  }

  public static func compileA2V(data: Data,
    driverTargetFrame: Int = 0, visibleDurationSeconds: Double? = nil,
    canvasAdmission: H3CanvasAdmission = .ordinary,
    resolveReference: (String, String, Double, Double, H3Geometry, H3ReferencePreparationControls?) throws -> H3Ref2VAReference)
    throws -> H3Ref2VAStillRequest {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var components = root["components"] as? [String: Any],
      components["task"] as? String == "ref2va",
      components["allow_fl2va_weights_for_ref2va"] == nil ||
        components["allow_fl2va_weights_for_ref2va"] as? Bool == false,
      let conditioning = root["conditioning"] as? [String: Any],
      let inputs = try? ConditioningV1.inputs(conditioning,
        task: "a2v", audioPolicy: "generated", count: 1...9),
      let config = root["config"] as? [String: Any],
      let duration = config["duration_seconds"] as? Double,
      duration.isFinite, (2.5...15).contains(duration) else {
      throw H3CheckpointError.invalid("Swift H3 A2V needs one timed audio driver and up to eight image anchors.")
    }
    let requiredAudioDuration = visibleDurationSeconds ?? duration
    guard requiredAudioDuration.isFinite, (2.5...15).contains(requiredAudioDuration),
      requiredAudioDuration <= duration + 0.001,
      visibleDurationSeconds == nil || driverTargetFrame > 0 else {
      throw H3CheckpointError.invalid("A2V visible audio interval needs a declared continuation overlap.")
    }
    var driver: (String, String, Double, Double)?
    var images: [(String, String, Any, H3ReferencePreparationControls?)] = []
    for input in inputs {
      let role = input["role"] as? String
      let kind = input["kind"] as? String
      let path = input["path"] as? String
      let digest = input["sha256"] as? String
      guard let path, path.hasPrefix("/"), !path.utf8.contains(0),
        let digest, digest.count == 64,
        digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        (input["strength"] == nil || (input["strength"] as? NSNumber)
          .map({ CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1 }) == true)
      else { throw H3CheckpointError.invalid("H3 A2V input path, hash or strength is invalid.") }
      if role == "audio_driver" && kind == "audio" && driver == nil {
        guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "sha256",
          "strength", "source_start_seconds", "source_duration_seconds"]),
          let start = input["source_start_seconds"] as? Double,
          let length = input["source_duration_seconds"] as? Double,
          start.isFinite, (0...86400).contains(start), length.isFinite,
          length >= requiredAudioDuration - 0.001, length <= 15 else {
          throw H3CheckpointError.invalid("H3 A2V needs a bounded source interval covering the clip.")
        }
        driver = (path, digest, start, length)
      } else if role == "keyframe" && kind == "image" && images.count < 8 {
        guard Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "sha256",
          "strength", "frame_index", "image_pixel_budget_percent"]), let placement = input["frame_index"] else {
          throw H3CheckpointError.invalid("H3 A2V image anchors need an explicit generated-frame index.")
        }
        images.append((path, digest, placement,
          try H3ReferencePreparationControls.parse(input, kind: "image")))
      } else {
        throw H3CheckpointError.invalid("H3 A2V accepts one audio driver and at most eight timed images.")
      }
    }
    guard let driver else { throw H3CheckpointError.invalid("H3 A2V audio driver is missing.") }
    let vision = (components["vision_encoder"] as? String) ??
      (components["text_encoder"] as? String)
    guard let vision, vision.hasPrefix("/") else {
      throw H3CheckpointError.invalid("H3 A2V needs an installed Qwen encoder.")
    }
    let noise = try H3ReferenceNoiseControls.parse(config)
    removeReferenceNoise(&root)
    components.removeValue(forKey: "vision_encoder")
    components.removeValue(forKey: "allow_fl2va_weights_for_ref2va")
    components["task"] = "t2va"
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v",
      "inputs": [], "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root), canvasAdmission: canvasAdmission)
    guard (0..<base.geometry.frames).contains(driverTargetFrame) else {
      throw H3CheckpointError.invalid("A2V driver position exceeds the sampled timeline.")
    }
    let placements = try images.map { try H3ReferencePlacement.frame($0.2, frames: base.geometry.frames)! }
    guard Set(placements).count == placements.count else {
      throw H3CheckpointError.invalid("A2V image anchor positions must be unique.")
    }
    var references: [H3Ref2VAReference] = []
    var imageIndex = 0
    for input in inputs {
      if input["role"] as? String == "keyframe" {
        let image = images[imageIndex]
        let frame = placements[imageIndex]
        imageIndex += 1
        let resolved = try resolveReference(image.0, image.1, 0, 0, base.geometry, image.3)
        guard case .image(let still) = resolved,
          still.pixelBudgetPercent == image.3?.imagePixelBudgetPercent else {
          throw H3CheckpointError.invalid("H3 A2V image anchor resolved to another media type.")
        }
        references.append(.timedImage(still, frame: frame))
      } else {
        let resolved = try resolveReference(driver.0, driver.1, driver.2, driver.3, base.geometry, nil)
        guard case .audio(let sound) = resolved else {
          throw H3CheckpointError.invalid("H3 A2V driver resolved to another media type.")
        }
        references.append(.timedAudio(sound, frame: driverTargetFrame))
      }
    }
    return try H3Ref2VAStillRequest(prompt: base.prompt,
      mediaReferences: references, width: base.geometry.width,
      height: base.geometry.height, durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages,
      qwenVision: URL(fileURLWithPath: vision), tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      turboLoRA: base.turboLoRA,
      turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs, loRAAdapters: base.loRAAdapters,
      videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      samplingMethod: base.samplingMethod, referenceNoise: noise, canvasAdmission: canvasAdmission)
  }

  /// Admit one already-preprocessed structure guide and validate every ordinary
  /// execution control before media decoding or weighted stage preparation.
  public static func compileControl(data: Data,
    resolveVideo: (String, String, H3Geometry) throws -> H3VideoReference) throws -> H3T2VARequest {
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var components = root["components"] as? [String: Any],
      components["task"] as? String == "t2va",
      let checkpoint = components.removeValue(forKey: "fun_controlnet") as? String,
      checkpoint.hasPrefix("/"), !checkpoint.utf8.contains(0),
      let conditioning = root["conditioning"] as? [String: Any],
      let inputs = try? ConditioningV1.inputs(conditioning,
        task: "control", audioPolicy: "generated", count: 1...1),
      let input = inputs.first,
      Set(conditioning.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      Set(input.keys).isSubset(of: ["id", "kind", "role", "path", "sha256", "strength", "control_type"]),
      input["kind"] as? String == "video", input["role"] as? String == "control",
      ["canny_edges", "depth_map", "hed_edges", "mlsd_lines", "pose_skeleton"]
        .contains(input["control_type"] as? String ?? ""),
      let path = input["path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
      let digest = input["sha256"] as? String, digest.count == 64,
      digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      let number = input["strength"] as? NSNumber,
      CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
      (0...1).contains(number.doubleValue) else {
      throw H3CheckpointError.invalid("H3 Fun control needs one hashed preprocessed Canny, depth, HED, MLSD or pose video and strength from 0 to 1.")
    }
    root["components"] = components
    root["conditioning"] = ["version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated"]
    let base = try compile(data: JSONSerialization.data(withJSONObject: root))
    guard base.geometry.width <= 2048,
      base.geometry.height <= 2048,
      base.geometry.frames * base.geometry.width * base.geometry.height * 3 <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("H3 Fun control requires dense sampling without LoRAs and a bounded guide canvas.")
    }
    let control = try H3FunControlGuide(checkpoint: URL(fileURLWithPath: checkpoint),
      strength: number.floatValue, video: resolveVideo(path, digest, base.geometry))
    return try H3T2VARequest(prompt: base.prompt, width: base.geometry.width,
      height: base.geometry.height, durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages, tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      turboLoRA: base.turboLoRA, turboLoRAStrength: base.turboLoRAStrength,
      additionalLoRAs: base.additionalLoRAs, loRAAdapters: base.loRAAdapters,
      funControl: control,
      videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      samplingMethod: base.samplingMethod)
  }

  private static func validMediaDigest(_ digest: String) -> Bool {
    digest.utf8.count == 64 && digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
  }
  private static func removeReferenceNoise(_ root: inout [String: Any]) {
    guard var config = root["config"] as? [String: Any] else { return }
    config.removeValue(forKey: "visual_condition_strength")
    config.removeValue(forKey: "audio_condition_strength")
    root["config"] = config
  }

  public static func compile(data: Data, canvasAdmission: H3CanvasAdmission = .ordinary) throws -> H3T2VARequest {
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
        (1...8).contains(entries.count) else { return nil }
      var adapters: [(URL, Float)] = []
      for pair in entries {
        guard pair.count == 2, let path = pair[0] as? String,
          path.hasPrefix("/"), !path.utf8.contains(0),
          let number = pair[1] as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let strength = number.floatValue
        guard strength.isFinite, (-10...10).contains(strength) else { return nil }
        adapters.append((URL(fileURLWithPath: path), strength))
      }
      guard Set(adapters.map { $0.0.standardizedFileURL.path }).count == adapters.count
      else { return nil }
      return adapters
    }
    func descriptorStack(_ value: Any?) throws -> [H3LoRAAdapter]? {
      guard let value else { return nil }
      guard let object = value as? [String: Any], Set(object.keys) == ["version", "adapters"],
        let version = object["version"] as? NSNumber, CFGetTypeID(version) != CFBooleanGetTypeID(), version.doubleValue == 1,
        let entries = object["adapters"] as? [[String: Any]], (1...8).contains(entries.count) else {
        throw H3CheckpointError.invalid("H3 LoRA v1 requires one to eight validated adapter descriptors.")
      }
      return try entries.map { item in
        guard Set(item.keys).isSubset(of: ["path", "strength", "profile", "adaln_input_grid", "qkv_layout", "start_after_evaluations"]),
          let path = item["path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
          let strength = item["strength"] as? NSNumber, CFGetTypeID(strength) != CFBooleanGetTypeID(),
          let profile = H3LoRAProfile(rawValue: item["profile"] as? String ?? "auto"),
          let qkv = H3LoRAQKVLayout(rawValue: item["qkv_layout"] as? String ?? "auto"),
          item["adaln_input_grid"] == nil || item["adaln_input_grid"] is NSNull,
          item["profile"] == nil || item["profile"] is String,
          item["qkv_layout"] == nil || item["qkv_layout"] is String else {
          throw H3CheckpointError.invalid("Unsupported H3 LoRA descriptor, layout or input grid.")
        }
        let start: Int
        if let n = item["start_after_evaluations"] as? NSNumber,
          CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite,
          n.doubleValue.rounded() == n.doubleValue, (0...99).contains(n.doubleValue) { start = n.intValue }
        else if item["start_after_evaluations"] == nil { start = 0 }
        else { throw H3CheckpointError.invalid("Invalid H3 deferred LoRA activation.") }
        return try H3LoRAAdapter(url: URL(fileURLWithPath: path), strength: strength.floatValue,
          profile: profile, qkvLayout: qkv, startAfterEvaluations: start)
      }
    }
    guard data.count <= 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(root.keys).isSubset(of: ["format", "engine", "candidate", "components",
        "config", "prompt", "conditioning", "ffmpeg", "block_residency",
        "negative_prompt", "loras"]),
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
      oneOf(config, "sampling_method", default: "euler", ["euler", "res_multistep"]),
      let samplingMethod = H3SamplingMethod(rawValue: config["sampling_method"] as? String ?? "euler"),
      oneOf(config, "inference_optimization", default: "off", ["off"]),
      zero(config, "paging_cache_gb"),
      let width = config["width"] as? Int,
      let height = config["height"] as? Int,
      let duration = config["duration_seconds"] as? Double,
      let steps = config["steps"] as? Int,
      let seed = config["seed"] as? Int, (0...Int(UInt32.max)).contains(seed) else {
      throw H3CheckpointError.invalid("Swift H3 currently admits only text-to-audiovisual Euler or res_multistep recipes with up to eight distinct compatible LoRAs and no unported controls.")
    }
    let explicit = try descriptorStack(root["loras"])
    guard explicit == nil || emptyArray(component, "loras") else {
      throw H3CheckpointError.invalid("Select either H3 component LoRA pairs or the explicit v1 descriptor stack.")
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
      additionalLoRAs: additional, loRAAdapters: explicit,
      videoDecodeMemoryMode: H3VideoDecodeMemoryMode(rawValue:
        config["memory_mode"] as? String ?? "normal"),
      samplingMethod: samplingMethod, canvasAdmission: canvasAdmission)
  }
}
