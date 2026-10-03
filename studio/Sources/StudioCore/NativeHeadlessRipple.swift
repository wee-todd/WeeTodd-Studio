import AVFoundation
import CoreFoundation
import Foundation

/// Frozen Ripple is a separate recipe/report subtype inside the compatible native-job-v1 envelope.
/// The existing worker still owns strict MLXRippleRequest decoding and all weighted execution.
enum NativeHeadlessRipple {
  static let keys: Set<String> = ["version", "engine", "task", "gemma_root", "transformer_root", "connector_checkpoint",
    "video_checkpoint", "audio_checkpoint", "adapter_path", "adapter_strength", "guide_path", "first_reference_path",
    "source_path", "source_sha256", "source_start", "duration", "editorial_frames", "width", "height", "frames", "fps", "seed",
    "prompt", "reference_strength", "anchors", "audio_policy", "ffmpeg_path", "output_directory"]
  static func object(_ data: Data) throws -> [String: Any] {
    guard data.count <= 1024 * 1024, let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw StudioError.invalid("Invalid frozen Ripple recipe or receipt.")
    }; return value
  }
  static func readObject(_ url: URL) throws -> [String: Any] {
    let source = try NativeHeadlessJob.Source.capture(url.path)
    guard source.size <= 1024 * 1024 else { throw StudioError.invalid("Ripple receipts must be under 1 MiB.") }
    return try object(Data(contentsOf: url))
  }
  static func boolean(_ value: Any?) -> Bool? {
    guard let value = value as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return nil }
    return value.boolValue
  }
  static func number(_ value: Any?) -> Double? {
    guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite else { return nil }
    return value.doubleValue
  }
  static func draft(_ recipe: NativeHeadlessJob.Recipe) throws -> RippleDraft? {
    let raw = try object(recipe.bytes)
    let report = recipe.report.isEmpty ? [:] : try object(recipe.report)
    guard raw["task"] as? String == "ripple" else {
      guard report["rippleDraft"] == nil else { throw StudioError.invalid("Ripple inputs require their dedicated edit recipe.") }
      return nil
    }
    guard let value = report["rippleDraft"] else { throw StudioError.invalid("A Ripple job needs its frozen editorial draft.") }
    return try JSONDecoder().decode(RippleDraft.self, from: JSONSerialization.data(withJSONObject: value))
  }
  static func references(_ raw: [String: Any]) throws -> [[String: Any]] {
    guard let first = raw["first_reference_path"] as? String, let strength = number(raw["reference_strength"]),
      let anchors = raw["anchors"] as? [[String: Any]] else { throw StudioError.invalid("Missing frozen Ripple references.") }
    return [["frame": 0, "path": first, "strength": strength]] + anchors
  }
  static func validate(_ recipe: NativeHeadlessJob.Recipe, clip: Clip, ffmpeg: String) throws -> RippleDraft? {
    guard let draft = try draft(recipe) else { return nil }
    try draft.validate()
    let raw = try object(recipe.bytes), refs = try references(raw), sorted = draft.references.sorted { $0.frame < $1.frame }
    let modelFrames = 1 + 8 * max(1, Int(ceil(Double(draft.frameCount - 1) / 8)), Int(ceil(0.25 * draft.frameRate / 8)))
    let prompt = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard Set(raw.keys) == keys, number(raw["version"]) == 1, raw["engine"] as? String == "ltx25",
      recipe.engine == "ltx25", clip.rippleDraft == draft, draft.sourceMatches(clip), recipe.signature == draft.inputFingerprint,
      raw["source_path"] as? String == draft.sourcePath, number(raw["source_start"]) == draft.sourceIn,
      number(raw["duration"]) == draft.duration, number(raw["fps"]) == draft.frameRate,
      number(raw["editorial_frames"]) == Double(draft.frameCount), number(raw["frames"]) == Double(modelFrames), modelFrames <= 1501,
      number(raw["width"]) == Double(draft.width), number(raw["height"]) == Double(draft.height), number(raw["seed"]) == Double(draft.seed),
      number(raw["adapter_strength"]).map({ Float($0) == Float(draft.loraStrength) }) == true,
      raw["prompt"] as? String == (prompt.isEmpty ? RippleDraft.defaultPrompt : prompt),
      raw["audio_policy"] as? String == draft.audioPolicy.rawValue, raw["ffmpeg_path"] as? String == ffmpeg,
      let hash = raw["source_sha256"] as? String, hash.count == 64, hash.allSatisfy({ "0123456789abcdef".contains($0) }),
      draft.sourceSHA256 == nil || draft.sourceSHA256 == hash, refs.count == sorted.count else {
      throw StudioError.invalid("The frozen Ripple request differs from its exact editor inputs.")
    }
    for key in ["gemma_root", "transformer_root", "connector_checkpoint", "video_checkpoint", "audio_checkpoint",
      "adapter_path", "guide_path", "first_reference_path", "source_path", "ffmpeg_path", "output_directory"] {
      guard let path = raw[key] as? String, path.hasPrefix("/"), path.utf8.count <= 4096, !path.utf8.contains(0) else {
        throw StudioError.invalid("Ripple requires absolute bounded local paths.")
      }
    }
    for (reference, expected) in zip(refs, sorted) {
      guard Set(reference.keys) == ["frame", "path", "strength"], number(reference["frame"]) == Double(expected.frame),
        number(reference["strength"]).map({ Float($0) == Float(expected.strength) }) == true,
        let path = reference["path"] as? String, path.hasPrefix("/"), path.utf8.count <= 4096, !path.utf8.contains(0) else {
        throw StudioError.invalid("Ripple reference timing or strength differs from the editor.")
      }
    }
    let guide = try NativeHeadlessJob.Source.capture(raw["guide_path"] as! String)
    guard guide.size == UInt64(modelFrames) * UInt64(draft.width) * UInt64(draft.height) * 3 else {
      throw StudioError.invalid("Ripple guide bytes differ from the frozen padded frame geometry.")
    }
    return draft
  }
  static func target(_ recipe: NativeHeadlessJob.Recipe) throws -> URL? {
    guard try draft(recipe) != nil else { return nil }
    return URL(fileURLWithPath: try object(recipe.bytes)["output_directory"] as! String)
  }
  static func contained(_ path: String, in target: URL) throws -> URL {
    let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    guard url.path.hasPrefix(target.standardizedFileURL.resolvingSymlinksInPath().path + "/") else {
      throw StudioError.invalid("Ripple publication escaped its frozen take directory.")
    }
    _ = try NativeHeadlessJob.Source.capture(url.path); return url
  }
  static func verifyResume(recipe: NativeHeadlessJob.Recipe, clip: Clip, completedPath: String,
    artifacts: [String: String]) throws {
    let draft = try self.draft(recipe)!, raw = try object(recipe.bytes)
    guard let take = clip.rippleTakes?.first(where: { $0.path == completedPath }),
      take.submittedDraftFingerprint == draft.inputFingerprint,
      take.draft.sourceSHA256 == raw["source_sha256"] as? String,
      clip.sourcePath == completedPath, clip.sourceIn == 0, clip.duration == draft.duration,
      take.artifactsDirectory == raw["output_directory"] as? String else {
      throw StudioError.invalid("Ripple resume requires its exact accepted take and editorial identities.")
    }
    var submitted = take.draft; submitted.sourceSHA256 = draft.sourceSHA256
    for index in submitted.references.indices {
      guard let original = draft.references.first(where: { $0.id == submitted.references[index].id }) else {
        throw StudioError.invalid("Ripple resume lost a frozen reference identity.")
      }
      submitted.references[index].path = original.path
    }
    guard submitted == draft else { throw StudioError.invalid("Ripple resume inputs differ from the frozen draft.") }
    let root = URL(fileURLWithPath: take.artifactsDirectory)
    let required = [take.receiptPath] + ["ripple-request.json", "result.json", "report.json"].map { root.appendingPathComponent($0).path }
      + take.draft.references.map(\.path)
    guard required.allSatisfy({ artifacts[$0] != nil }) else {
      throw StudioError.invalid("Ripple resume requires hashes for its receipt and replay artifacts.")
    }
  }
  static func accept(result: [String: Any], recipe: NativeHeadlessJob.Recipe, target: URL,
    project: inout StudioProject, index: Int) async throws -> (String, [String: String]) {
    let draft = try self.draft(recipe)!, raw = try object(recipe.bytes)
    guard let id = result["jobID"] as? String, UUID(uuidString: id) != nil,
      result["nativeRuntime"] as? String == "swift-mlx",
      let path = result["video_path"] as? String ?? result["path"] as? String, path != draft.sourcePath,
      number(result["duration"]).map({ abs($0 - draft.duration) <= 1 / draft.frameRate + 0.001 }) == true,
      number(result["frame_rate"]).map({ abs($0 - draft.frameRate) <= max(1, draft.frameRate) * 0.00001 }) == true,
      number(result["frames"]) == Double(draft.frameCount), number(result["width"]) == Double(draft.width), number(result["height"]) == Double(draft.height),
      let hasAudio = boolean(result["has_audio"]), result["scene"] == nil, draft.audioPolicy != .silent || !hasAudio,
      let receiptPath = result["receipt_path"] as? String,
      result["artifacts_directory"] as? String == target.path,
      result["source_sha256"] as? String == raw["source_sha256"] as? String,
      let frozen = result["frozen_references"] as? [[String: Any]], frozen.count == draft.references.count else {
      throw StudioError.invalid("Ripple returned incomplete or foreign take identities.")
    }
    let movie = try contained(path, in: target), receiptURL = try contained(receiptPath, in: target)
    let requestURL = try contained(target.appendingPathComponent("ripple-request.json").path, in: target)
    let resultURL = try contained(target.appendingPathComponent("result.json").path, in: target)
    let reportURL = try contained(target.appendingPathComponent("report.json").path, in: target)
    let receipt = try readObject(receiptURL), saved = try readObject(resultURL)
    guard try Data(contentsOf: requestURL) == recipe.bytes, saved["jobID"] as? String == id,
      saved["source_sha256"] as? String == raw["source_sha256"] as? String,
      saved["video_path"] as? String == path,
      receipt["format"] as? String == "weetodd-ripple-take-v1", receipt["status"] as? String == "complete",
      receipt["source_path"] as? String == draft.sourcePath, receipt["source_sha256"] as? String == raw["source_sha256"] as? String,
      number(receipt["source_start"]) == draft.sourceIn, number(receipt["duration"]) == draft.duration,
      number(receipt["editorial_frames"]) == Double(draft.frameCount),
      receipt["publication_audio"] as? String == (hasAudio ? "preserved source interval" : "silent"),
      let receiptRefs = receipt["reference_images"] as? [[String: Any]],
      NSDictionary(dictionary: ["refs": receiptRefs]).isEqual(to: ["refs": frozen]) else {
      throw StudioError.invalid("Ripple receipt does not identify the frozen request and editorial source.")
    }
    let sourceAudio = try await AVURLAsset(url: URL(fileURLWithPath: draft.sourcePath)).loadTracks(withMediaType: .audio)
    guard hasAudio == (draft.audioPolicy == .preserve && !sourceAudio.isEmpty) else {
      throw StudioError.invalid("Ripple output audio does not preserve the selected source policy.")
    }
    try await NativeRippleMedia.verifyPublishedTake(movie.path, draft: draft, hasAudio: hasAudio)
    var replay = draft; replay.sourceSHA256 = raw["source_sha256"] as? String
    var hashes: [String: String] = [:], seen = Set<Int>()
    let prepared = try references(raw)
    for ref in frozen {
      guard let frameValue = number(ref["frame"]), frameValue.rounded() == frameValue,
        frameValue >= 0, frameValue < Double(draft.frameCount) else { throw StudioError.invalid("Invalid Ripple frozen reference frame.") }
      let frame = Int(frameValue)
      guard seen.insert(frame).inserted, let position = replay.references.firstIndex(where: { $0.frame == frame }),
        number(ref["strength"]).map({ Float($0) == Float(replay.references[position].strength) }) == true,
        let path = ref["path"] as? String, let original = prepared.first(where: { number($0["frame"]) == frameValue })?["path"] as? String else {
        throw StudioError.invalid("Ripple returned inconsistent replay references.")
      }
      let url = try contained(path, in: target), digest = try NativeHeadlessJob.fileHash(url)
      guard digest == (try NativeHeadlessJob.fileHash(URL(fileURLWithPath: original))) else { throw StudioError.invalid("Ripple replay reference differs from its frozen edited image.") }
      hashes[url.path] = digest; replay.references[position].path = url.path
    }
    for url in [receiptURL, requestURL, resultURL, reportURL] { hashes[url.path] = try NativeHeadlessJob.fileHash(url) }
    let take = RippleTake(draft: replay, path: movie.path, receiptPath: receiptURL.path, artifactsDirectory: target.path,
      hasAudio: hasAudio, submittedDraftFingerprint: draft.inputFingerprint)
    let source = project.clips[index]
    if !source.versions.contains(where: { $0.path == source.sourcePath }) {
      project.clips[index].versions.append(RenderVersion(path: source.sourcePath, seed: source.seed, prompt: source.prompt,
        recipePath: "", usableSourceIn: source.sourceIn, usableDuration: source.duration))
    }
    project.clips[index].versions.append(RenderVersion(path: take.path, seed: draft.seed, prompt: draft.prompt,
      recipePath: take.receiptPath, usableSourceIn: 0, usableDuration: draft.duration))
    if project.clips[index].rippleTakes == nil { project.clips[index].rippleTakes = [] }
    project.clips[index].rippleTakes!.append(take)
    _ = project.separateContinuousSceneMember(clipID: source.id)
    project.clips[index].sourcePath = take.path; project.clips[index].sourceIn = 0; project.clips[index].duration = draft.duration
    project.clips[index].renderedSignature = ""; project.clips[index].motionResult = nil
    project.clips[index].reviewedTakeFingerprint = project.clips[index].reuseFingerprint
    var asset = MediaAsset(name: source.name + " · Ripple", kind: .video, path: take.path, scope: .clip, owner: source.id)
    asset.duration = draft.duration; asset.width = draft.width; asset.height = draft.height; asset.fps = draft.frameRate
    project.assets.append(asset)
    return (movie.path, hashes)
  }
}
