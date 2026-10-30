import Foundation

/// Orchestration only. The shared Swift/MLX movie worker owns all weighted inference.
public enum NativePixelCQService {
  public typealias Worker = NativeHeadlessExecutor.Worker
  private struct Checkpoint: Codable { let fingerprint: String; let attempt: UUID; let receiptSHA256: String }
  private static let preservationPrompt = "Preserve the source subject, face, clothing, expression, motion, camera, lighting and scene while refining detail."
  private static func call(_ runtime: NativePixelCQRuntime, recipe: URL, output: URL, mode: String,
    cancellation: NativeHeadlessCancellation, worker: Worker?, emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> [String: Any] {
    try cancellation.check(); try Task.checkCancellation()
    let task = Task.detached {
      if let worker { return try worker(runtime.workerPath, recipe, output, mode) }
      return try NativeHeadlessExecutor.worker(runtime.workerPath, recipe: recipe, output: output, mode: mode,
        cancellation: cancellation, emit: { bytes in
          if let value = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] {
            emit(.init(stage: value["stage"] as? String ?? mode, message: value["message"] as? String ?? "Native Pixel Spatial + CQ · " + (value["stage"] as? String ?? mode), fraction: value["fraction"] as? Double))
          }
        }, ffmpeg: runtime.ffmpegPath)
    }
    return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: {
      task.cancel(); cancellation.cancel()
    })
  }
  private static func prepare(source: URL, settings: NativePixelCQSettings, runtime: NativePixelCQRuntime,
    destination: URL, output: URL) async throws -> [String: Any] {
    try await NativeLTXMoviePreparation.prepare(source: source, settings: settings.movieSettings,
      components: runtime.components, prompt: preservationPrompt, seed: 20261028,
      ffmpeg: URL(fileURLWithPath: runtime.ffmpegPath), destination: destination, outputDirectory: output,
      refinementLoRAs: runtime.adapters(settings))
  }
  /// A one-frame synthetic movie admits model headers and adapter mappings before generation.
  /// It never invokes a weighted render, changes user media or loads tensor payloads.
  public static func preflight(recipe: VideoProcessingRecipe, runtime: NativePixelCQRuntime,
    worker: Worker? = nil, cancellation: NativeHeadlessCancellation = .init()) async throws -> NativePixelCQResources {
    try recipe.validate()
    guard recipe.isNative, let settings = recipe.steps[0].pixelCQ else { throw StudioError.invalid("Select native Pixel Spatial + CQ.") }
    try runtime.validate(settings: settings)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-native-admission-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var workerFiles = try [runtime.workerPath, runtime.ffmpegPath, runtime.ffprobePath].map { try FlashVSRFile.capture(URL(fileURLWithPath: $0)) }
    let workerFolder = URL(fileURLWithPath: runtime.workerPath).deletingLastPathComponent()
    let metalCandidates = [workerFolder.appendingPathComponent("mlx.metallib"),
      workerFolder.deletingLastPathComponent().appendingPathComponent("Resources/LTXNative/mlx.metallib")]
    if let metal = metalCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) { workerFiles.append(try .capture(metal)) }
    let source = root.appendingPathComponent("probe.mp4")
    try NativeMovieFrozenMedia.run(URL(fileURLWithPath: runtime.ffmpegPath), ["-v", "error", "-nostdin", "-n",
      "-f", "lavfi", "-i", "color=s=64x32:r=24", "-frames:v", "1", "-an", "-c:v", "libx264", source.path], log: root.appendingPathComponent("probe.log"))
    let prepared = try await prepare(source: source, settings: settings, runtime: runtime,
      destination: root.appendingPathComponent("prepared"), output: root.appendingPathComponent("admission"))
    let result = try await call(runtime, recipe: URL(fileURLWithPath: prepared["recipePath"] as! String), output: root.appendingPathComponent("admission"),
      mode: "preflight", cancellation: cancellation, worker: worker, emit: { _ in })
    guard let identity = result["componentIdentitySHA256"] as? String, identity.count == 64 else {
      throw StudioError.invalid("The native worker returned no component identity. Use a current complete Studio build.")
    }
    for file in workerFiles { try file.verify() }
    return .init(runtime: runtime, workerFiles: workerFiles, componentIdentitySHA256: identity)
  }
  private static func inspect(_ file: FlashVSRFile, runtime: NativePixelCQRuntime, directory: URL,
    cancellation: NativeHeadlessCancellation) throws -> FlashVSRMediaInfo {
    try file.verify()
    return try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: file.path), runtime: runtime.mediaRuntime,
      directory: directory, cancellation: cancellation)
  }
  private static func audioHash(_ file: FlashVSRFile, runtime: NativePixelCQRuntime, directory: URL,
    cancellation: NativeHeadlessCancellation) throws -> String {
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath), arguments: ["-v", "error", "-nostdin", "-i", file.path,
      "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"], directory: directory,
      log: directory.appendingPathComponent(UUID().uuidString + "-audio.log"), cancellation: cancellation)
    let hash = String(decoding: result.output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    guard result.exitCode == 0, hash.hasPrefix("SHA256="), hash.count == 71 else { throw StudioError.invalid("Cannot verify unchanged source audio packets.") }
    return hash
  }
  private static func verifiedRestored(result: [String: Any], output: URL, runtime: NativePixelCQRuntime,
    media: FlashVSRMediaInfo, directory: URL, cancellation: NativeHeadlessCancellation) throws -> FlashVSRFile? {
    guard let metadata = result["metadata"] as? [String: Any], let proof = metadata["restored"] as? [String: Any] else { return nil }
    let file = try FlashVSRFile.capture(output.appendingPathComponent("restored.mkv"))
    guard proof["file"] as? String == "restored.mkv", proof["sha256"] as? String == file.sha256,
      proof["codec"] as? String == "ffv1", proof["pixel_format"] as? String == "bgr0", proof["has_audio"] as? Bool == false,
      proof["frames"] as? Int == media.frames, let n = proof["fps_numerator"] as? Int, let d = proof["fps_denominator"] as? Int,
      (1...1_000_000).contains(n), (1...1_000_000).contains(d), Int64(n)*Int64(media.fps.denominator) == Int64(d)*Int64(media.fps.numerator),
      let rgb = proof["rgb_sha256"] as? String, rgb.count == 64 else { throw StudioError.invalid("Native restoration has no matching lossless byte/clock proof.") }
    let summary = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffprobePath),
      arguments: ["-v", "error", "-show_streams", "-of", "json", file.path], directory: directory,
      log: directory.appendingPathComponent(UUID().uuidString + "-restored.log"), cancellation: cancellation)
    guard summary.exitCode == 0, let body = try JSONSerialization.jsonObject(with: summary.output) as? [String: Any],
      let streams = body["streams"] as? [[String: Any]], streams.count == 1, streams[0]["codec_type"] as? String == "video",
      streams[0]["codec_name"] as? String == "ffv1", streams[0]["pix_fmt"] as? String == "bgr0" else {
      throw StudioError.invalid("Native restoration must contain one silent FFV1/bgr0 stream.")
    }
    let clock = try inspect(file, runtime: runtime, directory: directory, cancellation: cancellation)
    guard clock.frames == media.frames, abs(clock.fps.value - media.fps.value) < 0.000001, !clock.hasAudio,
      clock.width == proof["width"] as? Int, clock.height == proof["height"] as? Int else { throw StudioError.invalid("Native restoration changed source timing or its declared canvas.") }
    let decoded = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath),
      arguments: ["-v", "error", "-nostdin", "-i", file.path, "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "hash", "-hash", "sha256", "-"],
      directory: directory, log: directory.appendingPathComponent(UUID().uuidString + "-restored-rgb.log"), cancellation: cancellation)
    guard decoded.exitCode == 0, String(decoding: decoded.output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines) == "SHA256=" + rgb else {
      throw StudioError.invalid("Native restoration decoded RGB differs from its worker proof.")
    }
    try file.verify(); return file
  }
  /// Caller owns the processing-cache lock. Failed attempts are retained, never treated as completed.
  static func execute(resources: NativePixelCQResources, recipe: VideoProcessingRecipe, source: FlashVSRFile, directory: URL,
    cancellation: NativeHeadlessCancellation, worker: Worker?, emit: @escaping @Sendable (FlashVSREvent) -> Void, intermediateOnly: Bool = false) async throws -> NativeVideoProcessingResult {
    let settings = recipe.steps[0].pixelCQ!, runtime = resources.runtime
    try runtime.validate(settings: settings)
    for file in resources.workerFiles { try file.verify() }
    let attemptID = UUID(), staging = directory.appendingPathComponent("native-attempt-" + attemptID.uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    var preserveAttempt = false
    defer { if !preserveAttempt { try? FileManager.default.removeItem(at: staging) } }
    let media = try inspect(source, runtime: runtime, directory: staging, cancellation: cancellation)
    guard (1...60).contains(media.fps.value), media.width % 32 == 0, media.height % 32 == 0 else {
      throw StudioError.invalid("Native Pixel Spatial + CQ preserves 1–60 FPS constant-rate sources on a 32-pixel grid. Choose explicit source conversion or a different upscaler; the recipe will not change FPS or crop automatically.")
    }
    let output = staging.appendingPathComponent("native-render")
    let prepared = try await prepare(source: URL(fileURLWithPath: source.path), settings: settings, runtime: runtime,
      destination: staging.appendingPathComponent("prepared"), output: output)
    let recipeURL = URL(fileURLWithPath: prepared["recipePath"] as! String)
    if intermediateOnly {
      var wrapper = try JSONSerialization.jsonObject(with: Data(contentsOf: recipeURL)) as! [String: Any]
      var body = wrapper["movie_upscale"] as! [String: Any]; body["intermediate_only"] = true; wrapper["movie_upscale"] = body
      try JSONSerialization.data(withJSONObject: wrapper, options: [.sortedKeys, .withoutEscapingSlashes]).write(to: recipeURL, options: .atomic)
    }
    let admission = try await call(runtime, recipe: recipeURL, output: output, mode: "preflight", cancellation: cancellation, worker: worker, emit: emit)
    guard admission["componentIdentitySHA256"] as? String == resources.componentIdentitySHA256 else {
      throw StudioError.invalid("Native upscaler components changed after admission. Retry the upscale; the original generation is retained.")
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    let binding = try encoder.encode(resources) + encoder.encode(source) + Data((try recipe.fingerprint).utf8) + (intermediateOnly ? Data("lossless-intermediate-v1".utf8) : Data())
    let key = NativeHeadlessJob.hash(binding), checkpointURL = directory.appendingPathComponent("native-" + key + ".json")
    if FileManager.default.fileExists(atPath: checkpointURL.path) {
      do {
        guard let size = try FileManager.default.attributesOfItem(atPath: checkpointURL.path)[.size] as? NSNumber, size.intValue <= 1024 * 1024 else { throw StudioError.invalid("Invalid native processing checkpoint.") }
        let saved = try JSONDecoder().decode(Checkpoint.self, from: Data(contentsOf: checkpointURL))
        let folder = directory.appendingPathComponent("native-attempt-" + saved.attempt.uuidString), receiptURL = folder.appendingPathComponent("receipt.json")
        guard folder.resolvingSymlinksInPath().path.hasPrefix(directory.resolvingSymlinksInPath().path + "/"), saved.fingerprint == key,
          try NativeHeadlessJob.fileHash(receiptURL) == saved.receiptSHA256 else { throw StudioError.invalid("Native receipt changed.") }
        let receipt = try JSONDecoder().decode(NativePixelCQReceipt.self, from: Data(contentsOf: receiptURL))
        guard receipt.requestFingerprint == key, receipt.source == source else { throw StudioError.invalid("Native source or settings changed.") }
        guard ([receipt.video, receipt.workerMetadata, receipt.workerRecipe] + (receipt.restored.map { [$0] } ?? [])).allSatisfy({
          URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path.hasPrefix(folder.resolvingSymlinksInPath().path + "/")
        }) else { throw StudioError.invalid("The native receipt escaped its attempt folder.") }
        guard !intermediateOnly || (receipt.restored != nil && receipt.video == receipt.restored) else { throw StudioError.invalid("Legacy native receipt has no lossless restoration.") }
        try receipt.restored?.verify()
        try receipt.video.verify(); try receipt.workerMetadata.verify(); try receipt.workerRecipe.verify()
        try source.verify(); try cancellation.check(); try Task.checkCancellation()
        emit(.init(stage: "reuse", message: "Reusing verified native Pixel Spatial + CQ"))
        return .init(source: source, recipe: recipe, nativeReceipt: receipt, receiptPath: receiptURL.path, reusedSteps: 1)
      } catch { emit(.init(stage: "retry", message: "Prior native upscale changed or is incomplete; retaining it and retrying from the original.")) }
    }
    try cancellation.check(); try Task.checkCancellation(); preserveAttempt = true
    emit(.init(stage: "render", message: "Native Pixel Spatial + CQ · three refinement evaluations"))
    let started = Date()
    let result = try await call(runtime, recipe: recipeURL, output: output, mode: "render", cancellation: cancellation, worker: worker, emit: emit)
    try cancellation.check(); try Task.checkCancellation(); try source.verify()
    for file in resources.workerFiles { try file.verify() }
    let restored = try verifiedRestored(result: result, output: output, runtime: runtime, media: media, directory: staging, cancellation: cancellation)
    guard !intermediateOnly || restored != nil else { throw StudioError.invalid("This native worker did not publish verified lossless RGB. Update the complete Studio build.") }
    let raw = try intermediateOnly ? restored! : FlashVSRFile.capture(output.appendingPathComponent("render.mp4"))
    let rawMedia = try inspect(raw, runtime: runtime, directory: staging, cancellation: cancellation)
    let report = prepared["report"] as! [String: Any]
    guard let metadata = result["metadata"] as? [String: Any],
      metadata["component_identity_sha256"] as? String == resources.componentIdentitySHA256,
      metadata["transformer_evaluations_per_chunk"] as? Int == 3,
      let applied = metadata["applied_refinement_adapters"] as? [[String: Any]],
      applied.count == (settings.cqStrength > 0 ? 2 : 1),
      applied[0]["path"] as? String == runtime.components["pixel_spatial_adapter"], applied[0]["strength"] as? Double == Double(Float(settings.pixelStrength)),
      settings.cqStrength == 0 || (applied[1]["path"] as? String == runtime.cqAdapterPath && applied[1]["strength"] as? Double == Double(Float(settings.cqStrength))) else {
      throw StudioError.invalid("The native upscale output does not match its model and adapter contract.")
    }
    _ = try NativeLTXMovieAcceptance.duration(prepared: report, result: result, media: ["fps": rawMedia.fps.value,
      "width": rawMedia.width, "height": rawMedia.height, "videoDuration": Double(rawMedia.frames)/rawMedia.fps.value,
      "duration": Double(rawMedia.frames)/rawMedia.fps.value])
    let delivered: URL
    if intermediateOnly { delivered = URL(fileURLWithPath: restored!.path) }
    else {
      delivered = staging.appendingPathComponent("upscaled.mov")
      let mux = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath), arguments: ["-v", "error", "-nostdin", "-n",
        "-i", raw.path, "-i", source.path, "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-movflags", "+faststart", delivered.path],
        directory: staging, log: staging.appendingPathComponent("preserve-audio.log"), cancellation: cancellation)
      guard mux.exitCode == 0 else { throw StudioError.invalid("Original audio could not be copied unchanged into the native upscale. The original take is retained.") }
    }
    let video = try FlashVSRFile.capture(delivered), checked = try inspect(video, runtime: runtime, directory: staging, cancellation: cancellation)
    guard checked.frames == media.frames, abs(checked.fps.value - media.fps.value) < 0.0001,
      checked.width == report["width"] as? Int, checked.height == report["height"] as? Int, checked.hasAudio == (intermediateOnly ? false : media.hasAudio) else {
      throw StudioError.invalid("The native upscale changed the source frame rate, frame count, dimensions or audio presence.")
    }
    let sourceAudio = try media.hasAudio && !intermediateOnly ? audioHash(source, runtime: runtime, directory: staging, cancellation: cancellation) : nil
    let outputAudio = try checked.hasAudio ? audioHash(video, runtime: runtime, directory: staging, cancellation: cancellation) : nil
    guard sourceAudio == outputAudio else { throw StudioError.invalid("Native upscale audio packets differ from the original.") }
    try source.verify(); try cancellation.check(); try Task.checkCancellation()
    let receipt = NativePixelCQReceipt(requestFingerprint: key, source: source, video: video,
      workerMetadata: try .capture(output.appendingPathComponent("metadata.json")), workerRecipe: try .capture(recipeURL), media: checked,
      elapsedSeconds: Date().timeIntervalSince(started), sourceAudioSHA256: sourceAudio, outputAudioSHA256: outputAudio, restored: restored)
    let receiptURL = staging.appendingPathComponent("receipt.json")
    try encoder.encode(receipt).write(to: receiptURL, options: .withoutOverwriting)
    try encoder.encode(Checkpoint(fingerprint: key, attempt: attemptID, receiptSHA256: try NativeHeadlessJob.fileHash(receiptURL))).write(to: checkpointURL, options: .atomic)
    return .init(source: source, recipe: recipe, nativeReceipt: receipt, receiptPath: receiptURL.path, reusedSteps: 0)
  }
}
