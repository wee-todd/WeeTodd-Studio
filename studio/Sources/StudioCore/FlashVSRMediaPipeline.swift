import Darwin
import Foundation

/// Lossless interval preparation, clock/audio verification and delivery shared
/// by the explicitly selected engines. Model execution stays in its backend.
enum FlashVSRMediaPipeline {
  typealias Restore = (FlashVSRResolvedPlan, URL, URL, URL, NativeHeadlessCancellation,
    @escaping @Sendable (FlashVSREvent) -> Void) throws -> Void

  private static func pcmHash(_ url: URL, runtime: FlashVSRRuntime, directory: URL,
    cancellation: NativeHeadlessCancellation) throws -> String {
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath),
      arguments: ["-v", "error", "-nostdin", "-i", url.path, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "hash", "-hash", "sha256", "-"],
      directory: directory, log: directory.appendingPathComponent(UUID().uuidString + "-pcm-hash.log"),
      cancellation: cancellation, maximumOutput: 65536)
    let text = String(decoding: result.output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    guard result.exitCode == 0, text.hasPrefix("SHA256="), text.count == 71 else {
      throw StudioError.invalid("Cannot verify original source audio samples.")
    }
    return text
  }
  static func execute(plan: FlashVSRResolvedPlan, directory: URL, cancellation: NativeHeadlessCancellation,
    emit: @escaping @Sendable (FlashVSREvent) -> Void, intermediateOnly: Bool = false, restore: Restore) throws -> FlashVSRReceipt {
    try plan.request.settings.validateExecution()
    guard directory.isFileURL, !FileManager.default.fileExists(atPath: directory.path) else { throw StudioError.invalid("Choose a new local output directory for the upscale.") }
    let parent = directory.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let available = (try FileManager.default.attributesOfFileSystem(forPath: parent.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    guard available > plan.temporaryBytes else { throw StudioError.invalid("The upscale needs \(plan.temporaryBytes / 1_000_000) MB of temporary free space.") }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    let native = plan.request.settings.backend == .swiftMLX
    guard !intermediateOnly || native else { throw StudioError.invalid("Lossless sequence execution requires native FlashVSR+.") }
    let outputName = native ? "output.mov" : "output.mp4"
    var accepted = false
    defer { if !accepted { try? FileManager.default.removeItem(at: directory.appendingPathComponent(outputName)) } }
    let started = Date(), runtime = plan.request.runtime, settings = plan.request.settings
    func ffmpeg(_ args: [String], _ name: String) throws {
      let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath), arguments: ["-v", "error", "-nostdin", "-n"] + args,
        directory: directory, log: directory.appendingPathComponent(name + ".log"), cancellation: cancellation)
      guard result.exitCode == 0 else { throw StudioError.invalid("FlashVSR+ media preparation failed. See " + name + ".log.") }
    }
    try plan.source.verify(); for file in plan.engineFiles { try file.verify() }
    try cancellation.check()
    emit(.init(stage: "prepare", message: "Preparing the exact source interval", fraction: 0))
    let selected = directory.appendingPathComponent("selected.mkv")
    let start = plan.request.firstFrame
    try ffmpeg(["-i", plan.source.path, "-map", "0:v:0", "-vf", "trim=start_frame=\(start):end_frame=\(start + plan.frames),setpts=PTS-STARTPTS,setsar=1", "-fps_mode", "passthrough", "-an", "-c:v", "ffv1", selected.path], "prepare")
    let audio = directory.appendingPathComponent("source-audio.wav")
    var audioFile: FlashVSRFile?
    if !intermediateOnly && settings.preserveAudio && plan.media.hasAudio {
      let offset = plan.media.firstTimestamp + Double(start) / plan.media.fps.value
      guard let sampleRate = plan.media.audioRate else { throw StudioError.invalid("The source audio sample rate is unavailable.") }
      try FlashVSRAudioPreparation.prepare(source: URL(fileURLWithPath: plan.source.path), output: audio,
        ffmpeg: URL(fileURLWithPath: runtime.ffmpegPath), directory: directory, startTimestamp: offset,
        frames: plan.frames, fps: plan.media.fps, sampleRate: sampleRate, cancellation: cancellation)
      audioFile = try FlashVSRFile.capture(audio)
    }
    let restored = directory.appendingPathComponent("restored.nut")
    try restore(plan, selected, restored, directory, cancellation, emit)
    // vpipe's RGB stream header quantizes FPS to three decimal places.
    // Validate that internal clock, then reconstruct exact source cadence
    // from frame indices for final encoding without dropping any frames.
    let internalFPS = native ? plan.media.fps : try FlashVSRFrameRate(numerator: Int((plan.media.fps.value * 1000).rounded()), denominator: 1000)
    let restoredInfo = try FlashVSRMediaInfo.inspect(restored, runtime: runtime, directory: directory, cancellation: cancellation, expectedFrameRate: internalFPS)
    try verify(restoredInfo, plan: plan, intermediate: true)
    if intermediateOnly {
      let lossless = directory.appendingPathComponent("restored.mkv")
      try ffmpeg(["-r", plan.media.fps.argument, "-i", restored.path, "-map", "0:v:0", "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-fps_mode", "passthrough", lossless.path], "lossless-transport")
      let actual = try FlashVSRMediaInfo.inspect(lossless, runtime: runtime, directory: directory, cancellation: cancellation)
      try verify(actual, plan: plan)
      func rgbHash(_ url: URL) throws -> Data {
        let value = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath),
          arguments: ["-v", "error", "-nostdin", "-i", url.path, "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-f", "hash", "-hash", "sha256", "-"],
          directory: directory, log: directory.appendingPathComponent(UUID().uuidString + "-rgb-hash.log"), cancellation: cancellation, maximumOutput: 65536)
        guard value.exitCode == 0, value.output.count == 72, String(decoding: value.output, as: UTF8.self).hasPrefix("SHA256=") else {
          throw StudioError.invalid("Cannot verify lossless FlashVSR+ RGB transport.")
        }
        return value.output
      }
      guard !actual.hasAudio, try rgbHash(restored) == rgbHash(lossless) else {
        throw StudioError.invalid("The lossless FlashVSR+ transport changed model output pixels.")
      }
      try plan.source.verify(); try cancellation.check()
      let artifact = try FlashVSRFile.capture(lossless)
      let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: artifact,
        restored: artifact, audio: nil, elapsedSeconds: Date().timeIntervalSince(started))
      let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
      try encoder.encode(plan.request).write(to: directory.appendingPathComponent("request.json"), options: .atomic)
      try encoder.encode(receipt).write(to: directory.appendingPathComponent("receipt.json"), options: .atomic)
      try? FileManager.default.removeItem(at: selected)
      try? FileManager.default.removeItem(at: restored)
      accepted = true
      emit(.init(stage: "completed", message: "Lossless FlashVSR+ stage verified", fraction: 1))
      return receipt
    }
    emit(.init(stage: "finish", message: "Verifying video and restoring source audio"))
    let video = directory.appendingPathComponent(outputName)
    var final = ["-r", plan.media.fps.argument, "-i", restored.path]
    if audioFile != nil { final += ["-i", audio.path] }
    final += ["-map", "0:v:0"]
    if audioFile != nil { final += ["-map", "1:a:0", "-c:a", native ? "pcm_f32le" : "aac"] } else { final += ["-an"] }
    final += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-fps_mode", "passthrough", video.path]
    try ffmpeg(final, "finish")
    let outputInfo = try FlashVSRMediaInfo.inspect(video, runtime: runtime, directory: directory, cancellation: cancellation)
    try verify(outputInfo, plan: plan)
    guard outputInfo.hasAudio == (audioFile != nil), audioFile == nil || (outputInfo.audioRate == plan.media.audioRate && outputInfo.audioChannels == plan.media.audioChannels) else {
      throw StudioError.invalid("The final audio track differs from the selected source format.")
    }
    if audioFile != nil {
      let decoded = directory.appendingPathComponent("verified-audio.wav")
      try ffmpeg(["-i", video.path, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", decoded.path], "verify-audio")
      if native {
        guard try pcmHash(audio, runtime: runtime, directory: directory, cancellation: cancellation)
          == pcmHash(decoded, runtime: runtime, directory: directory, cancellation: cancellation) else {
          throw StudioError.invalid("Native restoration changed the original source audio samples.")
        }
      } else { try FlashVSRAudioVerification.verify(source: audio, decoded: decoded) }
      try? FileManager.default.removeItem(at: decoded)
    }
    try plan.source.verify(); try cancellation.check()
    var receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: try FlashVSRFile.capture(video),
      restored: try FlashVSRFile.capture(restored), audio: audioFile, elapsedSeconds: Date().timeIntervalSince(started))
    let preview = directory.appendingPathComponent("source-preview.mp4")
    try ffmpeg(["-i", selected.path, "-vf", "scale=w='min(1280,iw)':h=-2", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", preview.path], "preview")
    receipt.sourcePreview = try FlashVSRFile.capture(preview)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
    try encoder.encode(plan.request).write(to: directory.appendingPathComponent("request.json"), options: .atomic)
    try encoder.encode(receipt).write(to: directory.appendingPathComponent("receipt.json"), options: .atomic)
    try? FileManager.default.removeItem(at: selected)
    accepted = true
    emit(.init(stage: "completed", message: "Upscale verified", fraction: 1))
    return receipt
  }
  private static func verify(_ actual: FlashVSRMediaInfo, plan: FlashVSRResolvedPlan, intermediate: Bool = false) throws {
    guard actual.frames == plan.frames, actual.width == plan.request.settings.width,
      actual.height == plan.request.settings.height, abs(actual.fps.value - plan.media.fps.value) < (intermediate ? 0.00051 : 0.00001) else {
      throw StudioError.invalid("FlashVSR+ output lost frames or changed the requested dimensions or frame rate. The result was not accepted.")
    }
  }
}
