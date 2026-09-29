import AVFoundation
import Darwin
import Foundation
import MLX
import LTX25Engine
import InferenceMedia
import AdapterRuntime

/// Full-rate Ripple execution. The source-guide VAE, single-stage sampler and
/// output VAE are staged in one worker; Studio and headless callers share it.
public final class MLXRipplePipeline {
  public let request: MLXRippleRequest
  public let memory: MLXStudioMemoryPlan
  private let gate = NSLock()
  private var running = false

  public init(request: MLXRippleRequest, physicalMemory: UInt64,
    recommendedWorkingSet: UInt64) throws {
    memory = try MLXStudioMemoryPlan(ripple: request, physicalMemory: physicalMemory,
      recommendedWorkingSet: recommendedWorkingSet)
    self.request = request
  }

  /// Header and identity checks precede the first weighted stage. The guide
  /// must already have been prepared from the inspected, frozen source interval.
  public func preflight() throws {
    try request.validateSource()
    try request.validateGuide()
    try MLXRippleAdapterIdentity.verify(URL(fileURLWithPath: request.adapterPath))
    try MLXReferenceImage.inspect(URL(fileURLWithPath: request.firstReferencePath))
    for anchor in request.anchors {
      try MLXReferenceImage.inspect(URL(fileURLWithPath: anchor.path))
    }
    let text = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: request.gemmaRoot),
      connectorURL: URL(fileURLWithPath: request.connectorCheckpoint))
    _ = try MLXTextEncodingPlan(promptTokens: text.tokenize(request.prompt).count)
    _ = try MLXVideoEncoder(checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
    _ = try MLXVideoDecoder(checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
    guard FileManager.default.fileExists(atPath: request.transformerRoot) else {
      throw LTXError.invalid("Ripple distilled transformer is missing.")
    }
    _ = try MLXSingleStageRipple(geometry: request.geometry,
      referenceStrength: request.referenceStrength,
      imageAnchors: request.anchors.map {
        RippleImageAnchor(frame: $0.frame, strength: $0.strength)
      }, transformerRoot: URL(fileURLWithPath: request.transformerRoot),
      adapters: [LoRAAdapter(path: request.adapterPath,
        strength: request.adapterStrength)],
      maximumActivationBytes: memory.transformerActivationBytes)
    guard !FileManager.default.fileExists(atPath: request.outputDirectory) else {
      throw LTXError.invalid("Ripple output directory already exists.")
    }
  }

  public func run(ffmpeg: URL, decodedPreview: ((Int, Data) throws -> Void)? = nil,
    beforePublish: ((URL, [String: Any]) throws -> Void)? = nil,
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) async throws -> [String: Any] {
    guard gate.withLock({
      if running { return false }
      running = true
      return true
    }) else { throw LTXError.invalid("Ripple pipeline is already active.") }
    defer {
      Stream.gpu.synchronize(); Memory.clearCache()
      gate.withLock { running = false }
    }
    guard ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw LTXError.invalid("Ripple requires an executable FFmpeg.")
    }
    try preflight()
    let started = Date(), fm = FileManager.default
    let output = URL(fileURLWithPath: request.outputDirectory)
    let parent = output.deletingLastPathComponent()
    try fm.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".ripple-partial-" + UUID().uuidString)
    try fm.createDirectory(at: staging, withIntermediateDirectories: false)
    var published = false
    defer { if !published { try? fm.removeItem(at: staging) } }
    let g = request.geometry
    var timings: [String: Double] = [:]
    var tokenIDs: [Int] = []
    func report(_ stage: String, _ done: Int = 0, _ total: Int = 1) throws {
      try Task.checkCancellation()
      try progress(stage, done, total)
      try Task.checkCancellation()
    }
    Memory.peakMemory = 0
    let sampled: [Float] = try autoreleasepool {
      let textStart = Date()
      let contexts = try autoreleasepool {
        let encoder = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: request.gemmaRoot),
          connectorURL: URL(fileURLWithPath: request.connectorCheckpoint))
        return try encoder.encode(prompt: request.prompt) {
          try report("text:" + $0.stage, $0.completed, $0.total)
        }
      }
      tokenIDs = contexts.tokenIDs
      timings["text"] = Date().timeIntervalSince(textStart)
      try report("text_weights_released")

      let referenceStart = Date()
      let references: (MLXArray, [MLXArray]) = try autoreleasepool {
        let tilePlan = try MLXVideoEncodeTilePlan(frames: g.frames, width: g.width,
          height: g.height,
          maximumOwnedBufferBytes: min(4 * 1024 * 1024 * 1024,
            memory.activationCeilingBytes))
        let encodedGuide = try MLXTiledVideoEncoder.encode(
          guide: URL(fileURLWithPath: request.guidePath),
          checkpoint: URL(fileURLWithPath: request.videoCheckpoint), plan: tilePlan) {
            try report("guide_encode", $0, $1)
          }.reshaped([g.videoTokens, 128])
        var encodedAnchors: [MLXArray] = []
        if !request.anchors.isEmpty {
          let encoder = try MLXImageEncoder(checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
          let imageDirectory = staging.appendingPathComponent("reference-preparation")
          try fm.createDirectory(at: imageDirectory, withIntermediateDirectories: false)
          defer { try? fm.removeItem(at: imageDirectory) }
          for (index, anchor) in request.anchors.enumerated() {
            let pixels = try MLXReferenceImage.prepare(URL(fileURLWithPath: anchor.path),
              width: g.width, height: g.height, crf: 0, ffmpeg: ffmpeg,
              temporaryParent: imageDirectory)
            let latent = try encoder.encode(MLXArray(pixels, [1, g.height, g.width, 3])) {
              try report("anchor_encode", index * 42 + $0, request.anchors.count * 42)
            }
            encodedAnchors.append(latent)
          }
        }
        return (encodedGuide, encodedAnchors)
      }
      timings["reference_encode"] = Date().timeIntervalSince(referenceStart)
      try report("reference_weights_released")

      let samplingStart = Date()
      let latent = try autoreleasepool {
        let sampler = try MLXSingleStageRipple(geometry: g,
          referenceStrength: request.referenceStrength,
          imageAnchors: request.anchors.map {
            RippleImageAnchor(frame: $0.frame, strength: $0.strength)
          }, transformerRoot: URL(fileURLWithPath: request.transformerRoot),
          adapters: [LoRAAdapter(path: request.adapterPath,
            strength: request.adapterStrength)],
          maximumActivationBytes: memory.transformerActivationBytes)
        let result = try sampler.evaluate(videoContext: contexts.video,
          audioContext: contexts.audio, referenceVideo: references.0,
          imageAnchors: references.1, seed: request.seed, progress: report)
        return result["video"]!.asArray(Float.self)
      }
      timings["sampling"] = Date().timeIntervalSince(samplingStart)
      return latent
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    try report("sampling_weights_released")

    let videoStart = Date()
    let video = staging.appendingPathComponent("video.mp4")
    let writer = try RawVideoWriter(ffmpeg: ffmpeg, output: video,
      width: g.width, height: g.height, frames: request.editorialFrames, fps: g.fps)
    defer { writer.cancel() }
    let decoded = try g.unpackVideo(sampled)
    let decoder = try MLXVideoDecoder(checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
    let configuration = MLXMediaPipeline.videoConfiguration(for: g,
      activationBytes: memory.videoActivationBytes)
    var written = 0
    try decoder.decodeRGB8(latent: MLXArray(decoded, g.videoShape),
      configuration: configuration, progress: {
        try report("video_layers", $0, $1)
      }) { index, rgb in
        guard index < request.editorialFrames else { return }
        guard index == written else { throw LTXError.invalid("Ripple decoder skipped an editorial frame.") }
        try writer.append(rgb, frame: index)
        written += 1
        try report("video_decode", written, request.editorialFrames)
        try decodedPreview?(index, rgb)
      }
    guard written == request.editorialFrames else {
      throw LTXError.invalid("Ripple video decode did not cover the editorial interval.")
    }
    try writer.finish()
    timings["video_decode"] = Date().timeIntervalSince(videoStart)
    try report("video_weights_released")

    try request.validateSource()
    let audioStart = Date()
    let hasAudio: Bool
    if request.audioPolicy == "preserve" {
      let source = AVURLAsset(url: URL(fileURLWithPath: request.sourcePath))
      hasAudio = try await !source.loadTracks(withMediaType: .audio).isEmpty
    } else { hasAudio = false }
    let movie = staging.appendingPathComponent("ripple.mp4")
    if hasAudio {
      try Self.muxSourceAudio(ffmpeg: ffmpeg, video: video,
        source: URL(fileURLWithPath: request.sourcePath), target: movie,
        start: request.sourceStart, duration: request.duration,
        log: staging.appendingPathComponent("mux.log"))
    } else {
      try fm.moveItem(at: video, to: movie)
    }
    timings["source_audio"] = Date().timeIntervalSince(audioStart)
    try report("source_audio_published")

    let inputs = [(0, request.firstReferencePath, request.referenceStrength)] +
      request.anchors.map { ($0.frame, $0.path, $0.strength) }
    var references: [[String: Any]] = []
    for (index, input) in inputs.enumerated() {
      let name = String(format: "edited-%02d-frame-%06d", index + 1, input.0)
        + "." + URL(fileURLWithPath: input.1).pathExtension.lowercased()
      let frozen = staging.appendingPathComponent(name)
      try fm.copyItem(at: URL(fileURLWithPath: input.1), to: frozen)
      references.append(["frame": input.0,
        "path": output.appendingPathComponent(name).path,
        "strength": Double(input.2)])
    }
    let result: [String: Any] = [
      "video_path": output.appendingPathComponent("ripple.mp4").path,
      "path": output.appendingPathComponent("ripple.mp4").path,
      "duration": request.duration, "frames": request.editorialFrames,
      "frame_rate": request.fps, "width": request.width, "height": request.height,
      "has_audio": hasAudio, "artifacts_directory": output.path,
      "receipt_path": output.appendingPathComponent("receipt.json").path,
      "conditioning_mode": request.anchors.isEmpty && request.referenceStrength == 1
        ? "author_first_frame" : "studio_timed_anchors",
      "frozen_references": references, "source_sha256": request.sourceSHA256,
      "nativeRuntime": "swift-mlx"]
    var info = task_vm_info_data_t()
    var infoCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size /
      MemoryLayout<integer_t>.size)
    let memoryStatus = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &infoCount)
      }
    }
    guard memoryStatus == KERN_SUCCESS else {
      throw LTXError.invalid("Cannot measure Ripple process memory.")
    }
    let metadata: [String: Any] = [
      "status": "complete", "task": "ripple", "nativeRuntime": "swift-mlx",
      "production_qualified": false, "python_inference": false,
      "width": request.width, "height": request.height,
      "model_frames": request.frames, "editorial_frames": request.editorialFrames,
      "fps": request.fps, "seed": request.seed, "text_token_ids": tokenIDs,
      "reference_count": references.count, "stage_seconds": timings,
      "seconds": Date().timeIntervalSince(started),
      "peak_mlx_bytes": Memory.peakMemory,
      "peak_process_footprint_bytes": info.ledger_phys_footprint_peak,
      "current_process_footprint_bytes": info.phys_footprint,
      "process_memory_scope": "Swift worker; external FFmpeg excluded"]
    let receipt: [String: Any] = [
      "format": "weetodd-ripple-take-v1", "status": "complete",
      "source_sha256": request.sourceSHA256, "source_path": request.sourcePath,
      "source_start": request.sourceStart, "duration": request.duration,
      "editorial_frames": request.editorialFrames,
      "adapter_sha256": MLXRippleAdapterIdentity.sha256,
      "reference_images": references,
      "publication_audio": hasAudio ? "preserved source interval" : "silent",
      "qualification": "Native Swift MLX Ripple requires visual and production-size qualification."]
    try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
      .write(to: staging.appendingPathComponent("report.json"), options: .withoutOverwriting)
    try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
      .write(to: staging.appendingPathComponent("receipt.json"), options: .withoutOverwriting)
    try MLXMediaPipeline.publish(staging: staging, output: output) {
      try beforePublish?(staging, result)
      try report("ready_to_publish", 1, 1)
    }
    published = true
    return result
  }

  static func muxSourceAudio(ffmpeg: URL, video: URL, source: URL,
    target: URL, start: Double, duration: Double, log: URL) throws {
    let fd = Darwin.open(log.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0o600)
    guard fd >= 0 else { throw LTXError.invalid("Cannot create Ripple mux log.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-nostdin", "-n", "-i", video.path,
      "-ss", String(start), "-t", String(duration), "-i", source.path,
      "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
      "-af", "aresample=async=1:first_pts=0,apad=whole_dur=\(duration),atrim=duration=\(duration)",
      "-c:a", "aac", "-b:a", "192k", "-t", String(duration),
      "-movflags", "+faststart", target.path]
    process.standardOutput = handle
    process.standardError = handle
    try Task.checkCancellation()
    try process.run()
    defer {
      if process.isRunning {
        process.terminate(); usleep(100000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10000) }
    guard process.terminationStatus == 0 else {
      throw LTXError.invalid("Ripple source audio mux failed (exit \(process.terminationStatus)).")
    }
  }
}
