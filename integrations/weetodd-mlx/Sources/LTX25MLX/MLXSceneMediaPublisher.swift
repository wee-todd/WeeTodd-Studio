import Foundation
import Darwin
import MLX
import LTX25Engine
import LTX25Audio
import InferenceMedia

/// Decodes assembled scene latents once, after every sampling stage has gone
/// out of scope. Publication remains atomic and uses the same native VAEs and
/// FFmpeg path as an ordinary Studio clip.
public enum MLXSceneMediaPublisher {
  public struct StrictFrameRoute {
    public let windows: Range<Int>
    public let local: Range<Int>
    public let outputStart: Int
    public let frames: Int
  }
  public static func strictFrameRoutes(plan: LTX25ScenePlan,
    strictBoundaries: Set<Int>? = nil) throws -> [StrictFrameRoute] {
    let boundaries = strictBoundaries ?? Set(1..<plan.windowFrames.count)
    guard !boundaries.isEmpty,
      boundaries.allSatisfy({ (1..<plan.windowFrames.count).contains($0) }) else {
      throw LTXError.invalid("Strict scene image cuts require interior shot boundaries.")
    }
    let starts = [0] + boundaries.sorted()
    let ends = boundaries.sorted() + [plan.windowFrames.count]
    let routes = zip(starts, ends).map { start, end -> StrictFrameRoute in
      let localStart = start == 0 ? 0 : plan.overlapFrames - 1
      let delivered = plan.segmentFrames[start..<end].reduce(0, +)
      return StrictFrameRoute(windows: start..<end,
        local: localStart..<(localStart + delivered),
        outputStart: plan.segmentStarts[start],
        frames: plan.windowFrames[start] +
          (start + 1..<end).reduce(0) { $0 + plan.segmentFrames[$1] })
    }
    guard routes.allSatisfy({ $0.local.upperBound < $0.frames }),
      routes.reduce(0, { $0 + $1.local.count }) == deliveredFrames(plan: plan) else {
      throw LTXError.invalid("Strict scene image cuts do not cover the exact editorial frames.")
    }
    return routes
  }
  /// The final causal decoder frame supplies context but is outside the
  /// editorial scene. The audio stream must end at the same delivered time.
  public static func deliveredFrames(plan: LTX25ScenePlan) -> Int { plan.totalFrames - 1 }
  public static func deliveredAudioFrames(plan: LTX25ScenePlan) -> Int {
    Int((Double(deliveredFrames(plan: plan)) / plan.fps * 48000).rounded())
  }
  public static func requiredVideoActivationBytes(geometry: AVGeometry) throws -> Int {
    try MLXVideoDecodePlan(shape: geometry.videoShape,
      configuration: MLXMediaPipeline.videoConfiguration(for: geometry,
        activationBytes: Int.max)).admittedActivationBytes
  }

  public static func admit(geometry: AVGeometry,
    decodePlan: MLXSceneDecodeWindowPlan) throws {
    let config = MLXMediaPipeline.videoConfiguration(for: geometry,
      activationBytes: decodePlan.admittedActivationBytes)
    guard decodePlan.latentRanges.first?.lowerBound == 0,
      decodePlan.latentRanges.last?.upperBound == geometry.latentFrames else {
      throw LTXError.invalid("Scene video decode windows do not cover the admitted latent.")
    }
    for range in decodePlan.latentRanges {
      _ = try MLXVideoDecodePlan(shape: [1, 128, range.count,
        geometry.latentHeight, geometry.latentWidth], configuration: config)
    }
    _ = try MLXAudioDecoder.estimatedPeakBytes(latentFrames: geometry.audioFrames)
  }

  public static func publish(_ sampled: MLXSceneSampler.Result,
    plan: LTX25ScenePlan, videoCheckpoint: URL, audioCheckpoint: URL,
    ffmpeg: URL, output: URL, decodePlan: MLXSceneDecodeWindowPlan,
    decodeMode: MLXSceneDecodeMode,
    diffusionVAE:MLXDiffusionVideoSettings?=nil,
    sourceAudio: MLXSourceAudioInterval.Prepared? = nil,
    preview: ((Int, Data) throws -> Void)? = nil,
    beforePublish: ((URL, [String: Any]) throws -> Void)? = nil,
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> URL {
    let g = sampled.geometry
    guard g.frames == plan.totalFrames, g.fps == plan.fps,
      sampled.video.dtype == .float32,
      sampled.video.shape == [g.videoTokens, 128],
      sampled.audio.dtype == .float32,
      sampled.audio.shape == [g.audioFrames, 128],
      ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      output.isFileURL, !FileManager.default.fileExists(atPath: output.path) else {
      throw LTXError.invalid("Swift LTX scene publication does not match its admitted audiovisual timeline.")
    }
    if !sampled.strictBoundaries.isEmpty {
      guard sampled.strictBoundaries.allSatisfy({ (1..<plan.windowFrames.count).contains($0) }),
        sampled.videoWindows.count == plan.windowFrames.count else {
        throw LTXError.invalid("Strict image cuts require every sampled shot window.")
      }
      for index in sampled.videoWindows.indices {
        let window = try AVGeometry(width: g.width, height: g.height,
          frames: plan.windowFrames[index], fps: g.fps)
        guard sampled.videoWindows[index].dtype == .float32,
          sampled.videoWindows[index].shape == [window.videoTokens,128] else {
          throw LTXError.invalid("Strict image cut window differs from its admitted latent geometry.")
        }
      }
    }
    let selected=try MLXVideoDecoderSelection(checkpoint:videoCheckpoint,settings:diffusionVAE)
    if selected.isDiffusion {
      guard decodePlan.overlapFrames == 0,decodeMode.maximumWindowFrames == nil else { throw LTXError.invalid("DiffVAE scene publication requires complete groups with internal tiling.") }
      _ = try MLXDiffusionScenePublication.admit(geometry:g,plan:plan,strictBoundaries:sampled.strictBoundaries,
        checkpoint:videoCheckpoint,settings:diffusionVAE,maximumWorkspaceBytes:decodePlan.admittedActivationBytes)
    } else { try admit(geometry:g,decodePlan:decodePlan) }
    if case .single = decodeMode, decodePlan.latentRanges.count != 1,
      sampled.strictBoundaries.isEmpty {
      throw LTXError.invalid("Single-decode LTX scenes require one admitted video window.")
    }
    let fm = FileManager.default
    try fm.createDirectory(at: output.deletingLastPathComponent(),
      withIntermediateDirectories: true)
    let staging = output.deletingLastPathComponent()
      .appendingPathComponent(".weetodd-scene-" + UUID().uuidString)
    try fm.createDirectory(at: staging, withIntermediateDirectories: false)
    var published = false
    defer {
      if !published { try? fm.removeItem(at: staging) }
      Stream.gpu.synchronize(); Memory.clearCache()
    }
    func report(_ stage: String, _ completed: Int = 0, _ total: Int = 1) throws {
      try Task.checkCancellation()
      try progress(stage, completed, total)
      try Task.checkCancellation()
    }
    let videoStart = Date()
    let videoConfig = MLXMediaPipeline.videoConfiguration(for: g,
      activationBytes: decodePlan.admittedActivationBytes)
    let writer = try RawVideoWriter(ffmpeg: ffmpeg,
      output: staging.appendingPathComponent("video.mp4"),
      width: g.width, height: g.height, frames: deliveredFrames(plan: plan), fps: g.fps)
    defer { writer.cancel() }
    try autoreleasepool {
      if selected.isDiffusion {
        try MLXDiffusionScenePublication.decode(sampled,plan:plan,checkpoint:videoCheckpoint,settings:diffusionVAE,
          maximumWorkspaceBytes:decodePlan.admittedActivationBytes,
          progress:{ try report("video_layers",$0,$1) }) { index,bytes in
          try writer.append(bytes,frame:index);try preview?(index,bytes)
          try report("video_decode",index+1,deliveredFrames(plan:plan))
        }
      } else {
      let decoder = try MLXVideoDecoder(checkpoint: videoCheckpoint)
      if !sampled.strictBoundaries.isEmpty {
        let routes = try strictFrameRoutes(plan: plan,
          strictBoundaries: sampled.strictBoundaries)
        for route in routes {
          try Task.checkCancellation()
          let window = try AVGeometry(width: g.width, height: g.height,
            frames: route.frames, fps: g.fps)
          let windowPlan = try MLXSceneDecodeWindowPlan(geometry: window,
            maximumActivationBytes: decodePlan.admittedActivationBytes,
            maximumWindowFrames: decodeMode.maximumWindowFrames)
          let grouped = try MLXSceneLatentAssembly.assembleVideoGroup(
            video: sampled.videoWindows, plan: plan,
            windows: route.windows, latentHeight: g.latentHeight,
            latentWidth: g.latentWidth)
          let values = try window.unpackVideo(grouped.asArray(Float.self))
          let latent = MLXArray(values, window.videoShape)
          func deliver(_ local: Int, _ bytes: Data) throws {
            guard route.local.contains(local) else { return }
            let outputIndex = route.outputStart + local - route.local.lowerBound
            try writer.append(bytes, frame: outputIndex)
            try preview?(outputIndex, bytes)
            try report("video_decode", outputIndex + 1, deliveredFrames(plan: plan))
          }
          if windowPlan.latentRanges.count == 1 {
            try decoder.decodeRGB8(latent: latent, configuration: videoConfig,
              progress: { try report("video_layers", $0, $1) }) { index, bytes in
                try deliver(index, bytes)
              }
          } else {
            var joiner = try MLXSceneRGBJoiner(windowCount: windowPlan.latentRanges.count,
              frameBytes: g.width * g.height * 3)
            for (partIndex, range) in windowPlan.latentRanges.enumerated() {
              let part = latent[0..., 0..., range, 0..., 0...]
              let frames = (range.count - 1) * 8 + 1
              try decoder.decodeRGB8(latent: part, configuration: videoConfig,
                progress: { try report("video_layers", $0, $1) }) { index, bytes in
                  try joiner.receive(window: partIndex, frame: index, count: frames,
                    rgb: bytes) { local, value in try deliver(local, value) }
                }
              try joiner.finishWindow(window: partIndex)
            }
            try joiner.finish(expectedFrames: window.frames - 1)
          }
          Stream.gpu.synchronize(); Memory.clearCache()
        }
      } else {
      let unpacked = try g.unpackVideo(sampled.video.asArray(Float.self))
      let latent = MLXArray(unpacked, g.videoShape)
      let ranges = decodePlan.latentRanges
      if case .single = decodeMode {
        try decoder.decodeRGB8(latent: latent, configuration: videoConfig,
          progress: { try report("video_layers", $0, $1) }) { index, bytes in
            if index < deliveredFrames(plan: plan) {
              try writer.append(bytes, frame: index)
              try preview?(index, bytes)
            }
            try report("video_decode", index + 1, g.frames)
          }
      } else {
        var joiner = try MLXSceneRGBJoiner(windowCount: ranges.count,
          frameBytes: g.width * g.height * 3)
        for (window, range) in ranges.enumerated() {
          try Task.checkCancellation()
          let part = latent[0..., 0..., range, 0..., 0...]
          let frameCount = (range.count - 1) * 8 + 1
          try decoder.decodeRGB8(latent: part, configuration: videoConfig,
            progress: { completed, total in
              try report("video_layers", window * total + completed,
                ranges.count * total)
            }) { index, bytes in
              try joiner.receive(window: window, frame: index, count: frameCount,
                rgb: bytes) { outputIndex, outputBytes in
                  try writer.append(outputBytes, frame: outputIndex)
                  try preview?(outputIndex, outputBytes)
                  try report("video_decode", outputIndex + 1,
                    deliveredFrames(plan: plan))
                }
            }
          try joiner.finishWindow(window: window)
        }
        try joiner.finish(expectedFrames: deliveredFrames(plan: plan))
      }
      }
    }
    }
    try writer.finish()
    let videoSeconds = Date().timeIntervalSince(videoStart)
    Memory.clearCache(); try report("video_weights_released", 1, 1)
    let audioStart = Date()
    let deliveredAudio = deliveredAudioFrames(plan: plan)
    let decodedAudioSamples: Int
    if let sourceAudio {
      guard sourceAudio.publicationSamples == deliveredAudio else {
        throw LTXError.invalid("Swift LTX scene source audio does not match the editorial timeline.")
      }
      try MLXMediaPipeline.publishSourceAudio(sourceAudio,
        to: staging.appendingPathComponent("audio.wav"))
      decodedAudioSamples = 0
      try report("source_audio_published", 1, 1)
    } else {
      let unpackedAudio = try g.unpackAudio(sampled.audio.asArray(Float.self))
      let wave = try MLXMediaPipeline.decodeAudio(latent: unpackedAudio,
        latentFrames: g.audioFrames, checkpoint: audioCheckpoint,
        backend: .mlx) { try report("audio:" + $0) }
      guard wave.channels == 2, wave.sampleRate == 48000 else {
        throw LTXError.invalid("Swift LTX scene audio decoder returned an unexpected format.")
      }
      guard wave.frameCount >= deliveredAudio else {
        throw LTXError.invalid("Swift LTX scene audio ended before the delivered timeline.")
      }
      try MediaOutput.writeWAV(samples: Array(wave.samples.prefix(deliveredAudio * wave.channels)),
        sampleRate: wave.sampleRate, channels: wave.channels,
        to: staging.appendingPathComponent("audio.wav"))
      decodedAudioSamples = wave.frameCount
      Memory.clearCache(); try report("audio_weights_released", 1, 1)
    }
    let audioSeconds = Date().timeIntervalSince(audioStart)
    try MLXMediaPipeline.mux(ffmpeg: ffmpeg, directory: staging,
      fps: g.fps, rawVideo: true)
    var memory = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size /
      MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &memory) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS else {
      throw LTXError.invalid("Cannot measure Swift LTX scene process memory.")
    }
    var metadata: [String: Any] = [
      "status":"complete","task":"scene","nativeRuntime":"swift-mlx",
      "publication_mode":selected.isDiffusion ? "diffusion_internal_tiling_native_latent_chain":decodeMode.publicationMode,
      "video_decode_strategy":sampled.strictBoundaries.isEmpty
        ? "assembled-scene" : "separate-shot-windows-strict-image-cuts",
      "decode_window_latent_ranges":decodePlan.latentRanges.map { [$0.lowerBound,$0.upperBound] },
      "decode_admitted_activation_bytes":decodePlan.admittedActivationBytes,
      "frames":deliveredFrames(plan:plan),"latent_frames":g.frames,"fps":g.fps,"width":g.width,"height":g.height,
      "window_frames":plan.windowFrames,"window_starts":plan.windowStarts,
      "segment_frames":plan.segmentFrames,"segment_starts":plan.segmentStarts,
      "strict_image_boundaries":sampled.strictBoundaries.sorted(),
      "video_overlap_latent_frames":plan.videoOverlapLatentFrames,
      "join_audio_tokens":plan.joinAudioTokens,
      "window_seconds":sampled.windowSeconds,
      "video_decode_seconds":videoSeconds,"audio_decode_seconds":audioSeconds,
      "audio_samples":deliveredAudio,"decoded_audio_samples":decodedAudioSamples,"audio_sample_rate":48000,
      "audio_decoder":sourceAudio == nil ? "mlx" : "none-source-publication",
      "peak_mlx_bytes":Memory.peakMemory,
      "peak_process_footprint_bytes":memory.ledger_phys_footprint_peak,
      "process_memory_scope":"Swift process; external FFmpeg process excluded",
      "python_inference":false,"production_qualified":false]
    metadata.merge(MLXNativeVideoDecoder.publicationMetadata(isDiffusion:selected.isDiffusion,settings:diffusionVAE)) { _,new in new }
    try JSONSerialization.data(withJSONObject: metadata,
      options: [.prettyPrinted, .sortedKeys]).write(
        to: staging.appendingPathComponent("report.json"),
        options: .withoutOverwriting)
    try MLXMediaPipeline.publish(staging: staging, output: output) {
      try beforePublish?(staging, metadata)
      try report("ready_to_publish", 1, 1)
    }
    published = true
    return output.appendingPathComponent("render.mp4")
  }
}
