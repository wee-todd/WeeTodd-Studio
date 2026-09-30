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
    videoActivationBytes: Int) throws {
    let config = MLXMediaPipeline.videoConfiguration(for: geometry,
      activationBytes: videoActivationBytes)
    _ = try MLXVideoDecodePlan(shape: geometry.videoShape,
      configuration: config)
    _ = try MLXAudioDecoder.estimatedPeakBytes(latentFrames: geometry.audioFrames)
  }

  public static func publish(_ sampled: MLXSceneSampler.Result,
    plan: LTX25ScenePlan, videoCheckpoint: URL, audioCheckpoint: URL,
    ffmpeg: URL, output: URL, videoActivationBytes: Int,
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
    try admit(geometry: g, videoActivationBytes: videoActivationBytes)
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
      activationBytes: videoActivationBytes)
    let unpacked = try g.unpackVideo(sampled.video.asArray(Float.self))
    let writer = try RawVideoWriter(ffmpeg: ffmpeg,
      output: staging.appendingPathComponent("video.mp4"),
      width: g.width, height: g.height, frames: deliveredFrames(plan: plan), fps: g.fps)
    defer { writer.cancel() }
    try autoreleasepool {
      let decoder = try MLXVideoDecoder(checkpoint: videoCheckpoint)
      try decoder.decodeRGB8(latent: MLXArray(unpacked, g.videoShape),
        configuration: videoConfig,
        progress: { try report("video_layers", $0, $1) }) { index, bytes in
          if index < deliveredFrames(plan: plan) {
            try writer.append(bytes, frame: index)
            try preview?(index, bytes)
          }
          try report("video_decode", index + 1, g.frames)
        }
    }
    try writer.finish()
    let videoSeconds = Date().timeIntervalSince(videoStart)
    Memory.clearCache(); try report("video_weights_released", 1, 1)
    let audioStart = Date()
    let unpackedAudio = try g.unpackAudio(sampled.audio.asArray(Float.self))
    let wave = try MLXMediaPipeline.decodeAudio(latent: unpackedAudio,
      latentFrames: g.audioFrames, checkpoint: audioCheckpoint,
      backend: .mlx) { try report("audio:" + $0) }
    guard wave.channels == 2, wave.sampleRate == 48000 else {
      throw LTXError.invalid("Swift LTX scene audio decoder returned an unexpected format.")
    }
    let deliveredAudio = deliveredAudioFrames(plan: plan)
    guard wave.frameCount >= deliveredAudio else {
      throw LTXError.invalid("Swift LTX scene audio ended before the delivered timeline.")
    }
    try MediaOutput.writeWAV(samples: Array(wave.samples.prefix(deliveredAudio * wave.channels)), sampleRate: wave.sampleRate,
      channels: wave.channels, to: staging.appendingPathComponent("audio.wav"))
    let audioSeconds = Date().timeIntervalSince(audioStart)
    Memory.clearCache(); try report("audio_weights_released", 1, 1)
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
    let metadata: [String: Any] = [
      "status":"complete","task":"scene","nativeRuntime":"swift-mlx",
      "publication_mode":"single_decode_native_latent_chain",
      "frames":deliveredFrames(plan:plan),"latent_frames":g.frames,"fps":g.fps,"width":g.width,"height":g.height,
      "window_frames":plan.windowFrames,"window_starts":plan.windowStarts,
      "segment_frames":plan.segmentFrames,"segment_starts":plan.segmentStarts,
      "video_overlap_latent_frames":plan.videoOverlapLatentFrames,
      "join_audio_tokens":plan.joinAudioTokens,
      "window_seconds":sampled.windowSeconds,
      "video_decode_seconds":videoSeconds,"audio_decode_seconds":audioSeconds,
      "audio_samples":deliveredAudio,"decoded_audio_samples":wave.frameCount,"audio_sample_rate":wave.sampleRate,
      "peak_mlx_bytes":Memory.peakMemory,
      "peak_process_footprint_bytes":memory.ledger_phys_footprint_peak,
      "process_memory_scope":"Swift process; external FFmpeg process excluded",
      "python_inference":false,"production_qualified":false]
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
