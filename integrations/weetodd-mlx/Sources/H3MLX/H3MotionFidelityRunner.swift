import Foundation
import MLX
import TensorIO

/// Motion repair owns media expansion, but uses the ordinary native H3 sampler.
/// Video analysis/encoding, audio encoding, Qwen, DiT and decoding remain staged.
public enum H3MotionFidelityRunner {
  public struct Prepared {
    public let request: H3T2VARequest
    public let plan: H3MotionFidelityPlan
    public let rows: H3JointLatentArtifact.Rows?
    public let originalAudio: H3AudioReference
    let originalRGB: [UInt8]?
  }

  public static func preflight(base: H3T2VARequest,
    source: H3MotionFidelityMedia.Source, settings: H3MotionFidelitySettings) throws -> H3T2VARunner.Admission {
    try settings.validate(); try source.identity.verify()
    guard base.requestedSteps >= 16, base.funControl == nil, base.vdn == nil,
      base.geometry.width == source.width, base.geometry.height == source.height,
      base.loRAAdapters.allSatisfy({ $0.startAfterEvaluations == 0 && $0.profile != .turbo }) else {
      throw H3CheckpointError.invalid("Motion Fidelity requires a plain full-schedule T2VA repair recipe and immediate ordinary adapters.")
    }
    try validateBaselineMetadata(base.transformer)
    let maximum = settings.mode == .uniform ? try H3MotionFidelityPlan.alignedFrames(source.frames * settings.maxHold) : settings.maxFrames - (settings.maxFrames - 5) % 17
    let request = try expandedRequest(base: base, source: source, frames: maximum, settings: settings)
    let controls = try H3JointRefinement(strength: settings.strength,
      startVideoSigma: settings.strength, evaluations: settings.evaluations, preserveAudio: true)
    let admission = try H3T2VARunner.preflight(request, refinement: controls)
    try H3VideoVAEEncoder.preflight(checkpointURL: base.videoVAE)
    try H3AudioVAEEncoder.inspect(checkpointURL: base.audioVAE)
    for adapter in base.loRAAdapters {
      _ = try H3LoRAFile(url: adapter.url, strength: adapter.strength,
        requestedSteps: try controls.schedules(requestedSteps: base.requestedSteps).video.timesteps.count + 1,
        samplingMethod: base.samplingMethod, qkvLayout: adapter.qkvLayout,
        profile: adapter.profile, startAfterEvaluations: 0, requiresStandardProfile: true)
    }
    return admission
  }

  public static func prepare(base: H3T2VARequest, source: H3MotionFidelityMedia.Source,
    settings: H3MotionFidelitySettings, ffmpeg: URL, scratch: URL,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Prepared {
    _ = try preflight(base: base, source: source, settings: settings)
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    let decoded = try H3MotionFidelityMedia.decode(source, ffmpeg: ffmpeg)
    let jerk: [Float]?
    if settings.mode == .adaptive {
      jerk = try autoreleasepool {
        let count = try H3MotionFidelityPlan.alignedFrames(source.frames)
        let pixels = try expandedPixels(decoded.rgb8, width: source.width, height: source.height,
          indices: (0..<count).map { min($0, source.frames - 1) })
        let latent = try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: base.videoVAE,
          rgb8: pixels, frameCount: count, width: source.width, height: source.height) {
          progress("motion_analysis", $0, $1)
        }
        let layout = try H3VideoVAELayout(url: base.videoVAE)
        return try temporalJerk(latent, mean: layout.latentsMean,
          standardDeviation: layout.latentsStandardDeviation)
      }
      Stream.gpu.synchronize(); Memory.clearCache()
      progress("motion_analysis_weights_released", 1, 1)
    } else { jerk = nil }
    let plan = try H3MotionFidelityPlan(sourceFrames: source.frames,
      settings: settings, temporalJerk: jerk)
    let request = try expandedRequest(base: base, source: source, frames: plan.paddedFrames, settings: settings)
    if plan.noop {
      try source.identity.verify()
      return Prepared(request: request, plan: plan, rows: nil,
        originalAudio: decoded.audio, originalRGB: decoded.rgb8)
    }
    let video: [Float] = try autoreleasepool {
      let pixels = try expandedPixels(decoded.rgb8, width: source.width, height: source.height,
        indices: plan.expansionIndices)
      let latent = try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: base.videoVAE,
        rgb8: pixels, frameCount: plan.paddedFrames, width: source.width, height: source.height) {
        progress("motion_video_encode", $0, $1)
      }
      let metadata = try H3VideoVAELayout(url: base.videoVAE)
      return try H3LatentCodec.videoEncoderRows(latents: latent,
        mean: metadata.latentsMean, standardDeviation: metadata.latentsStandardDeviation).asArray(Float.self)
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    progress("motion_video_weights_released", 1, 1)
    let audio: [Float] = try autoreleasepool {
      let expanded = try H3MotionFidelityMedia.expandedAudio(original: decoded.audio,
        plan: plan, ffmpeg: ffmpeg, scratch: scratch)
      let latent = try H3AudioVAEEncoder.encode(checkpointURL: base.audioVAE,
        waveform: MLXArray(expanded.samples, [2, expanded.frames, 1]))
      let frames = request.geometry.audioLatentFrames
      // DAC pads to ceil(samples/800). Drop only its known final clock-padding token.
      let bounded = try cropAudioPadding(latent, samples: expanded.frames, expectedFrames: frames)
      let metadata = try H3AudioVAELayout(url: base.audioVAE)
      return try H3LatentCodec.audioEncoderRows(latents: bounded,
        mean: metadata.latentsMean, standardDeviation: metadata.latentsStandardDeviation).asArray(Float.self)
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    progress("motion_audio_weights_released", 1, 1)
    let rows = H3JointLatentArtifact.Rows(video: video, audio: audio)
    try rows.validate(geometry: request.geometry); try source.identity.verify()
    return Prepared(request: request, plan: plan, rows: rows,
      originalAudio: decoded.audio, originalRGB: nil)
  }

  public static func run(_ prepared: Prepared, source: H3MotionFidelityMedia.Source,
    onFrame: (Int, Data) throws -> Void, onAudio: ([Float], Int) throws -> Void,
    onLatents: (([Float], [Float]) throws -> Void)? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> H3T2VARunner.Result {
    try source.identity.verify(); try Task.checkCancellation()
    guard prepared.plan.sourceFrames == source.frames,
      prepared.request.geometry.width == source.width, prepared.request.geometry.height == source.height else {
      throw H3CheckpointError.invalid("Motion preparation no longer matches its source.")
    }
    if prepared.plan.noop {
      guard let pixels = prepared.originalRGB, pixels.count == source.frames * source.width * source.height * 3,
        prepared.rows == nil else { throw H3CheckpointError.invalid("Invalid motion no-op preparation.") }
      let frameBytes = source.width * source.height * 3
      for frame in 0..<source.frames {
        try Task.checkCancellation()
        try onFrame(frame, Data(pixels[(frame * frameBytes)..<((frame + 1) * frameBytes)]))
        progress("motion_recovery", frame + 1, source.frames)
      }
      try onAudio(prepared.originalAudio.samples, 32_000); try source.identity.verify()
      return .init(videoFrames: 0, audioSamplesPerChannel: prepared.originalAudio.frames,
        audioSampleRate: 32_000)
    }
    guard let rows = prepared.rows else { throw H3CheckpointError.invalid("Motion repair has no initialized AV rows.") }
    let controls = try H3JointRefinement(strength: prepared.plan.settings.strength,
      startVideoSigma: prepared.plan.settings.strength, evaluations: prepared.plan.settings.evaluations,
      preserveAudio: true)
    var cursor = 0
    _ = try H3T2VARunner.run(prepared.request, initialRows: rows, refinement: controls,
      publicationAudio: prepared.originalAudio,
      onFrame: { index, rgb in
        guard cursor < prepared.plan.recovery.count else { return }
        if index == prepared.plan.recovery[cursor] {
          try onFrame(cursor, rgb); cursor += 1
          progress("motion_recovery", cursor, source.frames)
        }
      }, onAudio: onAudio, onLatents: onLatents, progress: progress)
    guard cursor == source.frames else { throw H3CheckpointError.invalid("Recovered motion frame count differs from the source.") }
    try source.identity.verify()
    return .init(videoFrames: prepared.request.geometry.frames,
      audioSamplesPerChannel: prepared.originalAudio.frames, audioSampleRate: 32_000)
  }

  static func expandedPixels(_ pixels: [UInt8], width: Int, height: Int,
    indices: [Int]) throws -> [UInt8] {
    let frameBytes = width * height * 3
    guard width > 0, height > 0, frameBytes > 0, pixels.count.isMultiple(of: frameBytes),
      !indices.isEmpty, indices.count <= 362,
      indices.count * frameBytes <= 1024 * 1024 * 1024,
      indices.allSatisfy({ $0 >= 0 && $0 < pixels.count / frameBytes }) else {
      throw H3CheckpointError.invalid("Motion expansion exceeds complete bounded source RGB frames.")
    }
    var result: [UInt8] = []; result.reserveCapacity(indices.count * frameBytes)
    for frame in indices {
      try Task.checkCancellation()
      result.append(contentsOf: pixels[(frame * frameBytes)..<((frame + 1) * frameBytes)])
    }
    return result
  }
  static func temporalJerk(_ latents: MLXArray, mean: [Float], standardDeviation: [Float]) throws -> [Float] {
    guard latents.ndim == 5, latents.shape[0] == 1, latents.shape[1] >= 4,
      latents.shape[4] == 24, mean.count == 24, standardDeviation.count == 24,
      mean.allSatisfy(\.isFinite), standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid full-canvas latent motion analysis.")
    }
    let normalized = (latents.asType(.float32) - MLXArray(mean).reshaped([1, 1, 1, 1, 24])) /
      MLXArray(standardDeviation).reshaped([1, 1, 1, 1, 24])
    guard all(isFinite(normalized)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Non-finite motion analysis latents.")
    }
    var difference = normalized
    for _ in 0..<3 {
      let count = difference.shape[1]
      difference = difference[0..<1, 1..<count, 0..., 0..., 0..<24] -
        difference[0..<1, 0..<(count - 1), 0..., 0..., 0..<24]
    }
    let jerk = abs(difference).mean(axes: [0, 2, 3, 4]); eval(jerk)
    try Task.checkCancellation()
    return jerk.asArray(Float.self)
  }
  static func cropAudioPadding(_ latent: MLXArray, samples: Int, expectedFrames: Int) throws -> MLXArray {
    guard latent.ndim == 3, latent.shape[0] == 2, latent.shape[2] == 32,
      latent.shape[1] == expectedFrames || (samples % 800 != 0 &&
        latent.shape[1] == expectedFrames + 1 && (samples + 799) / 800 == expectedFrames + 1) else {
      throw H3CheckpointError.invalid("Motion audio encoder produced the wrong synchronized token count.")
    }
    return latent[0..<2, 0..<expectedFrames, 0..<32]
  }
  private static func expandedRequest(base: H3T2VARequest, source: H3MotionFidelityMedia.Source,
    frames: Int, settings: H3MotionFidelitySettings) throws -> H3T2VARequest {
    try H3T2VARequest(prompt: base.prompt, width: source.width, height: source.height,
      durationSeconds: Double(frames) / 24, seed: settings.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages, tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      loRAAdapters: base.loRAAdapters, videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      samplingMethod: base.samplingMethod)
  }
  private static func validateBaselineMetadata(_ url: URL) throws {
    var directory: ObjCBool = false
    let objects: [[String: Any]]
    if FileManager.default.fileExists(atPath: url.path, isDirectory: &directory), directory.boolValue {
      objects = try ["config.json", "paged_manifest.json", "quant_config.json"].compactMap { name in
        let location = url.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: location.path) else { return nil }
        let handle = try FileHandle(forReadingFrom: location); defer { try? handle.close() }
        let bytes = try handle.read(upToCount: 1_048_577) ?? Data()
        guard bytes.count <= 1_048_576, let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
          throw H3CheckpointError.invalid("Motion baseline metadata must be bounded JSON.")
        }
        return object
      }
    } else { objects = [try SafeTensorFile(url: url).metadata.mapValues { $0 as Any }] }
    for object in objects {
      let provenance = ["source", "model", "source_model"].compactMap { object[$0] as? String }.joined(separator: " ").lowercased()
      let sampling = object["sampling"] as? [String: Any]
      if object["vsa_gate"] as? Bool == true || ["fasth3", "fastvideo", "vdn", "turbo"].contains(where: provenance.contains) ||
        (sampling?["transformer_evaluations"] as? Int ?? 100) < 15 {
        throw H3CheckpointError.invalid("Distilled/FastH3 checkpoints are not qualified for Motion Fidelity.")
      }
    }
  }
}
