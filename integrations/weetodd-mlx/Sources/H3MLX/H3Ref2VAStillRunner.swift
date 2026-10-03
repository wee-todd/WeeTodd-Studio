import Foundation
import MLX

/// An in-memory, ordered visual-reference request. The worker is responsible
/// for bounded decode and source-file identity before constructing RGB bytes.
public struct H3Ref2VAStillRequest: Sendable {
  public let prompt: String
  public let references: [H3Ref2VAReference]
  public let geometry: H3Geometry
  public let seed: UInt64
  public let requestedSteps: Int
  public let transformer: URL
  public let qwenPages: URL
  public let qwenVision: URL
  public let tokenizer: URL
  public let videoDecodeMemoryMode: H3VideoDecodeMemoryMode?
  public let videoVAE: URL
  public let audioVAE: URL
  public let turboLoRA: URL?
  public let turboLoRAStrength: Float
  public let additionalLoRAs: [H3LoRAAdapter]
  public let loRAAdapters: [H3LoRAAdapter]

  public init(prompt: String, references: [H3StillReference], width: Int,
    height: Int, durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, qwenVision: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil) throws {
    try self.init(prompt: prompt, mediaReferences: references.map { .image($0) },
      width: width, height: height, durationSeconds: durationSeconds,
      seed: seed, requestedSteps: requestedSteps, transformer: transformer,
      qwenPages: qwenPages, qwenVision: qwenVision, tokenizer: tokenizer,
      videoVAE: videoVAE, audioVAE: audioVAE, turboLoRA: turboLoRA,
      turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs,
      videoDecodeMemoryMode: videoDecodeMemoryMode)
  }

  public init(prompt: String, mediaReferences: [H3Ref2VAReference], width: Int,
    height: Int, durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, qwenVision: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil) throws {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= 65_536, (2...101).contains(requestedSteps),
      [transformer, qwenPages, qwenVision, tokenizer, videoVAE, audioVAE]
        .allSatisfy({ $0.isFileURL && $0.path.hasPrefix("/") }),
      turboLoRA == nil || (turboLoRA!.isFileURL &&
        turboLoRA!.path.hasPrefix("/") && turboLoRAStrength.isFinite &&
        (0...2).contains(turboLoRAStrength)),
      (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count <= 4,
      Set(([turboLoRA].compactMap { $0 } + additionalLoRAs.map(\.url))
        .map { $0.standardizedFileURL.path }).count ==
        (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count else {
      throw H3CheckpointError.invalid("Invalid H3 visual-reference request.")
    }
    try H3VideoReferencePreparation.validate(mediaReferences)
    let geometry = try H3Geometry(width: width, height: height,
      durationSeconds: durationSeconds)
    let conditionRows = mediaReferences.reduce(0) { rows, reference in
      switch reference {
      case .image(let image), .timedImage(let image, _):
        rows + image.width * image.height / 1024
      case .video(let video):
        rows + ((video.frameCount - 5) / 17 * 5 + 2) *
          video.width * video.height / 1024
      case .audio, .timedAudio: rows
      }
    }
    let conditionAudioRows = mediaReferences.reduce(0) { rows, reference in
      switch reference {
      case .audio(let audio), .timedAudio(let audio, _):
        return rows + 2 * ((audio.frames + 799) / 800)
      default: break
      }
      if case .video(let video) = reference, let audio = video.audio {
        return rows + 2 * ((audio.frames + 799) / 800)
      }
      return rows
    }
    guard width * height <= H3Geometry.maximumCanvasPixels,
      try geometry.packedRows(textRows: 1,
        conditionVideoRows: conditionRows,
        conditionAudioRows: conditionAudioRows) <= 40_000 else {
      throw H3CheckpointError.invalid("H3 reference canvas exceeds packed-row admission.")
    }
    self.prompt = prompt
    self.references = mediaReferences
    self.geometry = geometry
    self.seed = seed
    self.requestedSteps = requestedSteps
    self.transformer = transformer
    self.qwenPages = qwenPages
    self.qwenVision = qwenVision
    self.tokenizer = tokenizer
    self.videoDecodeMemoryMode = videoDecodeMemoryMode
    self.videoVAE = videoVAE
    self.audioVAE = audioVAE
    self.turboLoRA = turboLoRA
    self.turboLoRAStrength = turboLoRAStrength
    self.additionalLoRAs = additionalLoRAs
    let primary = try turboLoRA.map {
      [try H3LoRAAdapter(url: $0, strength: turboLoRAStrength)]
    } ?? []
    self.loRAAdapters = primary + additionalLoRAs
  }
}

public enum H3Ref2VAStillRunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let textRows: Int
    public let packedRows: Int
    public let evaluations: Int
    let layout: H3ReferenceLayout
    let videoSchedule: H3Schedule
    let audioSchedule: H3Schedule
    let rowSchedule: H3ReferenceRowSchedule
  }

  public typealias Result = H3AVOutputDecoder.Result

  public static func preflight(_ request: H3Ref2VAStillRequest) throws -> Admission {
    try Task.checkCancellation()
    let prepared = try H3VideoReferencePreparation.prepare(prompt: request.prompt,
      geometry: request.geometry, references: request.references,
      tokenizerURL: request.tokenizer)
    let video = try H3Schedule(requestedSteps: request.requestedSteps, shift: 12)
    let audio = try H3Schedule(requestedSteps: request.requestedSteps, shift: 3)
    let rows = try H3ReferenceRowSchedule(layout: prepared.layout,
      video: video, audio: audio)
    guard prepared.layout.tags.count <= 40_000 else {
      throw H3CheckpointError.invalid("H3 reference packed rows exceed engine admission.")
    }
    _ = try H3QwenCheckpointLayout.inspect(root: request.qwenPages)
    if !prepared.qwenRequest.visualRanges.isEmpty {
      let vision = try H3QwenCheckpointLayout.inspect(root: request.qwenVision)
      guard vision.visionFile != nil else {
        throw H3CheckpointError.invalid("The H3 visual conditioner needs a vision tower.")
      }
    }
    _ = try H3CheckpointLayout(url: request.transformer)
    _ = try H3VideoVAELayout(url: request.videoVAE)
    _ = try H3AudioVAELayout(url: request.audioVAE)
    if !prepared.layout.conditionAudioIndices.isEmpty {
      try H3AudioVAEEncoder.inspect(checkpointURL: request.audioVAE)
    }
    for adapter in request.loRAAdapters {
      _ = try H3LoRAFile(url: adapter.url,
        strength: adapter.strength, requestedSteps: request.requestedSteps)
    }
    return Admission(geometry: request.geometry,
      textRows: prepared.qwenRequest.tags.count,
      packedRows: prepared.layout.tags.count,
      evaluations: video.timesteps.count, layout: prepared.layout,
      videoSchedule: video, audioSchedule: audio, rowSchedule: rows)
  }

  public static func run(_ request: H3Ref2VAStillRequest,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let admission = try preflight(request)
    let geometry = admission.geometry
    let textRows = try autoreleasepool { () throws -> [Float] in
      let prepared = try H3VideoReferencePreparation.prepare(prompt: request.prompt,
        geometry: geometry, references: request.references,
        tokenizerURL: request.tokenizer)
      let encoded = try H3QwenTextEncoder.encodeReferences(
        prompt: request.prompt, pixels: prepared.qwenPixels,
        references: prepared.qwenReferences,
        checkpointRoot: request.qwenPages,
        visionCheckpointURL: request.qwenVision,
        tokenizerURL: request.tokenizer) { completed, total in
          progress("text", completed, total)
        }
      guard encoded.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 reference conditioner changed admitted text rows.")
      }
      return encoded.hidden.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("text_weights_released", 1, 1)
    try Task.checkCancellation()

    let conditionVideoRows = try autoreleasepool { () throws -> [Float] in
      guard !admission.layout.conditionVideoIndices.isEmpty else { return [] }
      let rows = try H3VideoReferencePreparation.encodeVideoRows(
        references: request.references, layout: admission.layout,
        videoVAEURL: request.videoVAE)
      return rows.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("reference_video_weights_released", 1, 1)
    try Task.checkCancellation()

    let conditionAudioRows: [Float] = try autoreleasepool {
      guard admission.layout.conditionAudioIndices.count > 0 else { return [] }
      let rows = try H3VideoReferencePreparation.encodeAudioRows(
        references: request.references, layout: admission.layout,
        audioVAEURL: request.audioVAE)
      return rows.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    if !conditionAudioRows.isEmpty {
      progress("reference_audio_weights_released", 1, 1)
    }
    try Task.checkCancellation()

    let rawRows = try autoreleasepool { () throws -> ([Float], [Float]) in
      let state = try H3ReferenceDiTState(checkpointURL: request.transformer,
        layout: admission.layout,
        textEmbeddings: MLXArray(textRows,
          [1, admission.textRows, 5120]).asType(.bfloat16),
        timestepTable: admission.rowSchedule.table,
        turboLoRAURL: request.turboLoRA,
        turboLoRAStrength: request.turboLoRAStrength,
        additionalLoRAs: request.additionalLoRAs) { completed, total in
          progress("transformer_prepare", completed, total)
        }
      defer { state.unload() }
      let initial = try H3Noise.makeReference(seed: request.seed,
        conditionVideo: conditionVideoRows, conditionAudio: conditionAudioRows,
        videoLatentFrames: geometry.videoLatentFrames,
        latentHeight: geometry.height / 16, latentWidth: geometry.width / 16,
        audioLatentFrames: geometry.audioLatentFrames)
      let sampled = try H3ReferenceSampler.run(predictor: state,
        videoSchedule: admission.videoSchedule,
        audioSchedule: admission.audioSchedule,
        rowSchedule: admission.rowSchedule,
        videoLatents: initial.video, audioLatents: initial.audio,
        progress: { completed, total in progress("sampling", completed, total) },
        blockProgress: { step, completed, total in
          progress("sampling_block_\(step)", completed, total)
        })
      let targetVideo = sampled.video[0..<1,
        admission.layout.conditionVideoIndices.count..<admission.layout.videoIndices.count, 0..<96]
      let targetAudio = sampled.audio[0..<1,
        admission.layout.conditionAudioIndices.count..<admission.layout.audioIndices.count,
        0..<32]
      return (targetVideo.asType(.float32).asArray(Float.self),
        targetAudio.asType(.float32).asArray(Float.self))
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    return try H3AVOutputDecoder.decode(videoRows: rawRows.0,
      audioRows: rawRows.1, geometry: geometry, videoVAE: request.videoVAE,
      audioVAE: request.audioVAE, videoDecodeMemoryMode: request.videoDecodeMemoryMode,
      onFrame: onFrame, onAudio: onAudio,
      progress: progress)
  }
}
