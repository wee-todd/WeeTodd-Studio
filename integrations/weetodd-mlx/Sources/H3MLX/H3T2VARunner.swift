import Foundation
import MLX

/// Narrow text-to-audiovisual H3 vertical slice. It owns no movie files: a
/// worker must stage every decoded frame and waveform before atomic publish.
/// Other H3 tasks have separate conditioning contracts and cannot enter here.
public struct H3T2VARequest: Sendable {
  public let prompt: String
  public let geometry: H3Geometry
  public let durationSeconds: Double
  public let seed: UInt64
  public let requestedSteps: Int
  public let transformer: URL
  public let qwenPages: URL
  public let tokenizer: URL
  public let videoVAE: URL
  public let audioVAE: URL
  public let turboLoRA: URL?
  public let turboLoRAStrength: Float

  public init(prompt: String, width: Int, height: Int,
    durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1) throws {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= 65_536, (2...101).contains(requestedSteps),
      [transformer, qwenPages, tokenizer, videoVAE, audioVAE]
        .allSatisfy({ $0.isFileURL && $0.path.hasPrefix("/") }),
      turboLoRA == nil || (turboLoRA!.isFileURL &&
        turboLoRA!.path.hasPrefix("/") && turboLoRAStrength.isFinite &&
        (0...2).contains(turboLoRAStrength)) else {
      throw H3CheckpointError.invalid("Invalid H3 text-to-audiovisual request.")
    }
    let geometry = try H3Geometry(width: width, height: height,
      durationSeconds: durationSeconds)
    guard width * height <= 768 * 1344 else {
      throw H3CheckpointError.invalid("H3 video area exceeds the released canvas budget.")
    }
    guard try geometry.packedRows(textRows: 1,
      conditionVideoRows: 0, conditionAudioRows: 0) <= 40_000 else {
      throw H3CheckpointError.invalid("H3 packed rows exceed the current Swift engine limit.")
    }
    self.prompt = prompt
    self.geometry = geometry
    self.durationSeconds = durationSeconds
    self.seed = seed
    self.requestedSteps = requestedSteps
    self.transformer = transformer
    self.qwenPages = qwenPages
    self.tokenizer = tokenizer
    self.videoVAE = videoVAE
    self.audioVAE = audioVAE
    self.turboLoRA = turboLoRA
    self.turboLoRAStrength = turboLoRAStrength
  }
}

public enum H3T2VARunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let textRows: Int
    public let packedRows: Int
    public let evaluations: Int
    let layout: H3PackedLayout
    let videoSchedule: H3Schedule
    let audioSchedule: H3Schedule
    let rowSchedule: H3RowSchedule
  }

  public struct Result {
    public let videoFrames: Int
    public let audioSamplesPerChannel: Int
    public let audioSampleRate: Int
  }

  /// Inspect every installed component and all task geometry before the first
  /// weighted stage. A missing VAE cannot fail after a long transformer run.
  public static func preflight(_ request: H3T2VARequest) throws -> Admission {
    try Task.checkCancellation()
    let tokenizer = try H3QwenTokenizer(url: request.tokenizer)
    let qwen = try H3QwenRequest.text(request.prompt, tokenizer: tokenizer)
    let layout = try H3PackedLayout(geometry: request.geometry,
      textTags: qwen.tags, anchors: [])
    let video = try H3Schedule(requestedSteps: request.requestedSteps, shift: 12)
    let audio = try H3Schedule(requestedSteps: request.requestedSteps, shift: 3)
    let rows = try H3RowSchedule(layout: layout, video: video, audio: audio)
    guard rows.table.count <= 128, layout.tags.count <= 40_000 else {
      throw H3CheckpointError.invalid("H3 timestep table or packed rows exceed engine admission.")
    }
    _ = try H3QwenCheckpointLayout.inspect(root: request.qwenPages)
    _ = try H3CheckpointLayout(url: request.transformer)
    _ = try H3VideoVAELayout(url: request.videoVAE)
    _ = try H3AudioVAELayout(url: request.audioVAE)
    if let turboLoRA = request.turboLoRA {
      _ = try H3LoRAFile(url: turboLoRA,
        strength: request.turboLoRAStrength)
    }
    return Admission(geometry: request.geometry,
      textRows: qwen.tags.count, packedRows: layout.tags.count,
      evaluations: video.timesteps.count, layout: layout,
      videoSchedule: video, audioSchedule: audio, rowSchedule: rows)
  }

  /// `onFrame` receives exactly one RGB8 frame at a time. `onAudio` receives
  /// channel-major float32 stereo at 32 kHz after the video stage is released.
  public static func run(_ request: H3T2VARequest,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let admission = try preflight(request)
    let geometry = admission.geometry
    let rawRows = try autoreleasepool { () throws -> ([Float], [Float]) in
      try Task.checkCancellation()
      let conditioned = try H3QwenTextEncoder.encode(prompt: request.prompt,
        checkpointRoot: request.qwenPages,
        tokenizerURL: request.tokenizer) { completed, total in
        progress("text", completed, total)
      }
      guard conditioned.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 conditioner changed its admitted text rows.")
      }
      let state = try H3DiTState(checkpointURL: request.transformer,
        layout: admission.layout,
        textEmbeddings: conditioned.hidden
          .reshaped([1, admission.textRows, 5120]),
        timestepTable: admission.rowSchedule.table,
        turboLoRAURL: request.turboLoRA,
        turboLoRAStrength: request.turboLoRAStrength) { completed, total in
        progress("transformer_prepare", completed, total)
      }
      defer { state.unload() }
      progress("text_weights_released", 1, 1)
      let noise = try H3Noise.make(seed: request.seed,
        videoLatentFrames: geometry.videoLatentFrames,
        latentHeight: geometry.height / 16,
        latentWidth: geometry.width / 16,
        audioLatentFrames: geometry.audioLatentFrames)
      let sampled = try H3AVSampler.run(predictor: state,
        videoSchedule: admission.videoSchedule,
        audioSchedule: admission.audioSchedule,
        rowSchedule: admission.rowSchedule,
        videoLatents: noise.video, audioLatents: noise.audio,
        progress: { completed, total in progress("sampling", completed, total) },
        blockProgress: { step, completed, total in
          progress("sampling_block_\(step)", completed, total)
        })
      let video = sampled.video.asType(.float32).asArray(Float.self)
      let audio = sampled.audio.asType(.float32).asArray(Float.self)
      return (video, audio)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    try Task.checkCancellation()

    let result = try H3AVOutputDecoder.decode(videoRows: rawRows.0,
      audioRows: rawRows.1, geometry: geometry, videoVAE: request.videoVAE,
      audioVAE: request.audioVAE, onFrame: onFrame, onAudio: onAudio,
      progress: progress)
    return Result(videoFrames: result.videoFrames,
      audioSamplesPerChannel: result.audioSamplesPerChannel,
      audioSampleRate: result.audioSampleRate)
  }
}
