import Foundation
import MLX

/// A text-to-AV H3 continuation reuses the reference-conditioned transformer
/// with a fixed, normalized audiovisual latent tail at target frame zero.
public enum H3ContinuationRunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let packedRows: Int
    public let evaluations: Int
    let textRows: Int
    let layout: H3ReferenceLayout
    let videoSchedule: H3Schedule
    let audioSchedule: H3Schedule
    let rowSchedule: H3ReferenceRowSchedule
  }

  public static func preflight(_ request: H3T2VARequest,
    contextFrames: Int) throws -> Admission {
    try Task.checkCancellation()
    guard request.funControl == nil else {
      throw H3CheckpointError.invalid("H3 continuation cannot combine a Fun control guide.")
    }
    let tokenizer = try H3QwenTokenizer(url: request.tokenizer)
    let qwen = try H3QwenRequest.text(request.prompt, tokenizer: tokenizer)
    let contextAudioFrames = Int((Double(contextFrames) / 24 * 40)
      .rounded(.toNearestOrEven))
    let contextVideoFrames = ((contextFrames - 5) / 17) * 5 + 2
    guard H3Continuation.allowedContextFrames.contains(contextFrames) else {
      throw H3CheckpointError.invalid("Unsupported H3 continuation overlap.")
    }
    let layout = try H3ReferenceLayout(geometry: request.geometry,
      textTags: qwen.tags, references: [.video(
        latentFrames: contextVideoFrames,
        latentHeight: request.geometry.height / 16,
        latentWidth: request.geometry.width / 16,
        audioLatents: contextAudioFrames,
        sourceLatentFrames: contextVideoFrames, targetFrame: 0)])
    let video = try H3Schedule(requestedSteps: request.requestedSteps, shift: 12)
    let audio = try H3Schedule(requestedSteps: request.requestedSteps, shift: 3)
    let rows = try H3ReferenceRowSchedule(layout: layout, video: video,
      audio: audio, visualConditionStrength: 1,
      audioConditionStrength: 1)
    _ = try H3QwenCheckpointLayout.inspect(root: request.qwenPages)
    _ = try H3CheckpointLayout(url: request.transformer)
    _ = try H3VideoVAELayout(url: request.videoVAE)
    _ = try H3AudioVAELayout(url: request.audioVAE)
    for adapter in request.loRAAdapters {
      _ = try H3LoRAFile(url: adapter.url, strength: adapter.strength,
        requestedSteps: request.requestedSteps)
    }
    return Admission(geometry: request.geometry, packedRows: layout.tags.count,
      evaluations: video.timesteps.count, textRows: qwen.tags.count,
      layout: layout, videoSchedule: video, audioSchedule: audio,
      rowSchedule: rows)
  }

  public static func run(_ request: H3T2VARequest,
    contextFrames: Int, context: H3Continuation.Rows,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    onLatents: (([Float], [Float]) throws -> Void)? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws
    -> H3AVOutputDecoder.Result {
    let admission = try preflight(request, contextFrames: contextFrames)
    guard context.video.count == admission.layout.conditionVideoIndices.count * 96,
      context.audio.count == admission.layout.conditionAudioIndices.count * 32,
      context.video.allSatisfy(\.isFinite), context.audio.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 continuation latent tail disagrees with admission.")
    }
    let raw = try autoreleasepool { () throws -> ([Float], [Float]) in
      let encoded = try H3QwenTextEncoder.encode(prompt: request.prompt,
        checkpointRoot: request.qwenPages, tokenizerURL: request.tokenizer) {
          completed, total in progress("text", completed, total)
        }
      guard encoded.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 continuation text rows changed after admission.")
      }
      let state = try H3ReferenceDiTState(checkpointURL: request.transformer,
        layout: admission.layout,
        textEmbeddings: encoded.hidden.reshaped([1, admission.textRows, 5120]),
        timestepTable: admission.rowSchedule.table,
        turboLoRAURL: request.turboLoRA,
        turboLoRAStrength: request.turboLoRAStrength,
        additionalLoRAs: request.additionalLoRAs) { completed, total in
          progress("transformer_prepare", completed, total)
        }
      defer { state.unload() }
      progress("text_weights_released", 1, 1)
      let noise = try H3Noise.make(seed: request.seed,
        videoLatentFrames: request.geometry.videoLatentFrames,
        latentHeight: request.geometry.height / 16,
        latentWidth: request.geometry.width / 16,
        audioLatentFrames: request.geometry.audioLatentFrames)
      let video = concatenated([
        MLXArray(context.video, [1, admission.layout.conditionVideoIndices.count, 96]),
        noise.video], axis: 1)
      let audio = concatenated([
        MLXArray(context.audio, [1, admission.layout.conditionAudioIndices.count, 32]),
        noise.audio], axis: 1)
      let sampled = try H3ReferenceSampler.run(predictor: state,
        videoSchedule: admission.videoSchedule,
        audioSchedule: admission.audioSchedule,
        rowSchedule: admission.rowSchedule,
        videoLatents: video, audioLatents: audio,
        progress: { done, total in progress("sampling", done, total) },
        blockProgress: { step, done, total in
          progress("sampling_block_\(step)", done, total)
        })
      let targetVideo = sampled.video[0..<1,
        admission.layout.conditionVideoIndices.count..<admission.layout.videoIndices.count,
        0..<96]
      let targetAudio = sampled.audio[0..<1,
        admission.layout.conditionAudioIndices.count..<admission.layout.audioIndices.count,
        0..<32]
      return (targetVideo.asType(.float32).asArray(Float.self),
        targetAudio.asType(.float32).asArray(Float.self))
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    try Task.checkCancellation()
    try onLatents?(raw.0, raw.1)
    return try H3AVOutputDecoder.decode(videoRows: raw.0,
      audioRows: raw.1, geometry: request.geometry,
      videoVAE: request.videoVAE, audioVAE: request.audioVAE,
      onFrame: onFrame, onAudio: onAudio, progress: progress)
  }
}
