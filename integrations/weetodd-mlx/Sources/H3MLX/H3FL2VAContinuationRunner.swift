import Foundation
import MLX

/// FL2VA continuation reuses the reference sampler with a clean AV prefix and
/// separately noised timed images. The existing ordinary FL sampler is unchanged.
public enum H3FL2VAContinuationRunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let packedRows: Int
    public let evaluations: Int
    let ordinary: H3FL2VARunner.Admission
    let layout: H3ReferenceLayout
    let contextVideoRows: Int
    let contextAudioRows: Int
    let rowSchedule: H3ReferenceRowSchedule
  }

  public static func preflight(_ request: H3FL2VARequest,
    contextFrames: Int) throws -> Admission {
    try Task.checkCancellation()
    let ordinary = try H3FL2VARunner.preflight(request)
    let layout = try H3ReferenceLayout(geometry: request.base.geometry,
      textTags: Array(ordinary.layout.tags.prefix(ordinary.textRows)),
      anchors: request.anchors, contextFrames: contextFrames)
    let contextVideoRows = layout.conditionVideoIndices.count - ordinary.layout.conditionVideoRows
    let rows = try H3ReferenceRowSchedule(layout: layout,
      video: ordinary.videoSchedule, audio: ordinary.audioSchedule,
      visualConditionStrength: request.referenceNoise?.visual ?? 0.999,
      cleanVideoPrefixRows: contextVideoRows)
    return Admission(geometry: ordinary.geometry, packedRows: layout.tags.count,
      evaluations: ordinary.evaluations, ordinary: ordinary, layout: layout,
      contextVideoRows: contextVideoRows,
      contextAudioRows: layout.conditionAudioIndices.count, rowSchedule: rows)
  }

  public static func run(_ request: H3FL2VARequest,
    contextFrames: Int, context: H3Continuation.Rows,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    onLatents: (([Float], [Float]) throws -> Void)? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> H3AVOutputDecoder.Result {
    let admission = try preflight(request, contextFrames: contextFrames)
    let base = request.base
    guard context.video.count == admission.contextVideoRows * 96,
      context.audio.count == admission.contextAudioRows * 32,
      context.video.allSatisfy(\.isFinite), context.audio.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("FL2VA continuation latent tail disagrees with admission.")
    }
    let text = try H3FL2VARunner.encodeText(request, admission: admission.ordinary, progress: progress)
    let keyframes = try H3FL2VARunner.encodeKeyframes(request, admission: admission.ordinary, progress: progress)
    var backendReport: H3BackendReport?
    let raw = try autoreleasepool { () throws -> ([Float], [Float]) in
      let state = try H3ReferenceDiTState(checkpointURL: base.transformer,
        layout: admission.layout,
        textEmbeddings: MLXArray(text, [1, admission.ordinary.textRows, 5120]).asType(.bfloat16),
        timestepTable: admission.rowSchedule.table,
        turboLoRAURL: base.turboLoRA, turboLoRAStrength: base.turboLoRAStrength,
        additionalLoRAs: base.additionalLoRAs, loRAAdapters: base.loRAAdapters) { done, total in
          progress("transformer_prepare", done, total)
        }
      defer { state.unload() }
      let noise = try H3Noise.makeWithCondition(seed: base.seed,
        conditionRows: admission.ordinary.layout.conditionVideoRows,
        videoLatentFrames: base.geometry.videoLatentFrames,
        latentHeight: base.geometry.height / 16, latentWidth: base.geometry.width / 16,
        audioLatentFrames: base.geometry.audioLatentFrames)
      let clean = MLXArray(keyframes, [1, admission.ordinary.layout.conditionVideoRows, 96])
      let strength = request.referenceNoise?.visual ?? Float(0.999)
      let noiseStrength = request.referenceNoise.map { Float(1) - $0.visual } ?? Float(0.001)
      let anchored = strength * clean + noiseStrength * noise.condition
      let video = concatenated([
        MLXArray(context.video, [1, admission.contextVideoRows, 96]), anchored, noise.video], axis: 1)
      let audio = concatenated([
        MLXArray(context.audio, [1, admission.contextAudioRows, 32]), noise.audio], axis: 1)
      let sampled = try H3ReferenceSampler.run(predictor: state,
        videoSchedule: admission.ordinary.videoSchedule,
        audioSchedule: admission.ordinary.audioSchedule,
        rowSchedule: admission.rowSchedule, videoLatents: video, audioLatents: audio,
        samplingMethod: base.samplingMethod,
        progress: { done, total in progress("sampling", done, total) },
        blockProgress: { step, done, total in progress("sampling_block_\(step)", done, total) })
      let videoRows = sampled.video[0..<1,
        admission.layout.conditionVideoIndices.count..<sampled.video.shape[1], 0..<96]
        .asType(.float32).asArray(Float.self)
      let audioRows = sampled.audio[0..<1,
        admission.contextAudioRows..<sampled.audio.shape[1], 0..<32]
        .asType(.float32).asArray(Float.self)
      guard videoRows.allSatisfy(\.isFinite), audioRows.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("FL2VA continuation sampled nonfinite AV latents.")
      }
      backendReport = state.backendReport
      return (videoRows, audioRows)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    try Task.checkCancellation()
    try onLatents?(raw.0, raw.1)
    return try H3AVOutputDecoder.decode(videoRows: raw.0, audioRows: raw.1,
      geometry: base.geometry, videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      videoDecodeMemoryMode: base.videoDecodeMemoryMode,
      videoDecodePrecision: base.videoDecodePrecision,
      backendReport: backendReport,
      onFrame: onFrame, onAudio: onAudio, progress: progress)
  }
}
