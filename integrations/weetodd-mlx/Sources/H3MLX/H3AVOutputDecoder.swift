import Foundation
import MLX

/// Shared output stage for text-only and reference-conditioned H3. Host rows
/// keep the denoiser unloadable before either VAE is admitted into memory.
public enum H3AVOutputDecoder {
  public struct Result {
    public let videoFrames: Int
    public let audioSamplesPerChannel: Int
    public let audioSampleRate: Int
    public let backendReport: H3BackendReport?
    public let videoDecodePrecision: H3VideoDecodePrecision?

    public init(videoFrames: Int, audioSamplesPerChannel: Int, audioSampleRate: Int,
      backendReport: H3BackendReport? = nil, videoDecodePrecision: H3VideoDecodePrecision? = nil) {
      self.videoFrames = videoFrames
      self.audioSamplesPerChannel = audioSamplesPerChannel
      self.audioSampleRate = audioSampleRate
      self.backendReport = backendReport
      self.videoDecodePrecision = videoDecodePrecision
    }
  }

  public static func decode(videoRows: [Float], audioRows: [Float],
    geometry: H3Geometry, videoVAE: URL, audioVAE: URL,
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil,
    videoDecodePrecision: H3VideoDecodePrecision = .float32,
    publicationAudio: H3AudioReference? = nil,
    backendReport: H3BackendReport? = nil,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    try videoDecodePrecision.validate(memoryMode: videoDecodeMemoryMode)
    guard videoRows.count == geometry.videoRows * 96,
      audioRows.count == geometry.audioRows * 32 else {
      throw H3CheckpointError.invalid("H3 decoded AV row lengths disagree with geometry.")
    }
    if let publicationAudio {
      guard publicationAudio.frames > 0, publicationAudio.frames <= 480_000,
        publicationAudio.samples.count == publicationAudio.frames * 2,
        publicationAudio.samples.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("Invalid original soundtrack publication interval.")
      }
    }
    try Task.checkCancellation()
    let videoLayout = try H3VideoVAELayout(url: videoVAE)
    let video = try H3LatentCodec.videoDecoderInput(
      rows: MLXArray(videoRows, [1, geometry.videoRows, 96]),
      latentFrames: geometry.videoLatentFrames,
      latentHeight: geometry.height / 16,
      latentWidth: geometry.width / 16,
      mean: videoLayout.latentsMean,
      standardDeviation: videoLayout.latentsStandardDeviation)
    let frameBytes = geometry.width * geometry.height * 3
    var written = 0
    var appliedPrecision: H3VideoDecodePrecision?
    try H3VideoVAEDecoder.decodeChunks(checkpointURL: videoVAE,
      latent: video, retainWeights: true, memoryMode: videoDecodeMemoryMode,
      precision: videoDecodePrecision, onSessionClosed: { stats in
        appliedPrecision = H3VideoDecodePrecision(rawValue: stats.computePrecision)
      }) { chunk in
      try Task.checkCancellation()
      let bytes = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
      guard bytes.count == chunk.shape[1] * frameBytes else {
        throw H3CheckpointError.invalid("H3 video decoder returned incomplete RGB frames.")
      }
      for offset in 0..<chunk.shape[1] {
        let lower = offset * frameBytes
        try onFrame(written, Data(bytes[lower..<(lower + frameBytes)]))
        written += 1
        progress("video_decode", written, geometry.frames)
      }
    }
    guard appliedPrecision == videoDecodePrecision, written == geometry.frames else {
      throw H3CheckpointError.invalid("H3 video decoder frame count disagrees with the AV clock.")
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("video_weights_released", 1, 1)
    try Task.checkCancellation()

    if let publicationAudio {
      try onAudio(publicationAudio.samples, 32_000)
      progress("source_audio_preserved", 1, 1)
      return Result(videoFrames: written, audioSamplesPerChannel: publicationAudio.frames,
        audioSampleRate: 32_000, backendReport: backendReport, videoDecodePrecision: appliedPrecision)
    }
    let audioLayout = try H3AudioVAELayout(url: audioVAE)
    let audio = try H3LatentCodec.audioDecoderInput(
      rows: MLXArray(audioRows, [1, geometry.audioRows, 32]),
      latentFrames: geometry.audioLatentFrames,
      mean: audioLayout.latentsMean,
      standardDeviation: audioLayout.latentsStandardDeviation)
    let waveform = try H3AudioVAEDecoder.decode(
      checkpointURL: audioVAE, latent: audio) { completed, total in
      progress("audio_decode", completed, total)
    }
    let samples = waveform.asArray(Float.self)
    guard samples.count == 2 * geometry.audioLatentFrames * 800,
      samples.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 audio decoder returned invalid stereo timing.")
    }
    try onAudio(samples, 32_000)
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("audio_weights_released", 1, 1)
    return Result(videoFrames: written,
      audioSamplesPerChannel: samples.count / 2, audioSampleRate: 32_000, backendReport: backendReport, videoDecodePrecision: appliedPrecision)
  }
}
