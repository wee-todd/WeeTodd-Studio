import Foundation
import MLX
import MLXRandom

/// H3's seeded video-first, stereo-audio-second Gaussian draw. Sampling and
/// saved-take replay use the same MLX PRNG and channel-major row ordering.
public enum H3Noise {
  public struct Rows {
    public let video: MLXArray
    public let audio: MLXArray
  }

  public struct AnchoredRows {
    public let condition: MLXArray
    public let video: MLXArray
    public let audio: MLXArray
  }

  /// FL2VA draws anchor noise before target video and audio noise from the
  /// request seed. The fixed 0.999 condition strength is the released model's
  /// clean-frame augmentation, independent of the VAE posterior seed 42.
  public static func makeWithCondition(seed: UInt64, conditionRows: Int,
    videoLatentFrames: Int, latentHeight: Int, latentWidth: Int,
    audioLatentFrames: Int, canvasAdmission: H3CanvasAdmission = .ordinary) throws -> AnchoredRows {
    guard (1...(canvasAdmission == .ordinary ? 4096 : canvasAdmission.maximumPackedRows)).contains(conditionRows),
      (7...128).contains(videoLatentFrames),
      (2...256).contains(latentHeight), latentHeight.isMultiple(of: 2),
      (2...256).contains(latentWidth), latentWidth.isMultiple(of: 2),
      (1...601).contains(audioLatentFrames),
      videoLatentFrames * latentHeight * latentWidth <= 1_000_000 else {
      throw H3CheckpointError.invalid("H3 FL2VA seeded AV noise exceeds admitted geometry.")
    }
    return drawAnchored(seed: seed, conditionRows: conditionRows,
      videoLatentFrames: videoLatentFrames, latentHeight: latentHeight,
      latentWidth: latentWidth, audioLatentFrames: audioLatentFrames)
  }

  private static func drawAnchored(seed: UInt64, conditionRows: Int,
    videoLatentFrames: Int, latentHeight: Int, latentWidth: Int,
    audioLatentFrames: Int) -> AnchoredRows {
    MLXRandom.seed(seed)
    let condition = MLXRandom.normal([1, conditionRows, 96]).asType(.float32)
    let video = MLXRandom.normal([1, 24, videoLatentFrames,
      latentHeight, latentWidth]).asType(.float32)
    let videoRows = video.reshaped([1, 24, videoLatentFrames,
      latentHeight / 2, 2, latentWidth / 2, 2])
      .transposed(0, 2, 3, 5, 1, 4, 6)
      .reshaped([1, videoLatentFrames * latentHeight * latentWidth / 4, 96])
    let audio = MLXRandom.normal([2, 32, audioLatentFrames]).asType(.float32)
      .transposed(0, 2, 1).reshaped([1, 2 * audioLatentFrames, 32])
    eval(condition, videoRows, audio)
    return AnchoredRows(condition: condition, video: videoRows, audio: audio)
  }

  /// Ref2VA initialization, kept separate from clean latent-tail continuation.
  static func makeReference(seed: UInt64, conditionVideo: [Float],
    conditionAudio: [Float], videoLatentFrames: Int, latentHeight: Int,
    latentWidth: Int, audioLatentFrames: Int,
    referenceNoise: H3ReferenceNoiseControls? = nil) throws -> Rows {
    guard conditionVideo.count.isMultiple(of: 96),
      conditionAudio.count.isMultiple(of: 32),
      conditionVideo.count / 96 + conditionAudio.count / 32 <= 40_000,
      conditionVideo.allSatisfy(\.isFinite),
      conditionAudio.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("Invalid Ref2VA encoded condition rows.")
    }
    guard (7...128).contains(videoLatentFrames),
      (2...256).contains(latentHeight), latentHeight.isMultiple(of: 2),
      (2...256).contains(latentWidth), latentWidth.isMultiple(of: 2),
      (1...601).contains(audioLatentFrames),
      videoLatentFrames * latentHeight * latentWidth <= 1_000_000 else {
      throw H3CheckpointError.invalid("H3 Ref2VA seeded AV noise exceeds admitted geometry.")
    }
    let noise: Rows
    let video: MLXArray
    if conditionVideo.isEmpty {
      // A2V/audio-only conditioning consumes no visual draw and keeps the
      // ordinary target-only seed stream, with clean reference audio.
      noise = try make(seed: seed, videoLatentFrames: videoLatentFrames,
        latentHeight: latentHeight, latentWidth: latentWidth,
        audioLatentFrames: audioLatentFrames)
      video = noise.video
    } else {
      // Preserve the owned Python MLX request stream: one packed visual
      // condition draw, target video, then target audio. Upstream PyTorch
      // runtimes have their own RNG policies; this does not claim seed parity.
      let anchored = drawAnchored(seed: seed,
        conditionRows: conditionVideo.count / 96,
        videoLatentFrames: videoLatentFrames, latentHeight: latentHeight,
        latentWidth: latentWidth, audioLatentFrames: audioLatentFrames)
      noise = Rows(video: anchored.video, audio: anchored.audio)
      let strength = referenceNoise?.visual ?? Float(0.999)
      let clean = MLXArray(conditionVideo, [1, conditionVideo.count / 96, 96])
      // Use the float32 complement, matching scheduler.scale_noise; literal
      // .001 rounds differently from 1 - Float(.999).
      let augmented = strength * clean + (Float(1) - strength) * anchored.condition
      video = concatenated([augmented, noise.video], axis: 1)
    }
    let audio: MLXArray
    if conditionAudio.isEmpty { audio = noise.audio }
    else {
      var clean = MLXArray(conditionAudio, [1, conditionAudio.count / 32, 32])
      let strength = referenceNoise?.audio ?? 1
      if strength < 1 {
        // Independent owned-Python audio key: target PRNG draws stay unchanged.
        let keyed = MLXRandom.normal(clean.shape, key: MLXRandom.key(seed + 1)).asType(.float32)
        clean = strength * clean + (Float(1) - strength) * keyed
      }
      audio = concatenated([clean, noise.audio], axis: 1)
    }
    return Rows(video: video, audio: audio)
  }

  public static func make(seed: UInt64, videoLatentFrames: Int,
    latentHeight: Int, latentWidth: Int,
    audioLatentFrames: Int) throws -> Rows {
    guard (7...128).contains(videoLatentFrames),
      (2...256).contains(latentHeight), latentHeight.isMultiple(of: 2),
      (2...256).contains(latentWidth), latentWidth.isMultiple(of: 2),
      (1...601).contains(audioLatentFrames),
      videoLatentFrames * latentHeight * latentWidth <= 1_000_000 else {
      throw H3CheckpointError.invalid("H3 seeded AV noise exceeds admitted geometry.")
    }
    MLXRandom.seed(seed)
    let video = MLXRandom.normal([1, 24, videoLatentFrames,
      latentHeight, latentWidth]).asType(.float32)
    let videoRows = video.reshaped([1, 24, videoLatentFrames,
      latentHeight / 2, 2, latentWidth / 2, 2])
      .transposed(0, 2, 3, 5, 1, 4, 6)
      .reshaped([1, videoLatentFrames * latentHeight * latentWidth / 4, 96])
    let audio = MLXRandom.normal([2, 32, audioLatentFrames]).asType(.float32)
      .transposed(0, 2, 1).reshaped([1, 2 * audioLatentFrames, 32])
    eval(videoRows, audio)
    return Rows(video: videoRows, audio: audio)
  }
}
