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
    audioLatentFrames: Int) throws -> AnchoredRows {
    guard (1...4096).contains(conditionRows),
      (7...128).contains(videoLatentFrames),
      (2...256).contains(latentHeight), latentHeight.isMultiple(of: 2),
      (2...256).contains(latentWidth), latentWidth.isMultiple(of: 2),
      (1...601).contains(audioLatentFrames),
      videoLatentFrames * latentHeight * latentWidth <= 1_000_000 else {
      throw H3CheckpointError.invalid("H3 FL2VA seeded AV noise exceeds admitted geometry.")
    }
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
