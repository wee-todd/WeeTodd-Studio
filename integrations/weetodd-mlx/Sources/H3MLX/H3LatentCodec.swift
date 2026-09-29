import Foundation
import MLX

/// Convert the sampler's packed H3 rows into the installed VAE layouts.
/// The video decoder uses channels-last float32; stereo audio remains two
/// channel-major batch items, one mono waveform per batch item.
public enum H3LatentCodec {
  /// Normalize deterministic VAE posterior means and pack them in the same
  /// channel-major 2x2 patch order consumed by the H3 video projection.
  public static func videoEncoderRows(latents: MLXArray,
    mean: [Float], standardDeviation: [Float]) throws -> MLXArray {
    guard latents.ndim == 5, latents.shape[0] == 1,
      latents.shape[1] > 0, latents.shape[2] > 0, latents.shape[3] > 0,
      latents.shape[2].isMultiple(of: 2), latents.shape[3].isMultiple(of: 2),
      latents.shape[4] == 24, latents.dtype.isFloatingPoint,
      mean.count == 24, standardDeviation.count == 24,
      mean.allSatisfy(\.isFinite),
      standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 reference video encoder latents.")
    }
    let frames = latents.shape[1]
    let height = latents.shape[2]
    let width = latents.shape[3]
    let offset = MLXArray(mean).reshaped([1, 1, 1, 1, 24])
    let scale = MLXArray(standardDeviation).reshaped([1, 1, 1, 1, 24])
    let result = ((latents.asType(.float32) - offset) / scale)
      .reshaped([1, frames, height / 2, 2, width / 2, 2, 24])
      .transposed(0, 1, 2, 4, 6, 3, 5)
      .reshaped([1, frames * height * width / 4, 96])
    eval(result)
    return result
  }

  /// H3 stores stereo reference-audio rows channel-major, then by latent time.
  public static func audioEncoderRows(latents: MLXArray,
    mean: [Float], standardDeviation: [Float]) throws -> MLXArray {
    guard latents.ndim == 3, latents.shape[0] == 2,
      latents.shape[1] > 0, latents.shape[2] == 32,
      latents.dtype.isFloatingPoint,
      mean.count == 32, standardDeviation.count == 32,
      mean.allSatisfy(\.isFinite),
      standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 reference audio encoder latents.")
    }
    let offset = MLXArray(mean).reshaped([1, 1, 32])
    let scale = MLXArray(standardDeviation).reshaped([1, 1, 32])
    let result = ((latents.asType(.float32) - offset) / scale)
      .reshaped([1, latents.shape[0] * latents.shape[1], 32])
    eval(result)
    return result
  }

  public static func videoDecoderInput(rows: MLXArray,
    latentFrames: Int, latentHeight: Int, latentWidth: Int,
    mean: [Float], standardDeviation: [Float]) throws -> MLXArray {
    guard latentFrames > 0, latentHeight > 0, latentWidth > 0,
      latentHeight.isMultiple(of: 2), latentWidth.isMultiple(of: 2),
      rows.shape == [1, latentFrames * latentHeight * latentWidth / 4, 96],
      rows.dtype.isFloatingPoint, mean.count == 24,
      standardDeviation.count == 24,
      mean.allSatisfy(\.isFinite),
      standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 packed video decoder inputs.")
    }
    let patches = rows.asType(.float32)
      .reshaped([1, latentFrames, latentHeight / 2, latentWidth / 2, 24, 2, 2])
      .transposed(0, 1, 2, 5, 3, 6, 4)
      .reshaped([1, latentFrames, latentHeight, latentWidth, 24])
    let offset = MLXArray(mean).reshaped([1, 1, 1, 1, 24])
    let scale = MLXArray(standardDeviation).reshaped([1, 1, 1, 1, 24])
    let result = patches * scale + offset
    eval(result)
    return result
  }

  public static func audioDecoderInput(rows: MLXArray, latentFrames: Int,
    mean: [Float], standardDeviation: [Float]) throws -> MLXArray {
    guard latentFrames > 0, rows.shape == [1, 2 * latentFrames, 32],
      rows.dtype.isFloatingPoint, mean.count == 32,
      standardDeviation.count == 32,
      mean.allSatisfy(\.isFinite),
      standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid H3 packed audio decoder inputs.")
    }
    let latents = rows.asType(.float32).reshaped([2, latentFrames, 32])
    let offset = MLXArray(mean).reshaped([1, 1, 32])
    let scale = MLXArray(standardDeviation).reshaped([1, 1, 32])
    let result = latents * scale + offset
    eval(result)
    return result
  }

  public static func videoPixelsRGB8(_ normalized: MLXArray) throws -> MLXArray {
    guard normalized.ndim == 5, normalized.shape[0] == 1,
      normalized.shape[4] == 3, normalized.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 video VAE pixel tensor.")
    }
    let mean = MLXArray([Float(0.485), 0.456, 0.406])
      .reshaped([1, 1, 1, 1, 3])
    let std = MLXArray([Float(0.229), 0.224, 0.225])
      .reshaped([1, 1, 1, 1, 3])
    let result = floor(clip(normalized.asType(.float32) * std + mean,
      min: 0, max: 1) * Float(255) + Float(0.5)).asType(.uint8)
    eval(result)
    return result
  }
}
