import Foundation
import MLX
import MLXNN

/// Unweighted H3 causal-CNN operations used by the video VAE encoder. The
/// released checkpoint stores conv kernels in MLX's ODHWI layout.
public enum H3VideoVAEEncoderOps {
  public static func temporallyIsolatedGroupNorm(_ input: MLXArray,
    weight: MLXArray, bias: MLXArray) throws -> MLXArray {
    guard input.ndim == 5, input.shape[0] > 0, input.shape[1] > 0,
      input.shape[2] > 0, input.shape[3] > 0,
      input.shape[4] >= 32, input.shape[4].isMultiple(of: 32),
      input.dtype.isFloatingPoint,
      weight.shape == [input.shape[4]], bias.shape == weight.shape,
      weight.dtype.isFloatingPoint, bias.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 VAE frame-isolated group normalization.")
    }
    let batch = input.shape[0]
    let frames = input.shape[1]
    let folded = input.reshaped([batch * frames, input.shape[2], input.shape[3], input.shape[4]])
    let norm = GroupNorm(groupCount: 32, dimensions: input.shape[4],
      eps: 1e-6, affine: false, pytorchCompatible: true)
    let result = (norm(folded) * weight + bias).reshaped(input.shape)
    eval(result)
    return result
  }

  public static func reflectSpatial(_ input: MLXArray, axis: Int,
    before: Int, after: Int) throws -> MLXArray {
    guard input.ndim == 5, (axis == 2 || axis == 3),
      before >= 0, after >= 0,
      before < input.shape[axis], after < input.shape[axis] else {
      throw H3CheckpointError.invalid("Invalid H3 VAE spatial reflection geometry.")
    }
    if before == 0 && after == 0 { return input }
    let count = input.shape[axis]
    let indices = Array(stride(from: before, through: 1, by: -1))
      + Array(0..<count)
      + Array(stride(from: count - 2, through: count - 1 - after, by: -1))
    let result = take(input, MLXArray(indices.map(Int32.init)), axis: axis)
    eval(result)
    return result
  }

  public static func causalConv(_ input: MLXArray, weight: MLXArray,
    bias: MLXArray?, spatialPadding: Int, temporalPadding: Int,
    stride: IntOrTriple = 1) throws -> MLXArray {
    guard input.ndim == 5, weight.ndim == 5,
      input.shape[4] == weight.shape[4],
      weight.shape[0] > 0, weight.shape[1] > 0,
      weight.shape[2] > 0, weight.shape[3] > 0,
      input.dtype.isFloatingPoint, weight.dtype.isFloatingPoint,
      bias == nil || bias!.shape == [weight.shape[0]],
      spatialPadding >= 0, spatialPadding < input.shape[2],
      spatialPadding < input.shape[3],
      temporalPadding >= 0, temporalPadding <= weight.shape[1] - 1,
      stride.first > 0, stride.second > 0, stride.third > 0 else {
      throw H3CheckpointError.invalid("Invalid H3 VAE causal convolution geometry.")
    }
    var value = input
    if spatialPadding > 0 {
      value = try reflectSpatial(value, axis: 2,
        before: spatialPadding, after: spatialPadding)
      value = try reflectSpatial(value, axis: 3,
        before: spatialPadding, after: spatialPadding)
    }
    if temporalPadding > 0 {
      let widths: [IntOrPair] = [0, [temporalPadding, 0], 0, 0, 0]
      value = padded(value, widths: widths)
    }
    guard value.shape[1] >= weight.shape[1],
      value.shape[2] >= weight.shape[2],
      value.shape[3] >= weight.shape[3] else {
      throw H3CheckpointError.invalid("H3 VAE convolution kernel exceeds its padded input.")
    }
    let output = conv3d(value, weight, stride: stride)
    let result = bias.map { output + $0 } ?? output
    eval(result)
    return result
  }
}
