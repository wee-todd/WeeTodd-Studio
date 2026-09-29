import Foundation
import MLX

/// Independent rectified-flow schedule for one H3 modality. The caller builds
/// video with shift 12 and audio with shift 3, then advances both after one
/// shared audiovisual transformer evaluation.
public struct H3Schedule: Sendable {
  public let sigmas: [Float]
  public let timesteps: [Float]

  /// Matches the saved H3 `steps` setting: N grid points make N-1 model calls.
  public init(requestedSteps: Int, shift: Float) throws {
    guard (2...2001).contains(requestedSteps), shift.isFinite, shift > 0 else {
      throw H3GeometryError.invalid("H3 requires at least two bounded sigma grid points and a positive shift.")
    }
    let evaluations = requestedSteps - 1
    let step = Float(-1) / Float(evaluations)
    let last = evaluations
    var values: [Float] = []
    values.reserveCapacity(evaluations + 1)
    for index in 0...last {
      // ATen's endpoint-symmetric float32 grid uses a fused multiply-add.
      let base: Float = index < (evaluations + 1) / 2
        ? Float(1).addingProduct(step, Float(index))
        : Float(0).addingProduct(-step, Float(last - index))
      let shifted = (shift * base) / (Float(1) + (shift - 1) * base)
      if values.last != shifted { values.append(shifted) }
    }
    guard values.count >= 2, values.first == 1, values.last == 0 else {
      throw H3GeometryError.invalid("H3 sigma shift collapsed the entire sampling grid.")
    }
    sigmas = values
    timesteps = values.dropLast().map { Float(1) - $0 }
  }

  public func advance(sample: MLXArray, velocity: MLXArray, index: Int) throws -> MLXArray {
    guard (0..<timesteps.count).contains(index), sample.shape == velocity.shape,
      sample.dtype.isFloatingPoint, velocity.dtype.isFloatingPoint else {
      throw H3GeometryError.invalid("H3 sampling step or latent/velocity shape is invalid.")
    }
    let fromTimestep = Float(1) - timesteps[index]
    let denoised = sample + fromTimestep * velocity
    let ratio = sigmas[index + 1] / sigmas[index]
    let next = ratio * sample.asType(.float32)
      + (Float(1) - ratio) * denoised.asType(.float32)
    return next.asType(sample.dtype)
  }
}
