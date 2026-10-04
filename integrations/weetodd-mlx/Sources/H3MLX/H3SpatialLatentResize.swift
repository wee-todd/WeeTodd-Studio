import Foundation
import MLX

public enum H3SpatialLatentResizeMethod: String, Sendable, Codable {
  case nearestExact = "nearest exact", bilinear, bicubic, lanczos3 = "lanczos-3"
}

/// Independently implemented owned H3 latent interpolation. This only resizes
/// spatial axes of normalized video; it never resizes time, channels or audio.
public enum H3SpatialLatentResize {
  public static func rows(_ rows: MLXArray, source: H3Geometry,
    target: H3Geometry, method: H3SpatialLatentResizeMethod) throws -> MLXArray {
    guard source.frames == target.frames,
      source.width * source.height <= source.canvasAdmission.maximumPixels,
      target.width * target.height <= target.canvasAdmission.maximumPixels,
      rows.shape == [1, source.videoRows, 96], rows.dtype == .float32 else {
      throw H3CheckpointError.invalid("H3 spatial refinement requires unchanged AV duration and admitted full video rows.")
    }
    try source.canvasAdmission.validate(width: source.width, height: source.height)
    try target.canvasAdmission.validate(width: target.width, height: target.height)
    try target.canvasAdmission.validatePackedRows(target.videoRows + target.audioRows + 1)
    try Task.checkCancellation()
    if source.width == target.width && source.height == target.height { return rows }
    let height = source.height / 16, width = source.width / 16
    let targetHeight = target.height / 16, targetWidth = target.width / 16
    let latents = rows.reshaped([1, source.videoLatentFrames, height / 2, width / 2, 24, 2, 2])
      .transposed(0, 1, 2, 5, 3, 6, 4)
      .reshaped([1, source.videoLatentFrames, height, width, 24])
    var resized = try axis(latents, axis: 2, source: height, target: targetHeight, method: method)
    // Bound the six-tap Lanczos graph just as the owned implementation does.
    if method == .lanczos3 { eval(resized) }
    try Task.checkCancellation()
    resized = try axis(resized, axis: 3, source: width, target: targetWidth, method: method)
    let result = resized.reshaped([1, target.videoLatentFrames,
      targetHeight / 2, 2, targetWidth / 2, 2, 24])
      .transposed(0, 1, 2, 4, 6, 3, 5)
      .reshaped([1, target.videoRows, 96])
    eval(result); try Task.checkCancellation()
    return result
  }

  private static func axis(_ input: MLXArray, axis: Int, source: Int,
    target: Int, method: H3SpatialLatentResizeMethod) throws -> MLXArray {
    if source == target { return input }
    let destination = MLXArray(0..<target).asType(.float32)
    if method == .nearestExact {
      let index = clip(floor((destination + Float(0.5)) * Float(Double(source) / Double(target))),
        min: 0, max: Float(source - 1)).asType(.int32)
      return take(input, index, axis: axis)
    }
    let position = (destination + Float(0.5)) * Float(Double(source) / Double(target)) - Float(0.5)
    let lower = floor(position).asType(.int32)
    var shape = [Int](repeating: 1, count: input.ndim); shape[axis] = target
    func selected(_ offset: Int) -> MLXArray {
      take(input, clip(lower + Int32(offset), min: Int32(0), max: Int32(source - 1)), axis: axis)
    }
    if method == .bilinear {
      let bottom = selected(1), top = selected(0)
      let weight = (position - lower.asType(.float32)).reshaped(shape)
      return top + (bottom - top) * weight
    }
    let offsets = method == .bicubic ? [-1, 0, 1, 2] : [-2, -1, 0, 1, 2, 3]
    var weights: [MLXArray] = []
    for offset in offsets {
      let distance = position - (lower + Int32(offset)).asType(.float32)
      let absolute = abs(distance)
      if method == .bicubic {
        let insideOne = Float(1.5) * MLX.pow(absolute, 3)
          - Float(2.5) * MLX.pow(absolute, 2) + Float(1)
        let insideTwo = Float(-0.5) * MLX.pow(absolute, 3)
          + Float(2.5) * MLX.pow(absolute, 2) - Float(4) * absolute + Float(2)
        weights.append(MLX.where(absolute .<= Float(1), insideOne,
          MLX.where(absolute .< Float(2), insideTwo, zeros(like: absolute))))
      } else {
        func sinc(_ value: MLXArray) -> MLXArray {
          let angle = Float(Double.pi) * value
          let nearZero = abs(value) .< Float(1e-7)
          let denominator = MLX.where(nearZero, ones(like: angle), angle)
          return MLX.where(nearZero, ones(like: angle), sin(angle) / denominator)
        }
        weights.append(MLX.where(absolute .< Float(3), sinc(distance) * sinc(distance / Float(3)), zeros(like: absolute)))
      }
    }
    if method == .lanczos3 {
      var total = weights[0]
      for weight in weights.dropFirst() { total = total + weight }
      weights = weights.map { $0 / total }
    }
    var result = selected(offsets[0]) * weights[0].reshaped(shape)
    for index in 1..<offsets.count {
      try Task.checkCancellation()
      result = result + selected(offsets[index]) * weights[index].reshaped(shape)
    }
    return result
  }
}
