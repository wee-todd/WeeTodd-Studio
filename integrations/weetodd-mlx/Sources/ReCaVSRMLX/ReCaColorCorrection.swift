import Foundation
import MLX

enum ReCaImagePreparation {
  private static func cubic(_ value: Float) -> Float {
    let x = abs(value), a: Float = -0.75
    if x <= 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
    return 0
  }
  static func resize(_ input: MLXArray, width: Int, height: Int, bicubic: Bool = false) -> MLXArray {
    func axis(_ x: MLXArray, target: Int, axis: Int) -> MLXArray {
      let extent = x.dim(axis)
      if target == extent { return x }
      let locations = (0..<target).map { (Float($0) + 0.5) * Float(extent) / Float(target) - 0.5 }
      var result: MLXArray?
      for delta in (bicubic ? -1 : 0)...(bicubic ? 2 : 1) {
        let indices = locations.map { Int32(min(extent - 1, max(0, Int(floor($0)) + delta))) }
        let weights = locations.map { location -> Float in
          let fraction = location - floor(location)
          return bicubic ? cubic(Float(delta) - fraction) : delta == 0 ? 1 - fraction : fraction
        }
        var shape = Array(repeating: 1, count: x.ndim); shape[axis] = target
        let value = x.take(MLXArray(indices), axis: axis) * MLXArray(weights, shape)
        result = result.map { $0 + value } ?? value
      }
      return result!
    }
    return axis(axis(input.asType(.float32), target: width, axis: 2), target: height, axis: 1)
  }
  static func prepare(_ rgb: MLXArray, geometry: ReCaGeometry, frames: Int) throws -> MLXArray {
    guard rgb.ndim == 4, rgb.dim(0) > 0, rgb.dim(0) <= frames,
      rgb.shape == [rgb.dim(0), geometry.sourceHeight, geometry.sourceWidth, 3], rgb.dtype == .uint8 else {
      throw ReCaError.invalid("Source RGB block changed dimensions or dtype.")
    }
    // Reference OpenCV preprocessing rounds the enlarged RGB before normalization.
    var x = clip(round(resize(rgb, width: geometry.width, height: geometry.height)), min: 0, max: 255) / 127.5 - 1
    if rgb.dim(0) < frames { x = concatenated([x, repeated(x[(rgb.dim(0) - 1)..<rgb.dim(0)], count: frames - rgb.dim(0), axis: 0)], axis: 0) }
    let rows = MLXArray((0..<geometry.paddedHeight).map { Int32(geometry.sourceIndex($0, axisSize: geometry.height)) })
    let columns = MLXArray((0..<geometry.paddedWidth).map { Int32(geometry.sourceIndex($0, axisSize: geometry.width)) })
    return x.take(rows, axis: 1).take(columns, axis: 2)
  }
}

enum ReCaColorCorrection {
  static func quantizeNormalized(_ rgb: MLXArray) -> MLXArray {
    round((clip(rgb.asType(.float32), min: -1, max: 1) + 1) * 127.5).asType(.uint8)
  }
  static func adain(decodedNormalized: MLXArray, enlargedLQNormalized: MLXArray) throws -> MLXArray {
    guard decodedNormalized.ndim == 4, decodedNormalized.shape == enlargedLQNormalized.shape else {
      throw ReCaError.invalid("AdaIN needs matching cropped RGB and enlarged LQ frames.")
    }
    let x = (decodedNormalized.asType(.float32) + 1) * 0.5
    let ref = (enlargedLQNormalized.asType(.float32) + 1) * 0.5
    let xm = mean(x, axes: [1, 2], keepDims: true), rm = mean(ref, axes: [1, 2], keepDims: true)
    let n = x.dim(1) * x.dim(2), correction = Float(n) / Float(max(1, n - 1))
    let xv = mean((x - xm) * (x - xm), axes: [1, 2], keepDims: true) * correction
    let rv = mean((ref - rm) * (ref - rm), axes: [1, 2], keepDims: true) * correction
    return ((x - xm) * sqrt((rv + 1e-5) / (xv + 1e-5)) + rm) * 2 - 1
  }
  static func wavelet(decodedNormalized: MLXArray, originalLQUInt8: MLXArray) throws -> MLXArray {
    guard decodedNormalized.ndim == 4, originalLQUInt8.ndim == 4, originalLQUInt8.dtype == .uint8,
      decodedNormalized.dim(0) == originalLQUInt8.dim(0), decodedNormalized.dim(3) == 3 else {
      throw ReCaError.invalid("Wavelet correction needs the matching original uint8 LQ frames.")
    }
    var content = clip((decodedNormalized.asType(.float32) + 1) * 0.5, min: 0, max: 1)
    var reference = clip(ReCaImagePreparation.resize(originalLQUInt8.asType(.float32) / 255,
      width: content.dim(2), height: content.dim(1), bicubic: true), min: 0, max: 1)
    var high = MLXArray.zeros(content.shape)
    for radius in [1, 2, 4, 8, 16] {
      try Task.checkCancellation()
      let next = blur(content, radius: radius)
      high = high + (content - next); eval(high)
      content = next; reference = blur(reference, radius: radius); eval(content, reference)
    }
    // Keep the per-level sum and quantize directly; a normalized round trip changes ties.
    return round(clip(high + reference, min: 0, max: 1) * 255).asType(.uint8)
  }
  private static func blur(_ x: MLXArray, radius: Int) -> MLXArray {
    let h = x.dim(1), w = x.dim(2)
    let rows = MLXArray((-radius..<(h + radius)).map { Int32(min(h - 1, max(0, $0))) })
    let columns = MLXArray((-radius..<(w + radius)).map { Int32(min(w - 1, max(0, $0))) })
    let expanded = x.take(rows, axis: 1).take(columns, axis: 2)
    let kernel = MLXArray([Float(1), 2, 1, 2, 4, 2, 1, 2, 1], [1, 3, 3, 1]) / 16
    return conv2d(expanded, repeated(kernel, count: 3, axis: 0), dilation: IntOrPair(radius), groups: 3)
  }
}
