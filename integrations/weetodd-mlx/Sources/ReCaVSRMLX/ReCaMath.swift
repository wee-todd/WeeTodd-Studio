import Foundation
import MLX

enum ReCaMath {
  static func rms(_ x: MLXArray, weight: MLXArray, epsilon: Float, promote: Bool = false) -> MLXArray {
    let f = x.asType(.float32)
    let result = f * rsqrt(mean(f * f, axis: -1, keepDims: true) + epsilon) * weight.asType(.float32)
    return promote ? result : result.asType(x.dtype)
  }
  /// Gather forces an owning compact result rather than retaining a sliced backing allocation.
  static func own(_ x: MLXArray, range: Range<Int>, axis: Int = 0) -> MLXArray {
    let value = x.take(MLXArray(range.map(Int32.init)), axis: axis)
    eval(value); return value
  }
}

struct ReCaRotary {
  let cosine: MLXArray, sine: MLXArray
  init(height: Int, width: Int, times: [Int], axes: [Int], spatial: Bool = true,
    temporal: Bool = true, floatFrequencies: Bool = false) {
    let half = axes.reduce(0, +) / 2
    var angles: [Float] = []; angles.reserveCapacity(times.count * height * width * half)
    for time in times { for y in 0..<height { for x in 0..<width {
      for (axis, pair) in zip(axes, [temporal ? time : 0, spatial ? y : 0, spatial ? x : 0]).enumerated() {
        let (channels, position) = pair
        for frequency in 0..<(channels / 2) {
          if floatFrequencies {
            let inverse = 1 / pow(Float(10_000), Float(2 * frequency) / Float(channels))
            angles.append(Float(position) * inverse)
          } else {
            angles.append(Float(Double(position) * pow(10_000.0, -Double(2 * frequency) / Double(channels))))
          }
        }
        _ = axis
      }
    } } }
    let array = MLXArray(angles, [times.count * height * width, 1, half])
    cosine = cos(array); sine = sin(array)
    eval(cosine, sine)
  }
  func apply(_ x: MLXArray) -> MLXArray {
    let pairs = x.asType(.float32).reshaped([x.dim(0), x.dim(1), x.dim(2) / 2, 2])
    let even = pairs[.ellipsis, 0], odd = pairs[.ellipsis, 1]
    return stacked([even * cosine - odd * sine, even * sine + odd * cosine], axis: -1)
      .reshaped(x.shape).asType(x.dtype)
  }
}
