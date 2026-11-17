import Foundation
import MLX

/// NHWC activations partitioned into rows. Normalization statistics span every band.
struct MOSSImageBands {
  let bands: [MLXArray]
  let width, channels: Int
  var height: Int { bands.reduce(0) { $0 + $1.dim(1) } }
  init(_ image: MLXArray, rowLimit: Int) {
    precondition(image.ndim == 4 && image.dim(0) == 1 && rowLimit > 0)
    width = image.dim(2); channels = image.dim(3)
    bands = stride(from: 0, to: image.dim(1), by: rowLimit).map { image[0..., $0..<min($0 + rowLimit, image.dim(1)), 0..., 0...] }
  }
  init(bands: [MLXArray]) { self.bands = bands; width = bands[0].dim(2); channels = bands[0].dim(3) }
  func materialized() -> MLXArray { bands.count == 1 ? bands[0] : concatenated(bands, axis: 1) }
  func rows(_ range: Range<Int>) -> MLXArray {
    var pieces: [MLXArray] = [], start = 0
    for band in bands {
      let end = start + band.dim(1)
      let lower = max(range.lowerBound, start), upper = min(range.upperBound, end)
      if lower < upper { pieces.append(band[0..., (lower - start)..<(upper - start), 0..., 0...]) }
      start = end
      if start >= range.upperBound { break }
    }
    return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
  }
  func mapped(_ body: (MLXArray) -> MLXArray) -> MOSSImageBands {
    .init(bands: bands.map { let output = body($0); eval(output); return output })
  }
  func convolved(weight: MLXArray, bias: MLXArray, rowLimit: Int) -> MOSSImageBands {
    let kernel = weight.dim(1), halo = kernel / 2, h = height
    var output: [MLXArray] = []
    for start in stride(from: 0, to: h, by: max(1, rowLimit)) {
      let end = min(h, start + max(1, rowLimit)), sourceStart = max(0, start - halo), sourceEnd = min(h, end + halo)
      let neighborhood = rows(sourceStart..<sourceEnd)
      let convolved = conv2d(neighborhood, weight, padding: .init(halo)) + bias
      let band = convolved[0..., (start - sourceStart)..<(end - sourceStart), 0..., 0...]
      eval(band); output.append(band)
    }
    return .init(bands: output)
  }
  func groupNormalized(groups: Int = 32, weight: MLXArray, bias: MLXArray) -> MOSSImageBands {
    precondition(channels.isMultiple(of: groups))
    var total: Float = 0
    var center = MLXArray.zeros([1, 1, 1, groups, 1]), m2 = center
    for band in bands {
      let x = band.asType(.float32).reshaped(1, band.dim(1), width, groups, channels / groups)
      let count = Float(band.dim(1) * width * (channels / groups))
      let localMean = mean(x, axes: [1, 2, 4], keepDims: true)
      let delta = localMean - center
      let centered = x - localMean
      let localM2 = sum(centered * centered, axes: [1, 2, 4], keepDims: true)
      let combined = total + count
      m2 = m2 + localM2 + delta * delta * (total * count / combined)
      center = center + delta * (count / combined); total = combined
    }
    let inverse = rsqrt(m2 / total + 1e-6)
    eval(center, inverse)
    return mapped { band in
      let x = band.reshaped(1, band.dim(1), width, groups, channels / groups)
      return ((x - center) * inverse).reshaped(band.shape) * weight + bias
    }
  }
  /// FLUX.2 downsampling pads only the right and bottom, then uses stride two.
  func downsampled(weight: MLXArray, bias: MLXArray, rowLimit: Int) -> MOSSImageBands {
    precondition(height.isMultiple(of: 2) && width.isMultiple(of: 2) && weight.dim(1) == 3)
    var output: [MLXArray] = []
    for start in stride(from: 0, to: height / 2, by: max(1, rowLimit)) {
      let end = min(height / 2, start + max(1, rowLimit))
      let sourceEnd = min(height, end * 2 + 1)
      let input = padded(rows((start * 2)..<sourceEnd),
        widths: [0, .init((0, end * 2 + 1 - sourceEnd)), .init((0, 1)), 0])
      let value = conv2d(input, weight, stride: 2) + bias
      eval(value); output.append(value)
    }
    return .init(bands: output)
  }
  func adding(_ other: MOSSImageBands) -> MOSSImageBands {
    var start = 0
    return .init(bands: bands.map { band in
      let value = band + other.rows(start..<(start + band.dim(1)))
      start += band.dim(1); eval(value); return value
    })
  }
  func upsampled() -> MOSSImageBands {
    mapped { band in
      let h = band.dim(1)
      let rows = broadcast(band.expandedDimensions(axis: 2), to: [1, h, 2, width, channels]).reshaped(1, h * 2, width, channels)
      return broadcast(rows.expandedDimensions(axis: 3), to: [1, h * 2, width, 2, channels]).reshaped(1, h * 2, width * 2, channels)
    }
  }
}
