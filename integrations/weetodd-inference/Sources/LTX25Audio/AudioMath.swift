import Accelerate
import Foundation
import MetalPerformanceShaders

public enum AudioError: Error { case invalid(String) }

/// Internal channels-last storage, batch one; width is frequency for the VAE.
struct AudioTensor {
  var values: [Float]
  let height: Int
  let width: Int
  let channels: Int
  init(_ values: [Float], height: Int, width: Int, channels: Int) {
    self.values = values; self.height = height; self.width = width; self.channels = channels
    precondition(height > 0 && width > 0 && channels > 0 && values.count == height * width * channels)
  }
}

/// Bounded im2col convolution. The decoder supplies its Metal matrix engine;
/// nil selects Accelerate BLAS only for isolated primitive qualification tests.
/// No graph cache, full im2col, or retained decoded checkpoint is created.
enum AudioMath {
  static func conv(_ x: AudioTensor, weight: [Float], bias: [Float], outputChannels: Int,
    kernelHeight: Int, kernelWidth: Int = 1, padTop: Int, padLeft: Int = 0,
    dilation: Int = 1, stride: Int = 1, outputHeight: Int? = nil, matrix: AudioMatrixEngine? = nil) throws -> AudioTensor {
    let h = outputHeight ?? x.height, w = x.width, o = outputChannels
    let k = x.channels * kernelHeight * kernelWidth
    guard weight.count == o * k, bias.isEmpty || bias.count == o else { throw AudioError.invalid("Convolution weight shape mismatch") }
    var y = [Float](repeating: 0, count: h * w * o)
    let tile = min(512, h * w)
    var columns = [Float](repeating: 0, count: tile * k)
    // Checkpoint rows are [output, input, kernelHeight, kernelWidth].
    func execute(_ gpuWeight: MPSMatrix?) throws {
    for start in Swift.stride(from: 0, to: h * w, by: tile) {
      try Task.checkCancellation()
      let rows = min(tile, h * w - start)
      for row in 0..<rows {
        let position = start + row, oh = position / w, ow = position % w
        for c in 0..<x.channels {
          for kh in 0..<kernelHeight {
            let ih = oh * stride + kh * dilation - padTop
            for kw in 0..<kernelWidth {
              let iw = ow + kw - padLeft, j = (c * kernelHeight + kh) * kernelWidth + kw
              columns[row * k + j] = ih >= 0 && ih < x.height && iw >= 0 && iw < w ? x.values[(ih * w + iw) * x.channels + c] : 0
            }
          }
        }
      }
      if let matrix, let gpuWeight {
        let result = try matrix.multiply(columns, weight: gpuWeight, rows: rows, columns: o, inner: k)
        y.replaceSubrange(start * o..<(start + rows) * o, with: result)
      } else {
      columns.withUnsafeBufferPointer { a in weight.withUnsafeBufferPointer { b in y.withUnsafeMutableBufferPointer { result in
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(o), Int32(k),
          1, a.baseAddress!, Int32(k), b.baseAddress!, Int32(k), 0, result.baseAddress! + start * o, Int32(o))
      } } }
      }
      if !bias.isEmpty { for row in 0..<rows { for c in 0..<o { y[(start + row) * o + c] += bias[c] } } }
    }
    }
    if let matrix { try matrix.withWeight(weight, rows: o, columns: k) { try execute($0) } }
    else { try execute(nil) }
    guard y.allSatisfy(\.isFinite) else { throw AudioError.invalid("Nonfinite audio convolution result") }
    return AudioTensor(y, height: h, width: w, channels: o)
  }

  static func transposeConv(_ x: AudioTensor, weight: [Float], bias: [Float], outputChannels: Int,
    kernel: Int, stride: Int, padding: Int, matrix: AudioMatrixEngine? = nil) throws -> AudioTensor {
    let o = outputChannels, h = (x.height - 1) * stride - 2 * padding + kernel
    guard x.width == 1, weight.count == x.channels * o * kernel, bias.count == o else { throw AudioError.invalid("Transpose convolution shape mismatch") }
    // Gather strided input windows and reorder the compact kernel. This computes
    // scatter-add transpose convolution without a waveform-sized zero insertion.
    var reordered = [Float](repeating: 0, count: o * x.channels * kernel)
    for c in 0..<x.channels { for oc in 0..<o { for q in 0..<kernel {
      reordered[(oc * x.channels + c) * kernel + q] = weight[(c * o + oc) * kernel + q]
    } } }
    let k = x.channels * kernel, tile = min(512, h)
    var columns = [Float](repeating: 0, count: tile * k)
    var y = [Float](repeating: 0, count: h * o)
    func execute(_ gpuWeight: MPSMatrix?) throws {
    for start in Swift.stride(from: 0, to: h, by: tile) {
      try Task.checkCancellation()
      let rows = min(tile, h - start)
      for row in 0..<rows { for c in 0..<x.channels { for q in 0..<kernel {
        let source = start + row + padding - q
        columns[row * k + c * kernel + q] = source >= 0 && source % stride == 0 && source / stride < x.height ? x.values[source / stride * x.channels + c] : 0
      } } }
      if let matrix, let gpuWeight {
        let result = try matrix.multiply(columns, weight: gpuWeight, rows: rows, columns: o, inner: k)
        y.replaceSubrange(start * o..<(start + rows) * o, with: result)
      } else {
      columns.withUnsafeBufferPointer { a in reordered.withUnsafeBufferPointer { b in y.withUnsafeMutableBufferPointer { result in
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(o), Int32(k), 1,
          a.baseAddress!, Int32(k), b.baseAddress!, Int32(k), 0, result.baseAddress! + start * o, Int32(o))
      } } }
      }
      for row in 0..<rows { for c in 0..<o { y[(start + row) * o + c] += bias[c] } }
    }
    }
    if let matrix { try matrix.withWeight(reordered, rows: o, columns: k) { try execute($0) } }
    else { try execute(nil) }
    guard y.allSatisfy(\.isFinite) else { throw AudioError.invalid("Nonfinite audio transpose convolution result") }
    return AudioTensor(y, height: h, width: 1, channels: o)
  }

  static func normalizedSiLU(_ x: AudioTensor) -> AudioTensor {
    var result = x
    for p in 0..<(x.height * x.width) {
      let base = p * x.channels
      var sum: Float = 0
      for c in 0..<x.channels { sum += x.values[base + c] * x.values[base + c] }
      let scale = 1 / sqrt(sum / Float(x.channels) + 1e-6)
      for c in 0..<x.channels { let z = x.values[base + c] * scale; result.values[base + c] = z / (1 + exp(-z)) }
    }
    return result
  }
  static func add(_ x: AudioTensor, _ y: AudioTensor, scale: Float = 1) -> AudioTensor {
    precondition(x.values.count == y.values.count)
    var result = x
    for i in result.values.indices { result.values[i] += y.values[i] * scale }
    return result
  }
  static func nearest2x(_ x: AudioTensor) -> AudioTensor {
    var result = [Float](repeating: 0, count: x.values.count * 4)
    for h in 0..<(x.height * 2) { for w in 0..<(x.width * 2) { for c in 0..<x.channels {
      result[(h * x.width * 2 + w) * x.channels + c] = x.values[((h / 2) * x.width + w / 2) * x.channels + c]
    } } }
    return AudioTensor(result, height: x.height * 2, width: x.width * 2, channels: x.channels)
  }
  static func upsampleFilter(_ x: AudioTensor, filter: [Float], ratio: Int, inputPad: Int, cropLeft: Int) -> AudioTensor {
    let h = x.height * ratio, c = x.channels
    var y = [Float](repeating: 0, count: h * c)
    for t in 0..<h { for q in filter.indices {
      let s = t + cropLeft - q
      guard s >= 0, s % ratio == 0, s / ratio < x.height + 2 * inputPad else { continue }
      let source = min(x.height - 1, max(0, s / ratio - inputPad))
      for channel in 0..<c { y[t * c + channel] += Float(ratio) * filter[q] * x.values[source * c + channel] }
    } }
    return AudioTensor(y, height: h, width: 1, channels: c)
  }
  static func downsampleFilter(_ x: AudioTensor, filter: [Float]) -> AudioTensor {
    let h = x.height / 2, c = x.channels, left = (filter.count - 1) / 2
    var y = [Float](repeating: 0, count: h * c)
    for t in 0..<h { for q in filter.indices {
      let source = min(x.height - 1, max(0, 2 * t + q - left))
      for channel in 0..<c { y[t * c + channel] += filter[q] * x.values[source * c + channel] }
    } }
    return AudioTensor(y, height: h, width: 1, channels: c)
  }
  static func resample48k(_ x: AudioTensor) -> AudioTensor {
    let filter: [Float] = (0..<43).map { i in
      let t = (Double(i) / 3 - 7) * 0.99
      let window = pow(cos(min(6, max(-6, t)) * Double.pi / 12), 2)
      return Float((t == 0 ? 1 : sin(Double.pi * t) / (Double.pi * t)) * window * 0.99 / 3)
    }
    return upsampleFilter(x, filter: filter, ratio: 3, inputPad: 7, cropLeft: 42)
  }
}
