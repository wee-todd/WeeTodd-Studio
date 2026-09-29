import Foundation
import TensorIO

/// Bounded row decoding for dense and existing MLX group-64 affine Q8 weights.
struct TextWeights {
  let file: SafeTensorFile
  let prefix: String
  func vector(_ name: String, count: Int) throws -> [Float] {
    let key = prefix + name
    guard file.tensors[key]?.shape == [UInt64(count)] else { throw TextEncodingError.invalid("Missing/mismatched text vector: \(key)") }
    return try file.readFloat32(named: key)
  }
  func shape(_ name: String) throws -> [Int] {
    let key = prefix + name
    guard let descriptor = file.tensors[key] else { throw TextEncodingError.invalid("Missing text tensor: \(key)") }
    if descriptor.dtype == "U32" { return try MLXAffineQ8(file: file, weight: key, groupSize: 64).shape }
    guard ["F32", "F16", "BF16"].contains(descriptor.dtype) else { throw TextEncodingError.invalid("Unsupported text tensor dtype: \(key)") }
    return descriptor.shape.map(Int.init)
  }
  func rows(_ name: String, _ range: Range<Int>) throws -> [Float] {
    let key = prefix + name, shape = try shape(name)
    guard shape.count == 2, range.lowerBound >= 0, range.upperBound <= shape[0] else { throw TextEncodingError.invalid("Invalid text weight row slice.") }
    if file.tensors[key]?.dtype == "U32" { return try MLXAffineQ8(file: file, weight: key, groupSize: 64).readRows(range, maximumBytes: 64*1024*1024) }
    return try file.readFloat32(named: key, elements: UInt64(range.lowerBound*shape[1])..<UInt64(range.upperBound*shape[1]), maximumBytes: 64*1024*1024)
  }
  func linear(_ name: String, _ x: [Float], tokens: Int, width: Int, output: Int,
    bias: Bool = false, scale: Float = 1, gpu: TextMatrixGPU) throws -> [Float] {
    return try autoreleasepool {
    guard x.count == tokens*width, try shape(name + ".weight") == [output, width],
      output > 0, output <= 262144, width > 0, width <= 1048576 else {
      throw TextEncodingError.invalid("Text projection shape mismatch: \(prefix + name)")
    }
    var result = [Float](repeating: 0, count: tokens*output)
    let biases = bias ? try vector(name + ".bias", count: output) : []
    let chunk = max(1, min(output, 16*1024*1024/width))
    let left = try gpu.prepareLeft(x,rows: tokens,inner: width)
    for start in stride(from: 0, to: output, by: chunk) {
      try Task.checkCancellation()
      let count = min(chunk, output-start)
      let weight = try rows(name + ".weight", start..<(start+count))
      let block = try gpu.multiply(left, weight, columns: count, scale: scale)
      for t in 0..<tokens { for o in 0..<count { result[t*output+start+o] = block[t*count+o] + (bias ? biases[start+o] : 0) } }
    }
    return result
    }
  }
}
