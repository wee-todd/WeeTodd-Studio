import Foundation

extension SafeTensorFile {
  /// Decode a bounded slice without allocating a second full raw payload. The
  /// caller must admit the returned Float32 storage before requesting it.
  public func readFloat32(named name: String, elements: Range<UInt64>? = nil,
    maximumBytes: UInt64 = 512 * 1024 * 1024) throws -> [Float] {
    try readFloat32(named: name,elements: elements,maximumBytes: maximumBytes,access: .mapped)
  }
  public func readFloat32(named name: String, elements: Range<UInt64>? = nil,
    maximumBytes: UInt64 = 512 * 1024 * 1024,access: TensorAccess) throws -> [Float] {
    try Task.checkCancellation()
    guard let tensor = tensors[name], let width = ["F32": 4, "BF16": 2, "F16": 2][tensor.dtype] else {
      throw CheckpointError.invalid("Expected F32, BF16 or F16 tensor: \(name)")
    }
    let total = tensor.byteCount / UInt64(width)
    let range = elements ?? 0..<total
    let count = range.upperBound - range.lowerBound
    guard range.upperBound <= total, count <= maximumBytes / 4, count <= UInt64(Int.max / 4) else {
      throw CheckpointError.invalid("Float32 tensor slice exceeds its payload or memory budget: \(name)")
    }
    var result = [Float](repeating: 0, count: Int(count))
    // A bounded 4 MiB raw window avoids tens of thousands of mmap/fstat calls
    // for Gemma's dense aggregation matrix. Decoded storage is still admitted
    // above; no raw payload copy or persistent weight mapping is introduced.
    let windowElements = 4 * 1024 * 1024 / width
    for start in stride(from: 0, to: Int(count), by: windowElements) {
      try Task.checkCancellation()
      let end = min(start + windowElements, Int(count))
      let lower = (range.lowerBound + UInt64(start)) * UInt64(width)
      let upper = (range.lowerBound + UInt64(end)) * UInt64(width)
      try withTensorBytes(named: name, range: lower..<upper,access: access) { bytes in
        for index in 0..<(end - start) {
          switch tensor.dtype {
          case "F32": result[start + index] = Float(bitPattern: bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian)
          case "BF16": result[start + index] = Float(bitPattern: UInt32(bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self).littleEndian) << 16)
          default: result[start + index] = Float(Float16(bitPattern: bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self).littleEndian))
          }
        }
      }
    }
    try Task.checkCancellation()
    return result
  }
}

/// MLX affine Q8: packed little-endian uint32 lanes with one scale and bias per
/// group. This decoder produces Float32 weights; it does not silently round them
/// back to BF16/F16. Qualification must use the same dequantization precision.
public enum Q8Decoding: Sendable { case scalar, simd }

public struct MLXAffineQ8 {
  public let shape: [Int]
  public let groupSize: Int
  private let file: SafeTensorFile
  private let weight: String
  private let scales: String
  private let biases: String

  public init(file: SafeTensorFile, weight: String, groupSize: Int) throws {
    guard groupSize == 64, weight.hasSuffix(".weight"), let packed = file.tensors[weight],
          packed.dtype == "U32", packed.shape.count == 2, packed.shape.allSatisfy({ $0 > 0 }),
          packed.shape[1] <= UInt64(Int.max / 4) else {
      throw CheckpointError.invalid("Expected a group-64 affine Q8 matrix: \(weight)")
    }
    let stem = String(weight.dropLast(7))
    let scales = stem + ".scales", biases = stem + ".biases"
    let columns = packed.shape[1] * 4
    guard columns % UInt64(groupSize) == 0, let s = file.tensors[scales], let b = file.tensors[biases],
          ["F32", "BF16", "F16"].contains(s.dtype), b.dtype == s.dtype,
          s.shape == [packed.shape[0], columns / UInt64(groupSize)], b.shape == s.shape else {
      throw CheckpointError.invalid("Q8 scales and biases must match its rows, groups and dtype: \(weight)")
    }
    self.file = file; self.weight = weight; self.scales = scales; self.biases = biases
    self.groupSize = groupSize; shape = [Int(packed.shape[0]), Int(columns)]
  }

  public func readRows(_ rows: Range<Int>, maximumBytes: UInt64 = 512 * 1024 * 1024,
    decoding: Q8Decoding = .simd) throws -> [Float] {
    try Task.checkCancellation()
    guard rows.lowerBound >= 0, rows.upperBound <= shape[0] else {
      throw CheckpointError.invalid("Q8 row slice is outside the matrix: \(weight)")
    }
    let count = UInt64(rows.count).multipliedReportingOverflow(by: UInt64(shape[1]))
    guard !count.overflow, count.partialValue <= maximumBytes / 4,
          count.partialValue <= UInt64(Int.max / 4) else {
      throw CheckpointError.invalid("Decoded Q8 matrix exceeds its memory budget: \(weight)")
    }
    let total = Int(count.partialValue)
    let elementOffset = UInt64(rows.lowerBound) * UInt64(shape[1])
    var result = [Float](repeating: 0, count: total)
    // Bounded 4 MiB mapped input and two 256 KiB Float32 companion arrays.
    // Larger windows avoid thousands of mmap/munmap pairs per block without
    // retaining an entire page or a second decoded matrix.
    for start in stride(from: 0, to: total, by: 4 * 1024 * 1024) {
      try Task.checkCancellation()
      let end = min(start + 4 * 1024 * 1024, total)
      let lower = elementOffset + UInt64(start), upper = elementOffset + UInt64(end)
      let groups = lower / UInt64(groupSize)..<upper / UInt64(groupSize)
      let s = try file.readFloat32(named: scales, elements: groups)
      let b = try file.readFloat32(named: biases, elements: groups)
      guard s.allSatisfy(\.isFinite), b.allSatisfy(\.isFinite) else {
        throw CheckpointError.invalid("Q8 scales and biases must be finite: \(weight)")
      }
      try file.withTensorBytes(named: weight, range: lower..<upper) { bytes in
        switch decoding {
        case .scalar:
          for index in 0..<(end - start) {
            result[start + index] = b[index / groupSize].addingProduct(s[index / groupSize], Float(bytes[index]))
          }
        case .simd:
          // Group-aligned windows and rows: every group has exactly 64 lanes.
          // Explicit fused SIMD arithmetic preserves the scalar decoder's bits.
          result.withUnsafeMutableBytes { output in
            for group in 0..<((end - start) / groupSize) {
              let scale = SIMD16<Float>(repeating: s[group])
              let bias = SIMD16<Float>(repeating: b[group])
              for lane in stride(from: 0, to: groupSize, by: 16) {
                let index = group * groupSize + lane
                let q = SIMD16<Float>(bytes.loadUnaligned(fromByteOffset: index, as: SIMD16<UInt8>.self))
                output.storeBytes(of: bias.addingProduct(scale, q), toByteOffset: (start + index) * 4,
                  as: SIMD16<Float>.self)
              }
            }
          }
        }
      }
    }
    try Task.checkCancellation()
    return result
  }
}
