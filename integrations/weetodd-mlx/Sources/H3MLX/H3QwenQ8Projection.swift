import Foundation
import MLX
import TensorIO

/// The Qwen conditioner uses native MLX affine Q8 pages. Keep packed words and
/// group companions on device; never decode a 32B checkpoint to dense weights.
public struct H3QwenQ8Projection {
  public let rows: Int
  public let columns: Int
  public let storageBytes: Int
  private let packed: MLXArray
  private let scales: MLXArray
  private let biases: MLXArray

  public init(packed: MLXArray, scales: MLXArray, biases: MLXArray,
    columns: Int) throws {
    try self.init(packed: packed, scales: scales, biases: biases,
      columns: columns, materializeWeights: true)
  }

  // The single-layer Qwen owner defers these waits; public initialization
  // and the vision loader retain their established behavior.
  init(packed: MLXArray, scales: MLXArray, biases: MLXArray,
    columns: Int, materializeWeights: Bool) throws {
    guard packed.ndim == 2, packed.dtype == .uint32,
      columns > 0, columns.isMultiple(of: 64),
      packed.shape[1] == columns / 4,
      scales.shape == [packed.shape[0], columns / 64],
      biases.shape == scales.shape,
      scales.dtype.isFloatingPoint, biases.dtype == scales.dtype,
      packed.shape[0] > 0, packed.nbytes <= 512 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Invalid H3 Qwen affine Q8 projection shape.")
    }
    self.packed = packed
    self.scales = scales
    self.biases = biases
    rows = packed.shape[0]
    self.columns = columns
    storageBytes = packed.nbytes + scales.nbytes + biases.nbytes
    if materializeWeights { eval(parametersToMaterialize) }
  }

  public init(file: SafeTensorFile, name: String) throws {
    try self.init(file: file, name: name, tensor: nil)
  }

  init(file: SafeTensorFile, name: String,
    tensor: ((String) throws -> MLXArray)?,
    materializeWeights: Bool = true) throws {
    let layout = try MLXAffineQ8(file: file, weight: name, groupSize: 64)
    let stem = String(name.dropLast(".weight".count))
    func read(_ key: String) throws -> MLXArray {
      guard let descriptor = file.tensors[key], descriptor.byteCount <= 512 * 1024 * 1024 else {
        throw H3CheckpointError.invalid("Missing or oversized H3 Qwen projection: \(key)")
      }
      if let tensor { return try tensor(key) }
      return try file.withTensorBytes(named: key) { bytes in
        let shape = descriptor.shape.map(Int.init)
        switch descriptor.dtype {
        case "U32": return MLXArray(bytes, shape, type: UInt32.self)
        case "BF16": return MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
        case "F16": return MLXArray(bytes, shape, type: Float16.self)
        case "F32": return MLXArray(bytes, shape, type: Float.self)
        default: throw H3CheckpointError.invalid("Unsupported H3 Qwen projection dtype: \(key)")
        }
      }
    }
    try self.init(packed: read(name), scales: read(stem + ".scales"),
      biases: read(stem + ".biases"), columns: layout.shape[1],
      materializeWeights: materializeWeights)
    guard rows == layout.shape[0] else {
      throw H3CheckpointError.invalid("H3 Qwen projection changed after header admission.")
    }
  }

  public init(checkpointURL: URL, name: String) throws {
    let file = try SafeTensorFile(url: checkpointURL)
    try self.init(file: file, name: name)
    try file.checkUnchanged(at: checkpointURL)
  }

  // References to existing packed storage, never dense copies.
  var parametersToMaterialize: [MLXArray] { [packed, scales, biases] }

  public func project(_ input: MLXArray) throws -> MLXArray {
    guard input.ndim >= 2, input.shape.last == columns, input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("H3 Qwen projection input width or dtype differs.")
    }
    return quantizedMM(input, packed, scales: scales.asType(input.dtype),
      biases: biases.asType(input.dtype), groupSize: 64, bits: 8)
  }
}
