import Foundation
import MLX
import TensorIO

/// Converts an unrotated Comfy signed-int8 matrix to MLX's affine Q8 layout.
/// ConvRot weights require H3ComfyDecodedProjection and are rejected here;
/// treating their stored bytes as inference-ready changes the model.
public struct H3ComfyInt8Projection {
  public let rows: Int
  public let columns: Int
  public let storageBytes: Int
  private let packed: MLXArray
  private let scales: MLXArray
  private let biases: MLXArray

  /// Stream one checked projection from the already-open checkpoint. The MLX
  /// weights are materialized before the source mapping leaves its scope.
  public init(file: SafeTensorFile, name: String, rows: Int, columns: Int) throws {
    let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
    guard !overflow, rows > 0, columns > 0, columns.isMultiple(of: 64),
      let weight = file.tensors[name], weight.dtype == "I8",
      weight.shape == [UInt64(rows), UInt64(columns)],
      weight.byteCount == UInt64(count), name.hasSuffix(".weight") else {
      throw H3CheckpointError.invalid("Missing or incompatible Comfy INT8 weight: \(name)")
    }
    let stem = String(name.dropLast(".weight".count))
    let scaleName = stem + ".weight_scale"
    let markerName = stem + ".comfy_quant"
    guard let scale = file.tensors[scaleName], scale.dtype == "F32",
      scale.shape == [UInt64(rows), 1],
      let marker = file.tensors[markerName], marker.dtype == "U8",
      marker.byteCount > 0, marker.byteCount <= 4096 else {
      throw H3CheckpointError.invalid("Incomplete Comfy INT8 metadata: \(name)")
    }
    let markerData: Data = try file.withTensorBytes(named: markerName) { Data($0) }
    guard let object = try JSONSerialization.jsonObject(with: markerData) as? [String: Any],
      object["format"] as? String == "int8_tensorwise",
      Set(object.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]),
      object["convrot"] as? Bool != true else {
      throw H3CheckpointError.invalid("Unsupported Comfy INT8 marker: \(name)")
    }
    let rowScales = try file.readFloat32(named: scaleName)
    self = try file.withTensorBytes(named: name) {
      try Self(weightBytes: $0, rows: rows, columns: columns, rowScales: rowScales)
    }
  }

  public init(checkpointURL: URL, name: String, rows: Int, columns: Int) throws {
    let file = try SafeTensorFile(url: checkpointURL)
    try self.init(file: file, name: name, rows: rows, columns: columns)
    try file.checkUnchanged(at: checkpointURL)
  }

  public init(weightBytes: UnsafeRawBufferPointer, rows: Int, columns: Int,
    rowScales: [Float]) throws {
    let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
    guard !overflow, rows > 0, columns > 0, columns.isMultiple(of: 64),
      count == weightBytes.count, count <= 512 * 1024 * 1024,
      rowScales.count == rows, rowScales.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid Comfy INT8 projection shape or row scales.")
    }
    self.rows = rows
    self.columns = columns
    let groups = columns / 64
    let unsigned = MLXArray(weightBytes, [rows, columns / 4], type: UInt32.self)
      ^ UInt32(0x80808080)
    let scaleRows = MLXArray(rowScales, [rows, 1])
    let groupScales = broadcast(scaleRows, to: [rows, groups])
    let groupBiases = groupScales * Float(-128)
    eval([unsigned, groupScales, groupBiases])
    packed = unsigned
    scales = groupScales
    biases = groupBiases
    storageBytes = count + rows * groups * 8
  }

  public func project(_ input: MLXArray) throws -> MLXArray {
    guard input.ndim >= 2, input.shape.last == columns, input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("H3 projection input differs from its admitted width.")
    }
    return quantizedMM(input, packed, scales: scales.asType(input.dtype),
      biases: biases.asType(input.dtype), groupSize: 64, bits: 8)
  }
}
