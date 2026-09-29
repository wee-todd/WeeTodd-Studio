import Foundation
import MLX
import TensorIO

public enum H3ProjectionMode: Sendable {
  case weightDecoded
  case activationRotated
}

/// Apply a Comfy ConvRot INT8 matrix by rotating the small activation instead
/// of reconstructing every output weight row. The original installed bytes
/// remain the only checkpoint copy; one packed projection is resident at once.
public struct H3ComfyRotatedProjection {
  public let rows: Int
  public let columns: Int
  public let storageBytes: Int
  private let group: Int
  private let quantized: H3ComfyInt8Projection
  private let basis: MLXArray
  private let bias: MLXArray?

  public init(file: SafeTensorFile, name: String,
    rows: Int, columns: Int, biasName: String? = nil) throws {
    guard name.hasSuffix(".weight"), rows > 0, columns > 0,
      columns.isMultiple(of: 64),
      let weight = file.tensors[name], weight.dtype == "I8",
      weight.shape == [UInt64(rows), UInt64(columns)] else {
      throw H3CheckpointError.invalid("Invalid H3 ConvRot projection: \(name)")
    }
    let stem = String(name.dropLast(".weight".count))
    let markerName = stem + ".comfy_quant"
    guard let marker = file.tensors[markerName], marker.dtype == "U8",
      marker.byteCount <= 4096 else {
      throw H3CheckpointError.invalid("Missing H3 ConvRot metadata: \(name)")
    }
    let markerData = try file.withTensorBytes(named: markerName) { Data($0) }
    guard let object = try JSONSerialization.jsonObject(with: markerData) as? [String: Any],
      object["format"] as? String == "int8_tensorwise",
      object["convrot"] as? Bool == true,
      let group = object["convrot_groupsize"] as? Int,
      [4, 16, 64, 256, 1024].contains(group),
      columns.isMultiple(of: group) else {
      throw H3CheckpointError.invalid("Unsupported H3 ConvRot layout: \(name)")
    }
    let scales = try file.readFloat32(named: stem + ".weight_scale")
    let packed = try file.withTensorBytes(named: name) { bytes in
      try H3ComfyInt8Projection(weightBytes: bytes,
        rows: rows, columns: columns, rowScales: scales)
    }
    let bias: MLXArray?
    if let biasName {
      guard let descriptor = file.tensors[biasName],
        descriptor.dtype == "BF16", descriptor.shape == [UInt64(rows)] else {
        throw H3CheckpointError.invalid("Missing H3 ConvRot projection bias: \(biasName)")
      }
      bias = try file.withTensorBytes(named: biasName) { bytes in
        MLXArray(bytes, [rows], type: UInt16.self).view(dtype: .bfloat16)
      }
      eval(bias!)
    } else {
      bias = nil
    }
    self.rows = rows
    self.columns = columns
    self.group = group
    quantized = packed
    basis = H3ComfyDecodedProjection.basis(group: group)
    self.bias = bias
    storageBytes = packed.storageBytes + (bias?.nbytes ?? 0)
  }

  public func project(_ input: MLXArray,
    reorderQKV: Bool = false) throws -> MLXArray {
    guard input.ndim == 2 || input.ndim == 3,
      input.shape.last == columns, input.dtype.isFloatingPoint,
      !reorderQKV || (rows == 21504 && columns == 5376) else {
      throw H3CheckpointError.invalid("H3 ConvRot input or QKV layout differs from checkpoint.")
    }
    let count = input.size / columns
    let rotated = matmul(input.asType(.float32)
      .reshaped([count * (columns / group), group]), basis.T)
      .reshaped([count, columns])
    eval(rotated)
    var output = try quantized.project(rotated)
    if let bias { output = output + bias }
    if reorderQKV {
      output = output.reshaped([count, 3, 56, 128])
        .transposed(0, 2, 1, 3).reshaped([count, rows])
    }
    if input.ndim == 3 {
      output = output.reshaped([input.shape[0], input.shape[1], rows])
    }
    output = output.asType(input.dtype)
    eval(output)
    return output
  }
}
