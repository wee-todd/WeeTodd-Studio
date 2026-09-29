import Foundation
import MLX
import TensorIO

/// Decode active Comfy H3 INT8 rows through the checkpoint's inverse ConvRot
/// basis. This creates only transient MLX tensors; installed weights remain in
/// place, and callers can stream bounded row spans before loading another stage.
public enum H3ComfyDecodedProjection {
  private struct Metadata {
    let group: Int
    let scales: [Float]
  }

  private static func metadata(file: SafeTensorFile, name: String,
    rows: Int, columns: Int) throws -> Metadata {
    guard name.hasSuffix(".weight"), rows > 0, columns > 0,
      let weight = file.tensors[name], weight.dtype == "I8",
      weight.shape == [UInt64(rows), UInt64(columns)] else {
      throw H3CheckpointError.invalid("Invalid Comfy H3 INT8 weight: \(name)")
    }
    let stem = String(name.dropLast(".weight".count))
    let scaleName = stem + ".weight_scale"
    let markerName = stem + ".comfy_quant"
    guard file.tensors[scaleName] == nil ? false :
      file.tensors[scaleName]!.dtype == "F32" &&
      file.tensors[scaleName]!.shape == [UInt64(rows), 1],
      let marker = file.tensors[markerName], marker.dtype == "U8",
      marker.byteCount > 0, marker.byteCount <= 4096 else {
      throw H3CheckpointError.invalid("Incomplete Comfy H3 quantization metadata: \(name)")
    }
    let markerData: Data = try file.withTensorBytes(named: markerName) { Data($0) }
    guard let value = try JSONSerialization.jsonObject(with: markerData) as? [String: Any],
      value["format"] as? String == "int8_tensorwise",
      Set(value.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]),
      let rotated = value["convrot"] as? Bool else {
      throw H3CheckpointError.invalid("Unsupported Comfy H3 quantization marker: \(name)")
    }
    let group = rotated ? (value["convrot_groupsize"] as? Int ?? 256) : 0
    guard !rotated || ([4, 16, 64, 256, 1024].contains(group) && columns.isMultiple(of: group)) else {
      throw H3CheckpointError.invalid("Unsupported Comfy H3 ConvRot group: \(name)")
    }
    let scales = try file.readFloat32(named: scaleName)
    guard scales.count == rows, scales.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid Comfy H3 row scales: \(name)")
    }
    return Metadata(group: group, scales: scales)
  }

  static func basis(group: Int) -> MLXArray {
    var entries = [Float](repeating: 0, count: group * group)
    let normalization = Float(group).squareRoot()
    for row in 0..<group {
      for column in 0..<group {
        var sign: Float = 1
        var place = 1
        while place < group {
          let digitA = (row / place) % 4
          let digitB = (column / place) % 4
          if digitA + digitB == 3 { sign = -sign }
          place *= 4
        }
        entries[row * group + column] = sign / normalization
      }
    }
    return MLXArray(entries, [group, group])
  }

  public static func decodeRows(checkpointURL: URL, name: String,
    rows: Int, columns: Int, range: Range<Int>) throws -> MLXArray {
    let file = try SafeTensorFile(url: checkpointURL)
    let metadata = try metadata(file: file, name: name, rows: rows, columns: columns)
    let result = try decodeRows(file: file, name: name, rows: rows,
      columns: columns, range: range, metadata: metadata,
      rotation: metadata.group == 0 ? nil : basis(group: metadata.group))
    try file.checkUnchanged(at: checkpointURL)
    return result
  }

  /// Materialize only the active projection. The 8192-row default balances
  /// installed block latency against temporary MLX allocation; callers must
  /// release this projection before advancing to the next weighted stage.
  public static func load(checkpointURL: URL, name: String, rows: Int,
    columns: Int, reorderQKV: Bool = false,
    rowWindow: Int = 8192) throws -> MLXArray {
    let file = try SafeTensorFile(url: checkpointURL)
    return try load(file: file, checkpointURL: checkpointURL, name: name,
      rows: rows, columns: columns, reorderQKV: reorderQKV,
      rowWindow: rowWindow)
  }

  /// A transformer block already owns a validated file handle. Reuse its
  /// parsed tensor directory for each projection instead of reopening and
  /// reparsing the whole checkpoint header four times per block.
  static func load(file: SafeTensorFile, checkpointURL: URL, name: String,
    rows: Int, columns: Int, reorderQKV: Bool = false,
    rowWindow: Int = 8192) throws -> MLXArray {
    let metadata = try metadata(file: file, name: name, rows: rows, columns: columns)
    guard [1024, 2048, 4096, 8192, 16384].contains(rowWindow) else {
      throw H3CheckpointError.invalid("H3 decode row window is unsupported.")
    }
    guard !reorderQKV || (rows == 3 * 56 * 128 && columns == 5376) else {
      throw H3CheckpointError.invalid("H3 QKV reorder requires the released 56-head layout.")
    }
    let rotation = metadata.group == 0 ? nil : basis(group: metadata.group)
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    var parts: [MLXArray] = []
    parts.reserveCapacity((rows + rowWindow - 1) / rowWindow)
    for start in stride(from: 0, to: rows, by: rowWindow) {
      let stop = min(start + rowWindow, rows)
      parts.append(try decodeRows(file: file, name: name, rows: rows,
        columns: columns, range: start..<stop, metadata: metadata,
        rotation: rotation))
    }
    var value = concatenated(parts, axis: 0)
    if reorderQKV {
      value = value.reshaped([3, 56, 128, columns])
        .transposed(1, 0, 2, 3).reshaped([rows, columns])
    }
    eval(value)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return value
  }

  /// Apply a large row-major projection without ever materializing its full
  /// matrix. Only one bounded weight window and its small output are resident.
  public static func projectStreaming(checkpointURL: URL, name: String,
    rows: Int, columns: Int, input: MLXArray,
    rowWindow: Int = 1024) throws -> MLXArray {
    guard name.hasSuffix(".weight"), input.ndim == 2,
      (1...16).contains(input.shape[0]), input.shape[1] == columns,
      input.dtype == .bfloat16,
      [1024, 2048, 4096, 8192, 16384].contains(rowWindow) else {
      throw H3CheckpointError.invalid("H3 streaming projection requires bounded BF16 input.")
    }
    let file = try SafeTensorFile(url: checkpointURL)
    let metadata = try metadata(file: file, name: name, rows: rows, columns: columns)
    let biasName = String(name.dropLast(".weight".count)) + ".bias"
    guard let bias = file.tensors[biasName], bias.dtype == "BF16",
      bias.shape == [UInt64(rows)] else {
      throw H3CheckpointError.invalid("Missing H3 streaming projection bias: \(biasName)")
    }
    let rotation = metadata.group == 0 ? nil : basis(group: metadata.group)
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    var parts: [MLXArray] = []
    parts.reserveCapacity((rows + rowWindow - 1) / rowWindow)
    for start in stride(from: 0, to: rows, by: rowWindow) {
      let stop = min(start + rowWindow, rows)
      let weight = try decodeRows(file: file, name: name, rows: rows,
        columns: columns, range: start..<stop, metadata: metadata,
        rotation: rotation)
      let biasStart = UInt64(start) * 2
      let biasStop = UInt64(stop) * 2
      let biasChunk = try file.withTensorBytes(named: biasName,
        range: biasStart..<biasStop) { bytes in
        MLXArray(bytes, [stop - start], type: UInt16.self).view(dtype: .bfloat16)
      }
      let output = addMM(biasChunk, input, weight.T)
      eval(output)
      parts.append(output)
      try Task.checkCancellation()
    }
    let result = concatenated(parts, axis: 1)
    eval(result)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return result
  }

  private static func decodeRows(file: SafeTensorFile, name: String,
    rows: Int, columns: Int, range: Range<Int>, metadata: Metadata,
    rotation: MLXArray?) throws -> MLXArray {
    guard range.lowerBound >= 0, range.upperBound <= rows,
      !range.isEmpty, range.count <= 16384 else {
      throw H3CheckpointError.invalid("Comfy H3 decode rows exceed the bounded window.")
    }
    try Task.checkCancellation()
    let lower = UInt64(range.lowerBound) * UInt64(columns)
    let upper = UInt64(range.upperBound) * UInt64(columns)
    let quantized = try file.withTensorBytes(named: name, range: lower..<upper) { bytes in
      MLXArray(bytes, [range.count, columns], type: Int8.self)
    }
    let scales = MLXArray(Array(metadata.scales[range]), [range.count, 1])
    let decoded = quantized.asType(.float32) * scales
    let restored: MLXArray
    if let rotation {
      restored = matmul(decoded.reshaped([-1, metadata.group]), rotation)
        .reshaped([range.count, columns])
    } else {
      restored = decoded
    }
    let result = restored.asType(.bfloat16)
    eval(result)
    try Task.checkCancellation()
    return result
  }
}
