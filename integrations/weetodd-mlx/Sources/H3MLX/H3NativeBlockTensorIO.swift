import Foundation
import MLX
import TensorIO
import Darwin

/// Stage-local transfer to the compiled block worker. Export reuses evaluated
/// MLX storage where contiguous; each write/validation window is at most 4 MiB.
/// Import maps one immutable F32 result and copies directly into MLX storage,
/// avoiding a second full-size Swift Float array.
enum H3NativeBlockTensorIO {
  private static let maximumWindow = 4 * 1024 * 1024

  private static func validateRequest(x: MLXArray, indices: MLXArray, tableRows: Int) throws {
    guard x.ndim == 3, x.shape[0] == 1, (1...40_000).contains(x.shape[1]),
      x.shape[2] == 5376, x.dtype == .bfloat16,
      (1...300).contains(tableRows), indices.shape == [x.shape[1]], indices.dtype == .int32,
      indices.asArray(Int32.self).allSatisfy({ (0..<tableRows).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("Invalid compiled H3 packed input or modulation indices.")
    }
  }

  private static func finite(_ value: MLXArray) throws {
    let data = value.asData(access: .noCopyIfContiguous).data
    try withExtendedLifetime(value) {
      try data.withUnsafeBytes { bytes in
        for start in stride(from: 0, to: bytes.count, by: maximumWindow) {
          try Task.checkCancellation()
          let chunk = UnsafeRawBufferPointer(rebasing: bytes[start..<min(bytes.count, start + maximumWindow)])
          switch value.dtype {
          case .bfloat16:
            guard chunk.bindMemory(to: UInt16.self).allSatisfy({ $0 & 0x7f80 != 0x7f80 }) else {
              throw H3CheckpointError.invalid("Nonfinite compiled H3 BF16 transport input.")
            }
          case .float32:
            guard chunk.bindMemory(to: Float.self).allSatisfy(\.isFinite) else {
              throw H3CheckpointError.invalid("Nonfinite compiled H3 FP32 transport input.")
            }
          case .int32: break
          default: throw H3CheckpointError.invalid("Unsupported compiled H3 transport dtype.")
          }
        }
      }
    }
  }

  private static func write(_ arrays: [String: MLXArray], to url: URL) throws {
    // Reject invalid tensors before a usable transfer file is created.
    for name in arrays.keys.sorted() { try finite(arrays[name]!) }
    var header: [String: SafeTensorStreamWriter.Tensor] = [:]
    for (name, array) in arrays {
      let dtype: String
      switch array.dtype {
      case .bfloat16: dtype = "BF16"
      case .float32: dtype = "F32"
      case .int32: dtype = "I32"
      default: throw H3CheckpointError.invalid("Unsupported compiled H3 transport dtype.")
      }
      header[name] = try .init(dtype: dtype, shape: array.shape.map(UInt64.init))
    }
    let writer = try SafeTensorStreamWriter(url: url, tensors: header)
    for name in arrays.keys.sorted() {
      let array = arrays[name]!, data = array.asData(access: .noCopyIfContiguous).data
      try withExtendedLifetime(array) {
        try data.withUnsafeBytes { bytes in
          for start in stride(from: 0, to: bytes.count, by: maximumWindow) {
            try writer.append(tensor: name, bytes: UnsafeRawBufferPointer(rebasing:
              bytes[start..<min(bytes.count, start + maximumWindow)]))
          }
        }
      }
    }
    _ = try writer.finish()
  }

  static func writeRequest(x: MLXArray, indices: MLXArray, tableRows: Int, to url: URL) throws {
    try validateRequest(x: x, indices: indices, tableRows: tableRows)
    try write(["x": x.reshaped([x.shape[1], 5376]), "indices": indices], to: url)
  }
  static func writeInitial(x: MLXArray, indices: MLXArray, modulations: [MLXArray], angles: H3RotaryAngles, to url: URL) throws {
    guard modulations.count == 50, let first = modulations.first,
      first.ndim == 2, (1...100).contains(first.shape[0]), first.shape[1] == 96768,
      modulations.allSatisfy({ $0.shape == first.shape && $0.dtype == .bfloat16 }),
      x.ndim == 3, angles.rows == x.shape[1],
      angles.cosine.shape == [1, 1, angles.rows, 96], angles.sine.shape == angles.cosine.shape,
      angles.cosine.dtype == .bfloat16, angles.sine.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("Compiled H3 requires all 50 matching BF16 modulation and rotary tables.")
    }
    let rows = first.shape[0] * 3
    try validateRequest(x: x, indices: indices, tableRows: rows)
    var arrays = ["x": x.reshaped([x.shape[1], 5376]), "indices": indices,
      "cos": angles.cosine.reshaped([angles.rows, 1, 96]),
      "sin": angles.sine.reshaped([angles.rows, 1, 96])]
    for block in 0..<50 {
      let table = modulations[block].reshaped([rows, 6 * 5376])
      for mod in 0..<6 { arrays["block\(block).mod\(mod)"] = table[0..<rows, (mod * 5376)..<((mod + 1) * 5376)] }
    }
    try write(arrays, to: url)
  }
  static func readOutput(url: URL, rows: Int) throws -> MLXArray {
    guard url.isFileURL, (1...40_000).contains(rows) else {
      throw H3CheckpointError.invalid("Invalid compiled H3 output location or rows.")
    }
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard fd >= 0 else { throw H3CheckpointError.invalid("Cannot open compiled H3 output.") }
    defer { Darwin.close(fd) }
    var before = stat()
    let count = rows * 5376 * MemoryLayout<Float>.size
    guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG, before.st_size == count else {
      throw H3CheckpointError.invalid("Compiled H3 output must contain exactly the declared FP32 residual.")
    }
    guard let mapping = mmap(nil, count, PROT_READ, MAP_PRIVATE, fd, 0), mapping != MAP_FAILED else {
      throw H3CheckpointError.invalid("Cannot map compiled H3 output.")
    }
    defer { munmap(mapping, count) }
    let bytes = UnsafeRawBufferPointer(start: mapping, count: count)
    for start in stride(from: 0, to: count, by: maximumWindow) {
      try Task.checkCancellation()
      let chunk = UnsafeRawBufferPointer(rebasing: bytes[start..<min(count, start + maximumWindow)])
      guard chunk.bindMemory(to: Float.self).allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("Compiled H3 output contains a nonfinite FP32 residual.")
      }
    }
    let data = Data(bytesNoCopy: mapping, count: count, deallocator: .none)
    let result = MLXArray(data, [1, rows, 5376], dtype: .float32)
    eval(result)
    var after = stat(), named = stat()
    func identity(_ s: stat) -> [Int64] {
      [Int64(s.st_dev), Int64(s.st_ino), s.st_size,
        Int64(s.st_mtimespec.tv_sec), Int64(s.st_mtimespec.tv_nsec),
        Int64(s.st_ctimespec.tv_sec), Int64(s.st_ctimespec.tv_nsec)]
    }
    guard fstat(fd, &after) == 0, Darwin.lstat(url.path, &named) == 0,
      identity(before) == identity(after), identity(before) == identity(named) else {
      throw H3CheckpointError.invalid("Compiled H3 output changed during import.")
    }
    try Task.checkCancellation()
    return result
  }
}
