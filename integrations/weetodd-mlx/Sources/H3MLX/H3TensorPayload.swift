import Foundation
import TensorIO

/// Transient H3 payload acquisition. One released I8 row window is assembled
/// from the existing bounded buffered API; larger optional factors stay mapped.
enum H3TensorPayload {
  static let bufferedSpanBytes = 4 * 1024 * 1024
  static let maximumOwnedBytes: UInt64 = 88_080_384

  static func access(byteCount: UInt64, maximumBufferedBytes: UInt64) -> TensorAccess {
    byteCount <= min(maximumBufferedBytes, maximumOwnedBytes) ? .buffered : .mapped
  }

  static func withTensorBytes<T>(file: SafeTensorFile, name: String,
    range: Range<UInt64>? = nil,
    maximumBufferedBytes: UInt64 = maximumOwnedBytes,
    _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
    guard let tensor = file.tensors[name] else {
      throw H3CheckpointError.invalid("Missing H3 tensor payload: \(name)")
    }
    let range = range ?? 0..<tensor.byteCount
    guard range.upperBound <= tensor.byteCount else {
      throw H3CheckpointError.invalid("H3 tensor payload range exceeds its bytes: \(name)")
    }
    try Task.checkCancellation()
    if access(byteCount: range.upperBound - range.lowerBound,
      maximumBufferedBytes: maximumBufferedBytes) == .mapped {
      // Keep previously supported large factors usable without retaining an
      // oversized owned copy or increasing TensorIO's buffered-read cap.
      return try file.withTensorBytes(named: name, range: range, body)
    }
    let owned = try readBuffered(file: file, name: name, range: range)
    let result = try owned.withUnsafeBytes(body)
    try file.checkUnchanged()
    try Task.checkCancellation()
    return result
  }

  /// The optional observer permits focused unwind/cancellation checks between
  /// spans. Ordinary acquisition does not retain or cache the returned bytes.
  static func readBuffered(file: SafeTensorFile, name: String,
    range: Range<UInt64>, afterSpan: ((Int) throws -> Void)? = nil) throws -> Data {
    guard let tensor = file.tensors[name], range.upperBound <= tensor.byteCount,
      range.upperBound - range.lowerBound <= maximumOwnedBytes else {
      throw H3CheckpointError.invalid("H3 owned payload window is missing, out of bounds or oversized.")
    }
    try file.checkUnchanged()
    try Task.checkCancellation()
    let count = Int(range.upperBound - range.lowerBound)
    var owned = Data(count: count)
    try owned.withUnsafeMutableBytes { destination in
      var offset = 0, spans = 0
      while offset < count {
        try Task.checkCancellation()
        let end = min(offset + bufferedSpanBytes, count)
        try file.withTensorBytes(named: name,
          range: (range.lowerBound + UInt64(offset))..<(range.lowerBound + UInt64(end)),
          access: .buffered) { bytes in
          destination.baseAddress!.advanced(by: offset)
            .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        offset = end; spans += 1
        try afterSpan?(spans)
      }
    }
    try file.checkUnchanged()
    try Task.checkCancellation()
    return owned
  }
}
