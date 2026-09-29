import Darwin
import Foundation

/// Bounded buffered reads avoid repeated VM mappings while a large model is
/// resident. Mapped remains the default for established checkpoint readers.
public enum TensorAccess: Sendable { case mapped, buffered }

public struct TensorDescriptor: Sendable, Equatable {
  public let dtype: String
  public let shape: [UInt64]
  public let byteOffset: UInt64
  public let byteCount: UInt64
}

/// Header-only inspection with scoped, read-only tensor mappings. The caller must keep
/// checkpoint files immutable while they are open. A mapping is valid only inside its closure.
public final class SafeTensorFile {
  public let metadata: [String: String]
  public let tensors: [String: TensorDescriptor]
  public let fileByteCount: UInt64
  private let fd: Int32
  private let payloadOffset: UInt64
  private let identity: FileIdentity

  private struct FileIdentity: Equatable {
    let device: dev_t
    let inode: ino_t
    let size: off_t
    let modifiedSeconds: Int
    let modifiedNanos: Int
    let changedSeconds: Int
    let changedNanos: Int
    init(_ s: stat) {
      device = s.st_dev; inode = s.st_ino; size = s.st_size
      modifiedSeconds = s.st_mtimespec.tv_sec; modifiedNanos = s.st_mtimespec.tv_nsec
      changedSeconds = s.st_ctimespec.tv_sec; changedNanos = s.st_ctimespec.tv_nsec
    }
  }

  private struct Record: Decodable {
    let dtype: String
    let shape: [UInt64]
    let data_offsets: [UInt64]
  }

  private struct Header: Decodable {
    let metadata: [String: String]
    let records: [String: Record]
    struct Key: CodingKey {
      let stringValue: String
      var intValue: Int? { nil }
      init?(intValue: Int) { return nil }
      init(stringValue: String) { self.stringValue = stringValue }
    }
    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: Key.self)
      metadata = try container.decodeIfPresent([String: String].self,
        forKey: Key(stringValue: "__metadata__")) ?? [:]
      var result: [String: Record] = [:]
      for key in container.allKeys where key.stringValue != "__metadata__" {
        result[key.stringValue] = try container.decode(Record.self, forKey: key)
      }
      records = result
    }
  }

  public init(url: URL, maximumHeaderBytes: UInt64 = 64 * 1024 * 1024) throws {
    guard url.isFileURL else { throw CheckpointError.invalid("Checkpoint must be a local file.") }
    let opened = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard opened >= 0 else { throw CheckpointError.invalid("Cannot open checkpoint: \(url.lastPathComponent)") }
    var accepted = false
    defer { if !accepted { Darwin.close(opened) } }
    var status = stat()
    guard fstat(opened, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
          status.st_size >= 10 else { throw CheckpointError.invalid("Checkpoint is not a regular safetensors file.") }
    let size = UInt64(status.st_size)
    let lengthData = try Self.read(opened, offset: 0, count: 8)
    let length = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard length >= 2, length <= maximumHeaderBytes, length <= size - 8,
          length <= UInt64(Int.max) else { throw CheckpointError.invalid("Safetensors header size is invalid or exceeds its limit.") }
    let data = try Self.read(opened, offset: 8, count: Int(length))
    guard data.first == UInt8(ascii: "{") else { throw CheckpointError.invalid("Safetensors header must begin with a JSON object.") }
    let header = try JSONDecoder().decode(Header.self, from: data)
    let widths: [String: UInt64] = ["F64": 8, "F32": 4, "F16": 2, "BF16": 2,
      "I64": 8, "U64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2,
      "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1]
    var parsed: [String: TensorDescriptor] = [:]
    for (name, record) in header.records {
      guard let width = widths[record.dtype], record.shape.count <= 16,
            record.data_offsets.count == 2 else { throw CheckpointError.invalid("Unsupported tensor descriptor: \(name)") }
      var count: UInt64 = record.shape.contains(0) ? 0 : 1
      for dimension in record.shape {
        guard dimension <= UInt64(Int.max) else { throw CheckpointError.invalid("Tensor dimension exceeds addressable memory: \(name)") }
        let result = count.multipliedReportingOverflow(by: dimension)
        guard !result.overflow else { throw CheckpointError.invalid("Tensor shape overflows: \(name)") }
        count = result.partialValue
      }
      let bytes = count.multipliedReportingOverflow(by: width)
      let start = record.data_offsets[0], end = record.data_offsets[1]
      guard !bytes.overflow, end >= start, end - start == bytes.partialValue,
            end <= size - length - 8 else { throw CheckpointError.invalid("Tensor byte range does not match its shape or file: \(name)") }
      parsed[name] = TensorDescriptor(dtype: record.dtype, shape: record.shape,
        byteOffset: start, byteCount: bytes.partialValue)
    }
    var cursor: UInt64 = 0
    let ordered = parsed.values.sorted {
      ($0.byteOffset, $0.byteCount) < ($1.byteOffset, $1.byteCount)
    }
    for tensor in ordered {
      guard tensor.byteOffset == cursor else { throw CheckpointError.invalid("Checkpoint contains overlapping or unindexed tensor bytes.") }
      cursor += tensor.byteCount
    }
    guard cursor == size - length - 8 else { throw CheckpointError.invalid("Checkpoint contains trailing unindexed bytes.") }
    fd = opened; payloadOffset = length + 8; fileByteCount = size
    identity = FileIdentity(status); metadata = header.metadata; tensors = parsed
    // Ownership has transferred to self. A throwing check now invokes deinit;
    // the initializer defer must not close the same descriptor a second time.
    accepted = true
    try checkUnchanged()
  }

  deinit { Darwin.close(fd) }

  public func checkUnchanged() throws {
    var status = stat()
    guard fstat(fd, &status) == 0, FileIdentity(status) == identity else {
      throw CheckpointError.invalid("Checkpoint changed while open; reopen it before inference.")
    }
  }

  /// A native loader opens the pathname independently of this validated file
  /// descriptor. Reject atomic replacement as well as edits to the open inode.
  public func checkUnchanged(at url:URL) throws {
    try checkUnchanged()
    var status=stat()
    guard url.isFileURL,fstatat(AT_FDCWD,url.path,&status,0)==0,FileIdentity(status)==identity else {
      throw CheckpointError.invalid("Checkpoint pathname changed after validation.")
    }
  }

  public func withTensorBytes<T>(named name: String,
    _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
    guard let tensor = tensors[name] else { throw CheckpointError.invalid("Missing tensor: \(name)") }
    return try withTensorBytes(named: name, range: 0..<tensor.byteCount, body)
  }

  /// Map only the requested bytes, so a streaming loader need not materialize a full tensor.
  public func withTensorBytes<T>(named name: String, range: Range<UInt64>,
    _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
    try withTensorBytes(named: name,range: range,access: .mapped,body)
  }
  public func withTensorBytes<T>(named name: String, range: Range<UInt64>,access: TensorAccess,
    _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
    guard let tensor = tensors[name], range.upperBound <= tensor.byteCount else {
      throw CheckpointError.invalid("Tensor slice is missing or outside its payload: \(name)")
    }
    try checkUnchanged()
    let count = range.upperBound - range.lowerBound
    if count == 0 { return try body(UnsafeRawBufferPointer(start: nil, count: 0)) }
    let start = payloadOffset + tensor.byteOffset + range.lowerBound
    if access == .buffered {
      guard count <= 4*1024*1024 else { throw CheckpointError.invalid("Buffered tensor read exceeds 4 MiB window.") }
      try Task.checkCancellation()
      let data = try Self.read(fd,offset: off_t(start),count: Int(count))
      let result = try data.withUnsafeBytes(body)
      try checkUnchanged()
      try Task.checkCancellation()
      return result
    }
    let pageSize = UInt64(getpagesize())
    let alignedStart = start - start % pageSize
    let prefix = start - alignedStart
    let mappedCount = Int(prefix + count) // bounded by the signed file size at inspection
    guard let mapped = mmap(nil, mappedCount, PROT_READ, MAP_PRIVATE, fd, off_t(alignedStart)),
          mapped != MAP_FAILED else { throw CheckpointError.invalid("Cannot map tensor: \(name)") }
    defer { munmap(mapped, mappedCount) }
    let result = try body(UnsafeRawBufferPointer(start: mapped.advanced(by: Int(prefix)), count: Int(count)))
    try checkUnchanged()
    return result
  }

  private static func read(_ fd: Int32, offset: off_t, count: Int) throws -> Data {
    var data = Data(count: count)
    try data.withUnsafeMutableBytes { bytes in
      var done = 0
      while done < count {
        let n = pread(fd, bytes.baseAddress!.advanced(by: done), count - done, offset + off_t(done))
        if n < 0 && errno == EINTR { continue }
        guard n > 0 else { throw CheckpointError.invalid("Checkpoint header is truncated or unreadable.") }
        done += n
      }
    }
    return data
  }
}
