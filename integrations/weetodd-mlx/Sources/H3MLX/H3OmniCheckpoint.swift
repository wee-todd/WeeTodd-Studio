import CryptoKit
import Darwin
import Foundation
import MLX
import TensorIO

public enum H3OmniVariant: String, Sendable { case affineQ8V1 = "omni-affine-q8-v1" }

/// Explicit execution admission for the pinned native Omni export. Header-only
/// inspection never yields this token. No model parameters are retained here.
public final class H3OmniCheckpoint {
  public static let sourceSHA256 = "44fff6e29adde8ec485b1dddad02d9a134d75c52df2218e19bbf760199080ba6"
  public static let sourceBytes: UInt64 = 37_304_406_432
  public let url: URL
  public let layout: H3CheckpointLayout
  let file: SafeTensorFile
  private let identity: Identity

  private struct Identity: Equatable {
    let device: dev_t, inode: ino_t, bytes: off_t
    let modified: Int, modifiedNS: Int, changed: Int, changedNS: Int
    init(_ status: stat) {
      device = status.st_dev; inode = status.st_ino; bytes = status.st_size
      modified = status.st_mtimespec.tv_sec; modifiedNS = status.st_mtimespec.tv_nsec
      changed = status.st_ctimespec.tv_sec; changedNS = status.st_ctimespec.tv_nsec
    }
    init(url: URL) throws {
      var status = stat()
      guard url.isFileURL, url.path.hasPrefix("/"), lstat(url.path, &status) == 0,
        status.st_mode & S_IFMT == S_IFREG else {
        throw H3CheckpointError.invalid("Omni requires an absolute regular source file, not a symlink.")
      }
      self.init(status)
    }
  }
  private final class Cache: @unchecked Sendable {
    let lock = NSLock()
    var sources: [String: H3OmniCheckpoint] = [:]
  }
  private static let cache = Cache()

  /// Strict names, shapes and source dtypes; metadata is an intrinsic format
  /// discriminator, not evidence that payload bytes match the release digest.
  public static func validate(tensors: [String: H3TensorInfo], metadata: [String: String]) throws -> H3CheckpointLayout {
    let required = ["modified_from": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
      "modified_by": "mlx-serve tests/convert_minimax_h3_weights.py",
      "quantization": "affine 8-bit g64", "quantized_tensors": "310", "dense_tensors": "275"]
    guard required.allSatisfy({ metadata[$0.key] == $0.value }) else {
      throw H3CheckpointError.invalid("Unsupported Omni source identity or affine Q8 recipe.")
    }
    var expected: [String: H3TensorInfo] = [:]
    func dense(_ name: String, _ shape: [UInt64], _ dtype: String = "BF16") {
      expected[name] = H3TensorInfo(dtype: dtype, shape: shape)
    }
    func affine(_ name: String, _ rows: UInt64, _ columns: UInt64) {
      dense(name + ".weight", [rows, columns / 4], "U32")
      dense(name + ".scales", [rows, columns / 64])
      dense(name + ".biases", [rows, columns / 64])
    }
    for (stem, rows, columns) in [("video_patch_proj", 5376, 96),
      ("audio_patch_proj", 5376, 32), ("time_embedder.proj_in", 5376, 256),
      ("time_embedder.proj_out", 2688, 5376), ("final_layer.video_out", 3072, 5376),
      ("final_layer.audio_out", 1024, 5376)] {
      dense(stem + ".weight", [UInt64(rows), UInt64(columns)])
      dense(stem + ".bias", [UInt64(rows)])
    }
    affine("condition_proj", 5376, 5120); dense("condition_proj.bias", [5376])
    affine("final_layer.adaln_proj.linear", 10752, 2688)
    dense("final_layer.adaln_proj.linear.bias", [10752])
    dense("final_layer.norm.weight", [5376])
    dense("token_refiner.final_norm.weight", [5376])
    dense("rope.inv_freq", [16], "F32")
    for index in 0..<52 {
      let body = index < 50
      let root = body ? "blocks.\(index)." : "token_refiner.blocks.\(index - 50)."
      for name in ["norm1.weight", "norm2.weight"] { dense(root + name, [5376]) }
      for name in ["attn.q_norm.weight", "attn.k_norm.weight"] { dense(root + name, [128]) }
      for (stem, rows, columns) in [("attn.qkv_proj", 21504, 5376),
        ("attn.out_proj", 5376, 7168), ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336)] {
        affine(root + stem, UInt64(rows), UInt64(columns))
      }
      if body {
        affine(root + "attn.to_gate_compress", 7168, 5376)
        affine(root + "adaln_proj.linear", 96768, 2688)
        dense(root + "adaln_proj.linear.bias", [96768])
      }
    }
    guard tensors == expected else {
      let mismatch = expected.keys.sorted().first { tensors[$0] != expected[$0] } ?? "unexpected tensor"
      throw H3CheckpointError.invalid("Missing or incompatible Omni tensor: \(mismatch)")
    }
    return H3CheckpointLayout(validatedOmni: .affineQ8V1)
  }

  public static func inspect(url: URL) throws -> H3CheckpointLayout {
    try Task.checkCancellation()
    let before = try Identity(url: url)
    let file = try SafeTensorFile(url: url, maximumHeaderBytes: 1_048_576)
    let layout = try validate(tensors: file.tensors.mapValues {
      H3TensorInfo(dtype: $0.dtype, shape: $0.shape)
    }, metadata: file.metadata)
    try file.checkUnchanged(at: url)
    guard try Identity(url: url) == before else {
      throw H3CheckpointError.invalid("Omni source changed during header inspection.")
    }
    return layout
  }

  /// One streaming SHA256 pass per unchanged source identity per process. Cache
  /// entries contain only metadata/file descriptors and are bounded to four.
  /// Changed entries are evicted and must pass complete digest admission again.
  public static func admit(url: URL) throws -> H3OmniCheckpoint {
    try Task.checkCancellation()
    cache.lock.lock(); defer { cache.lock.unlock() }
    let key = url.standardizedFileURL.path
    if let cached = cache.sources[key] {
      do { try cached.checkUnchanged(); return cached }
      catch is CancellationError { throw CancellationError() }
      catch { cache.sources.removeValue(forKey: key) }
    }
    let source = try H3OmniCheckpoint(url: url)
    if cache.sources.count >= 4 { cache.sources.removeAll() }
    cache.sources[key] = source
    return source
  }

  private init(url: URL) throws {
    self.url = url
    identity = try Identity(url: url)
    guard identity.bytes == Self.sourceBytes else {
      throw H3CheckpointError.invalid("Omni source size differs from the pinned release.")
    }
    file = try SafeTensorFile(url: url, maximumHeaderBytes: 1_048_576)
    layout = try Self.validate(tensors: file.tensors.mapValues {
      H3TensorInfo(dtype: $0.dtype, shape: $0.shape)
    }, metadata: file.metadata)
    try checkUnchanged()
    guard try Self.checkedDigest(file: file, url: url) == Self.sourceSHA256 else {
      throw H3CheckpointError.invalid("Omni source SHA256 differs from the pinned release.")
    }
    try checkUnchanged()
  }

  /// Digest the opened source with bounded host storage. This returns only a
  /// digest, never an execution token; admit() alone compares the fixed pin.
  static func checkedDigest(file: SafeTensorFile, url: URL) throws -> String {
    try Task.checkCancellation()
    let identity = try Identity(url: url)
    try file.checkUnchanged(at: url)
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(handle.fileDescriptor, &status) == 0, Identity(status) == identity else {
      throw H3CheckpointError.invalid("Omni source changed before digest verification.")
    }
    var digest = SHA256(), count: UInt64 = 0
    while true {
      try Task.checkCancellation()
      // Foundation reads may retain autoreleased backing storage until the
      // caller's pool drains. Bound that lifetime to one checksum chunk.
      let consumed = try autoreleasepool {
        let bytes = try handle.read(upToCount: 4 * 1024 * 1024) ?? Data()
        count += UInt64(bytes.count)
        guard count <= file.fileByteCount else { throw H3CheckpointError.invalid("Omni source grew during verification.") }
        digest.update(data: bytes)
        return bytes.count
      }
      if consumed == 0 { break }
    }
    guard fstat(handle.fileDescriptor, &status) == 0, Identity(status) == identity,
      count == file.fileByteCount, try Identity(url: url) == identity else {
      throw H3CheckpointError.invalid("Omni source changed during digest verification.")
    }
    try file.checkUnchanged(at: url)
    try Task.checkCancellation()
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }

  public func checkUnchanged() throws {
    try Task.checkCancellation()
    try file.checkUnchanged(at: url)
    guard try Identity(url: url) == identity else {
      throw H3CheckpointError.invalid("Omni source changed after admission.")
    }
  }

  func withParameters<T>(_ body: (H3OmniParameterScope) throws -> T) throws -> T {
    try checkUnchanged()
    let scope = H3OmniParameterScope(file: file, url: url)
    defer { scope.close() }
    let result = try body(scope)
    try checkUnchanged()
    return result
  }
}

/// One explicitly bounded stage of native parameters. Consumers must evaluate
/// lazy outputs before closing; closure entrypoints guarantee close on unwind.
final class H3OmniParameterScope {
  private let file: SafeTensorFile
  private let url: URL
  private var tensors: [String: MLXArray] = [:]
  private(set) var isClosed = false
  var storageBytes: Int { tensors.values.reduce(0) { $0 + $1.nbytes } }
  init(file: SafeTensorFile, url: URL) { self.file = file; self.url = url }
  deinit { close() }

  func read(_ name: String, rows: Range<Int>? = nil,
    maximumBytes: Int = 512 * 1024 * 1024) throws -> MLXArray {
    try checkUnchanged()
    guard let descriptor = file.tensors[name], !descriptor.shape.isEmpty,
      descriptor.shape[0] > 0, ["BF16", "F32", "U32"].contains(descriptor.dtype) else {
      throw H3CheckpointError.invalid("Missing or unsupported Omni parameter: \(name)")
    }
    let selection = rows ?? 0..<Int(descriptor.shape[0])
    guard selection.lowerBound >= 0, !selection.isEmpty,
      selection.upperBound <= Int(descriptor.shape[0]) else {
      throw H3CheckpointError.invalid("Omni parameter row range is outside its source.")
    }
    let stride = descriptor.byteCount / descriptor.shape[0]
    let range = UInt64(selection.lowerBound) * stride..<UInt64(selection.upperBound) * stride
    guard maximumBytes > 0, maximumBytes <= 512 * 1024 * 1024,
      range.count <= maximumBytes else {
      throw H3CheckpointError.invalid("Omni parameter read exceeds its bounded byte budget.")
    }
    let key = "\(name):\(selection.lowerBound):\(selection.upperBound)"
    if let value = tensors[key] { return value }
    let shape = [selection.count] + descriptor.shape.dropFirst().map(Int.init)
    let value = try file.withTensorBytes(named: name, range: range) { bytes in
      switch descriptor.dtype {
      case "BF16": return MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      case "F32": return MLXArray(bytes, shape, type: Float.self)
      default: return MLXArray(bytes, shape, type: UInt32.self)
      }
    }
    eval(value)
    try checkUnchanged()
    tensors[key] = value
    return value
  }

  func projection(_ stem: String, rows: Range<Int>? = nil) throws -> H3QwenQ8Projection {
    try checkUnchanged()
    let affine = try MLXAffineQ8(file: file, weight: stem + ".weight", groupSize: 64)
    return try H3QwenQ8Projection(packed: read(stem + ".weight", rows: rows),
      scales: read(stem + ".scales", rows: rows), biases: read(stem + ".biases", rows: rows),
      columns: affine.shape[1])
  }

  func project(_ stem: String, input: MLXArray, qkvHeads: Int = 56,
    headWidth: Int = 128) throws -> MLXArray {
    let weight = try projection(stem)
    var result = try weight.project(input)
    if stem.hasSuffix("attn.qkv_proj") {
      result = try Self.headMajorQKV(result, heads: qkvHeads, headWidth: headWidth)
    }
    eval(result)
    try checkUnchanged()
    return result
  }

  /// Reorder the small activation, never decode/repack the large Q8 matrix.
  /// Source rows are Q-all-heads, K-all-heads, V-all-heads for body AND refiner.
  static func headMajorQKV(_ value: MLXArray, heads: Int = 56, headWidth: Int = 128) throws -> MLXArray {
    guard value.ndim >= 2, (1...56).contains(heads), (1...128).contains(headWidth),
      value.shape.last == 3 * heads * headWidth else {
      throw H3CheckpointError.invalid("Invalid Omni QKV projection geometry.")
    }
    let leading = Array(value.shape.dropLast())
    let rank = leading.count
    var axes = Array(0..<(rank + 3)); axes.swapAt(rank, rank + 1)
    return value.reshaped(leading + [3, heads, headWidth]).transposed(axes: axes).reshaped(value.shape)
  }

  func checkUnchanged() throws {
    guard !isClosed else { throw H3CheckpointError.invalid("Omni parameter scope is closed.") }
    try Task.checkCancellation()
    try file.checkUnchanged(at: url)
  }
  func close() {
    guard !isClosed else { return }
    Stream.gpu.synchronize()
    tensors.removeAll()
    isClosed = true
  }
}
