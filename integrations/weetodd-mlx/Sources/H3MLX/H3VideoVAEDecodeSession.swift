import Foundation
import MLX
import TensorIO

/// Packed decoder weights live only for one video decode stage. This is not a
/// process-wide model cache: close drops all arrays before the audio VAE starts.
final class H3VideoVAEDecodeSession {
  struct Statistics {
    let projectionLoads: Int
    let tensorLoads: Int
    let maximumResidentBytes: Int
    let closed: Bool
    let remainingResidentBytes: Int
  }
  let memoryMode: H3VideoDecodeMemoryMode?
  let checkpointURL: URL
  private let file: SafeTensorFile
  private var tensors: [String: MLXArray] = [:]
  private var projections: [String: H3QwenQ8Projection] = [:]
  private let previousCacheLimit: Int
  private(set) var projectionLoads = 0
  private(set) var tensorLoads = 0
  private(set) var residentBytes = 0
  private var maximumResidentBytes = 0
  private(set) var isClosed = false

  init(checkpointURL: URL, memoryMode: H3VideoDecodeMemoryMode? = nil) throws {
    self.memoryMode = memoryMode
    self.checkpointURL = checkpointURL
    previousCacheLimit = Memory.cacheLimit
    file = try SafeTensorFile(url: checkpointURL)
    _ = try H3VideoVAELayout(file: file, url: checkpointURL)
    // Keep the established decoder allocation-cache bound, once per stage.
    // Active packed weights are separate from MLX's reusable allocation cache.
    Memory.cacheLimit = 128 * 1024 * 1024
  }

  deinit { close() }

  static func withSession<T>(checkpointURL: URL,
    memoryMode: H3VideoDecodeMemoryMode? = nil,
    onClose: (Statistics) -> Void = { _ in },
    _ body: (H3VideoVAEDecodeSession) throws -> T) throws -> T {
    let session = try H3VideoVAEDecodeSession(checkpointURL: checkpointURL, memoryMode: memoryMode)
    defer {
      session.close()
      onClose(Statistics(projectionLoads: session.projectionLoads,
        tensorLoads: session.tensorLoads,
        maximumResidentBytes: session.maximumResidentBytes,
        closed: session.isClosed, remainingResidentBytes: session.residentBytes))
    }
    return try body(session)
  }

  func checkUnchanged() throws {
    guard !isClosed else {
      throw H3CheckpointError.invalid("H3 video decoder session is closed.")
    }
    try file.checkUnchanged(at: checkpointURL)
  }

  func read(_ name: String, shape: [Int]) throws -> MLXArray {
    guard !isClosed, let descriptor = file.tensors[name], descriptor.dtype == "F16",
      descriptor.shape == shape.map(UInt64.init),
      name.hasPrefix("decoder.") || name.hasPrefix("post_quant_conv.") else {
      throw H3CheckpointError.invalid("Missing H3 video decoder session tensor: \(name)")
    }
    if let value = tensors[name] { return value }
    try Task.checkCancellation()
    let value = try file.withTensorBytes(named: name) { bytes in
      MLXArray(bytes, shape, type: Float16.self)
    }
    eval(value)
    tensors[name] = value
    tensorLoads += 1
    residentBytes += value.nbytes
    maximumResidentBytes = max(maximumResidentBytes, residentBytes)
    return value
  }

  func projection(_ name: String) throws -> H3QwenQ8Projection {
    guard !isClosed, name.hasPrefix("decoder.transformer_blocks.") else {
      throw H3CheckpointError.invalid("Invalid H3 video decoder session projection.")
    }
    if let value = projections[name] { return value }
    try Task.checkCancellation()
    let value = try H3QwenQ8Projection(file: file, name: name)
    projections[name] = value
    projectionLoads += 1
    residentBytes += value.storageBytes
    maximumResidentBytes = max(maximumResidentBytes, residentBytes)
    return value
  }

  func materializeProjection(_ value: MLXArray) {
    if H3VideoDecodeMemoryMode.materializesProjection(for: memoryMode) { eval(value) }
  }

  func materializeFirstResidual(_ value: MLXArray) {
    if H3VideoDecodeMemoryMode.materializesFirstResidual(for: memoryMode) { eval(value) }
  }

  func close() {
    guard !isClosed else { return }
    Stream.gpu.synchronize()
    projections.removeAll(keepingCapacity: false)
    tensors.removeAll(keepingCapacity: false)
    residentBytes = 0
    isClosed = true
    Memory.clearCache()
    Memory.cacheLimit = previousCacheLimit
  }
}
