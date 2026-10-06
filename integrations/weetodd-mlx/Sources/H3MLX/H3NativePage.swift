import Foundation
import MLX
import TensorIO

/// Lazy MLX file handles for one already admitted block. Factors leave this
/// owner as they materialize; unused AdaLN factors never become resident.
final class H3NativePage {
  private let file: SafeTensorFile
  private let url: URL
  private var pending: [String: MLXArray] = [:]
  private var loaded = false
  private var closed = false

  init(file: SafeTensorFile, url: URL) { self.file = file; self.url = url }

  func clear() { pending.removeAll(); closed = true }

  func read(_ name: String) throws -> MLXArray {
    do {
      try Task.checkCancellation()
      guard !closed, let descriptor = file.tensors[name], descriptor.byteCount <= 512 * 1024 * 1024 else {
        throw H3CheckpointError.invalid("Missing or oversized native H3 factor.")
      }
      try file.checkUnchanged(at: url)
      if !loaded {
        pending = try loadArrays(url: url)
        guard Set(pending.keys) == Set(file.tensors.keys) else {
          throw H3CheckpointError.invalid("Native H3 page header changed.")
        }
        loaded = true
      }
      let dtype: DType
      switch descriptor.dtype {
      case "U32": dtype = .uint32
      case "BF16": dtype = .bfloat16
      case "F16": dtype = .float16
      case "F32": dtype = .float32
      default: throw H3CheckpointError.invalid("Unsupported native H3 factor dtype.")
      }
      guard let value = pending.removeValue(forKey: name),
        value.shape == descriptor.shape.map(Int.init), value.dtype == dtype else {
        throw H3CheckpointError.invalid("Native H3 factor shape or dtype changed, or factor was already consumed.")
      }
      eval(value)
      try file.checkUnchanged(at: url)
      try Task.checkCancellation()
      return value
    } catch { clear(); throw error }
  }
}
