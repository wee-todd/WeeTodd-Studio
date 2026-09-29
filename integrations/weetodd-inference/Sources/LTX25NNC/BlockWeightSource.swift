import Foundation
import LTX25Engine
import TensorIO

/// Header-only preflight of one selected block. Supports released BF16/F32/F16
/// weights and WeeTodd's existing group-64 affine Q8 pages, without conversion.
/// The full selected block must match; tensors from other blocks are not loaded.
public final class BlockWeightSource {
  private enum Storage { case dense(String), q8(MLXAffineQ8) }
  private let file: SafeTensorFile
  private let storage: [String: Storage]
  private let shapes: [String: [Int]]
  public let decodedWeightBytes: UInt64
  public let largestDecodedTensorBytes: UInt64
  public let tensorCount: Int
  public let tensorPayloadBytes: UInt64

  public init(url: URL, blockIndex: Int, expectedShapes: [String: [Int]],
    maximumDecodedWeightBytes: UInt64 = 2 * 1024 * 1024 * 1024,
    requireSingleBlock: Bool = false) throws {
    guard (0..<48).contains(blockIndex), !expectedShapes.isEmpty else {
      throw BlockError.invalid("Expected one LTX 2.5 block (0–47) and a nonempty weight layout.")
    }
    let file = try SafeTensorFile(url: url, maximumHeaderBytes: requireSingleBlock ? 512 * 1024 : 64 * 1024 * 1024)
    let prefix = "transformer_blocks.\(blockIndex)."
    var selected: [String: String] = [:]
    for name in file.tensors.keys {
      guard let normalized = LTXAdapterCompatibility.normalize(name), normalized.hasPrefix(prefix) else { continue }
      let short = String(normalized.dropFirst(prefix.count))
      guard selected.updateValue(name, forKey: short) == nil else {
        throw BlockError.invalid("Checkpoint aliases map to the same block weight: \(short)")
      }
    }
    guard !requireSingleBlock || selected.count == file.tensors.count else {
      throw BlockError.invalid("A block page contains tensors outside its declared block.")
    }
    var storage: [String: Storage] = [:], consumed: Set<String> = []
    var total: UInt64 = 0, largest: UInt64 = 0
    for (name, shape) in expectedShapes {
      guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }), let original = selected[name],
            let tensor = file.tensors[original] else {
        throw BlockError.invalid("Missing or invalid block weight: \(name)")
      }
      var bytes: UInt64 = 4
      for dimension in shape {
        let product = bytes.multipliedReportingOverflow(by: UInt64(dimension))
        guard !product.overflow else { throw BlockError.invalid("Block weight size overflows: \(name)") }
        bytes = product.partialValue
      }
      guard bytes <= 512 * 1024 * 1024, total <= maximumDecodedWeightBytes,
            bytes <= maximumDecodedWeightBytes - total else {
        throw BlockError.invalid("Block weights exceed the admitted decoded-weight budget.")
      }
      total += bytes; largest = max(largest, bytes)
      if tensor.dtype == "U32" {
        let q8 = try MLXAffineQ8(file: file, weight: original, groupSize: 64)
        guard q8.shape == shape else { throw BlockError.invalid("Q8 block weight shape differs: \(name)") }
        storage[name] = .q8(q8)
        let stem = String(name.dropLast(7))
        consumed.formUnion([stem + ".scales", stem + ".biases"])
      } else {
        guard ["BF16", "F16", "F32"].contains(tensor.dtype), tensor.shape == shape.map(UInt64.init) else {
          throw BlockError.invalid("Block weight shape or dtype differs: \(name)")
        }
        storage[name] = .dense(original)
      }
      consumed.insert(name)
    }
    guard consumed == Set(selected.keys) else {
      throw BlockError.invalid("Unsupported tensors in selected block: \(Set(selected.keys).subtracting(consumed).sorted())")
    }
    self.file = file; self.storage = storage; shapes = expectedShapes
    decodedWeightBytes = total; largestDecodedTensorBytes = largest
    tensorCount = file.tensors.count
    tensorPayloadBytes = file.tensors.values.reduce(0) { $0 + $1.byteCount }
  }

  /// Return at most one decoded matrix. The caller owns its lifetime and GPU copy.
  public func read(_ name: String, shape: [Int], decoding: Q8Decoding = .simd) throws -> [Float] {
    guard shapes[name] == shape, let entry = storage[name] else {
      throw BlockError.invalid("Weight request differs from the validated block layout: \(name)")
    }
    try Task.checkCancellation()
    switch entry {
    case .dense(let original): return try file.readFloat32(named: original)
    case .q8(let q8): return try q8.readRows(0..<q8.shape[0], decoding: decoding)
    }
  }
}
