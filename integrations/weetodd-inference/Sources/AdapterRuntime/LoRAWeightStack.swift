import Accelerate
import Foundation
import TensorIO

/// Ordered local adapter selection. Disabled entries do not open checkpoints.
public struct LoRAAdapter: Codable, Sendable {
  public let path: String
  public let strength: Float
  public let enabled: Bool
  public init(path: String, strength: Float, enabled: Bool = true) {
    self.path = path; self.strength = strength; self.enabled = enabled
  }
  private enum Key: String, CodingKey { case path, strength, enabled }
  private struct AnyKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  public init(from decoder: Decoder) throws {
    let all = try decoder.container(keyedBy: AnyKey.self)
    guard Set(all.allKeys.map(\.stringValue)).isSubset(of: ["path","strength","enabled"]) else {
      throw AdapterError.invalid("Unsupported LoRA selection field.")
    }
    let c = try decoder.container(keyedBy: Key.self)
    path = try c.decode(String.self,forKey: .path)
    strength = try c.decode(Float.self,forKey: .strength)
    enabled = try c.contains(.enabled) ? c.decode(Bool.self,forKey: .enabled) : true
    try validate()
  }
  public func validate() throws {
    guard path.hasPrefix("/"), !path.utf8.contains(0), path.utf8.count <= 4096, strength.isFinite else {
      throw AdapterError.invalid("LoRA needs an absolute local path and finite strength.")
    }
  }
}

/// Applies B @ A directly into the just-decoded base matrix, in selection order.
/// Only headers persist. No merged checkpoint or full delta matrix is created.
/// The bound covers explicit A/B arrays and a 4 MiB raw read window, not BLAS
/// internal scratch, base weights, header metadata or a process footprint cap.
public final class LoRAWeightStack {
  private struct Entry { let file: SafeTensorFile; let pair: LoRAPlan.Pair }
  private let entries: [String:[Entry]]
  private let rowsPerTile: Int
  private let rowsPerRead: Int
  private let lock = NSLock()
  public let admittedWorkspaceBytes: UInt64
  public let activePairCount: Int

  public init(adapters: [LoRAAdapter], maximumWorkspaceBytes: UInt64 = 64*1024*1024,
    rowsPerTile: Int = 1024,rowsPerRead: Int = 2048,
    plan: (SafeTensorFile,Float) throws -> LoRAPlan) throws {
    guard adapters.count <= 16, (1...1024).contains(rowsPerTile),
      (rowsPerTile...16384).contains(rowsPerRead), rowsPerRead % rowsPerTile == 0 else {
      throw AdapterError.invalid("Use at most 16 LoRAs, compute rows 1...1024, and aligned read rows up to 16384.")
    }
    var entries: [String:[Entry]] = [:]
    var peak: UInt64 = 0, count = 0
    for adapter in adapters {
      try Task.checkCancellation(); try adapter.validate()
      guard adapter.enabled else { continue }
      let file = try SafeTensorFile(url: URL(fileURLWithPath: adapter.path),maximumHeaderBytes: 4*1024*1024)
      let mapping = try plan(file,adapter.strength)
      for pair in mapping.pairs {
        guard pair.shape.allSatisfy({ $0 <= UInt64(Int32.max) }), pair.rank <= UInt64(Int32.max) else {
          throw AdapterError.invalid("LoRA matrix exceeds BLAS dimensions: \(pair.target)")
        }
        // Header compatibility is checked even at zero strength. No payload is
        // needed for a zero delta, and no adapter workspace is reserved for it.
        guard pair.scale != 0 else { continue }
        let elements = pair.rank * (pair.shape[1] + min(UInt64(rowsPerRead),pair.shape[0]))
        guard elements <= (UInt64.max-4*1024*1024)/4,
          elements*4+4*1024*1024 <= maximumWorkspaceBytes else {
          throw AdapterError.invalid("LoRA workspace exceeds admission before loading base weights: \(pair.target)")
        }
        peak = max(peak,elements*4+4*1024*1024)
        if let existing = entries[pair.target]?.first,existing.pair.shape != pair.shape {
          throw AdapterError.invalid("LoRA stack destination shapes disagree: \(pair.target)")
        }
        entries[pair.target,default: []].append(Entry(file: file,pair: pair)); count += 1
      }
    }
    self.entries = entries; self.rowsPerTile = rowsPerTile; self.rowsPerRead = rowsPerRead
    admittedWorkspaceBytes = peak; activePairCount = count
  }

  public func validateTargets(_ shapes: [String:[Int]]) throws {
    for (target,matching) in entries {
      guard let shape = shapes[target],shape.allSatisfy({ $0 > 0 }),
        shape.map(UInt64.init) == matching[0].pair.shape else {
        throw AdapterError.invalid("LoRA target is not consumed by this model configuration: \(target)")
      }
    }
  }

  /// The base loader transfers its array to this call. Failed/cancelled matrices
  /// are never published. Overlapping calls reject rather than multiply scratch.
  public func read(_ name: String,shape: [Int],
    checkCancelled: () throws -> Void = { try Task.checkCancellation() },
    base: () throws -> [Float]) throws -> [Float] {
    guard name.hasSuffix(".weight"),let matching = entries[String(name.dropLast(7))] else {
      try checkCancelled(); return try base()
    }
    guard shape.count == 2, shape.allSatisfy({ $0 > 0 }),
      shape.map(UInt64.init) == matching[0].pair.shape else {
      throw AdapterError.invalid("Unvalidated LoRA destination: \(name)")
    }
    guard lock.try() else { throw AdapterError.invalid("Overlapping LoRA weight preparation is not admitted.") }
    defer { lock.unlock() }
    try checkCancelled()
    for entry in matching { try entry.file.checkUnchanged() }
    var result = try base()
    guard result.count == shape[0]*shape[1],FloatValidation.allFinite(result) else {
      throw AdapterError.invalid("Invalid base matrix for LoRA: \(name)")
    }
    for entry in matching {
      try checkCancelled()
      let pair = entry.pair, rank = Int(entry.pair.rank)
      let down = try Self.read(entry.file,name: pair.downTensor,range: 0..<(pair.rank*pair.shape[1]))
      guard FloatValidation.allFinite(down) else { throw AdapterError.invalid("Nonfinite LoRA down matrix: \(name)") }
      // Read a larger bounded up-factor window once, then retain the same
      // bounded BLAS tiles. This avoids thousands of mmap calls
      // per block while a model is GPU-resident, without a full adapter cache.
      for firstRow in stride(from: 0,to: shape[0],by: rowsPerRead) {
        try checkCancelled()
        let readRows = min(rowsPerRead,shape[0]-firstRow)
        let up = try Self.read(entry.file,name: pair.upTensor,
          range: UInt64(firstRow*rank)..<UInt64((firstRow+readRows)*rank))
        guard FloatValidation.allFinite(up) else { throw AdapterError.invalid("Nonfinite LoRA up matrix: \(name)") }
        for offset in stride(from: 0,to: readRows,by: rowsPerTile) {
          try checkCancelled()
          let rows = min(rowsPerTile,readRows-offset),row = firstRow+offset
          result.withUnsafeMutableBufferPointer { dst in
            down.withUnsafeBufferPointer { a in
              up.withUnsafeBufferPointer { b in
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,Int32(rows),Int32(shape[1]),Int32(rank),
                  pair.scale,b.baseAddress!.advanced(by: offset*rank),Int32(rank),a.baseAddress!,Int32(shape[1]),
                  1,dst.baseAddress!.advanced(by: row*shape[1]),Int32(shape[1]))
              }
            }
          }
        }
      }
      guard FloatValidation.allFinite(result) else {
        throw AdapterError.invalid("LoRA produced nonfinite weights: \(name)")
      }
      try entry.file.checkUnchanged()
    }
    try checkCancelled()
    return result
  }

  private static func read(_ file: SafeTensorFile,name: String,range: Range<UInt64>) throws -> [Float] {
    if file.tensors[name]?.dtype != "F64" {
      return try file.readFloat32(named: name,elements: range,access: .buffered)
    }
    // F64 adapters are structurally supported too. Convert through bounded raw
    // windows, rejecting overflow to Float32 in the caller's finite validation.
    var output = [Float](repeating: 0,count: Int(range.count))
    for start in stride(from: 0,to: output.count,by: 512*1024) {
      try Task.checkCancellation()
      let end = min(start+512*1024,output.count)
      try file.withTensorBytes(named: name,range: (range.lowerBound+UInt64(start))*8..<(range.lowerBound+UInt64(end))*8,access: .buffered) { bytes in
        for i in 0..<(end-start) {
          output[start+i] = Float(Double(bitPattern: bytes.loadUnaligned(fromByteOffset: i*8,as: UInt64.self).littleEndian))
        }
      }
    }
    return output
  }
}
