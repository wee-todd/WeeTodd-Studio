import Foundation
import MLX
import LTX25Engine
import TensorIO

/// Unmerged low-rank factors. Evaluation performs (x Aᵀ) Bᵀ on the GPU; no
/// output-width × input-width delta is constructed on the CPU or retained.
public struct MLXLoRA {
  public let down: MLXArray
  public let up: MLXArray
  public let scale: Float
  public init(down: MLXArray, up: MLXArray, strength: Float, alpha: Float? = nil) throws {
    guard down.ndim == 2, up.ndim == 2, down.shape.allSatisfy({ $0 > 0 }),
      up.shape.allSatisfy({ $0 > 0 }), down.shape[0] == up.shape[1],
      down.dtype.isFloatingPoint, up.dtype.isFloatingPoint,
      strength.isFinite, alpha?.isFinite ?? true else {
      throw LTXError.invalid("LoRA requires finite strength/alpha and compatible floating-point factors.")
    }
    let scale = strength * ((alpha ?? Float(down.shape[0])) / Float(down.shape[0]))
    guard scale.isFinite else { throw LTXError.invalid("LoRA scale overflow.") }
    self.down = down; self.up = up; self.scale = scale
  }
}

/// Owns packed MLX arrays copied directly from the existing checkpoint. Dense
/// BF16/F16 tensors retain their original storage type. Not Sendable: each model
/// and stream belongs to one serial worker execution context.
public struct MLXWeight {
  public let shape: [Int]
  public let storageBytes: Int
  private let values: MLXArray
  private let scales: MLXArray?
  private let offsets: MLXArray?
  private var materializationCheck:(() throws -> Void)?
  /// Validated packed storage, passed as explicit compiled graph arguments.
  /// Callers must never retain these after releasing their active block.
  var graphArrays:[MLXArray] {
    if let scales,let offsets { return [values,scales,offsets] }
    return [values]
  }

  public init(dense: MLXArray) throws {
    guard dense.dtype.isFloatingPoint, !dense.shape.isEmpty,
      dense.shape.allSatisfy({ $0 > 0 }), dense.nbytes <= 512*1024*1024 else {
      throw LTXError.invalid("Expected a bounded floating-point tensor.")
    }
    shape = dense.shape; storageBytes = dense.nbytes
    values = dense; scales = nil; offsets = nil
  }

  public init(file: SafeTensorFile, name: String, shape: [Int], access: TensorAccess = .mapped) throws {
    try self.init(file:file,name:name,shape:shape,tensor:{ try Self.read(file,$0,access:access) })
  }

  init(file:SafeTensorFile,name:String,shape:[Int],verifyMaterialization:(() throws -> Void)?=nil,tensor:(String) throws -> MLXArray) throws {
    try Task.checkCancellation()
    func read(_ key:String) throws -> MLXArray {
      guard let descriptor=file.tensors[key] else { throw LTXError.invalid("Missing weight tensor.") }
      let value=try tensor(key)
      let dtype:[String:DType]=["U32":.uint32,"F32":.float32,"F16":.float16,"BF16":.bfloat16]
      guard value.shape==descriptor.shape.map(Int.init),value.dtype==dtype[descriptor.dtype] else {
        throw LTXError.invalid("Native tensor differs from validated checkpoint header.")
      }
      return value
    }
    guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }), let descriptor = file.tensors[name],
      descriptor.byteCount <= 512*1024*1024 else {
      throw LTXError.invalid("Missing or oversized MLX weight: \(name)")
    }
    if descriptor.dtype == "U32" {
      let layout = try MLXAffineQ8(file:file,weight:name,groupSize:64)
      guard shape == layout.shape else { throw LTXError.invalid("Q8 shape mismatch: \(name)") }
      let stem = String(name.dropLast(7))
      values = try read(name)
      scales = try read(stem+".scales")
      offsets = try read(stem+".biases")
      storageBytes = values.nbytes + scales!.nbytes + offsets!.nbytes
    } else {
      guard descriptor.shape == shape.map(UInt64.init), ["F32","F16","BF16"].contains(descriptor.dtype) else {
        throw LTXError.invalid("Dense shape or dtype mismatch: \(name)")
      }
      values = try read(name); scales = nil; offsets = nil
      storageBytes = values.nbytes
    }
    self.shape = shape
    materializationCheck=verifyMaterialization
    try Task.checkCancellation()
  }

  /// Complete one block's native reads together, within the measured load phase.
  /// Check every source before AND after evaluation; no unchecked lazy I/O is
  /// allowed to escape into the later compute phase.
  static func materialize(_ weights:[MLXWeight]) throws {
    try Task.checkCancellation()
    for weight in weights { try weight.materializationCheck?() }
    eval(weights.flatMap(\.graphArrays))
    for weight in weights { try weight.materializationCheck?() }
    try Task.checkCancellation()
  }

  /// Slice contiguous output rows directly from dense or packed affine-Q8
  /// storage. Large embedding/aggregation matrices are never fully mapped or
  /// expanded; companion quantization tensors use the same bounded row range.
  public init(file:SafeTensorFile,name:String,shape:[Int],rows:Range<Int>,maximumBytes:Int=64*1024*1024) throws {
    try Task.checkCancellation()
    guard shape.count == 2, shape.allSatisfy({ $0 > 0 }), !rows.isEmpty,
      rows.lowerBound >= 0, rows.upperBound <= shape[0], (1...512*1024*1024).contains(maximumBytes),
      let d=file.tensors[name] else { throw LTXError.invalid("Invalid sliced weight request.") }
    let isQ8=d.dtype == "U32"
    if isQ8 {
      guard try MLXAffineQ8(file:file,weight:name,groupSize:64).shape == shape else { throw LTXError.invalid("Sliced Q8 shape mismatch.") }
    } else {
      guard d.shape == shape.map(UInt64.init), ["F32","F16","BF16"].contains(d.dtype) else { throw LTXError.invalid("Sliced dense shape/dtype mismatch.") }
    }
    let stem=isQ8 ? String(name.dropLast(7)) : ""
    let keys=isQ8 ? [name,stem+".scales",stem+".biases"] : [name]
    var amount:UInt64=0
    for key in keys {
      let record=file.tensors[key]!
      let bytes=record.byteCount/UInt64(shape[0])*UInt64(rows.count)
      guard bytes <= UInt64(maximumBytes), amount <= UInt64(maximumBytes)-bytes else { throw LTXError.invalid("Sliced weight exceeds byte budget.") }
      amount += bytes
    }
    func slice(_ key:String) throws -> MLXArray {
      let record=file.tensors[key]!, stride=record.byteCount/UInt64(shape[0])
      let selectedShape=[rows.count]+record.shape.dropFirst().map(Int.init)
      return try file.withTensorBytes(named:key,range:UInt64(rows.lowerBound)*stride..<UInt64(rows.upperBound)*stride) { bytes in
        switch record.dtype {
        case "U32": return MLXArray(bytes,selectedShape,type:UInt32.self)
        case "F32": return MLXArray(bytes,selectedShape,type:Float.self)
        case "F16": return MLXArray(bytes,selectedShape,type:Float16.self)
        case "BF16": return MLXArray(bytes,selectedShape,type:UInt16.self).view(dtype:.bfloat16)
        default: throw LTXError.invalid("Unsupported sliced tensor dtype.")
        }
      }
    }
    values=try slice(name)
    scales=isQ8 ? try slice(stem+".scales") : nil
    offsets=isQ8 ? try slice(stem+".biases") : nil
    self.shape=[rows.count,shape[1]]; storageBytes=Int(amount)
    try Task.checkCancellation()
  }

  /// MLX's raw-buffer initializer copies the bytes before the scoped mmap ends.
  /// BF16 is reinterpreted on-device, never expanded to a Swift [Float].
  public static func read(_ file: SafeTensorFile, _ name: String, access:TensorAccess = .mapped) throws -> MLXArray {
    try Task.checkCancellation()
    guard let d = file.tensors[name], d.byteCount <= 512*1024*1024 else {
      throw LTXError.invalid("Missing or oversized MLX tensor: \(name)")
    }
    let shape = d.shape.map(Int.init)
    func array(_ bytes:UnsafeRawBufferPointer) throws -> MLXArray {
      switch d.dtype {
      case "U32": return MLXArray(bytes,shape,type:UInt32.self)
      case "F32": return MLXArray(bytes,shape,type:Float.self)
      case "F16": return MLXArray(bytes,shape,type:Float16.self)
      case "BF16": return MLXArray(bytes,shape,type:UInt16.self).view(dtype:.bfloat16)
      default: throw LTXError.invalid("Unsupported MLX tensor dtype: \(d.dtype)")
      }
    }
    switch access {
    case .mapped: return try file.withTensorBytes(named:name,array)
    case .buffered:
      // At most one packed factor plus a 4 MiB read window. Large LoRA files
      // otherwise spend most load time creating repeated VM mappings.
      var packed=Data(count:Int(d.byteCount))
      try packed.withUnsafeMutableBytes { destination in
        for start in stride(from:0,to:Int(d.byteCount),by:4*1024*1024) {
          let end=min(start+4*1024*1024,Int(d.byteCount))
          try file.withTensorBytes(named:name,range:UInt64(start)..<UInt64(end),access:.buffered) { bytes in
            destination.baseAddress!.advanced(by:start).copyMemory(from:bytes.baseAddress!,byteCount:end-start)
          }
        }
      }
      try Task.checkCancellation()
      return try packed.withUnsafeBytes(array)
    }
  }

  public func tensor() throws -> MLXArray {
    guard scales == nil else { throw LTXError.invalid("Packed Q8 weights must use quantized projection.") }
    return values
  }

  public func projected(_ input: MLXArray, adapters: [MLXLoRA] = []) throws -> MLXArray {
    try Task.checkCancellation()
    guard shape.count == 2, input.ndim >= 2, input.shape.last == shape[1], input.dtype.isFloatingPoint else {
      throw LTXError.invalid("Projection input differs from admitted weight shape.")
    }
    // Validate every adapter before constructing the lazy compute graph.
    for adapter in adapters {
      guard adapter.down.shape[1] == shape[1], adapter.up.shape[0] == shape[0] else {
        throw LTXError.invalid("LoRA factors differ from projection shape.")
      }
    }
    var result: MLXArray
    if let scales, let offsets {
      result = quantizedMM(input,values,scales:scales.asType(input.dtype),
        biases:offsets.asType(input.dtype),groupSize:64,bits:8)
    } else {
      result = matmul(input,values.T)
    }
    for adapter in adapters where adapter.scale != 0 {
      let lowRank = matmul(input,adapter.down.asType(input.dtype).T)
      result = result + matmul(lowRank,adapter.up.asType(input.dtype).T) * adapter.scale
    }
    return result
  }
}
