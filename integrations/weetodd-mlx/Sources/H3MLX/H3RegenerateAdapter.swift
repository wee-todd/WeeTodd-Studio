import Foundation
import MLX
import TensorIO

/// Process-local, serial adapter scope. Headers remain open; factors are acquired
/// only for one projection. Merged BF16 weights remain only for the active
/// block so token/query windows do not repeat the FP32 merge.
/// No stage mutates the base or caches a second full transformer.
public final class H3RegenerateAdapter: H3LoRAApplying {
  public enum Stage: Sendable { case draft, regenerate }
  public let profile: H3RegenerateProfile
  public let url: URL
  public private(set) var activeStage: Stage?
  private var file: SafeTensorFile?
  let mergedWeights=H3RegenerateBlockWeightCache()

  public init(url: URL, profile: H3RegenerateProfile,
    companions: [H3RegenerateCompanion]) throws {
    try profile.validateCompanions(companions)
    let file = try SafeTensorFile(url:url)
    let found = try H3RegenerateProfile.admit(metadata:file.metadata,
      tensors:file.tensors.mapValues { .init(dtype:$0.dtype,shape:$0.shape.map(Int.init)) })
    guard found == profile else { throw H3CheckpointError.invalid("Regenerate profile differs from the selected file.") }
    if profile == .extractedRefiner16 {
      for name in file.tensors.keys where name.hasSuffix(".alpha") {
        let alpha = try file.withTensorBytes(named:name) { Float(bitPattern: UInt32($0.loadUnaligned(as:UInt16.self).littleEndian) << 16) }
        guard alpha == 16 else { throw H3CheckpointError.invalid("Reduced refiner requires the released BF16 alpha=16 scalars.") }
      }
    }
    try file.checkUnchanged(at:url)
    self.file=file; self.url=url; self.profile=profile
  }

  /// Always returns to inactive on success, error or cancellation. The caller
  /// should give Sharp the same stage lifetime; draft never uses Sharp deltas.
  public func withStage<T>(_ stage: Stage, _ body: () throws -> T) throws -> T {
    guard file != nil, activeStage == nil else { throw H3CheckpointError.invalid("Closed or already active regenerate adapter.") }
    try Task.checkCancellation()
    mergedWeights.clear();activeStage=stage
    defer { mergedWeights.clear();activeStage=nil }
    let result=try body()
    try Task.checkCancellation()
    return result
  }
  public func close() { mergedWeights.clear();activeStage=nil; file=nil }
  deinit { close() }
  var requiresWeightMerge: Bool { activeStage == .regenerate }

  /// Activation addition is deliberately unavailable for the publisher path:
  /// BF16 post-projection addition does not equal FP32 premerge then cast.
  func apply(base: MLXArray, input: MLXArray, target: String, reorderQKV: Bool) throws -> MLXArray {
    guard file != nil, activeStage == .draft else {
      throw H3CheckpointError.invalid("Regenerate requires an active stage and a weight-merge projection hook.")
    }
    try Task.checkCancellation()
    return base
  }

  func projectMerged(input: MLXArray, target: String, rows: Int, columns: Int,
    baseRows: (Range<Int>) throws -> MLXArray) throws -> MLXArray {
    guard activeStage == .regenerate, let file,
      let entry=H3RegenerateProfile.nativeTargets[target], entry.rows == rows, entry.columns == columns,
      input.ndim >= 2,input.shape.last == columns,input.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("Inactive or mismatched regenerate projection.")
    }
    try file.checkUnchanged(at:url); try Task.checkCancellation()
    let weight=try mergedWeights.weight(block:entry.block,target:target,shape:[rows,columns],dtype:input.dtype) {
    let sources=try profile.sources(for:target)
    let pairs=try sources.map { source -> H3RegenerateProjection.Pair in
      func read(_ suffix: String) throws -> MLXArray {
        let name=source+suffix
        guard let tensor=file.tensors[name] else { throw H3CheckpointError.invalid("Missing regenerate factor.") }
        return try H3TensorPayload.withTensorBytes(file:file,name:name,maximumBufferedBytes:4*1024*1024) { bytes in
          tensor.dtype == "F32" ? MLXArray(bytes,tensor.shape.map(Int.init),type:Float.self)
            : MLXArray(bytes,tensor.shape.map(Int.init),type:UInt16.self).view(dtype:.bfloat16)
        }
      }
      // Official strengths and extracted r384 scale are already folded.
      // Exact r16 alpha=rank was validated separately; never divide again.
      return try .init(a:read(".lora_A.weight"),b:read(".lora_B.weight"),scale:1)
    }
    let layout: H3RegenerateProjection.Layout
    if entry.suffix == "attn.qkv_proj" {
      layout = profile == .publisherThreeStep ? .splitQKV(heads:56,headWidth:128) : .contiguousQKV(heads:56,headWidth:128)
    } else { layout = entry.suffix == "mlp.fc1" && profile == .publisherThreeStep ? .swapSwiGLU : .native }
    return try H3RegenerateProjection.mergedWeight(pairs:pairs,layout:layout,
      rows:rows,columns:columns,dtype:input.dtype,baseRows:baseRows)
    }
    let result=matmul(input,weight.T)
    eval(result)
    try file.checkUnchanged(at:url); try Task.checkCancellation()
    return result
  }
}

/// Small analytic kernels also used by the bounded production projection. The
/// row permutation acts on output rows, never expands the low-rank dimension.
enum H3RegenerateProjection {
  struct Pair { let a: MLXArray; let b: MLXArray; let scale: Float }
  enum Layout { case native, swapSwiGLU, splitQKV(heads:Int,headWidth:Int), contiguousQKV(heads:Int,headWidth:Int) }

  private static func dimensions(_ pairs:[Pair],_ layout:Layout) throws -> (rows:Int,columns:Int) {
    guard let first=pairs.first, first.a.ndim == 2, first.b.ndim == 2,
      first.a.shape.allSatisfy({ $0 > 0 }), first.b.shape.allSatisfy({ $0 > 0 }),
      pairs.allSatisfy({ $0.a.ndim == 2 && $0.b.ndim == 2 && $0.a.shape[1] == first.a.shape[1]
        && $0.a.shape[0] == $0.b.shape[1] && $0.b.shape[0] == first.b.shape[0]
        && $0.a.dtype.isFloatingPoint && $0.b.dtype.isFloatingPoint && $0.scale.isFinite }) else {
      throw H3CheckpointError.invalid("Malformed regenerate factors.")
    }
    let rows:Int
    switch layout {
    case .splitQKV(let heads,let width):
      guard heads > 0, width > 0, pairs.count == 3, first.b.shape[0] == heads*width else {
        throw H3CheckpointError.invalid("Split regenerate QKV shape differs.")
      }
      rows=3*heads*width
    case .contiguousQKV(let heads,let width):
      guard heads > 0, width > 0, pairs.count == 1, first.b.shape[0] == 3*heads*width else {
        throw H3CheckpointError.invalid("Fused regenerate QKV shape differs.")
      }
      rows=first.b.shape[0]
    case .native, .swapSwiGLU:
      guard pairs.count == 1 else { throw H3CheckpointError.invalid("Unexpected regenerate factor count.") }
      rows=first.b.shape[0]
      if case .swapSwiGLU=layout, !rows.isMultiple(of:2) { throw H3CheckpointError.invalid("Odd SwiGLU output width.") }
    }
    return (rows,first.a.shape[1])
  }

  private static func mappedRows(_ range:Range<Int>, rows:Int,layout:Layout) -> [Int32] {
    range.map { row in
      switch layout {
      case .native: return Int32(row)
      case .swapSwiGLU: return Int32((row + rows/2) % rows)
      case .splitQKV(_,let width): return Int32((row/(3*width))*width + row%width)
      case .contiguousQKV(let heads,let width):
        return Int32(((row/width)%3)*heads*width + (row/(3*width))*width + row%width)
      }
    }
  }
  private static func updateRows(_ range:Range<Int>,rows:Int,pairs:[Pair],layout:Layout) -> MLXArray {
    let indices=MLXArray(mappedRows(range,rows:rows,layout:layout))
    var delta:MLXArray?
    for (modality,pair) in pairs.enumerated() {
      var b=take(pair.b,indices,axis:0).asType(.float32)
      if case .splitQKV(_,let width)=layout {
        let mask=MLXArray(range.map { Float(($0/width)%3 == modality ? 1:0) }).reshaped([range.count,1])
        b=b*mask
      }
      let part=matmul(b,pair.a.asType(.float32))*pair.scale
      delta=delta.map { $0+part } ?? part
    }
    return delta!
  }

  /// Comparison path only. Production uses projectMerged to retain cast order.
  static func delta(input:MLXArray,pairs:[Pair],layout:Layout) throws -> MLXArray {
    let d=try dimensions(pairs,layout)
    guard input.ndim >= 2, input.shape.last == d.columns, input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Regenerate activation shape differs.")
    }
    var parts:[MLXArray]=[]
    for pair in pairs {
      parts.append(matmul(matmul(input.asType(.float32),pair.a.asType(.float32).T),pair.b.asType(.float32).T)*pair.scale)
    }
    let contiguous=parts.count == 3 ? concatenated(parts,axis:-1):parts[0]
    let order:Layout
    if case .splitQKV(let heads,let width)=layout { order = .contiguousQKV(heads:heads,headWidth:width) } else { order=layout }
    return take(contiguous,MLXArray(mappedRows(0..<d.rows,rows:d.rows,layout:order)),axis:-1)
  }

  static func merge(base:MLXArray,pairs:[Pair],layout:Layout,rowChunk:Int=512) throws -> MLXArray {
    let d=try dimensions(pairs,layout)
    guard base.shape == [d.rows,d.columns], base.dtype.isFloatingPoint, (1...512).contains(rowChunk) else {
      throw H3CheckpointError.invalid("Invalid regenerate merge window or base matrix.")
    }
    var parts:[MLXArray]=[]
    for start in stride(from:0,to:d.rows,by:rowChunk) {
      try Task.checkCancellation()
      let range=start..<min(start+rowChunk,d.rows)
      let part=(base[range].asType(.float32)+updateRows(range,rows:d.rows,pairs:pairs,layout:layout)).asType(base.dtype)
      eval(part); parts.append(part)
    }
    return concatenated(parts,axis:0)
  }

  static func mergedWeight(pairs:[Pair],layout:Layout,rows:Int,columns:Int,dtype:DType,
    rowChunk:Int=512,baseRows:(Range<Int>) throws -> MLXArray) throws -> MLXArray {
    let d=try dimensions(pairs,layout)
    guard rows == d.rows,columns == d.columns,dtype == .bfloat16,(1...512).contains(rowChunk) else {
      throw H3CheckpointError.invalid("Invalid cached regenerate merge dimensions or precision.")
    }
    var parts:[MLXArray]=[]
    for start in stride(from:0,to:rows,by:rowChunk) {
      try Task.checkCancellation()
      let range=start..<min(start+rowChunk,rows),original=try baseRows(range)
      guard original.shape == [range.count,columns],original.dtype == dtype else {
        throw H3CheckpointError.invalid("Cached regenerate decoded source row differs.")
      }
      let merged=(original.asType(.float32)+updateRows(range,rows:rows,pairs:pairs,layout:layout)).asType(dtype)
      eval(merged);parts.append(merged)
    }
    let result=concatenated(parts,axis:0);eval(result);return result
  }

  static func projectMerged(input:MLXArray,pairs:[Pair],layout:Layout,rows:Int,columns:Int,
    rowChunk:Int=512,baseRows:(Range<Int>) throws -> MLXArray) throws -> MLXArray {
    let d=try dimensions(pairs,layout)
    guard rows == d.rows, columns == d.columns, input.ndim >= 2, input.shape.last == columns,
      input.dtype.isFloatingPoint, (1...512).contains(rowChunk) else {
      throw H3CheckpointError.invalid("Invalid regenerate projection dimensions.")
    }
    var outputs:[MLXArray]=[]
    for start in stride(from:0,to:rows,by:rowChunk) {
      try Task.checkCancellation()
      let range=start..<min(start+rowChunk,rows)
      let original=try baseRows(range)
      guard original.shape == [range.count,columns], original.dtype == input.dtype else {
        throw H3CheckpointError.invalid("Regenerate decoded base row shape or precision differs.")
      }
      let merged=(original.asType(.float32)+updateRows(range,rows:rows,pairs:pairs,layout:layout)).asType(original.dtype)
      let output=matmul(input,merged.T)
      eval(output) // Retire FP32 update and decoded weight window before advancing.
      outputs.append(output)
    }
    return concatenated(outputs,axis:-1)
  }

  /// Defined reuse of affine-Q8 source weights. This preserves that source's
  /// decode, not the original dense publisher checkpoint's numerical identity.
  static func affineRows(parameters:[MLXArray],range:Range<Int>) throws -> MLXArray {
    guard parameters.count == 3, parameters[0].ndim == 2, parameters[0].dtype == .uint32,
      parameters[1].ndim == 2, parameters[1].shape == parameters[2].shape,
      parameters[1].shape[0] == parameters[0].shape[0], parameters[0].shape[1] == parameters[1].shape[1]*16,
      parameters[1].dtype == .bfloat16, parameters[2].dtype == .bfloat16,
      !range.isEmpty,range.lowerBound >= 0,range.upperBound <= parameters[0].shape[0] else {
      throw H3CheckpointError.invalid("Invalid regenerate affine-Q8 source row window.")
    }
    return dequantized(parameters[0][range],scales:parameters[1][range],biases:parameters[2][range],groupSize:64,bits:8,dtype:.bfloat16)
  }
}
