import CryptoKit
import Foundation
import MLX
import TensorIO

/// Sharp has its own 1025×8 basis. It must never consume the native base's
/// independent 1001×64 coordinates, or an approximate inverse of those coordinates.
public final class H3SharpCurve {
  public static let tableSHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
  private let branchesURL: URL
  private var file: SafeTensorFile?
  private var table: MLXArray?

  public init(tableURL:URL,branchesURL:URL) throws {
    let file=try SafeTensorFile(url:branchesURL)
    try Self.validate(metadata:file.metadata,tensors:file.tensors.mapValues { .init(dtype:$0.dtype,shape:$0.shape.map(Int.init)) })
    self.table=try Self.readTable(url:tableURL)
    try file.checkUnchanged(at:branchesURL)
    self.branchesURL=branchesURL; self.file=file
  }
  public func close() { file=nil; table=nil }
  deinit { close() }

  static func validate(metadata:[String:String],tensors:[String:H3RegenerateTensor]) throws {
    guard metadata == ["format":"lms-sharp-curve-adaln","strength":"1.0 folded"],tensors.count == 100 else {
      throw H3CheckpointError.invalid("Sharp requires all 50 released modulation branches and folded unit strength.")
    }
    for block in 0..<50 {
      let stem=branchName(block)
      guard tensors[stem+".curve_A.weight"] == .init(dtype:"F32",shape:[64,8]),
        tensors[stem+".curve_B.weight"] == .init(dtype:"F32",shape:[96768,64]) else {
        throw H3CheckpointError.invalid("Missing or incompatible Sharp modulation branch \(block).")
      }
    }
  }
  private static func branchName(_ block:Int) -> String { "transformer.transformer_blocks.\(block).adaln_proj.linear" }

  static func readTable(url:URL) throws -> MLXArray {
    guard url.isFileURL else { throw H3CheckpointError.invalid("Sharp table must be a local file.") }
    let handle=try FileHandle(forReadingFrom:url)
    defer { try? handle.close() }
    let data=try handle.read(upToCount:1025*8*4+1) ?? Data()
    guard data.count == 1025*8*4,
      SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == tableSHA256 else {
      throw H3CheckpointError.invalid("Sharp requires the pinned 1025×8 F32 curve table identity.")
    }
    let floats=data.withUnsafeBytes { bytes in
      (0..<(1025*8)).map { Float(bitPattern:bytes.loadUnaligned(fromByteOffset:$0*4,as:UInt32.self).littleEndian) }
    }
    guard floats.allSatisfy(\.isFinite) else { throw H3CheckpointError.invalid("Nonfinite Sharp curve table.") }
    return MLXArray(floats,[1025,8])
  }

  public func coordinates(modelTimes:[Float]) throws -> MLXArray {
    guard let table,file != nil else { throw H3CheckpointError.invalid("Sharp curve scope is closed.") }
    return try Self.coordinates(table:table,modelTimes:modelTimes)
  }
  static func coordinates(table:MLXArray,modelTimes:[Float]) throws -> MLXArray {
    guard table.shape == [1025,8],table.dtype == .float32,(1...128).contains(modelTimes.count),
      modelTimes.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
      throw H3CheckpointError.invalid("Sharp requires scalar model times in [0,1] and its independent 1025×8 basis.")
    }
    let positions=modelTimes.map { $0*1024 }
    let indices=positions.map { Int32(min(Int($0.rounded(.down)),1023)) }
    let fraction=MLXArray(zip(positions,indices).map { $0-Float($1) }).reshaped([modelTimes.count,1])
    let low=take(table,MLXArray(indices),axis:0),high=take(table,MLXArray(indices.map { $0+1 }),axis:0)
    return low+(high-low)*fraction
  }

  public func modulationDelta(block:Int,times:[Float]) throws -> MLXArray {
    guard (0..<50).contains(block),let file else { throw H3CheckpointError.invalid("Closed or invalid Sharp branch.") }
    try file.checkUnchanged(at:branchesURL); try Task.checkCancellation()
    let coordinates=try coordinates(modelTimes:times)
    func read(_ suffix:String,_ shape:[Int]) throws -> MLXArray {
      try H3TensorPayload.withTensorBytes(file:file,name:Self.branchName(block)+suffix,maximumBufferedBytes:4*1024*1024) {
        MLXArray($0,shape,type:Float.self)
      }
    }
    let delta=try Self.project(coordinates:coordinates,a:read(".curve_A.weight",[64,8]),b:read(".curve_B.weight",[96768,64]))
    eval(delta)
    try file.checkUnchanged(at:branchesURL); try Task.checkCancellation()
    return delta
  }

  static func project(coordinates:MLXArray,a:MLXArray,b:MLXArray) throws -> MLXArray {
    guard coordinates.ndim == 2,a.ndim == 2,b.ndim == 2,
      coordinates.shape[1] == a.shape[1],a.shape[0] == b.shape[1],
      coordinates.dtype == .float32,a.dtype == .float32,b.dtype == .float32 else {
      throw H3CheckpointError.invalid("Sharp coordinate and modulation projection dimensions differ.")
    }
    return matmul(matmul(coordinates,a.T),b.T)
  }
  /// Output remains [model-time row, modality × six modulation groups × width].
  /// Existing modulationIndices performs row selection after this addition.
  static func add(delta:MLXArray,to base:MLXArray) throws -> MLXArray {
    guard delta.ndim == 2,delta.shape == base.shape,base.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Sharp deltas do not match the base modulation time rows.")
    }
    return base+delta.asType(base.dtype)
  }
}
