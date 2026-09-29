import Foundation
import LTX25Engine
import TensorIO
import MLX

/// Header-only validation of exactly one installed block page. No model export,
/// CPU dequantization or eager reading of other pages is performed.
public final class MLXBlockSource {
  public let storageBytes: Int
  public var tensorCount:Int { file.tensors.count }
  public var tensorBytes:UInt64 { file.tensors.values.reduce(0) { $0+$1.byteCount } }
  private let file: SafeTensorFile
  private let names: [String:String]
  private let shapes: [String:[Int]]
  private let url:URL
  private let nativeLoading:Bool
  private var pending:[String:MLXArray]=[:]
  var pendingTensorCount:Int { pending.count }
  public init(url:URL,blockIndex:Int,expectedShapes:[String:[Int]],maximumWeightBytes:Int=2*1024*1024*1024,nativeLoading:Bool=true) throws {
    guard (0..<48).contains(blockIndex), !expectedShapes.isEmpty, maximumWeightBytes > 0 else {
      throw LTXError.invalid("Expected one LTX block and a positive storage budget.")
    }
    let file=try SafeTensorFile(url:url,maximumHeaderBytes:512*1024)
    let prefix="transformer_blocks.\(blockIndex)."
    var selected: [String:String]=[:]
    for name in file.tensors.keys {
      guard let full=LTXAdapterCompatibility.normalize(name), full.hasPrefix(prefix) else {
        throw LTXError.invalid("A block page contains tensors outside its selected block.")
      }
      let key=String(full.dropFirst(prefix.count))
      guard selected.updateValue(name,forKey:key) == nil else {
        throw LTXError.invalid("Duplicate normalized weight: \(key)")
      }
    }
    var consumed: Set<String>=[], total: UInt64=0
    for (name,shape) in expectedShapes {
      guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }), let original=selected[name], let d=file.tensors[original] else {
        throw LTXError.invalid("Missing MLX block weight: \(name)")
      }
      var tensorBytes=d.byteCount
      if d.dtype == "U32" {
        let q8=try MLXAffineQ8(file:file,weight:original,groupSize:64)
        guard q8.shape == shape else { throw LTXError.invalid("Q8 shape mismatch: \(name)") }
        let stem=String(name.dropLast(7)), fullStem=String(original.dropLast(7))
        for suffix in [".scales",".biases"] {
          consumed.insert(stem+suffix); tensorBytes += file.tensors[fullStem+suffix]!.byteCount
        }
      } else {
        guard ["F32","F16","BF16"].contains(d.dtype), d.shape == shape.map(UInt64.init) else {
          throw LTXError.invalid("Dense shape or dtype mismatch: \(name)")
        }
      }
      guard tensorBytes <= 512*1024*1024, total <= UInt64(maximumWeightBytes),
        tensorBytes <= UInt64(maximumWeightBytes)-total else {
        throw LTXError.invalid("MLX block exceeds admitted packed-weight budget.")
      }
      total += tensorBytes; consumed.insert(name)
    }
    guard consumed == Set(selected.keys) else { throw LTXError.invalid("Unexpected MLX block weights.") }
    self.file=file; names=selected; shapes=expectedShapes; storageBytes=Int(total)
    self.url=url;self.nativeLoading=nativeLoading
  }
  public func read(_ name:String,shape:[Int],deferEvaluation:Bool=false) throws -> MLXWeight {
    guard shapes[name] == shape, let original=names[name] else { throw LTXError.invalid("Unexpected block weight request.") }
    if nativeLoading {
      do {
        try Task.checkCancellation();try file.checkUnchanged(at:url)
        if pending[original] == nil {
          pending.removeAll()
          pending=try loadArrays(url:url)
          guard Set(pending.keys)==Set(file.tensors.keys) else { throw LTXError.invalid("Native page header changed.") }
        }
        let file=self.file,url=self.url
        let verification:(() throws -> Void)?=deferEvaluation ? { try file.checkUnchanged(at:url) } : nil
        let weight=try MLXWeight(file:file,name:original,shape:shape,verifyMaterialization:verification,tensor:{ key in
          guard let value=self.pending.removeValue(forKey:key) else { throw LTXError.invalid("Missing native packed weight factor.") }
          return value
        })
        // Materialize within the reported load phase. Remaining page handles
        // are lazy; remove every returned factor so readers retain no payload.
        if !deferEvaluation { eval(weight.graphArrays) }
        try file.checkUnchanged(at:url);try Task.checkCancellation()
        return weight
      } catch { pending.removeAll();throw error }
    }
    // A bounded buffered read avoids mapping every packed tensor and its
    // companions again at every diffusion step. Storage and arithmetic stay unchanged.
    return try MLXWeight(file:file,name:original,shape:shape,access:.buffered)
  }
}
