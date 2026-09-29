import Foundation
import LTX25Engine
import AdapterRuntime
import TensorIO

/// Validates the complete selected standard adapters once, retaining only file
/// descriptors and headers. Only the active block's low-rank factors are read.
public final class MLXLoRAStack {
  private struct Entry { let file:SafeTensorFile; let pair:LoRAPlan.Pair }
  private let entries:[Entry]
  public init(adapters:[LoRAAdapter],maximumFactorBytes:Int=512*1024*1024) throws {
    guard adapters.count <= 16, maximumFactorBytes > 0 else { throw LTXError.invalid("Invalid adapter count or factor budget.") }
    var result:[Entry]=[], bytes:[String:UInt64]=[:]
    for adapter in adapters {
      try Task.checkCancellation(); try adapter.validate()
      guard adapter.enabled else { continue }
      let file=try SafeTensorFile(url:URL(fileURLWithPath:adapter.path),maximumHeaderBytes:4*1024*1024)
      let plan=try LTXAdapterCompatibility.standardPlan(file:file,strength:adapter.strength)
      for pair in plan.pairs {
        guard [pair.downTensor,pair.upTensor].allSatisfy({ ["BF16","F16","F32"].contains(file.tensors[$0]!.dtype) }) else {
          throw LTXError.invalid("Swift MLX adapters require BF16/F16/F32 factors.")
        }
        guard pair.scale != 0 else { continue }
        let parts=pair.target.split(separator:".")
        let group=parts.first == "transformer_blocks" ? parts.prefix(2).joined(separator:".") : "fixed"
        let amount=file.tensors[pair.downTensor]!.byteCount + file.tensors[pair.upTensor]!.byteCount
        let prior=bytes[group,default:0]
        guard amount <= UInt64(maximumFactorBytes), prior <= UInt64(maximumFactorBytes)-amount else {
          throw LTXError.invalid("LoRA factors exceed admitted per-block storage before loading weights.")
        }
        bytes[group]=prior+amount; result.append(Entry(file:file,pair:pair))
      }
    }
    entries=result
  }
  public func load(block:Int) throws -> [String:[MLXLoRA]] {
    guard (0..<48).contains(block) else { throw LTXError.invalid("Invalid adapter block index.") }
    let prefix="transformer_blocks.\(block)."
    var result:[String:[MLXLoRA]]=[:]
    for entry in entries where entry.pair.target.hasPrefix(prefix) {
      try Task.checkCancellation()
      let pair=entry.pair
      let down=try MLXWeight.read(entry.file,pair.downTensor,access:.buffered)
      let up=try MLXWeight.read(entry.file,pair.upTensor,access:.buffered)
      let key=String(pair.target.dropFirst(prefix.count))+".weight"
      result[key,default:[]].append(try MLXLoRA(down:down,up:up,strength:pair.scale))
    }
    try Task.checkCancellation()
    return result
  }

  /// A selected adapter must be fully executable by the current model. Header
  /// compatibility alone must not silently omit fixed heads or unused blocks.
  public func validateTargets(_ targets:[String:[Int]]) throws {
    for entry in entries {
      guard let shape=targets[entry.pair.target], shape.map(UInt64.init) == entry.pair.shape else {
        throw LTXError.invalid("Selected model cannot consume LoRA target: \(entry.pair.target)")
      }
    }
  }

  /// Load one fixed projection's factors, not the entire fixed adapter stack.
  public func loadFixed(_ weightName:String) throws -> [MLXLoRA] {
    guard weightName.hasSuffix(".weight"), !weightName.hasPrefix("transformer_blocks.") else {
      throw LTXError.invalid("Expected a fixed projection weight name.")
    }
    let target=String(weightName.dropLast(7))
    guard LTXAdapterCompatibility.targetShapes[target] != nil else {
      throw LTXError.invalid("Unsupported fixed adapter target: \(target)")
    }
    var result:[MLXLoRA]=[]
    for entry in entries where entry.pair.target == target {
      try Task.checkCancellation()
      result.append(try MLXLoRA(
        down:MLXWeight.read(entry.file,entry.pair.downTensor,access:.buffered),
        up:MLXWeight.read(entry.file,entry.pair.upTensor,access:.buffered),strength:entry.pair.scale))
    }
    try Task.checkCancellation()
    return result
  }
}
