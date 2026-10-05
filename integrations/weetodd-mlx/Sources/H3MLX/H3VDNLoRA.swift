import Foundation
import MLX
import TensorIO

/// Reads the original named VDN PEFT adapters in place. This is separate from
/// ordinary Comfy-format LoRA admission; no fused weights or converted copy.
final class H3VDNLoRAFile:H3LoRAApplying {
  enum Kind { case standard,turbo }
  private struct Target {
    let native:String
    let source:String
    let rows:Int
    let columns:Int
    let rank:Int
    let slot:Int?
    let swap:Bool
  }
  private let file:SafeTensorFile
  private let url:URL
  private let name:String
  private let targets:[String:[Target]]
  let targetCount:Int
  let modulationTargetCount:Int

  init(directory:URL,kind:Kind) throws {
    let configURL=directory.appendingPathComponent("adapter_config.json")
    let size=(try FileManager.default.attributesOfItem(atPath:configURL.path)[.size] as? NSNumber)?.intValue ?? 0
    guard size > 0,size <= 2*1024*1024,
      let document=try JSONSerialization.jsonObject(with:Data(contentsOf:configURL)) as? [String:Any],
      document["type"] as? String == "lora",document["version"] as? Int == 1,
      let config=document["config"] as? [String:Any],config["rank"] as? Int == 64,
      config["alpha"] as? Int == 64 else {
      throw H3CheckpointError.invalid("Unsupported VDN named PEFT adapter config.")
    }
    let name=kind == .standard ? "default" : "turbo"
    var expected:[Target]=[]
    for group in 0..<52 {
      let source=group < 50 ? "transformer_blocks.\(group)." : "token_refiner.refiner_blocks.\(group-50)."
      let native=group < 50 ? "diffusion_model.blocks.\(group)." : "diffusion_model.token_refiner.blocks.\(group-50)."
      let attention=group < 50 ? "attn.orig." : "attn."
      for (slot,suffix) in ["to_q","to_k","to_v"].enumerated() {
        expected.append(.init(native:native+"attn.qkv_proj",source:source+attention+suffix,
          rows:7168,columns:5376,rank:64,slot:slot,swap:false))
      }
      expected.append(.init(native:native+"attn.out_proj",source:source+attention+"to_out.0",
        rows:5376,columns:7168,rank:64,slot:nil,swap:false))
      if kind == .turbo {
        expected.append(.init(native:native+"mlp.fc1",source:source+"ff.net.0.proj",
          rows:28672,columns:5376,rank:64,slot:nil,swap:true))
        expected.append(.init(native:native+"mlp.fc2",source:source+"ff.net.2",
          rows:5376,columns:14336,rank:64,slot:nil,swap:false))
        if group < 50 {
          expected.append(.init(native:native+"adaln_proj.linear",source:source+"adaln_proj.linear",
            rows:96768,columns:2688,rank:16,slot:nil,swap:false))
        }
      }
    }
    if kind == .turbo {
      expected.append(.init(native:"diffusion_model.final_layer.adaln_proj.linear",source:"norm_out.linear",
        rows:10752,columns:2688,rank:16,slot:nil,swap:false))
      let patterns=Dictionary(uniqueKeysWithValues:expected.filter { $0.rank == 16 }.map { ($0.source,16) })
      let exact=Set(expected.map(\.source))
      guard config["name"] as? String == "turbo",config["exact_targets"] as? Bool == true,
        config["family"] as? String == "larryvrh_v4_step600_ema",
        config["rank_pattern"] as? [String:Int] == patterns,
        config["alpha_pattern"] as? [String:Int] == patterns,
        let requested=config["targets"] as? [String],requested.count == exact.count,Set(requested) == exact else {
        throw H3CheckpointError.invalid("VDN Turbo requires its complete released target/rank/alpha config.")
      }
    } else {
      let patterns:Set<String>=["attn.orig.to_q","attn.orig.to_k","attn.orig.to_v","attn.orig.to_out.0",
        "token_refiner.refiner_blocks.*.attn.to_q","token_refiner.refiner_blocks.*.attn.to_k",
        "token_refiner.refiner_blocks.*.attn.to_v","token_refiner.refiner_blocks.*.attn.to_out.0"]
      guard let requested=config["targets"] as? [String],requested.count == patterns.count,
        Set(requested) == patterns,(config["rank_pattern"] as? [String:Int] ?? [:]).isEmpty,
        (config["alpha_pattern"] as? [String:Int] ?? [:]).isEmpty else {
        throw H3CheckpointError.invalid("VDN original adapter targets differ from the released attention projections.")
      }
    }
    let url=directory.appendingPathComponent("adapter_model.safetensors")
    let file=try SafeTensorFile(url:url)
    guard file.tensors.count == expected.count*2 else {
      throw H3CheckpointError.invalid("VDN named adapter is incomplete or has extra tensors.")
    }
    for target in expected {
      for (projection,shape) in [("A",[target.rank,target.columns]),("B",[target.rows,target.rank])] {
        let key=target.source+".lora_\(projection).\(name).weight"
        guard let item=file.tensors[key],item.dtype == "BF16",item.shape == shape.map(UInt64.init) else {
          throw H3CheckpointError.invalid("Invalid VDN named adapter tensor: \(key)")
        }
      }
    }
    self.url=url;self.file=file;self.name=name
    targets=Dictionary(grouping:expected,by:\.native)
    targetCount=expected.count;modulationTargetCount=expected.filter { $0.rank == 16 }.count
  }

  func apply(base:MLXArray,input:MLXArray,target:String,reorderQKV:Bool=false) throws -> MLXArray {
    guard let selected=targets[target] else { return base }
    guard input.ndim >= 2,input.dtype.isFloatingPoint,base.dtype.isFloatingPoint,
      Array(input.shape.dropLast()) == Array(base.shape.dropLast()),
      input.dim(-1) == selected[0].columns,
      base.dim(-1) == selected[0].rows*(selected[0].slot == nil ? 1 : 3) else {
      throw H3CheckpointError.invalid("VDN LoRA projection shape mismatch; modulation adapters need original 2688-wide coordinates.")
    }
    try Task.checkCancellation();try file.checkUnchanged(at:url)
    func delta(_ target:Target) throws -> MLXArray {
      let aName=target.source+".lora_A.\(name).weight",bName=target.source+".lora_B.\(name).weight"
      let a=try file.withTensorBytes(named:aName) { MLXArray($0,[target.rank,target.columns],type:UInt16.self).view(dtype:.bfloat16) }
      let b=try file.withTensorBytes(named:bName) { MLXArray($0,[target.rows,target.rank],type:UInt16.self).view(dtype:.bfloat16) }
      // Both release configs have alpha == rank. A second scaling factor would
      // silently alter the trained adapter; the required stack uses strength 1.
      let value=matmul(matmul(input.asType(.bfloat16),a.T),b.T)
      eval(value);try Task.checkCancellation();return value
    }
    let correction:MLXArray
    if selected[0].slot != nil {
      guard reorderQKV,selected.count == 3 else {
        throw H3CheckpointError.invalid("VDN split QKV requires native head-major projection ordering.")
      }
      correction=try Self.fuseQKV(selected.sorted { $0.slot! < $1.slot! }.map(delta),heads:56,headWidth:128)
    } else {
      let value=try delta(selected[0])
      correction=try selected[0].swap ? Self.swapSwiGLU(value) : value
    }
    let result=base+correction.asType(base.dtype)
    eval(result);try file.checkUnchanged(at:url);return result
  }

  static func fuseQKV(_ parts:[MLXArray],heads:Int,headWidth:Int) throws -> MLXArray {
    guard parts.count == 3,(1...56).contains(heads),(1...128).contains(headWidth),
      parts[0].ndim >= 2,parts[0].dim(-1) == heads*headWidth,
      parts.allSatisfy({ $0.shape == parts[0].shape && $0.dtype == parts[0].dtype }) else {
      throw H3CheckpointError.invalid("Invalid split VDN QKV correction.")
    }
    let leading=Array(parts[0].shape.dropLast())
    let shaped=parts.map { $0.reshaped(leading+[heads,headWidth]) }
    return stacked(shaped,axis:-2).reshaped(leading+[heads*3*headWidth])
  }

  static func swapSwiGLU(_ value:MLXArray) throws -> MLXArray {
    guard value.ndim >= 1,value.dim(-1) > 0,value.dim(-1).isMultiple(of:2) else {
      throw H3CheckpointError.invalid("VDN SwiGLU correction requires two equally sized halves.")
    }
    let width=value.dim(-1),half=width/2
    return concatenated([value[.ellipsis,half..<width],value[.ellipsis,0..<half]],axis:-1)
  }
}
