import Foundation
import MLX
import TensorIO

/// Header-only admission. Payload access is explicitly one block at a time;
/// neither the backbone nor the four-gigabyte branch is copied or preloaded.
struct H3VDNCheckpoint {
  static let blockShapes:[String:[Int]] = [
    "linear_attention.alpha.A_log":[56],
    "linear_attention.alpha.down.weight":[128,5376],
    "linear_attention.alpha.dt_bias":[7168],
    "linear_attention.alpha.up.weight":[7168,128],
    "linear_attention.beta_proj.weight":[56,5376],
    "linear_attention.norm.weight":[128],
    "linear_attention.output_gate.down.weight":[128,5376],
    "linear_attention.output_gate.up.bias":[7168],
    "linear_attention.output_gate.up.weight":[7168,128],
    "linear_attention.short_conv.k_sp.weight":[7168,1,5,5],
    "linear_attention.short_conv.k_tm.weight":[7168,1,5],
    "linear_attention.short_conv.v_sp.weight":[7168,1,5,5],
    "linear_attention.short_conv.v_tm.weight":[7168,1,5],
    "softmax_gate.up.bias":[56],"softmax_gate.up.weight":[56,5376],
    "to_out_linear.weight":[5376,7168],
  ]
  private let file:SafeTensorFile
  private let url:URL
  let tensorCount:Int

  init(stage:URL) throws {
    let specURL=stage.appendingPathComponent("model_spec.json")
    let size=(try FileManager.default.attributesOfItem(atPath:specURL.path)[.size] as? NSNumber)?.intValue ?? 0
    guard size > 0,size <= 2*1024*1024,
      let spec=try JSONSerialization.jsonObject(with:Data(contentsOf:specURL)) as? [String:Any],
      spec["format_version"] as? Int == 2,
      (spec["base"] as? [String:Any])?["class_name"] as? String == "MiniMaxH3Transformer3DModel",
      let transforms=spec["transforms"] as? [[String:Any]],transforms.count == 1,
      transforms[0]["type"] as? String == "hybrid_attention",transforms[0]["version"] as? Int == 2,
      let config=transforms[0]["config"] as? [String:Any],
      config["anchor_frames"] as? String == "both",config["enable_softmax_gate"] as? Bool == true,
      let linear=config["linear_attention"] as? [String:Any],
      linear["delta_rule"] as? String == "vdn_solve",linear["linear_head_dim"] as? Int == 128,
      linear["enable_text_state"] as? Bool == true,linear["bridge"] as? String == "alpha",
      linear["a_fp32"] as? Bool == true,
      (linear["short_conv"] as? [String:Any])?["targets"] as? [String] == ["k","v"],
      let softmax=config["softmax_attention"] as? [String:Any],
      softmax["chunk"] as? Int == 5,softmax["radius"] as? Int == 1 else {
      throw H3CheckpointError.invalid("Unsupported VDN model specification.")
    }
    let url=stage.appendingPathComponent("linear_branch/model.safetensors")
    let file=try SafeTensorFile(url:url)
    guard file.tensors.count == 800 else {
      throw H3CheckpointError.invalid("VDN requires the complete 800-tensor branch.")
    }
    for index in 0..<50 {
      for (suffix,shape) in Self.blockShapes {
        let name="transformer_blocks.\(index).attn."+suffix
        guard let item=file.tensors[name],item.shape == shape.map(UInt64.init),
          ["BF16","F16","F32"].contains(item.dtype) else {
          throw H3CheckpointError.invalid("Invalid VDN branch tensor: \(name)")
        }
      }
    }
    self.file=file;self.url=url;tensorCount=file.tensors.count
  }

  func readBlock(_ index:Int) throws -> [String:MLXArray] {
    guard (0..<50).contains(index) else { throw H3CheckpointError.invalid("Invalid VDN block index.") }
    try Task.checkCancellation();try file.checkUnchanged(at:url)
    var result:[String:MLXArray]=[:]
    for (suffix,shape) in Self.blockShapes {
      let name="transformer_blocks.\(index).attn."+suffix
      let value=try file.withTensorBytes(named:name) { bytes -> MLXArray in
        switch file.tensors[name]!.dtype {
        case "F32":return MLXArray(bytes,shape,type:Float.self)
        case "F16":return MLXArray(bytes,shape,type:Float16.self)
        default:return MLXArray(bytes,shape,type:UInt16.self).view(dtype:.bfloat16)
        }
      }
      eval(value);result[suffix]=value
      try Task.checkCancellation()
    }
    try file.checkUnchanged(at:url)
    return result
  }
}
