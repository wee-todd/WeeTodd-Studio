import Foundation
import InferenceContracts
import MLX
import TensorIO

/// A validated file view; the runner verifies full payload identities once before
/// constructing stage readers. Scoped reads copy bytes into owning MLX arrays.
final class ReCaCheckpoint {
  enum Component { case transformer, flashDecoder, prompt }
  let file: SafeTensorFile
  let asset: ReCaFileIdentity
  let component: Component
  init(asset: ReCaFileIdentity, component: Component, precision: ReCaPrecision = .bf16) throws {
    self.asset = asset; self.component = component
    file = try SafeTensorFile(url: URL(fileURLWithPath: asset.path))
    let shapes = Self.shapes(component: component)
    var expected = Set(shapes.keys)
    let quantNames = precision == .int8ConvRot && component == .transformer ? Self.quantizedNames : []
    for name in quantNames { expected.insert(name + ".weight_scale"); expected.insert(name + ".comfy_quant") }
    guard Set(file.tensors.keys) == expected else { throw ReCaError.invalid("Checkpoint tensor set is not the released ReCaVSR architecture.") }
    for (name, shape) in shapes {
      let quant = name.hasSuffix(".weight") && quantNames.contains(String(name.dropLast(7)))
      let dtype = component == .flashDecoder ? "F32" : quant ? "I8" : "BF16"
      guard let descriptor = file.tensors[name], descriptor.shape == shape.map(UInt64.init), descriptor.dtype == dtype else {
        throw ReCaError.invalid("Incompatible ReCaVSR tensor \(name).")
      }
    }
    for name in quantNames {
      guard file.tensors[name + ".weight_scale"]?.shape == [UInt64(shapes[name + ".weight"]![0]), 1],
        ["F32", "BF16"].contains(file.tensors[name + ".weight_scale"]?.dtype ?? ""),
        file.tensors[name + ".comfy_quant"]?.dtype == "U8",
        (file.tensors[name + ".comfy_quant"]?.byteCount ?? UInt64.max) <= 1024 else {
        throw ReCaError.invalid("Invalid ConvRot scales or format marker.")
      }
      let data = try file.withTensorBytes(named: name + ".comfy_quant") { Data($0) }
      guard let marker = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        marker["format"] as? String == "int8_tensorwise", marker["convrot"] as? Bool == true,
        marker["convrot_groupsize"] as? Int == 256 else { throw ReCaError.invalid("Unsupported ConvRot layout.") }
    }
    try file.checkUnchanged(at: URL(fileURLWithPath: asset.path))
  }
  func tensor(_ name: String) throws -> MLXArray {
    try Task.checkCancellation(); try file.checkUnchanged(at: URL(fileURLWithPath: asset.path))
    guard let d = file.tensors[name] else { throw ReCaError.invalid("Missing ReCaVSR tensor \(name).") }
    let shape = d.shape.map(Int.init)
    let result = try file.withTensorBytes(named: name) { bytes -> MLXArray in
      switch d.dtype {
      case "F32": return MLXArray(bytes, shape, type: Float.self)
      case "BF16": return MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      case "I8": return MLXArray(bytes, shape, type: Int8.self)
      default: throw ReCaError.invalid("Unsupported ReCaVSR tensor dtype.")
      }
    }
    eval(result); return result
  }
  func linear(_ name: String, bias: Bool = true) throws -> ReCaLinear {
    let weight = try tensor(name + ".weight")
    return ReCaLinear(weight: weight, bias: bias ? try tensor(name + ".bias") : nil,
      scale: weight.dtype == .int8 ? try tensor(name + ".weight_scale") : nil)
  }
  func causalConv(_ name: String, temporalStride: Int, spatialStride: Int) throws -> ReCaCausalConv {
    ReCaCausalConv(weight: try tensor(name + ".weight").transposed(0, 2, 3, 4, 1),
      bias: try tensor(name + ".bias"), temporalStride: temporalStride, spatialStride: spatialStride)
  }
  static var quantizedNames: Set<String> {
    Set((0..<30).flatMap { layer in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
      "attn2.to_q", "attn2.to_out.0", "ffn.net.0.proj", "ffn.net.2"].map { "blocks.\(layer)." + $0 } })
  }
  static func shapes(component: Component) -> [String: [Int]] {
    if component == .prompt { return ["positive": [1, 512, 4096]] }
    var s: [String: [Int]] = [:]
    func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = true) {
      s[name + ".weight"] = [output, input]; if bias { s[name + ".bias"] = [output] }
    }
    if component == .flashDecoder {
      linear("latent_projection", 48, 512); linear("temporal_projection", 512, 2048)
      linear("spatial_projection.0", 512, 2048); linear("spatial_projection.2", 2048, 768)
      linear("lr_stem", 768, 48); linear("lr_early_projection", 192, 512, bias: false)
      linear("lr_late_projection", 48, 512, bias: false)
      for (stack, depth) in [("backbone", 12), ("temporal_refinement", 2)] { for i in 0..<depth {
        let p = "\(stack).blocks.\(i)."
        s[p + "attention_norm.weight"] = [512]; s[p + "feed_forward_norm.weight"] = [512]
        for part in ["query", "key", "value", "output"] { linear(p + "attention." + part, 512, ["key", "value"].contains(part) ? 128 : 512) }
        for part in ["key", "value"] { s[p + "attention." + part + "_norm.weight"] = [64] }
        linear(p + "feed_forward.gate", 512, 2048); linear(p + "feed_forward.up", 512, 2048)
        linear(p + "feed_forward.down", 2048, 512)
      } }
      return s
    }
    linear("condition_embedder.text_embedder.linear_1", 4096, 3072)
    linear("condition_embedder.text_embedder.linear_2", 3072, 3072)
    linear("condition_embedder.time_embedder.linear_1", 256, 3072)
    linear("condition_embedder.time_embedder.linear_2", 3072, 3072)
    linear("condition_embedder.time_proj", 3072, 18432)
    linear("lq_proj.out_proj", 3072, 3072); linear("sr_proj", 3072, 3072); linear("proj_out", 3072, 192)
    s["scale_shift_table"] = [1, 2, 3072]
    s["patch_embedding.weight"] = [3072, 48, 1, 2, 2]; s["patch_embedding.bias"] = [3072]
    s["lq_proj.conv1.weight"] = [2048, 768, 4, 3, 3]; s["lq_proj.conv1.bias"] = [2048]
    s["lq_proj.conv2.weight"] = [3072, 2048, 4, 3, 3]; s["lq_proj.conv2.bias"] = [3072]
    s["lq_proj.norm1.weight"] = [2048, 1, 1, 1]; s["lq_proj.norm2.weight"] = [3072, 1, 1, 1]
    for i in 0..<30 {
      let p = "blocks.\(i)."
      for a in ["attn1", "attn2"] {
        for part in ["to_q", "to_k", "to_v", "to_out.0"] { linear(p + a + "." + part, 3072, 3072) }
        for part in ["norm_q", "norm_k"] { s[p + a + "." + part + ".weight"] = [3072] }
      }
      linear(p + "ffn.net.0.proj", 3072, 14336); linear(p + "ffn.net.2", 14336, 3072)
      s[p + "norm2.weight"] = [3072]; s[p + "norm2.bias"] = [3072]
      s[p + "scale_shift_table"] = [1, 6, 3072]
    }
    return s
  }
}
