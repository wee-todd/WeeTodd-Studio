import Foundation
import TensorIO

/// Header-only admission for the released five-block Union and ten-block
/// Union 2.0 H3 Fun branches. Filenames never
/// establish compatibility with a base model or its AdaLN coordinates.
public struct H3FunControlLayout: Sendable {
  public static let v1InjectionLayers = [0, 10, 20, 30, 40]
  public static let v2InjectionLayers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
  public let injectionLayers: [Int]
  public var blockCount: Int { injectionLayers.count }
  public let prefix: String
  public let splitQKV: Bool
  public let timeWidth: Int

  public init(tensors: [String: H3TensorInfo], metadata: [String: String] = [:]) throws {
    let roots = ["model.diffusion_model.", "diffusion_model.", "controlnet.", ""]
      .filter { tensors[$0 + "control_proj_in.weight"] != nil }
    guard roots.count == 1 else {
      throw H3CheckpointError.invalid("Expected one H3 Fun ControlNet namespace.")
    }
    let root = roots[0]
    var expected = Set<String>()
    func require(_ suffix: String, _ shape: [UInt64], quantized: Bool = false) throws {
      let name = root + suffix
      expected.insert(name)
      guard let tensor = tensors[name], tensor.shape == shape,
        ["BF16", "F16", "F32"].contains(tensor.dtype) ||
          (quantized && tensor.dtype == "I8") else {
        throw H3CheckpointError.invalid("Missing or incompatible H3 Fun control tensor: \(suffix)")
      }
      if tensor.dtype == "I8" {
        let stem = String(name.dropLast(".weight".count))
        for metadata in [stem + ".weight_scale", stem + ".comfy_quant"] {
          expected.insert(metadata)
        }
        guard tensors[stem + ".weight_scale"] ==
          H3TensorInfo(dtype: "F32", shape: [shape[0], 1]),
          let marker = tensors[stem + ".comfy_quant"], marker.dtype == "U8",
          marker.shape.count == 1, (1...4096).contains(marker.shape[0]) else {
          throw H3CheckpointError.invalid("Incomplete H3 Fun control quantization metadata.")
        }
      }
    }
    try require("control_proj_in.weight", [5376, 196])
    try require("control_proj_in.bias", [5376])
    guard let adaln = tensors[root + "control_blocks.0.adaln_proj.linear.weight"],
      adaln.shape.count == 2, [UInt64(64), 2688].contains(adaln.shape[1]) else {
      throw H3CheckpointError.invalid("H3 Fun control AdaLN must use 64 curve coordinates or 2688 full-width coordinates.")
    }
    let time = adaln.shape[1]
    let split = tensors[root + "control_blocks.0.attn.to_q.weight"] != nil
    let indices = Set(tensors.keys.compactMap { name -> Int? in
      guard name.hasPrefix(root + "control_blocks.") else { return nil }
      return Int(name.dropFirst((root + "control_blocks.").count).split(separator: ".")[0])
    })
    let layers: [Int]
    switch indices {
    case Set(0..<5): layers = Self.v1InjectionLayers
    case Set(0..<10): layers = Self.v2InjectionLayers
    default: throw H3CheckpointError.invalid("H3 Fun control blocks must match released contiguous 0–4 or 0–9 topology.")
    }
    if let declared = metadata["control_blocks_places"] {
      guard let bytes = declared.data(using: .utf8),
        let places = try? JSONSerialization.jsonObject(with: bytes) as? [NSNumber],
        places.count == layers.count,
        zip(places, layers).allSatisfy({ number, layer in
          CFGetTypeID(number) != CFBooleanGetTypeID() && number.doubleValue == Double(layer)
        }) else {
        throw H3CheckpointError.invalid("H3 Fun control injection metadata differs from the verified topology.")
      }
    }
    for index in 0..<layers.count {
      let block = "control_blocks.\(index)."
      try require(block + "adaln_proj.linear.weight", [96768, time], quantized: true)
      try require(block + "adaln_proj.linear.bias", [96768])
      for name in ["norm1.weight", "norm2.weight"] {
        try require(block + name, [5376])
      }
      for name in split ? ["attn.norm_q.weight", "attn.norm_k.weight"] :
        ["attn.q_norm.weight", "attn.k_norm.weight"] {
        try require(block + name, [128])
      }
      if split {
        for name in ["to_q", "to_k", "to_v"] {
          try require(block + "attn.\(name).weight", [7168, 5376])
        }
      } else {
        try require(block + "attn.qkv_proj.weight", [21504, 5376], quantized: true)
      }
      try require(block + (split ? "attn.to_out.0.weight" : "attn.out_proj.weight"),
        [5376, 7168], quantized: !split)
      try require(block + (split ? "ff.net.0.proj.weight" : "mlp.fc1.weight"),
        [28672, 5376], quantized: !split)
      try require(block + (split ? "ff.net.2.weight" : "mlp.fc2.weight"),
        [5376, 14336], quantized: !split)
      try require(block + "after_proj.weight", [5376, 5376])
      try require(block + "after_proj.bias", [5376])
    }
    try require("control_blocks.0.before_proj.weight", [5376, 5376])
    try require("control_blocks.0.before_proj.bias", [5376])
    guard Set(tensors.keys) == expected else {
      throw H3CheckpointError.invalid("Unsupported H3 Fun control topology or extra checkpoint tensors.")
    }
    prefix = root; splitQKV = split; timeWidth = Int(time); injectionLayers = layers
  }

  public init(url: URL, base: H3CheckpointLayout? = nil) throws {
    guard url.isFileURL, url.path.hasPrefix("/"), url.pathExtension.lowercased() == "safetensors" else {
      throw H3CheckpointError.invalid("H3 Fun control needs a local safetensors checkpoint.")
    }
    let file = try SafeTensorFile(url: url)
    try self.init(tensors: file.tensors.mapValues {
      H3TensorInfo(dtype: $0.dtype, shape: $0.shape)
    }, metadata: file.metadata)
    if let base, timeWidth != (base.curveRank ?? 2688) {
      throw H3CheckpointError.invalid("H3 Fun control and base AdaLN widths differ; use matching converted checkpoints.")
    }
    // Parse INT8 markers during admission, before any weighted stage.
    for (name, descriptor) in file.tensors where descriptor.dtype == "I8" {
      let marker = String(name.dropLast(".weight".count)) + ".comfy_quant"
      let data = try file.withTensorBytes(named: marker) { Data($0) }
      guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        value["format"] as? String == "int8_tensorwise",
        Set(value.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]),
        let rotated = value["convrot"] as? Bool,
        !rotated || ([4, 16, 64, 256, 1024].contains(value["convrot_groupsize"] as? Int ?? 256)
          && descriptor.shape[1].isMultiple(of: UInt64(value["convrot_groupsize"] as? Int ?? 256))) else {
        throw H3CheckpointError.invalid("Unsupported H3 Fun control INT8 marker.")
      }
    }
    try file.checkUnchanged(at: url)
  }
}
