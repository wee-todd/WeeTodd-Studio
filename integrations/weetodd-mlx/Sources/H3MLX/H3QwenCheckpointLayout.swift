import Foundation
import TensorIO

/// Weight-free inspection of the Qwen3-VL-32B state H3 reads after decoder
/// layer 50. The paged text export and original compact Q8 checkpoint stay in
/// their installed locations; this type never copies or dequantizes them.
public struct H3QwenCheckpointLayout: Sendable {
  public let embeddingFile: URL
  public let layerFiles: [URL]
  public let visionFile: URL?
  public var hasVision: Bool { visionFile != nil }

  private struct Manifest: Decodable {
    struct Page: Decodable {
      let file: String
      let sha256: String
      let tensor_bytes: UInt64
      let tensor_count: Int
    }
    let format: String
    let num_layers: Int
    let fixed: Page
    let layers: [Page]
    let vision: Page?
  }

  private static func headers(_ file: SafeTensorFile) -> [String: H3TensorInfo] {
    file.tensors.mapValues { H3TensorInfo(dtype: $0.dtype, shape: $0.shape) }
  }

  private static func validateEmbedding(_ tensors: [String: H3TensorInfo]) throws {
    guard tensors["model.embed_tokens.weight"] == H3TensorInfo(dtype: "BF16", shape: [151936, 5120]) else {
      throw H3CheckpointError.invalid("H3 Qwen requires the 151936×5120 BF16 embedding.")
    }
  }

  static func validateLayer(index: Int, tensors: [String: H3TensorInfo]) throws {
    guard (0..<50).contains(index) else {
      throw H3CheckpointError.invalid("H3 reads exactly 50 Qwen decoder layers.")
    }
    let root = "model.layers.\(index)."
    func require(_ name: String, _ shape: [UInt64]) throws {
      guard let value = tensors[root + name], value.shape == shape,
        ["BF16", "F16"].contains(value.dtype) else {
        throw H3CheckpointError.invalid("Missing H3 Qwen tensor: \(root + name)")
      }
    }
    try require("input_layernorm.weight", [5120])
    try require("post_attention_layernorm.weight", [5120])
    try require("self_attn.q_norm.weight", [128])
    try require("self_attn.k_norm.weight", [128])
    for (name, rows, columns) in [
      ("self_attn.q_proj", 8192, 5120), ("self_attn.k_proj", 1024, 5120),
      ("self_attn.v_proj", 1024, 5120), ("self_attn.o_proj", 5120, 8192),
      ("mlp.gate_proj", 25600, 5120), ("mlp.up_proj", 25600, 5120),
      ("mlp.down_proj", 5120, 25600),
    ] {
      let stem = root + name
      guard tensors[stem + ".weight"] == H3TensorInfo(dtype: "U32",
          shape: [UInt64(rows), UInt64(columns / 4)]),
        let scales = tensors[stem + ".scales"],
        scales.shape == [UInt64(rows), UInt64(columns / 64)],
        ["BF16", "F16", "F32"].contains(scales.dtype),
        tensors[stem + ".biases"] == scales else {
        throw H3CheckpointError.invalid("Incomplete H3 Qwen affine Q8 projection: \(stem)")
      }
    }
  }

  /// Admit the complete installed Q8 vision tower before a visual request.
  /// A pair of recognizable keys is insufficient: late blocks and deepstack
  /// mergers are needed to produce the features injected into text layers.
  static func validateVision(tensors: [String: H3TensorInfo]) throws {
    func require(_ name: String, _ shape: [UInt64], dtype: String = "BF16") throws {
      guard tensors[name] == H3TensorInfo(dtype: dtype, shape: shape) else {
        throw H3CheckpointError.invalid("Missing or incompatible H3 Qwen vision tensor: \(name)")
      }
    }
    func affine(_ stem: String, rows: UInt64, columns: UInt64, bias: Bool = true) throws {
      try require(stem + ".weight", [rows, columns / 4], dtype: "U32")
      try require(stem + ".scales", [rows, columns / 64])
      try require(stem + ".biases", [rows, columns / 64])
      if bias { try require(stem + ".bias", [rows]) }
    }
    try require("visual.patch_embed.proj.weight", [1152, 3, 2, 16, 16])
    try require("visual.patch_embed.proj.bias", [1152])
    try require("visual.pos_embed.weight", [2304, 1152])
    for index in 0..<27 {
      let root = "visual.blocks.\(index)."
      for norm in ["norm1", "norm2"] {
        try require(root + norm + ".weight", [1152])
        try require(root + norm + ".bias", [1152])
      }
      try affine(root + "attn.qkv", rows: 3456, columns: 1152)
      try affine(root + "attn.proj", rows: 1152, columns: 1152)
      try affine(root + "mlp.linear_fc1", rows: 4304, columns: 1152)
      try require(root + "mlp.linear_fc2.weight", [1152, 4304])
      try require(root + "mlp.linear_fc2.bias", [1152])
    }
    for root in ["visual.merger"] + (0..<3).map({ "visual.deepstack_merger_list.\($0)" }) {
      let normWidth: UInt64 = root == "visual.merger" ? 1152 : 4608
      try require(root + ".norm.weight", [normWidth])
      try require(root + ".norm.bias", [normWidth])
      try affine(root + ".linear_fc1", rows: 4608, columns: 4608)
      try affine(root + ".linear_fc2", rows: 5120, columns: 4608)
    }
    guard tensors.keys.filter({ $0.hasPrefix("visual.") }).count == 529 else {
      throw H3CheckpointError.invalid("H3 Qwen vision tower contains unexpected tensor keys.")
    }
  }

  public static func inspect(root: URL) throws -> Self {
    let manifestURL = root.appendingPathComponent("paged_text_encoder_manifest.json")
    if FileManager.default.fileExists(atPath: manifestURL.path) {
      let attributes = try FileManager.default.attributesOfItem(atPath: manifestURL.path)
      guard attributes[.type] as? FileAttributeType == .typeRegular,
        let bytes = attributes[.size] as? NSNumber, bytes.intValue <= 256 * 1024 else {
        throw H3CheckpointError.invalid("Invalid or oversized Qwen page manifest.")
      }
      let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
      let visionPaged = manifest.format == "weetodd-h3-qwen-paged-v2"
      guard ["weetodd-h3-qwen-paged-v1", "weetodd-h3-qwen-paged-v2"].contains(manifest.format),
        manifest.num_layers == 50, manifest.layers.count == 50,
        (manifest.vision != nil) == visionPaged else {
        throw H3CheckpointError.invalid("Qwen page manifest is not a 50-layer H3 export.")
      }
      func page(_ descriptor: Manifest.Page, expected: String, count: Int) throws -> (URL, SafeTensorFile) {
        guard descriptor.file == expected, descriptor.tensor_count == count,
          descriptor.sha256.count == 64,
          descriptor.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
          throw H3CheckpointError.invalid("Qwen page identity differs from its manifest.")
        }
        let url = root.appendingPathComponent(expected)
        let file = try SafeTensorFile(url: url)
        let tensorBytes = file.tensors.values.reduce(UInt64(0)) { $0 + $1.byteCount }
        guard file.tensors.count == count, tensorBytes == descriptor.tensor_bytes else {
          throw H3CheckpointError.invalid("Qwen page header differs from its manifest.")
        }
        try file.checkUnchanged(at: url)
        return (url, file)
      }
      let (embeddingURL, fixed) = try page(manifest.fixed,
        expected: "pages/fixed.safetensors", count: 1)
      try validateEmbedding(headers(fixed))
      var layers: [URL] = []
      for index in 0..<50 {
        let expected = String(format: "pages/layer-%03d.safetensors", index)
        let (url, file) = try page(manifest.layers[index], expected: expected, count: 25)
        try validateLayer(index: index, tensors: headers(file))
        layers.append(url)
      }
      var visionURL: URL?
      if let vision = manifest.vision {
        let (url, file) = try page(vision, expected: "pages/vision.safetensors", count: 529)
        try validateVision(tensors: headers(file))
        visionURL = url
      }
      return Self(embeddingFile: embeddingURL, layerFiles: layers, visionFile: visionURL)
    }

    let fileURL = root.hasDirectoryPath ? root.appendingPathComponent("text_encoder.safetensors") : root
    let file = try SafeTensorFile(url: fileURL)
    let tensors = headers(file)
    try validateEmbedding(tensors)
    for index in 0..<50 { try validateLayer(index: index, tensors: tensors) }
    let vision = tensors.keys.contains(where: { $0.hasPrefix("visual.") })
    if vision { try validateVision(tensors: tensors) }
    try file.checkUnchanged(at: fileURL)
    return Self(embeddingFile: fileURL, layerFiles: [URL](repeating: fileURL, count: 50),
      visionFile: vision ? fileURL : nil)
  }
}
