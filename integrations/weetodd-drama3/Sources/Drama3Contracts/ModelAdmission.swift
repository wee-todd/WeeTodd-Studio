import CryptoKit
import Foundation

public struct Drama3TensorInfo: Codable, Sendable {
  public let dtype: String
  public let shape: [Int]
  public let data_offsets: [Int64]
  public var count: Int64 { shape.reduce(1) { $0 * Int64($1) } }
}
public struct Drama3TensorFile: Sendable {
  public let url: URL
  public let headerBytes: Int64
  public let tensors: [String: Drama3TensorInfo]
  public init(_ url: URL) throws {
    let h = try FileHandle(forReadingFrom: url)
    defer { try? h.close() }
    let prefix = try h.read(upToCount: 8) ?? Data()
    guard prefix.count == 8 else {
      throw Drama3Error.invalid("Invalid tensor file: \(url.lastPathComponent)")
    }
    let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard length > 0 && length <= 16 * 1024 * 1024 else {
      throw Drama3Error.invalid("Tensor header exceeds its bound.")
    }
    let data = try h.read(upToCount: Int(length)) ?? Data()
    guard data.count == length,
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { throw Drama3Error.invalid("Invalid tensor header.") }
    var result: [String: Drama3TensorInfo] = [:]
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    for (key, value) in object where key != "__metadata__" {
      let info = try JSONDecoder().decode(
        Drama3TensorInfo.self, from: JSONSerialization.data(withJSONObject: value))
      guard !info.shape.isEmpty, info.shape.count <= 8,
        info.shape.allSatisfy({ $0 > 0 && $0 <= 2_000_000 }), info.data_offsets.count == 2,
        info.data_offsets[0] >= 0, info.data_offsets[1] >= info.data_offsets[0],
        info.data_offsets[1] <= Int64(size) - Int64(length) - 8
      else { throw Drama3Error.invalid("Invalid shape or tensor offsets for \(key).") }
      guard
        let width = [
          "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2,
          "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
        ][info.dtype]
      else { throw Drama3Error.invalid("Unsupported tensor storage type for \(key).") }
      var count: Int64 = 1
      for dimension in info.shape {
        let product = count.multipliedReportingOverflow(by: Int64(dimension))
        guard !product.overflow else { throw Drama3Error.invalid("Tensor element count overflow.") }
        count = product.partialValue
      }
      let storage = count.multipliedReportingOverflow(by: Int64(width))
      guard !storage.overflow, storage.partialValue == info.data_offsets[1] - info.data_offsets[0]
      else { throw Drama3Error.invalid("Tensor byte count differs from its shape: \(key).") }
      result[key] = info
    }
    self.url = url
    headerBytes = Int64(length) + 8
    tensors = result
  }
  public func rows(_ name: String, indices: [Int]) throws -> Data {
    guard let t = tensors[name], t.shape.count == 2,
      let bytes = ["BF16": 2, "F16": 2, "F32": 4][t.dtype],
      indices.allSatisfy({ $0 >= 0 && $0 < t.shape[0] })
    else { throw Drama3Error.invalid("Invalid output-head rows.") }
    let stride = t.shape[1] * bytes
    let h = try FileHandle(forReadingFrom: url)
    defer { try? h.close() }
    var result = Data()
    for row in indices {
      try h.seek(toOffset: UInt64(headerBytes + t.data_offsets[0]) + UInt64(row * stride))
      let b = try h.read(upToCount: stride) ?? Data()
      guard b.count == stride else { throw Drama3Error.invalid("Truncated head tensor.") }
      result.append(b)
    }
    return result
  }
}
public struct Drama3ModelAdmission: Codable, Sendable {
  public let path: String
  public let fingerprint: String
  public let files: [String]
  public let missing: [String]
  public let modes: [Drama3Mode]
  public let weightBytes: Int64
  public var ready: Bool { missing.isEmpty }
}
public enum Drama3ModelLoader {
  public static func inspect(at root: URL) throws -> Drama3ModelAdmission {
    let cURL = root.appendingPathComponent("config.json")
    guard (try cURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) < 2 * 1024 * 1024,
      let c = try JSONSerialization.jsonObject(with: Data(contentsOf: cURL)) as? [String: Any],
      c["model_type"] as? String == "fish_drama3",
      c["speech_template_version"] as? String == "reference-bracket-v1",
      let text = c["text_config"] as? [String: Any],
      text["hidden_size"] as? Int == 2048, text["num_hidden_layers"] as? Int == 40,
      let tower = c["audio_decoder_config"] as? [String: Any], tower["num_codebooks"] as? Int == 10
    else {
      throw Drama3Error.invalid(
        "Choose a verified Drama 3 MLX checkpoint. Original FP8 and ComfyUI offload layouts need native-compatible setup."
      )
    }
    let children = try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
      options: .skipsHiddenFiles)
    let weights = children.filter {
      $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("model")
    }.sorted { $0.path < $1.path }
    var missing: [String] = []
    var names = Set<String>()
    var tensorInfo: [String: Drama3TensorInfo] = [:]
    var bytes: Int64 = 0
    var fingerprint = Data()
    for file in weights {
      let tensor = try Drama3TensorFile(file)
      for (name, info) in tensor.tensors {
        guard names.insert(name).inserted else {
          throw Drama3Error.invalid("Duplicate tensor \(name).")
        }
        tensorInfo[name] = info
      }
    }
    for file in children.filter({
      $0.pathExtension == "safetensors"
        || ["config.json", "tokenizer.json"].contains($0.lastPathComponent)
    }).sorted(by: { $0.path < $1.path }) {
      let v = try file.resourceValues(forKeys: [
        .fileSizeKey, .contentModificationDateKey, .fileResourceIdentifierKey,
      ])
      bytes += Int64(v.fileSize ?? 0)
      fingerprint.append(
        Data(
          "\(file.lastPathComponent):\(v.fileSize ?? 0):\(v.contentModificationDate?.timeIntervalSince1970 ?? 0):\(String(describing:v.fileResourceIdentifier))\n"
            .utf8))
    }
    let quant = c["quantization"] as? [String: Any] ?? [:]
    func shape(_ name: String, _ expected: [Int], _ dtype: String? = nil) throws {
      guard let info = tensorInfo[name] else {
        missing.append(name)
        return
      }
      guard info.shape == expected, dtype == nil || info.dtype == dtype else {
        throw Drama3Error.invalid("Incompatible tensor shape or type: \(name).")
      }
    }
    func linear(_ name: String, _ output: Int, _ input: Int, _ experts: Bool = false) throws {
      let leading = experts ? [256, output] : [output]
      if tensorInfo[name + ".scales"] != nil {
        let settings = quant[name] as? [String: Any] ?? quant
        guard let bits = settings["bits"] as? Int, [2, 3, 4, 5, 6, 8].contains(bits),
          let group = settings["group_size"] as? Int, [32, 64, 128].contains(group),
          input % group == 0
        else { throw Drama3Error.invalid("Unsupported per-layer quantization for \(name).") }
        try shape(name + ".weight", leading + [input * bits / 32], "U32")
        try shape(name + ".scales", leading + [input / group])
        try shape(name + ".biases", leading + [input / group])
      } else {
        try shape(name + ".weight", leading + [input])
      }
    }
    try shape("backbone.model.embed_tokens.weight", [252288, 2048])
    try shape("backbone.model.norm.weight", [2048])
    try shape("tower.codebook_embeddings.weight", [40960, 2048])
    try linear("semantic_head", 4097, 2048)
    for i in 0..<40 {
      let p = "backbone.model.layers.\(i)"
      try shape(p + ".input_layernorm.weight", [2048])
      try shape(p + ".post_attention_layernorm.weight", [2048])
      if i % 4 == 3 {
        let a = p + ".self_attn"
        try linear(a + ".q_proj", 8192, 2048)
        try linear(a + ".k_proj", 512, 2048)
        try linear(a + ".v_proj", 512, 2048)
        try linear(a + ".o_proj", 2048, 4096)
        try shape(a + ".q_norm.weight", [256])
        try shape(a + ".k_norm.weight", [256])
      } else {
        let a = p + ".linear_attn"
        try linear(a + ".in_proj_qkv", 8192, 2048)
        try linear(a + ".in_proj_z", 4096, 2048)
        try linear(a + ".in_proj_a", 32, 2048)
        try linear(a + ".in_proj_b", 32, 2048)
        try linear(a + ".out_proj", 2048, 4096)
        try shape(a + ".conv1d.weight", [8192, 4, 1])
        try shape(a + ".A_log", [32])
        try shape(a + ".dt_bias", [32])
        try shape(a + ".norm.weight", [128])
      }
      let m = p + ".mlp"
      try linear(m + ".gate", 256, 2048)
      try linear(m + ".shared_expert_gate", 1, 2048)
      for experts in [true, false] {
        let e = m + (experts ? ".switch_mlp" : ".shared_expert")
        try linear(e + ".gate_proj", 512, 2048, experts)
        try linear(e + ".up_proj", 512, 2048, experts)
        try linear(e + ".down_proj", 2048, 512, experts)
      }
    }
    for i in 0..<13 {
      let p = i < 4 ? "tower.shared_layers.\(i)" : "tower.residual_layers.\(i-4)"
      try shape(
        p + (i < 4 ? ".semantic_embedding.weight" : ".codebook_embedding.weight"), [4096, 2048])
      try linear(p + ".eh_proj", 2048, 4096)
      for name in [
        "enorm", "hnorm", "final_layernorm", "transformer_layer.attention_norm",
        "transformer_layer.ffn_norm",
      ] { try shape(p + "." + name + ".weight", [2048]) }
      let a = p + ".transformer_layer.attention"
      try linear(a + ".wqkv", 9216, 2048)
      try linear(a + ".wo", 2048, 4096)
      try shape(a + ".q_norm.weight", [256])
      try shape(a + ".k_norm.weight", [256])
      let f = p + ".transformer_layer.feed_forward"
      try linear(f + ".w1", 6144, 2048)
      try linear(f + ".w3", 6144, 2048)
      try linear(f + ".w2", 2048, 6144)
      if i >= 4 { try linear("tower.codebook_heads.\(i-4)", 4096, 2048) }
    }
    for file in ["tokenizer.json", "codec.safetensors"]
    where !FileManager.default.fileExists(atPath: root.appendingPathComponent(file).path) {
      missing.append(file)
    }
    let codec = root.appendingPathComponent("codec.safetensors")
    if FileManager.default.fileExists(atPath: codec.path) { try Drama3CodecAdmission.inspect(codec) }
    let controls = root.appendingPathComponent("control_head.safetensors")
    var modes: [Drama3Mode] = [.speak, .lineContinuation, .direct, .infill, .streamingText]
    if FileManager.default.fileExists(atPath: controls.path) {
      let control = try Drama3TensorFile(controls)
      if let head = control.tensors["control_head.weight"], head.shape == [1, 2048],
        ["BF16", "F16", "F32"].contains(head.dtype)
      {
        modes += [.scene, .lineSeparated]
      } else {
        throw Drama3Error.invalid("Invalid native scene-control head.")
      }
    }
    return .init(
      path: root.path,
      fingerprint: SHA256.hash(data: fingerprint).map { String(format: "%02x", $0) }.joined(),
      files: weights.map(\.lastPathComponent), missing: missing, modes: modes, weightBytes: bytes)
  }
}
public enum Drama3ControlRows {
  public static func extract(values: [Float], width: Int, rows: [Int]) throws -> [Float] {
    guard width > 0, values.count % width == 0,
      rows.allSatisfy({ $0 >= 0 && $0 < values.count / width })
    else { throw Drama3Error.invalid("Invalid output-head row selection.") }
    return rows.flatMap { Array(values[($0 * width)..<(($0 + 1) * width)]) }
  }
  public static func supplement(source: URL, destination: URL, tokenIDs: [Int]) throws {
    let file = try Drama3TensorFile(source)
    guard let info = file.tensors["lm_head.weight"] else {
      throw Drama3Error.invalid("Source shard has no lm_head.weight.")
    }
    let bytes = try file.rows("lm_head.weight", indices: tokenIDs)
    let metadata: [String: Any] = [
      "control_head.weight": [
        "dtype": info.dtype, "shape": [tokenIDs.count, info.shape[1]],
        "data_offsets": [0, bytes.count],
      ]
    ]
    var header = try JSONSerialization.data(withJSONObject: metadata, options: .sortedKeys)
    while header.count % 8 != 0 { header.append(32) }
    var length = UInt64(header.count).littleEndian
    var output = Data(bytes: &length, count: 8)
    output.append(header)
    output.append(bytes)
    try output.write(to: destination, options: .atomic)
  }
}
