import Foundation
import TensorIO

/// Shared header-only inventory and embedded tokenizer; no weighted inference.
/// Both native text backends reuse these exact checkpoint/tokenizer contracts.
public final class TextCheckpoint {
  public let fixed:SafeTensorFile
  public let layers:[SafeTensorFile]
  public let layerURLs:[URL]
  public let connector:SafeTensorFile
  public let connectorPrefix:String
  public let tokenizer:GemmaTokenizer
  public let types:[String]
  public init(gemmaRoot: URL, connectorURL: URL) throws {
    guard gemmaRoot.isFileURL, connectorURL.isFileURL else { throw TextEncodingError.invalid("Text checkpoints must be local files.") }
    let root = gemmaRoot.resolvingSymlinksInPath().standardizedFileURL
    let manifestURL = root.appendingPathComponent("paged_manifest.json")
    let attributes = try manifestURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard attributes.isRegularFile == true, let size = attributes.fileSize, size <= 1024*1024,
      let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any],
      manifest["format"] as? String == "weetodd-ltx25-gemma-paged-q8-v1",
      manifest["kind"] as? String == "gemma", manifest["bits"] as? Int == 8,
      manifest["group_size"] as? Int == 64, manifest["num_layers"] as? Int == 48,
      let metadata = manifest["metadata"] as? [String: Any],
      let config = metadata["gemma_config"] as? [String: Any],
      let text = config["text_config"] as? [String: Any],
      text["attention_bias"] as? Bool == false, text["hidden_activation"] as? String == "gelu_pytorch_tanh",
      text["vocab_size"] as? Int == 262144,
      text["hidden_size"] as? Int == 3840, text["num_hidden_layers"] as? Int == 48,
      text["intermediate_size"] as? Int == 15360, text["num_attention_heads"] as? Int == 16,
      text["num_key_value_heads"] as? Int == 8, text["num_global_key_value_heads"] as? Int == 1,
      text["head_dim"] as? Int == 256, text["global_head_dim"] as? Int == 512,
      text["num_kv_shared_layers"] as? Int == 0, text["hidden_size_per_layer_input"] as? Int == 0,
      text["enable_moe_block"] as? Bool == false, text["use_double_wide_mlp"] as? Bool == false,
      text["attention_k_eq_v"] as? Bool == true, text["sliding_window"] as? Int == 1024,
      text["rms_norm_eps"] as? Double == 1e-6,
      let types = text["layer_types"] as? [String], types.count == 48,
      types.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }),
      let rope = text["rope_parameters"] as? [String: [String: Any]],
      rope["full_attention"]?["rope_theta"] as? Double == 1000000,
      rope["full_attention"]?["partial_rotary_factor"] as? Double == 0.25,
      rope["full_attention"]?["rope_type"] as? String == "proportional",
      rope["sliding_attention"]?["rope_theta"] as? Double == 10000,
      rope["sliding_attention"]?["rope_type"] as? String == "default",
      let fixedRecord = manifest["fixed"] as? [String: Any],
      let records = manifest["layers"] as? [[String: Any]], records.count == 48 else {
      throw TextEncodingError.invalid("Gemma checkpoint architecture differs from the supported 12B LTX2.5 pack.")
    }
    var paths:Set<String>=[],openedURLs:[URL]=[]
    func open(_ record: [String: Any]) throws -> SafeTensorFile {
      guard let relative = record["file"] as? String, !relative.hasPrefix("/"),
        !relative.split(separator:"/",omittingEmptySubsequences:false).contains(where:{ $0.isEmpty || $0 == "." || $0 == ".." }) else { throw TextEncodingError.invalid("Unsafe Gemma page path.") }
      let url = root.appendingPathComponent(relative).resolvingSymlinksInPath().standardizedFileURL
      guard url.path.hasPrefix(root.path + "/"), paths.insert(url.path).inserted else { throw TextEncodingError.invalid("Gemma page escapes its checkpoint root.") }
      let file = try SafeTensorFile(url: url)
      guard record["tensor_count"] as? Int == file.tensors.count,
        (record["tensor_bytes"] as? NSNumber)?.uint64Value == file.tensors.values.reduce(UInt64(0), { $0 + $1.byteCount }) else {
        throw TextEncodingError.invalid("Gemma page does not match declared tensor counts/bytes.")
      }
      openedURLs.append(url)
      return file
    }
    let fixed = try open(fixedRecord), layers = try records.map(open)
    let fw = TextWeights(file: fixed, prefix: "")
    try Self.validate(fw, shapes: ["model.embed_tokens.weight": [262144,3840],
      "text_embedding_projection.video_aggregate_embed.weight": [4096,188160],
      "text_embedding_projection.video_aggregate_embed.bias": [4096],
      "text_embedding_projection.audio_aggregate_embed.weight": [2048,188160],
      "text_embedding_projection.audio_aggregate_embed.bias": [2048]])
    for index in layers.indices {
      try Task.checkCancellation()
      let w = TextWeights(file: layers[index], prefix: "model.layers.\(index).")
      let shapes = GemmaLayer.expectedShapes(Self.configuration(types[index]))
      try Self.validate(w, shapes: shapes)
      var names = Set(shapes.keys.map { w.prefix + $0 })
      for name in Array(names) where layers[index].tensors[name]?.dtype == "U32" {
        let stem = String(name.dropLast(7)); names.insert(stem + ".scales"); names.insert(stem + ".biases")
      }
      guard names == Set(layers[index].tensors.keys) else { throw TextEncodingError.invalid("Unsupported weights in Gemma layer \(index).") }
    }
    let connector = try SafeTensorFile(url: connectorURL)
    let prefix = connector.tensors["model.diffusion_model.video_embeddings_connector.learnable_registers"] != nil ? "model.diffusion_model." : ""
    for (modality,width) in [("video",4096),("audio",2048)] {
      let stem = prefix + modality + "_embeddings_connector."
      try Self.validate(TextWeights(file: connector, prefix: stem), shapes: ["learnable_registers": [128,width]])
      var admitted = Set([stem + "learnable_registers"])
      for index in 0..<8 {
        let blockPrefix = stem + "transformer_1d_blocks.\(index)."
        admitted.formUnion(TextConnector.expectedShapes(width: width).keys.map { blockPrefix + $0 })
        try Self.validate(TextWeights(file: connector, prefix: stem + "transformer_1d_blocks.\(index)."),
          shapes: TextConnector.expectedShapes(width: width))
      }
      for name in Array(admitted) where connector.tensors[name]?.dtype == "U32" {
        let matrixStem = String(name.dropLast(7)); admitted.insert(matrixStem + ".scales"); admitted.insert(matrixStem + ".biases")
      }
      guard admitted == Set(connector.tensors.keys.filter { $0.hasPrefix(stem) }) else {
        throw TextEncodingError.invalid("Unsupported text connector tensor layout.")
      }
    }
    self.fixed = fixed; self.layers = layers; layerURLs=Array(openedURLs.dropFirst())
    self.connector = connector; connectorPrefix = prefix
    // The embedded tokenizer JSON creates many Foundation parsing temporaries.
    // Drain them before any weighted stage, while retaining the tokenizer itself.
    guard let tokenizerRecord=fixed.tensors["tokenizer_json"], tokenizerRecord.byteCount <= 64*1024*1024,
      let tokenizerConfig=fixed.tensors["hf_asset__tokenizer_config.json"], tokenizerConfig.byteCount <= 1024*1024 else {
      throw TextEncodingError.invalid("Missing or oversized embedded tokenizer assets.")
    }
    tokenizer = try autoreleasepool { try GemmaTokenizer(fixed: fixed) }; self.types = types
  }
  private static func validate(_ weights: TextWeights, shapes: [String: [Int]]) throws {
    for (name, shape) in shapes {
      guard try weights.shape(name) == shape else { throw TextEncodingError.invalid("Text weight shape differs: \(weights.prefix + name)") }
    }
  }
  private static func configuration(_ type: String) -> GemmaLayerConfiguration {
    var c = GemmaLayerConfiguration()
    if type == "full_attention" {
      c.kvHeads = 1; c.headWidth = 512; c.window = nil; c.keyEqualsValue = true
      c.theta = 1000000; c.rotaryFraction = 0.25
    }
    return c
  }
  public func configuration(forLayer index:Int) throws -> GemmaLayerConfiguration {
    guard types.indices.contains(index) else { throw TextEncodingError.invalid("Invalid Gemma layer index.") }
    return Self.configuration(types[index])
  }
}

public enum TextModelLayout {
  public static func gemmaShapes(_ configuration:GemmaLayerConfiguration) throws -> [String:[Int]] {
    try configuration.validate()
    return GemmaLayer.expectedShapes(configuration)
  }
  public static func connectorShapes(width:Int,heads:Int=32) throws -> [String:[Int]] {
    guard (4...8192).contains(width), (1...128).contains(heads), width % heads == 0,
      width/heads % 2 == 0, width/heads <= 512 else { throw TextEncodingError.invalid("Invalid connector shape.") }
    return TextConnector.expectedShapes(width:width,heads:heads)
  }
}
