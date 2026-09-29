import Foundation
import AdapterRuntime
import XCTest
import InferenceTestSupport
@testable import LTX25NNC

final class PagedBlockWeightsTests: XCTestCase {
  private func fixture(denoiser: Bool = false, _ body: (URL, [String: Any], AVBlockConfiguration) throws -> Void) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let config = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let shapes = try AVBlockRunner.expectedWeightShapes(configuration: config)
    var records: [[String: Any]] = []
    for index in 0..<48 {
      let filename = "layer-\(index).safetensors"
      let tensors = shapes.map { ("model.diffusion_model.transformer_blocks.\(index)." + $0.key, $0.value, "F32") }
      try withTensorFile(tensors: tensors) { source in
        try FileManager.default.copyItem(at: source, to: root.appendingPathComponent(filename))
      }
      records.append(["file": filename, "tensor_count": shapes.count,
        "tensor_bytes": shapes.values.reduce(0) { $0 + $1.reduce(4, *) }, "sha256": String(repeating: "0", count: 64)])
    }
    var architecture: [String: Any] = ["num_layers": 48, "num_attention_heads": 2,
      "audio_num_attention_heads": 2, "attention_head_dim": 16, "audio_attention_head_dim": 8,
      "cross_attention_dim": 32, "audio_cross_attention_dim": 16, "ff_bias": false,
      "apply_gated_attention": true, "cross_attention_adaln": true, "use_audio_video_cross_attention": true,
      "rope_type": "split", "qk_norm": "rms_norm", "norm_eps": 1e-6, "activation_fn": "gelu-approximate",
      "attention_bias": true, "double_self_attention": false, "only_cross_attention": false,
      "share_ff": false, "av_cross_ada_norm": true, "norm_elementwise_affine": false]
    architecture.merge(["timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000,
      "positional_embedding_theta": 10000, "frequencies_precision": "float64",
      "positional_embedding_max_pos": [20, 2048, 2048], "audio_positional_embedding_max_pos": [20],
      "in_channels": 128, "out_channels": 128, "audio_out_channels": 128]) { _, new in new }
    var manifest: [String: Any] = ["format": "weetodd-ltx25-transformer-paged-q8-v1", "kind": "transformer",
      "num_layers": 48, "bits": 8, "group_size": 64, "layers": records,
      "metadata": ["config": ["transformer": architecture]]]
    if denoiser {
      let shapes = DenoiserLayout.weightShapes(config)
      try withTensorFile(tensors: shapes.map { ("model.diffusion_model." + $0.key, $0.value, "F32") }) {
        try FileManager.default.copyItem(at: $0, to: root.appendingPathComponent("fixed.safetensors"))
      }
      manifest["fixed"] = ["file": "fixed.safetensors", "tensor_count": shapes.count,
        "tensor_bytes": shapes.values.reduce(0) { $0 + $1.reduce(4, *) }]
    }
    try body(root, manifest, config)
  }
  private func write(_ manifest: [String: Any], _ root: URL) throws {
    try JSONSerialization.data(withJSONObject: manifest).write(to: root.appendingPathComponent("paged_manifest.json"))
  }

  func testDenoiserValidatesFixedFileTopLevelSemanticsAndAllBlocks() throws {
    try fixture(denoiser: true) { root, manifest, config in
      try write(manifest, root)
      let weights = try DenoiserWeights(root: root, configuration: config)
      XCTAssertEqual(try weights.readFixed("proj_out.bias", shape: [128]).count, 128)
      XCTAssertThrowsError(try weights.readFixed("proj_out.bias", shape: [32]))
      let metadata = manifest["metadata"] as! [String: Any]
      let model = metadata["config"] as! [String: Any]
      let architecture = model["transformer"] as! [String: Any]
      for (key, value): (String, Any) in [("timestep_scale_multiplier", 1),
        ("av_ca_timestep_scale_multiplier", 1), ("frequencies_precision", "float32"),
        ("positional_embedding_theta", 1000), ("out_channels", 64),
        ("audio_positional_embedding_max_pos", [2048])] {
        var changed = architecture; changed[key] = value
        var invalid = manifest; invalid["metadata"] = ["config": ["transformer": changed]]
        try write(invalid, root)
        XCTAssertThrowsError(try DenoiserWeights(root: root, configuration: config), "Unsupported \(key)")
      }
      for (key, value): (String, Any) in [("file", "../fixed.safetensors"), ("tensor_bytes", 1)] {
        var record = manifest["fixed"] as! [String: Any]; record[key] = value
        var invalid = manifest; invalid["fixed"] = record; try write(invalid, root)
        XCTAssertThrowsError(try DenoiserWeights(root: root, configuration: config))
      }
      try write(manifest, root)
      try FileManager.default.removeItem(at: root.appendingPathComponent("layer-47.safetensors"))
      XCTAssertThrowsError(try DenoiserWeights(root: root, configuration: config))
    }
  }

  func testValidatesEveryPageWithoutDecodingWeights() throws {
    try fixture { root, manifest, config in
      try write(manifest, root)
      let weights = try PagedBlockWeights(root: root, configuration: config)
      XCTAssertEqual(weights.blockCount, 48)
      XCTAssertEqual(try weights.read(block: 47, name: "attn1.to_q.bias", shape: [32]), [Float](repeating: 0, count: 32))
      XCTAssertThrowsError(try weights.read(block: 48, name: "attn1.to_q.bias", shape: [32]))
      try FileManager.default.removeItem(at: root.appendingPathComponent("layer-47.safetensors"))
      XCTAssertThrowsError(try PagedBlockWeights(root: root, configuration: config), "Late pages must preflight before GPU work")
    }
  }

  func testRejectsUnsupportedFormatArchitectureAndEscapingPage() throws {
    try fixture { root, manifest, config in
      for (key, value): (String, Any) in [("bits", 4), ("num_layers", 47), ("group_size", 32)] {
        var invalid = manifest; invalid[key] = value; try write(invalid, root)
        XCTAssertThrowsError(try PagedBlockWeights(root: root, configuration: config))
      }
      var invalid = manifest
      var records = manifest["layers"] as! [[String: Any]]
      records[0]["file"] = "../outside.safetensors"; invalid["layers"] = records; try write(invalid, root)
      XCTAssertThrowsError(try PagedBlockWeights(root: root, configuration: config))
      let metadata = manifest["metadata"] as! [String: Any]
      let modelConfig = metadata["config"] as! [String: Any]
      let architecture = modelConfig["transformer"] as! [String: Any]
      for (key, value): (String, Any) in [("rope_type", "interleaved"), ("norm_eps", 1e-5),
        ("apply_gated_attention", false), ("ff_bias", true), ("audio_ff_bias", false)] {
        var changed = architecture; changed[key] = value
        invalid = manifest
        invalid["metadata"] = ["config": ["transformer": changed]]
        try write(invalid, root)
        XCTAssertThrowsError(try PagedBlockWeights(root: root, configuration: config), "Unsupported \(key) must fail after decoding")
      }
      records = manifest["layers"] as! [[String: Any]]
      records[47]["tensor_bytes"] = 1; invalid = manifest; invalid["layers"] = records
      try write(invalid, root)
      XCTAssertThrowsError(try PagedBlockWeights(root: root, configuration: config))
    }
  }
}

extension PagedBlockWeightsTests {
  func testActiveAdaptersReachFixedAndPagedWeightsAndRejectUnusedTargets() throws {
    try fixture(denoiser: true) { root,manifest,config in
      try write(manifest,root)
      let shapes = ["proj_out":[128,32],"transformer_blocks.47.attn1.to_q":[32,32]]
      let tensors = shapes.flatMap { name,shape in
        [(name+".lora_A.weight",[1,shape[1]],"F32"),(name+".lora_B.weight",[shape[0],1],"F32")]
      }
      let payloads = Dictionary(uniqueKeysWithValues: tensors.map { name,shape,_ in
        (name,Array(repeating: Float(1),count: shape.reduce(1,*)).withUnsafeBytes { Data($0) })
      })
      try withTensorFile(tensors: tensors,payloads: payloads) { url in
        let adapters = try LoRAWeightStack(adapters: [LoRAAdapter(path: url.path,strength: 0.5)]) {
          try LoRAPlan(file: $0,strength: $1,targetShapes: shapes.mapValues { $0.map(UInt64.init) })
        }
        let weights = try DenoiserWeights(root: root,configuration: config,adapters: adapters)
        XCTAssertEqual(try weights.readFixed("proj_out.weight",shape: [128,32]),Array(repeating: 0.5,count: 128*32))
        XCTAssertEqual(try weights.readBlock(47,name: "attn1.to_q.weight",shape: [32,32]),Array(repeating: 0.5,count: 32*32))
        XCTAssertEqual(try weights.readBlock(46,name: "attn1.to_q.weight",shape: [32,32]),Array(repeating: 0,count: 32*32))
        XCTAssertThrowsError(try weights.readBlock(47,name: "attn1.to_q.weight",shape: [16,64]))
        let different = try AVBlockConfiguration(videoDimension: 64,audioDimension: 16,heads: 2,
          videoHeadDimension: 32,audioHeadDimension: 8,videoTokens: 5,audioTokens: 3,textTokens: 4)
        XCTAssertThrowsError(try DenoiserWeights(root: root,configuration: different,adapters: adapters))
      }
    }
  }
}
