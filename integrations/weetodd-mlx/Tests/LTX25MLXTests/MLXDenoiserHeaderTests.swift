import Foundation
import XCTest
import LTX25Engine
import LTX25MLX
import InferenceTestSupport

final class MLXDenoiserHeaderTests: XCTestCase {
  private func fixture(denoiser: Bool = false, _ body: (URL, [String: Any], AVBlockConfiguration) throws -> Void) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let config = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let shapes = try MLXAVBlock(configuration: config).weightShapes
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

  func testAllPagesAndFixedHeadersValidateWithoutReadingPayloads() throws {
    try fixture(denoiser:true) { root,manifest,config in
      try write(manifest,root)
      let weights=try MLXDenoiserWeights(root:root,configuration:config)
      XCTAssertEqual(weights.blockCount,48)
      XCTAssertGreaterThan(weights.largestBlockBytes,0)
      for (key,value):(String,Any) in [("bits",4),("group_size",32),("num_layers",47)] {
        var invalid=manifest; invalid[key]=value; try write(invalid,root)
        XCTAssertThrowsError(try MLXDenoiserWeights(root:root,configuration:config))
      }
      var changed=manifest["fixed"] as! [String:Any]; changed["tensor_bytes"]=1
      var invalid=manifest; invalid["fixed"]=changed; try write(invalid,root)
      XCTAssertThrowsError(try MLXDenoiserWeights(root:root,configuration:config))
      let metadata=manifest["metadata"] as! [String:Any]
      let model=metadata["config"] as! [String:Any]
      let original=model["transformer"] as! [String:Any]
      for (key,value):(String,Any) in [("rope_type","interleaved"),("timestep_scale_multiplier",1),
        ("ff_bias",true),("audio_ff_bias",false),("frequencies_precision","float32")] {
        var architecture=original; architecture[key]=value
        invalid=manifest; invalid["metadata"]=["config":["transformer":architecture]]
        try write(invalid,root)
        XCTAssertThrowsError(try MLXDenoiserWeights(root:root,configuration:config))
      }
      for filename in ["../fixed.safetensors","/tmp/fixed.safetensors","./fixed.safetensors"] {
        changed=manifest["fixed"] as! [String:Any]; changed["file"]=filename
        invalid=manifest; invalid["fixed"]=changed; try write(invalid,root)
        XCTAssertThrowsError(try MLXDenoiserWeights(root:root,configuration:config))
      }
      try write(manifest,root)
      try FileManager.default.removeItem(at:root.appendingPathComponent("layer-47.safetensors"))
      XCTAssertThrowsError(try MLXDenoiserWeights(root:root,configuration:config))
    }
  }
  func testFixedSourceRejectsMissingDuplicateOrWrongShapedWeights() throws {
    let c=try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:5,audioTokens:3,textTokens:4)
    let shapes=DenoiserLayout.weightShapes(c)
    let tensors=shapes.map { ($0.key,$0.value,"F32") }
    try withTensorFile(tensors:tensors) { url in
      XCTAssertNoThrow(try MLXFixedSource(url:url,configuration:c))
    }
    try withTensorFile(tensors:Array(tensors.dropLast())) { url in
      XCTAssertThrowsError(try MLXFixedSource(url:url,configuration:c))
    }
    try withTensorFile(tensors:tensors+[("model.diffusion_model.proj_out.bias",[128],"F32")]) { url in
      XCTAssertThrowsError(try MLXFixedSource(url:url,configuration:c))
    }
    let wrong=tensors.map { $0.0 == "proj_out.bias" ? ($0.0,[127],$0.2) : $0 }
    try withTensorFile(tensors:wrong) { url in
      XCTAssertThrowsError(try MLXFixedSource(url:url,configuration:c))
    }
  }
}
