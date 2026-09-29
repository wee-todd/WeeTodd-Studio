import XCTest
import TensorIO
import NNC
@testable import LTX25NNC

/// Optional stage-boundary diagnostics. Normal tests never require model files.
final class InstalledDenoiserTests: XCTestCase {
  func testFixedStagesAndRotaryAgainstInstalledModelTrace() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_DENOISER_WEIGHTS"],
      let tracePath = ProcessInfo.processInfo.environment["WEETODD_DENOISER_TRACE"] else {
      throw XCTSkip("Set WEETODD_DENOISER_WEIGHTS and WEETODD_DENOISER_TRACE for installed-model diagnostics.")
    }
    let originalFlags = DynamicGraph.flags
    defer { DynamicGraph.flags = originalFlags }
    if ProcessInfo.processInfo.environment["WEETODD_DENOISER_MPS_GEMM"] == "1" {
      DynamicGraph.flags.insert(.disableMFAGEMM)
    }
    let traceURL = URL(fileURLWithPath: tracePath)
    let trace = try SafeTensorFile(url: traceURL.appendingPathExtension("trace.safetensors"))
    let config = try JSONDecoder().decode(AVBlockConfiguration.self, from: Data(contentsOf: traceURL.appendingPathExtension("config.json")))
    let input = try SafeTensorFile(url: traceURL.appendingPathExtension("inputs.safetensors"))
    let weights = try DenoiserWeights(root: URL(fileURLWithPath: root), configuration: config)
    func compare(_ key: String, _ actual: [Float], tolerance: Double = 0.0001) throws {
      let reference = try trace.readFloat32(named: key)
      XCTAssertEqual(actual.count, reference.count)
      let absolute = zip(actual, reference).map { abs($0 - $1) }.max()!
      let relative = sqrt(zip(actual, reference).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) }
        / max(1e-30, reference.reduce(0.0) { $0 + Double($1) * Double($1) }))
      print("STAGE \(key) abs=\(absolute) relative=\(relative)")
      XCTAssertLessThan(relative, tolerance, key)
    }
    let time = try DenoiserMath.timestep(0.731)
    try compare("time", time)
    let keys = ["video_adaln_params", "audio_adaln_params", "video_prompt_adaln_params", "audio_prompt_adaln_params",
      "av_ca_video_params", "av_ca_audio_params", "av_ca_a2v_gate_params", "av_ca_v2a_gate_params"]
    for (head, key) in zip(DenoiserLayout.heads(config), keys) {
      let values = try autoreleasepool {
        try FixedStage.adaptive(head.name, dimension: head.dimension, parameters: head.parameters)
          .evaluate([time], weights: { try weights.readFixed($0, shape: $1) })
      }
      try compare(key, values[0])
      if head.name == "adaln_single" { try compare("video_embedded", values[1]) }
      if head.name == "audio_adaln_single" { try compare("audio_embedded", values[1]) }
    }
    for (name, prefix, rows, width) in [("video", "", config.videoTokens, config.videoDimension),
      ("audio", "audio_", config.audioTokens, config.audioDimension)] {
      let latent = try input.readFloat32(named: name + "_latent").map(DenoiserMath.bfloat16)
      let projected = try autoreleasepool {
        try FixedStage.projection(prefix + "patchify_proj", rows: rows, width: 128, output: width)
          .evaluate([latent], weights: { try weights.readFixed($0, shape: $1) })[0]
      }
      try compare(name + "_hidden", projected)
      let hidden = try trace.readFloat32(named: name + "_final_hidden")
      let embedded = try trace.readFloat32(named: name + "_embedded")
      let velocity = try autoreleasepool {
        try FixedStage.projection(prefix + "proj_out", rows: rows, width: width, output: 128,
          table: prefix + "scale_shift_table").evaluate([hidden, embedded], weights: { try weights.readFixed($0, shape: $1) })[0]
      }
      try compare(name + "_velocity", velocity)
    }
    let vp = try input.readFloat32(named: "video_positions"), ap = try input.readFloat32(named: "audio_positions")
    let temporal = stride(from: 0, to: vp.count, by: 3).map { vp[$0] }
    for (name, positions, axes, tokens, width, maximum): (String, [Float], Int, Int, Int, [Float]) in [
      ("video", vp, 3, config.videoTokens, config.videoHeadDimension, [20, 2048, 2048]),
      ("audio", ap, 1, config.audioTokens, config.audioHeadDimension, [20]),
      ("video_cross", temporal, 1, config.videoTokens, config.audioHeadDimension, [20]),
      ("audio_cross", ap, 1, config.audioTokens, config.audioHeadDimension, [20])] {
      let rotary = try DenoiserMath.rotary(positions: positions, axes: axes, tokens: tokens,
        heads: config.heads, headWidth: width, maximumPositions: maximum)
      try compare(name + "_rope_freqs_cos", rotary.cos)
      try compare(name + "_rope_freqs_sin", rotary.sin)
    }
    if ProcessInfo.processInfo.environment["WEETODD_DENOISER_TRACE_STACK"] == "1" {
      var prepared: [String: [Float]] = [:]
      var names = Dictionary(uniqueKeysWithValues: zip(DenoiserLayout.heads(config).map(\.input), keys))
      names.merge(["video": "video_hidden", "audio": "audio_hidden",
        "video_text": "video_text_embeds", "audio_text": "audio_text_embeds"]) { _, new in new }
      for name in ["video", "audio", "video_cross", "audio_cross"] {
        for suffix in ["cos", "sin"] { names[name + "_rope_" + suffix] = name + "_rope_freqs_" + suffix }
      }
      for (key, name) in names { prepared[key] = try trace.readFloat32(named: name) }
      if ProcessInfo.processInfo.environment["WEETODD_DENOISER_TRACE_BLOCKS"] == "1" {
        let block = try AVBlockRunner(configuration: config)
        var referenceInputs = prepared, cumulative = prepared
        for index in 0..<48 {
          try block.load { try weights.blocks.read(block: index, name: $0, shape: $1) }
          let single = try block.evaluate(referenceInputs)
          print("ISOLATED BLOCK \(index)")
          try compare("block_\(index)_video", single.video)
          try compare("block_\(index)_audio", single.audio)
          let chained = try block.evaluate(cumulative)
          print("CUMULATIVE BLOCK \(index)")
          try compare("block_\(index)_video", chained.video, tolerance: 0.002)
          try compare("block_\(index)_audio", chained.audio)
          cumulative["video"] = chained.video; cumulative["audio"] = chained.audio
          referenceInputs["video"] = try trace.readFloat32(named: "block_\(index)_video")
          referenceInputs["audio"] = try trace.readFloat32(named: "block_\(index)_audio")
        }
        return
      }
      let output = try autoreleasepool {
        try AVStackRunner(configuration: config, blockCount: 48).evaluate(prepared,
          weights: { try weights.blocks.read(block: $0, name: $1, shape: $2) })
      }
      try compare("video_final_hidden", output.video, tolerance: 0.002)
      try compare("audio_final_hidden", output.audio)
      for (name, prefix, rows, width, hidden) in [("video", "", config.videoTokens, config.videoDimension, output.video),
        ("audio", "audio_", config.audioTokens, config.audioDimension, output.audio)] {
        let embedded = try trace.readFloat32(named: name + "_embedded")
        let velocity = try autoreleasepool {
          try FixedStage.projection(prefix + "proj_out", rows: rows, width: width, output: 128,
            table: prefix + "scale_shift_table").evaluate([hidden, embedded], weights: { try weights.readFixed($0, shape: $1) })[0]
        }
        try compare(name + "_velocity", velocity, tolerance: name == "video" ? 0.002 : 0.0001)
      }
    }
  }
}
