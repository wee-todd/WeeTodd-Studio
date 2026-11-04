import XCTest
@testable import StudioCore

final class NativeLTXConfigResolutionTests: XCTestCase {
  func testExplicitDiffusionVAEControlsResolveTheirCompatibleDecoder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let helper = NativeModelSetupControlTests(), decoder = try NativeLTXDiffusionVAETests().fixture()
    defer { try? FileManager.default.removeItem(at: decoder) }
    var components: [String: Any] = ["transformer_path": try helper.devPack(root, name: "distilled",
      source: "ltx-2.5-22b-distilled-transformer-bf16.safetensors").path]
    for (key, header) in try supportHeaders() { components[key] = try helper.write(root, name: key, header: header).path }
    var diffusion = components; diffusion["video_vae_path"] = decoder.path
    var clip = Clip(engine: .ltx25)
    try clip.applyLTX25Config(.distilled)
    clip.generationSelection?.ltx25DiffusionVAE = LTX25DiffusionVAESettings(experimentalEnabled: true)
    let result = try NativeLTXConfigResolution.resolve(clip: clip, project: StudioProject(), sources: [
      ("ordinary.json", ["components": components]), ("diffusion.json", ["components": diffusion])])
    XCTAssertEqual((result.recipe["components"] as? [String: Any])?["video_vae_path"] as? String, decoder.path)
    XCTAssertTrue(result.sourcePaths.contains("diffusion.json"))
  }

  func testDevAndDistilledResolveSharedInstalledComponentsWithoutMatchingRecipes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let helper = NativeModelSetupControlTests()
    let distilled = try helper.devPack(root, name: "fast-pages",
      source: "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
    let dev = try helper.devPack(root, name: "full-pages")
    let adapter = try helper.write(root, name: "refinement", header: helper.distilledHelperHeader())
    var components: [String: Any] = ["transformer_path": distilled.path]
    for (key, header) in try supportHeaders() {
      components[key] = try helper.write(root, name: key, header: header).path
    }
    func write(_ name: String, _ components: [String: Any], _ mode: String) throws {
      // An old incompatible schedule cannot prevent reuse of its installed components.
      let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
        "components": components, "config": ["pipeline_mode": mode, "stage1_steps": 7, "stage2_steps": 0],
        "ffmpeg": "/usr/bin/true"]
      try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent(name))
    }
    try write("shared.json", components, "distilled")
    try write("dev.json", ["transformer_path": dev.path, "distilled_lora_path": adapter.path], "guided_hq")
    let badEncoder = try helper.write(root, name: "wrong-encoder", header: ["unrelated.weight": ["dtype": "BF16", "shape": [1]]])
    var bad = components; bad["text_encoder_path"] = badEncoder.path
    try write("a-incompatible-binding.json", bad, "distilled")
    var clip = Clip(engine: .ltx25); clip.prompt = "A dancer turns."; clip.duration = 2
    clip.negativePrompt = "blur"
    var project = StudioProject(); project.settings.fps = 25
    let runtime: [String: Any] = ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"]
    for (config, mode, transformer, steps) in [(LTX25GenerationConfig.distilled, "distilled", distilled.path, 8),
      (.dev, "guided", dev.path, 30), (.devHQ, "guided_hq", dev.path, 15)] {
      try clip.applyLTX25Config(config); project.clips = [clip]
      let request: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
        "clipID": clip.id.uuidString, "runtime": runtime]
      let result = try NativeLTXPreparation.compose(request: request)
      let recipe = try XCTUnwrap(result["recipe"] as? [String: Any])
      let actual = recipe["config"] as! [String: Any], linked = recipe["components"] as! [String: Any]
      XCTAssertEqual(actual["pipeline_mode"] as? String, mode)
      XCTAssertEqual(actual["stage1_steps"] as? Int, steps)
      XCTAssertEqual(actual["stage2_steps"] as? Int, 3)
      XCTAssertEqual(actual["frame_rate"] as? Double, 25)
      XCTAssertEqual(actual["negative_prompt"] as? String, mode == "distilled" ? "" : "blur")
      XCTAssertEqual(clip.negativePrompt, "blur")
      XCTAssertEqual(linked["transformer_path"] as? String, transformer)
      XCTAssertEqual(linked["text_encoder_path"] as? String, components["text_encoder_path"] as? String)
      XCTAssertEqual(linked["distilled_lora_path"] as? String, mode == "distilled" ? nil : adapter.path)
      let description = try NativeLTXPreparation.describe(request: request)
      XCTAssertTrue((description["sourcePaths"] as? [String] ?? []).contains(root.appendingPathComponent("shared.json").path))
    }
  }

  func supportHeaders() throws -> [String: [String: Any]] {
    func json(_ object: [String: Any]) throws -> String {
      String(data: try JSONSerialization.data(withJSONObject: object), encoding: .utf8)!
    }
    let tensor: [String: Any] = ["dtype": "BF16", "shape": [1]]
    return [
      "text_encoder_path": ["__metadata__": ["gemma_config": try json(["model_type": "gemma4_unified"])], "model.embed_tokens.weight": tensor],
      "video_vae_path": ["__metadata__": ["config": try json(["vae": [:] as [String: Any]])], "decoder.weight": tensor],
      "audio_vae_path": ["audio_vae.decoder.weight": tensor, "vocoder.weight": tensor],
      "spatial_upscaler_path": ["__metadata__": ["config": try json(["_class_name": "LatentUpsampler", "in_channels": 128,
        "dims": 3, "spatial_upsample": true, "temporal_upsample": false])], "weight": tensor]
    ]
  }

  func testMissingComponentsAndWrongVariantExplainWhatMustBeLinked() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let dev = try NativeModelSetupControlTests().devPack(root, name: "misleading-distilled-name")
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "config": ["pipeline_mode": "distilled"], "components": ["transformer_path": dev.path]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("wrong.json"))
    var clip = Clip(engine: .ltx25); clip.prompt = "A cup rises."
    try clip.applyLTX25Config(.distilled)
    var project = StudioProject(); project.clips = [clip]
    let request: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": clip.id.uuidString, "runtime": ["profilesDirectory": root.path]]
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request)) {
      XCTAssertTrue($0.localizedDescription.contains("distilled transformer"))
      XCTAssertTrue($0.localizedDescription.contains("text encoder"))
      XCTAssertFalse($0.localizedDescription.contains("Import a distilled recipe"))
    }
  }
}
