import XCTest
@testable import StudioCore

final class LTX25FastDevConfigTests: XCTestCase {
  func fastConfig() throws -> LTX25GenerationConfig {
    try XCTUnwrap(LTX25GenerationConfig.builtIns.first { $0.id == "builtin-ltx25-dev-fast" })
  }

  func testApplyingFastDevPreservesAuthoredShotAndKeepsDevTaskBoundaries() throws {
    let config = try fastConfig()
    try config.validate()
    var clip = Clip(engine: .ltx25)
    clip.prompt = "A dancer turns."; clip.negativePrompt = "blur"
    clip.seed = 123; clip.duration = 7; clip.generationWidth = 1344; clip.generationHeight = 768
    clip.attachments = [Attachment(assetID: UUID(), role: .first), Attachment(assetID: UUID(), role: .lora)]
    var settings = MovieSettings(); settings.fps = 25; clip.settingsOverride = settings
    clip.generationSelection = GenerationSelection(task: "i2v")
    clip.generationSelection?.ltx25DFR = LTX25DFRSettings(experimentalEnabled: true)
    let original = clip
    try clip.applyLTX25Config(config)
    let selected = try XCTUnwrap(clip.generationSelection)
    XCTAssertEqual(selected.task, "i2v")
    XCTAssertEqual(selected.ltx25Guidance?.mode, .guided)
    XCTAssertTrue(selected.ltx25Guidance?.experimentalEnabled == true)
    XCTAssertTrue(selected.ltx25DiffusionVAE?.experimentalEnabled == true)
    XCTAssertNil(selected.ltx25DFR)
    XCTAssertNil(selected.ltx25SingleStage)
    XCTAssertEqual(clip.profileID, "auto")
    clip.generationSelection = original.generationSelection; clip.profileID = original.profileID
    XCTAssertEqual(clip, original)
    clip.continuity = ClipContinuity(mode: "motion", sourceClipID: UUID())
    let before = clip
    XCTAssertThrowsError(try clip.applyLTX25Config(config))
    XCTAssertEqual(clip, before)
  }

  func testFastDevExportUsesQualifiedScheduleAndDiffVAEWithoutConvFallback() throws {
    let config = try fastConfig()
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let helper = NativeModelSetupControlTests(), decoder = try NativeLTXDiffusionVAETests().fixture()
    defer { try? FileManager.default.removeItem(at: decoder) }
    let dev = try helper.devPack(root, name: "dev-pages")
    let adapter = try helper.write(root, name: "refinement", header: helper.distilledHelperHeader())
    var components: [String: Any] = ["transformer_path": dev.path, "distilled_lora_path": adapter.path]
    for (key, header) in try NativeLTXConfigResolutionTests().supportHeaders() {
      components[key] = try helper.write(root, name: key, header: header).path
    }
    func write(_ name: String, _ components: [String: Any]) throws {
      let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
        "components": components, "config": ["pipeline_mode": "guided_hq", "stage1_steps": 7,
          "diffvae_query_chunk_size": 512], "ffmpeg": "/usr/bin/true"]
      try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent(name))
    }
    try write("ordinary.json", components)
    var diffusion = components; diffusion["video_vae_path"] = decoder.path
    try write("diffusion.json", diffusion)
    var clip = Clip(engine: .ltx25)
    clip.prompt = "A dancer turns."; clip.negativePrompt = "blur"; clip.duration = 2
    clip.generationWidth = 1344; clip.generationHeight = 768; clip.seed = 123
    try clip.applyLTX25Config(config)
    var project = StudioProject(); project.settings.fps = 24; project.clips = [clip]
    let request: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": clip.id.uuidString, "runtime": ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"]]
    let recipe = try XCTUnwrap(NativeLTXPreparation.compose(request: request)["recipe"] as? [String: Any])
    let actual = try XCTUnwrap(recipe["config"] as? [String: Any])
    let linked = try XCTUnwrap(recipe["components"] as? [String: Any])
    XCTAssertEqual(recipe["prompt"] as? String, clip.prompt)
    XCTAssertEqual(linked["transformer_path"] as? String, dev.path)
    XCTAssertEqual(linked["distilled_lora_path"] as? String, adapter.path)
    XCTAssertEqual(linked["video_vae_path"] as? String, decoder.path)
    XCTAssertEqual(actual["pipeline_mode"] as? String, "guided")
    XCTAssertEqual(actual["stage1_sampler"] as? String, "euler_guided")
    XCTAssertEqual(actual["stage2_sampler"] as? String, "euler")
    for (key, value) in ["stage1_steps": 30, "stage2_steps": 3, "width": 1344, "height": 768,
      "seed": 123, "diffvae_query_chunk_size": 8192, "diffvae_context_width_chunks": 4,
      "diffvae_stage4_tile_width": 0] {
      XCTAssertEqual(actual[key] as? Int, value, key)
    }
    for (key, value) in ["video_cfg_scale": 3.0, "audio_cfg_scale": 7, "stg_scale": 1,
      "video_rescale_scale": 0.7, "audio_rescale_scale": 0.7, "modality_scale": 3,
      "frame_rate": 24, "duration_seconds": 2] {
      XCTAssertEqual(actual[key] as? Double, value, key)
    }
    XCTAssertEqual(actual["stg_blocks"] as? [Int], [28])
    XCTAssertTrue(actual["stage1_sigmas"] is NSNull)
    XCTAssertEqual(actual["negative_prompt"] as? String, "blur")
    XCTAssertEqual(actual["low_memory"] as? Bool, true)
    XCTAssertEqual(actual["low_ram_streaming"] as? Bool, false)
    XCTAssertEqual(actual["diffvae_optimization"] as? String, "metal_na3d_query_tiled_experimental")
    try FileManager.default.removeItem(at: root.appendingPathComponent("diffusion.json"))
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request)) {
      XCTAssertTrue($0.localizedDescription.contains("Diffusion VAE"))
    }
  }
}
