import XCTest
@testable import StudioCore

final class LTX25AdapterConfigTests: XCTestCase {
  func testPreparedGuideAssetActionsAttachTheSelectedFamilyAndCrossViewRole() throws {
    let movie = MediaAsset(name: "Guide", kind: .video)
    for (kind, wanted) in [(LTXLibraryAdapterKind.motionTrack, ["motion_track"]),
      (.crossview, ["crossview_warp", "crossview_warp"])] {
      var clip = Clip(engine: .ltx25)
      try clip.applyLTX25Config(.adapter(LTX25AdapterBinding(kind: kind, path: "/models/guide.safetensors")))
      let actions = clip.referenceActions(for: movie)
      XCTAssertEqual(actions.map(\.controlType), wanted)
      for action in actions { try clip.attachReference(movie, action: action) }
      XCTAssertEqual(clip.attachments.map(\.controlType), wanted)
      if kind == .crossview { XCTAssertEqual(clip.attachments.compactMap(\.referenceRole), ["warp", "source"]) }
      XCTAssertEqual(clip.inferredTask, "control")
      XCTAssertFalse(clip.canAssignMedia(MediaAsset(name: "Still", kind: .image), role: .keyframe))
    }
  }

  func testSavedAutomaticDurationRefinementConfigRejectsAudioDrivenShots() throws {
    var selection = LTX25GenerationConfig.adapter(
      LTX25AdapterBinding(kind: .distilledRefinement, path: "/models/refinement.safetensors")).selection
    selection.ltx25AutomaticDuration = LTX25AutomaticDurationSettings(experimentalEnabled: true)
    let config = try LTX25GenerationConfig(name: "Dev with timing", selection: selection)
    var clip = Clip(engine: .ltx25); clip.generationSelection = GenerationSelection(task: "a2v")
    XCTAssertThrowsError(try clip.applyLTX25Config(config))
  }

  func testAssetActionsHonorTheSelectedAdapterAcrossAllInputEntryPoints() throws {
    var clip = Clip(engine: .ltx25)
    let image = MediaAsset(name: "Portrait", kind: .image), audio = MediaAsset(name: "Voice", kind: .audio)
    let appearance = try XCTUnwrap(clip.referenceActions(for: image).first { $0.id == "imageAppearance" })
    try clip.applyLTX25Config(.adapter(LTX25AdapterBinding(kind: .motionTrack, path: "/models/motion.safetensors")))
    XCTAssertTrue(clip.referenceActions(for: image).isEmpty)
    XCTAssertThrowsError(try clip.attachReference(image, action: appearance))
    XCTAssertEqual(clip.inferredTask, "control")
    try clip.applyLTX25Config(.adapter(LTX25AdapterBinding(kind: .msr, path: "/models/msr-v1.safetensors")))
    XCTAssertTrue(clip.referenceActions(for: audio).isEmpty)
    XCTAssertEqual(clip.referenceActions(for: image).map(\.id), ["imageAppearance"])
    XCTAssertFalse(clip.supportsEndpoint(.first))
  }

  func testOrdinaryOnlyAdaptersRejectAudioAndReferenceTasksThatTheyCannotExecute() throws {
    var clip = Clip(engine: .ltx25)
    clip.generationSelection = GenerationSelection(task: "a2v")
    let before = clip
    XCTAssertThrowsError(try clip.applyLTX25Config(.adapter(
      LTX25AdapterBinding(kind: .dfr, path: "/models/dfr.safetensors"))))
    XCTAssertEqual(clip, before)
    for task in ["ref2va", "control"] {
      clip.generationSelection = GenerationSelection(task: task)
      XCTAssertThrowsError(try clip.applyLTX25Config(.adapter(
        LTX25AdapterBinding(kind: .distilledRefinement, path: "/models/refinement.safetensors"))))
    }
  }

  func testSpecializedConfigCanBeSelectedBeforeItsInputsAndPreservesTheShot() throws {
    var clip = Clip(engine: .ltx25)
    clip.prompt = "Preserve the shot."; clip.negativePrompt = "blur"; clip.seed = 42
    clip.duration = 3; clip.generationWidth = 768; clip.generationHeight = 512
    let attachment = Attachment(assetID: UUID(), role: .first); clip.attachments = [attachment]
    let binding = LTX25AdapterBinding(kind: .motionTrack, path: "/models/motion.safetensors")
    try clip.applyLTX25Config(.adapter(binding))
    XCTAssertEqual(clip.profileID, "auto")
    XCTAssertEqual(clip.inferredTask, "control")
    XCTAssertEqual(clip.generationSelection?.ltx25AdapterBinding, binding)
    XCTAssertEqual(clip.prompt, "Preserve the shot."); XCTAssertEqual(clip.negativePrompt, "blur")
    XCTAssertEqual(clip.attachments, [attachment]); XCTAssertEqual(clip.seed, 42)
    XCTAssertEqual(clip.duration, 3); XCTAssertEqual(clip.generationHeight, 512)
    XCTAssertFalse(binding.inputRequirements(for: clip, assets: []).allSatisfy(\.satisfied))
  }

  func testSavedConfigRetainsTheAdapterAndChangingToDevClearsItsTask() throws {
    var clip = Clip(engine: .ltx25)
    try clip.applyLTX25Config(.adapter(LTX25AdapterBinding(kind: .msr, path: "/models/msr.safetensors")))
    let saved = try LTX25GenerationConfig(name: "My references", selection: clip.generationSelection!)
    let restored = try JSONDecoder().decode(LTX25GenerationConfig.self, from: JSONEncoder().encode(saved))
    var another = Clip(engine: .ltx25); try another.applyLTX25Config(restored)
    XCTAssertEqual(another.generationSelection?.ltx25AdapterBinding?.path, "/models/msr.safetensors")
    XCTAssertEqual(another.inferredTask, "ref2va")
    XCTAssertTrue(LTX25GenerationConfig.dev.supports(another))
    try another.applyLTX25Config(.dev)
    XCTAssertNil(another.generationSelection?.ltx25AdapterBinding)
    XCTAssertEqual(another.inferredTask, "t2v")
  }

  func testAutomaticResolutionPinsTheSelectedAdapterWithoutASpecializedRecipe() throws {
    let helper = NativeModelSetupControlTests(), root = try helper.directory()
    defer { try? FileManager.default.removeItem(at: root) }
    let selected = try helper.write(root, name: "selected-motion", header: helper.factors(downscale: "2", version: "2.3.0"))
    let other = try helper.write(root, name: "other-motion", header: helper.factors(downscale: "2", version: "2.3.0"))
    var components: [String: Any] = ["transformer_path": try helper.devPack(root, name: "distilled",
      source: "ltx-2.5-22b-distilled-transformer-bf16.safetensors").path,
      "motion_track_lora_path": other.path]
    for (key, header) in try NativeLTXConfigResolutionTests().supportHeaders() {
      components[key] = try helper.write(root, name: key, header: header).path
    }
    var clip = Clip(engine: .ltx25)
    try clip.applyLTX25Config(.adapter(try NativeModelSetup.adapterBinding(at: selected, kind: .motionTrack)))
    let result = try NativeLTXConfigResolution.resolve(clip: clip, project: StudioProject(),
      sources: [("distilled.json", ["components": components])])
    let resolved = try XCTUnwrap(result.recipe["components"] as? [String: Any])
    XCTAssertEqual((resolved["ic_loras"] as? [[Any]])?.first?.first as? String, selected.path)
    XCTAssertEqual((result.recipe["conditioning"] as? [String: Any])?["control_family"] as? String, "motion_track")
    XCTAssertEqual((result.recipe["config"] as? [String: Any])?["stage2_steps"] as? Int, 3)
    XCTAssertTrue(result.sourcePaths.contains(selected.path))
    try FileManager.default.removeItem(at: selected)
    XCTAssertThrowsError(try NativeLTXConfigResolution.resolve(clip: clip, project: StudioProject(),
      sources: [("distilled.json", ["components": components])]))
  }
}
