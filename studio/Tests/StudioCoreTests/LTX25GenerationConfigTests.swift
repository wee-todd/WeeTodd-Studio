import XCTest
@testable import StudioCore

final class LTX25GenerationConfigTests: XCTestCase {
  func testUserConfigsRejectUnsupportedOverridesAndConflictingRefinementControls() throws {
    var selection = GenerationSelection()
    selection.cfg = 2
    XCTAssertThrowsError(try LTX25GenerationConfig(name: "Invalid distilled", selection: selection))
    selection.cfg = nil
    selection.shift = 5
    XCTAssertThrowsError(try LTX25GenerationConfig(name: "Unsupported shift", selection: selection))
    selection.shift = nil
    selection.ltx25Guidance = LTX25GuidanceSettings(mode: .guided, experimentalEnabled: true)
    selection.ltx25Guidance?.videoRescale = 2
    XCTAssertThrowsError(try LTX25GenerationConfig(name: "Invalid rescale", selection: selection))
    selection.ltx25Guidance = nil
    selection.ltx25DFR = LTX25DFRSettings(experimentalEnabled: true)
    selection.ltx25Keyframes = LTX25KeyframeSettings(experimentalEnabled: true)
    XCTAssertThrowsError(try LTX25GenerationConfig(name: "Conflicting refinements", selection: selection))
  }

  func testFirstSwitchToLTX25StartsWithRecommendedConfigAndKeepsInputs() {
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    clip.selectLocalModel(.ltx25)
    XCTAssertEqual(clip.generationSelection?.ltx25ConfigID, LTX25GenerationConfig.distilled.id)
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertEqual(clip.profileID, "auto")
  }

  func testSingleStageConfigCannotBeAppliedToMotionContinuity() throws {
    var selection = GenerationSelection()
    selection.ltx25SingleStage = LTX25SingleStageSettings(experimentalEnabled: true)
    let config = try LTX25GenerationConfig(name: "Full resolution", selection: selection)
    var clip = Clip(engine: .ltx25)
    clip.continuity = ClipContinuity(mode: "motion", sourceClipID: UUID())
    let before = clip
    XCTAssertThrowsError(try clip.applyLTX25Config(config))
    XCTAssertEqual(clip, before)
  }

  func testApplyingDevConfigReplacesConflictingExecutionSettingsAndPreservesShot() throws {
    var clip = Clip(engine: .ltx25)
    clip.profileID = "/old/distilled.json"
    clip.prompt = "A cup rises."; clip.negativePrompt = "blur"
    clip.seed = 123; clip.duration = 7; clip.generationWidth = 1024
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    clip.generationSelection = GenerationSelection(task: "i2v")
    clip.generationSelection?.ltx25DFR = LTX25DFRSettings(experimentalEnabled: true)
    clip.generationSelection?.ltx25SingleStage = LTX25SingleStageSettings(experimentalEnabled: true)
    let media = clip.attachments
    try clip.applyLTX25Config(.dev)
    XCTAssertEqual(clip.profileID, "auto")
    XCTAssertEqual(clip.generationSelection?.task, "i2v")
    XCTAssertEqual(clip.generationSelection?.ltx25Guidance?.mode, .guided)
    XCTAssertEqual(clip.generationSelection?.ltx25Guidance?.experimentalEnabled, true)
    XCTAssertNil(clip.generationSelection?.ltx25DFR)
    XCTAssertNil(clip.generationSelection?.ltx25SingleStage)
    XCTAssertEqual(clip.prompt, "A cup rises.")
    XCTAssertEqual(clip.negativePrompt, "blur")
    XCTAssertEqual(clip.attachments, media)
    XCTAssertEqual(clip.seed, 123); XCTAssertEqual(clip.duration, 7)
    XCTAssertEqual(clip.generationWidth, 1024)
    clip.selectLocalModel(.h3); clip.selectLocalModel(.ltx25)
    XCTAssertEqual(clip.generationSelection?.ltx25ConfigID, "builtin-ltx25-dev")
  }

  func testCustomConfigSnapshotSurvivesSerializationWithoutARecipePath() throws {
    var selection = GenerationSelection(task: "a2v", preset: .custom)
    selection.ltx25Guidance = LTX25GuidanceSettings(mode: .guidedHQ, experimentalEnabled: true)
    selection.steps = 17; selection.cfg = 2.5
    let custom = try LTX25GenerationConfig(name: "My guided setup", selection: selection)
    let decoded = try JSONDecoder().decode(LTX25GenerationConfig.self, from: JSONEncoder().encode(custom))
    var clip = Clip(engine: .ltx25)
    clip.generationSelection = GenerationSelection(task: "fflf")
    try clip.applyLTX25Config(decoded)
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertEqual(clip.generationSelection?.steps, 17)
    XCTAssertEqual(clip.generationSelection?.cfg, 2.5)
    XCTAssertEqual(clip.generationSelection?.ltx25ConfigName, "My guided setup")
    XCTAssertEqual(clip.profileID, "auto")
    XCTAssertThrowsError(try LTX25GenerationConfig(name: "  ", selection: selection))
  }

  func testLegacySelectionsRemainByteCompatibleAndDistilledClearsDevOverrides() throws {
    let old = try JSONDecoder().decode(GenerationSelection.self, from: Data(#"{"task":"t2v","preset":"balanced"}"#.utf8))
    let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as! [String: Any]
    XCTAssertEqual(Set(object.keys), ["task", "preset"])
    var clip = Clip(engine: .ltx25)
    try clip.applyLTX25Config(.dev)
    clip.generationSelection?.steps = 42
    try clip.applyLTX25Config(.distilled)
    XCTAssertNil(clip.generationSelection?.ltx25Guidance)
    XCTAssertNil(clip.generationSelection?.steps)
    XCTAssertEqual(clip.generationSelection?.ltx25ConfigID, "builtin-ltx25-distilled")
    var h3 = Clip(engine: .h3)
    XCTAssertThrowsError(try h3.applyLTX25Config(.dev))
  }
}
