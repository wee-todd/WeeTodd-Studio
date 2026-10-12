import XCTest
@testable import StudioCore

final class GenerationPresentationTests: XCTestCase {
  func testSettingsDetailSeparatesOrdinaryAndSpecializedControls() {
    XCTAssertTrue(GenerationSettingsDetail.basic.showsBasic)
    XCTAssertFalse(GenerationSettingsDetail.basic.showsAdvanced)
    XCTAssertFalse(GenerationSettingsDetail.advanced.showsBasic)
    XCTAssertTrue(GenerationSettingsDetail.advanced.showsAdvanced)
    XCTAssertTrue(GenerationSettingsDetail.all.showsBasic)
    XCTAssertTrue(GenerationSettingsDetail.all.showsAdvanced)
  }

  func testIngredientsSheetIsPresentedAsAReferenceMethod() {
    for engine in [Engine.ltx23, .ltx25] {
      var clip = Clip(engine: engine)
      var sheet = Attachment(assetID: UUID(), role: engine == .ltx23 ? .reference : .control)
      sheet.controlType = "ingredients_reference_sheet"
      clip.attachments = [sheet]
      XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Reference sheet · Ingredients")
      XCTAssertEqual(GenerationPresentation.inputLabel(for: sheet, engine: engine), "Reference sheet")
      XCTAssertEqual(clip.inferredTask, engine == .ltx23 ? "ref2va" : "control")
    }
  }

  func testH3MovieReferenceRetainsAppearancePurposeWithoutPreparingASheet() throws {
    var clip = Clip(engine: .h3)
    let movie = MediaAsset(name: "Motion reference", kind: .video)
    let action = try XCTUnwrap(clip.referenceActions(for: movie).first { $0.id == "movieAppearance" })
    XCTAssertNil(action.preparation)
    try clip.attachReference(movie, action: action)
    let attachment = try XCTUnwrap(clip.attachments.first)
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Appearance references")
    XCTAssertEqual(GenerationPresentation.inputLabel(for: attachment, engine: .h3), "Appearance reference")
    XCTAssertEqual(clip.inferredTask, "ref2va")
  }

  func testResolvedIngredientsFamilyLabelsAnEmptyControlDraft() {
    var clip = Clip(engine: .ltx25)
    clip.generationSelection = GenerationSelection(task: "control")
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip, referenceFamily: "ingredients"),
      "Reference sheet · Ingredients")
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Motion / scene guide")
  }

  func testReferenceMethodsDistinguishMSRAndH3() {
    var clip = Clip(engine: .ltx25)
    clip.generationSelection = GenerationSelection(task: "ref2va")
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip, referenceFamily: "msr"),
      "Appearance references · MSR")
    clip.engine = .h3
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Appearance references")
  }

  func testFrameMethodReflectsActualFirstLastAndTimedInputs() {
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Animate image")
    clip.attachments.append(Attachment(assetID: UUID(), role: .last))
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "First and last frames")
    clip.attachments.removeLast()
    clip.attachments.append(Attachment(assetID: UUID(), role: .keyframe, time: 2.5))
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Timed keyframes")
    clip.generationSelection = GenerationSelection(task: "i2v")
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Timed keyframes")
  }

  func testOtherGenerationMethodsRemainExplicit() {
    var clip = Clip(engine: .ltx25)
    for (task, label) in [("t2v", "Prompt video"), ("control", "Motion / scene guide"),
      ("a2v", "Audio-driven video"), ("extension", "Video extension"),
      ("video_upscale", "Source movie 2× upscale")] {
      clip.generationSelection = GenerationSelection(task: task)
      XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), label)
    }
    clip.engine = .movie
    XCTAssertEqual(GenerationPresentation.methodLabel(for: clip), "Imported media")
  }

  func testInputLabelsDistinguishFrameTimingAudioPurposeAndGuides() {
    XCTAssertEqual(GenerationPresentation.inputLabel(for: Attachment(assetID: UUID(), role: .first), engine: .h3), "First frame")
    XCTAssertEqual(GenerationPresentation.inputLabel(for: Attachment(assetID: UUID(), role: .last), engine: .h3), "Last frame")
    XCTAssertEqual(GenerationPresentation.inputLabel(for: Attachment(assetID: UUID(), role: .keyframe, time: 2.5), engine: .h3), "Keyframe · 2.5 s")
    XCTAssertEqual(GenerationPresentation.inputLabel(for: Attachment(assetID: UUID(), role: .audioDriver), engine: .ltx25), "Audio driver")
    var voice = Attachment(assetID: UUID(), role: .reference)
    voice.audioSourceStart = 0
    voice.audioSourceDuration = 5
    XCTAssertEqual(GenerationPresentation.inputLabel(for: voice, engine: .ltx25), "Voice reference")
    voice.audioSourceStart = nil
    voice.audioSourceDuration = nil
    voice.msrAudioReferenceID = UUID()
    XCTAssertEqual(GenerationPresentation.inputLabel(for: voice, engine: .ltx25), "Voice reference")
    var control = Attachment(assetID: UUID(), role: .control)
    XCTAssertEqual(GenerationPresentation.inputLabel(for: control, engine: .ltx25), "Control guide")
    control.controlType = "crossview_warp"
    control.referenceRole = "warp"
    XCTAssertEqual(GenerationPresentation.inputLabel(for: control, engine: .ltx25), "CrossView guide")
    control.referenceRole = "source"
    XCTAssertEqual(GenerationPresentation.inputLabel(for: control, engine: .ltx25), "Original source and audio")
    control.controlType = "motion_track"
    XCTAssertEqual(GenerationPresentation.inputLabel(for: control, engine: .ltx25), "Motion guide")
  }

  func testOrdinaryReferenceCannotBeMisidentifiedAsAnEndpointOrVoice() {
    var reference = Attachment(assetID: UUID(), role: .reference)
    XCTAssertEqual(GenerationPresentation.inputLabel(for: reference, engine: .ltx25), "Appearance reference")
    reference.referenceRole = "background"
    XCTAssertEqual(GenerationPresentation.inputLabel(for: reference, engine: .ltx25), "Background reference")
    reference.role = .first
    reference.msrAudioReferenceID = UUID()
    XCTAssertEqual(GenerationPresentation.inputLabel(for: reference, engine: .ltx25), "First frame")
  }

  func testProfilesMatchTasksAndOnlyAliasNativeFirstOnlyInput() {
    for engine in [Engine.h3, .ltx23, .ltx25] {
      var clip = Clip(engine: engine)
      clip.attachments = [Attachment(assetID: UUID(), role: .first)]
      XCTAssertTrue(GenerationPresentation.canUseProfile(supportedTasks: ["i2v"], for: clip))
      clip.generationSelection = GenerationSelection(task: "i2v")
      XCTAssertTrue(GenerationPresentation.canUseProfile(supportedTasks: ["fflf"], for: clip))
      clip.attachments.append(Attachment(assetID: UUID(), role: .last))
      clip.generationSelection = GenerationSelection(task: "fflf")
      XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["i2v"], for: clip))
      XCTAssertTrue(GenerationPresentation.canUseProfile(supportedTasks: ["fflf"], for: clip))
      clip.attachments.removeLast()
      clip.attachments.append(Attachment(assetID: UUID(), role: .keyframe, time: 2))
      XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["i2v"], for: clip))
    }
  }

  func testProfileCompatibilityDoesNotGrantUnadvertisedMethods() {
    var clip = Clip(engine: .drawThings)
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["fflf"], for: clip))
    clip.engine = .ltx25
    clip.generationSelection = GenerationSelection(task: "control")
    XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["fflf", "t2v"], for: clip))
    XCTAssertTrue(GenerationPresentation.canUseProfile(supportedTasks: ["control"], for: clip))
    XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: [], for: clip))
    clip.generationSelection = GenerationSelection(task: "extension")
    XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["fflf"], for: clip))
    clip.generationSelection = GenerationSelection(task: "i2v")
    clip.attachments = []
    XCTAssertFalse(GenerationPresentation.canUseProfile(supportedTasks: ["fflf"], for: clip))
  }

  func testGenerationDescriptorRetainsReferenceFamilyWithoutRequiringItInOldDocuments() throws {
    var metadata: [String: Any] = [
      "supportedTasks": ["control"],
      "controls": ["stepsEditable": true, "refinementStepsEditable": false,
        "cfgEditable": false, "shiftEditable": false, "stepsExplanation": "",
        "cfgExplanation": "", "shiftExplanation": ""],
      "presets": []
    ]
    let old = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: metadata))
    XCTAssertNil(old.referenceFamily)
    metadata["referenceFamily"] = "ingredients"
    let current = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: metadata))
    XCTAssertEqual(current.referenceFamily, "ingredients")
    let reloaded = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONEncoder().encode(current))
    XCTAssertEqual(reloaded.referenceFamily, "ingredients")
    XCTAssertEqual(reloaded.supportedTasks, ["control"])
  }

  func testChoosingProfilePreservesAuthoredSettingsAndPersistsThem() throws {
    var clip = authoredClip()
    var expected = clip
    expected.profileID = "ingredients-custom"
    clip.selectGenerationProfile("ingredients-custom")
    XCTAssertEqual(clip, expected)
    let reloaded = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(reloaded, expected)
  }

  func testSelectedModelDefaultsClearOnlyExplicitGenerationOverrides() throws {
    var clip = authoredClip()
    let task = clip.inferredTask
    var expected = clip
    expected.generationSelection = GenerationSelection(task: task, preset: .custom)
    expected.h3PagingCacheGB = nil
    clip.useSelectedModelDefaults()
    XCTAssertEqual(clip, expected)
    let reloaded = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(reloaded, expected)
  }

  func testSelectedModelDefaultsPreserveInferredTaskWithoutSelection() {
    var clip = Clip(engine: .h3)
    clip.profileID = "h3-reference"
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    clip.useSelectedModelDefaults()
    XCTAssertEqual(clip.profileID, "h3-reference")
    XCTAssertEqual(clip.generationSelection?.task, "ref2va")
    XCTAssertEqual(clip.generationSelection?.preset, .custom)
    XCTAssertEqual(clip.attachments.count, 1)
  }

  private func authoredClip() -> Clip {
    var clip = Clip(name: "Authored scene", engine: .h3)
    clip.profileID = "existing-custom"
    clip.prompt = "A person turns toward the window."
    clip.negativePrompt = "Blur"
    clip.soundscape = "A quiet room."
    clip.seed = 76
    clip.duration = 6
    clip.generationWidth = 1024
    clip.generationHeight = 576
    clip.h3PagingCacheGB = 8
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    clip.generationSelection = GenerationSelection(task: "ref2va", preset: .lowMemory)
    clip.generationSelection?.steps = 18
    clip.generationSelection?.cfg = 1.5
    clip.generationSelection?.memoryPolicy = "staged"
    clip.generationSelection?.h3VideoDecodePrecision = .float16
    clip.savedNativeGenerations = ["ltx25": SavedNativeGenerationSettings(profileID: "saved-ltx", selection: GenerationSelection(task: "control"))]
    return clip
  }
}
