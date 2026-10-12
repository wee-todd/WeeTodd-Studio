import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class GenerationInputChoicesTests: XCTestCase {
  func testAppearanceAndReferenceSheetRemainDistinctPurposes() {
    let image = MediaAsset(name: "Character sheet", kind: .image)
    let clip = Clip(engine: .ltx25)
    XCTAssertEqual(GenerationInputPurpose.appearance.actions(for: image, clip: clip).map(\.id), ["imageAppearance"])
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: clip).map(\.id), ["ingredients"])
    XCTAssertTrue(GenerationInputPurpose.audioDriver.actions(for: image, clip: clip).isEmpty)
  }

  func testPinnedRecipeOnlyOffersItsAdvertisedReferenceTasks() {
    let image = MediaAsset(name: "Character", kind: .image)
    let clip = Clip(engine: .ltx25)
    XCTAssertTrue(GenerationInputPurpose.appearance.actions(for: image, clip: clip, supportedTasks: ["control"]).isEmpty)
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: clip, supportedTasks: ["control"]).map(\.id), ["ingredients"])
    XCTAssertTrue(GenerationInputPurpose.ingredients.actions(for: image, clip: clip, supportedTasks: ["ref2va"]).isEmpty)
    XCTAssertEqual(GenerationInputPurpose.appearance.actions(for: image, clip: clip, supportedTasks: ["ref2va"]).map(\.id), ["imageAppearance"])
  }

  func testLTX23SheetUsesReferenceTaskAndMoviePreparationRemainsExplicit() {
    let clip = Clip(engine: .ltx23)
    let image = MediaAsset(name: "Sheet", kind: .image)
    let movie = MediaAsset(name: "Story", kind: .video)
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: clip, supportedTasks: ["ref2va"]).map(\.id), ["ingredients"])
    XCTAssertTrue(GenerationInputPurpose.appearance.actions(for: movie, clip: clip).isEmpty)
    let sheet = GenerationInputPurpose.ingredients.actions(for: movie, clip: clip)
    XCTAssertEqual(sheet.map(\.id), ["movieAppearance"])
    XCTAssertEqual(sheet.first?.preparation, "sheet")
    XCTAssertEqual(GenerationInputPurpose.motionGuide.actions(for: movie, clip: clip).map(\.id), ["movieMotion", "preprocessedControl"])
  }

  func testAudioDriverAndVoiceReferenceNeverShareAMenuPurpose() {
    let audio = MediaAsset(name: "Voice", kind: .audio)
    let clip = Clip(engine: .ltx25)
    XCTAssertEqual(GenerationInputPurpose.audioDriver.actions(for: audio, clip: clip).map(\.id), ["audioDriver"])
    XCTAssertEqual(GenerationInputPurpose.appearance.actions(for: audio, clip: clip).map(\.id), ["msrVoice"])
    XCTAssertTrue(GenerationInputPurpose.audioDriver.actions(for: audio, clip: clip, supportedTasks: ["ref2va"]).isEmpty)
  }

  func testReplacingReferencePreservesItsMediaPurpose() {
    let image = MediaAsset(name: "Face", kind: .image)
    let audio = MediaAsset(name: "Voice", kind: .audio)
    let attachment = Attachment(assetID: image.id, role: .reference)
    let clip = Clip(engine: .ltx25)
    XCTAssertTrue(GenerationInputPurpose.canReplace(attachment, original: image, with: image, clip: clip))
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: image, with: audio, clip: clip))
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: image, with: image, clip: clip, supportedTasks: ["control"]))
  }

  func testReplacingCrossViewKeepsGuideTypeAndRequiresMovieMedia() {
    let clip = Clip(engine: .ltx25)
    let movie = MediaAsset(name: "Camera guide", kind: .video)
    var attachment = Attachment(assetID: movie.id, role: .control)
    attachment.controlType = "crossview_warp"
    attachment.referenceRole = "source"
    XCTAssertTrue(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie, clip: clip, supportedTasks: ["control"]))
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: movie, with: MediaAsset(name: "Still", kind: .image), clip: clip))
  }

  func testIngredientsAndMSRExcludeOtherKnownFamilies() {
    let clip = Clip(engine: .ltx25)
    let image = MediaAsset(name: "Subject", kind: .image)
    let audio = MediaAsset(name: "Voice", kind: .audio)
    for family in ["union", "motion_track", "crossview_warp", "ordinary", "msr"] {
      XCTAssertTrue(GenerationInputPurpose.ingredients.actions(for: image, clip: clip,
        supportedTasks: ["control", "ref2va"], referenceFamily: family).isEmpty, family)
    }
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: clip,
      supportedTasks: ["control"], referenceFamily: "ingredients").map(\.id), ["ingredients"])
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: clip,
      supportedTasks: ["control"], referenceFamily: "crossview_ingredients").map(\.id), ["ingredients"])
    XCTAssertEqual(GenerationInputPurpose.appearance.actions(for: image, clip: clip,
      supportedTasks: ["ref2va"], referenceFamily: "msr").map(\.id), ["imageAppearance"])
    XCTAssertEqual(GenerationInputPurpose.appearance.actions(for: audio, clip: clip,
      supportedTasks: ["ref2va"], referenceFamily: "msr").map(\.id), ["msrVoice"])
    XCTAssertTrue(GenerationInputPurpose.appearance.actions(for: image, clip: clip,
      supportedTasks: ["ref2va"], referenceFamily: "ingredients").isEmpty)
  }

  func testUnknownFamilyKeepsAdvertisedTaskFallback() {
    let image = MediaAsset(name: "Subject", kind: .image)
    XCTAssertEqual(GenerationInputPurpose.ingredients.actions(for: image, clip: Clip(engine: .ltx25),
      supportedTasks: ["control"], referenceFamily: "future_adapter").map(\.id), ["ingredients"])
  }

  func testKnownGuideFamiliesRejectWrongAndUnknownSavedTypes() {
    let movie = MediaAsset(name: "Guide", kind: .video)
    let clip = Clip(engine: .ltx25)
    var attachment = Attachment(assetID: movie.id, role: .control)
    attachment.controlType = "motion_track"
    XCTAssertTrue(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: clip, supportedTasks: ["control"], referenceFamily: "motion_track"))
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: clip, supportedTasks: ["control"], referenceFamily: "union"))
    attachment.controlType = "unknown_guide"
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: clip, supportedTasks: ["control"]))
    attachment.controlType = "crossview_warp"
    attachment.referenceRole = "warp"
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: Clip(engine: .h3), supportedTasks: ["control"]))
    XCTAssertTrue(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: clip, supportedTasks: ["control"], referenceFamily: "crossview_warp"))
    attachment.referenceRole = nil
    XCTAssertFalse(GenerationInputPurpose.canReplace(attachment, original: movie, with: movie,
      clip: clip, supportedTasks: ["control"], referenceFamily: "crossview_warp"))
  }

  func testDrawThingsReferenceModelOffersAppearanceBeforeAnyReferenceIsAttached() throws {
    var clip = Clip(engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3_ref", modelFamily: "minimaxh3")
    clip.drawThings?.modelModifier = "ref2va"
    let image = MediaAsset(name: "Subject", kind: .image)
    XCTAssertEqual(clip.inferredTask, "t2v")
    let actions = GenerationInputPurpose.appearance.actions(for: image, clip: clip)
    XCTAssertEqual(actions.map(\.id), ["imageAppearance"])
    XCTAssertTrue(clip.canAssignDrawThingsInput(image, role: .reference))
    try clip.attachReference(image, action: XCTUnwrap(actions.first))
    XCTAssertEqual(clip.inferredTask, "ref2va")
    XCTAssertEqual(clip.attachments.first?.role, .reference)
    XCTAssertFalse(clip.attachments.contains { [.first, .last].contains($0.role) })
  }

  func testPreparedGuideChoicesUseTheKnownAdapterFamily() {
    XCTAssertEqual(GenerationControlPurpose.available(engine: .ltx25, referenceFamily: "union",
      supportedTasks: ["control"]), [.canny, .depth, .pose])
    XCTAssertEqual(GenerationControlPurpose.available(engine: .ltx25, referenceFamily: "motion_track",
      supportedTasks: ["control"]), [.motionTracks])
    XCTAssertEqual(GenerationControlPurpose.available(engine: .ltx25, referenceFamily: "crossview_ingredients",
      supportedTasks: ["control"]), [.crossViewWarp, .crossViewSource])
    XCTAssertEqual(GenerationControlPurpose.available(engine: .h3, referenceFamily: nil,
      supportedTasks: ["control"]), [.canny, .depth, .pose, .hed, .mlsd])
    XCTAssertTrue(GenerationControlPurpose.available(engine: .ltx25, referenceFamily: nil,
      supportedTasks: ["control"]).isEmpty)
    XCTAssertTrue(GenerationControlPurpose.available(engine: .ltx25, referenceFamily: "motion_track",
      supportedTasks: ["t2v"]).isEmpty)
  }

  @MainActor private func fixture(family: String?, tasks: [String], task: String = "t2v") throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var object: [String: Any] = ["supportedTasks": tasks, "presets": [], "controls": [
      "stepsEditable": false, "refinementStepsEditable": false, "cfgEditable": false,
      "shiftEditable": false, "stepsExplanation": "", "cfgExplanation": "", "shiftExplanation": ""]]
    if let family { object["referenceFamily"] = family }
    let generation = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: object))
    store.profiles = [ModelProfile(id: "selected", name: "Selected", engine: "ltx25", task: task, generation: generation)]
    var clip = Clip(engine: .ltx25)
    clip.profileID = "selected"
    clip.generationSelection = GenerationSelection(task: task, preset: .custom)
    clip.generationSelection?.steps = 13
    clip.generationSelection?.cfg = 2.5
    clip.seed = 41
    let image = MediaAsset(name: "First", kind: .image, path: root.appendingPathComponent("first.png").path)
    var first = Attachment(assetID: image.id, role: .first)
    first.strength = 0.4
    clip.attachments = [first]
    store.project.assets = [image]
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    return store
  }

  @MainActor func testPreparedGuideSelectsExactTypeInOneUndoableChange() throws {
    let store = try fixture(family: "union", tasks: ["control", "fflf"])
    let movie = MediaAsset(name: "Depth movie", kind: .video, path: "/tmp/depth.mp4")
    store.project.assets.append(movie)
    let before = store.project
    XCTAssertTrue(store.usePreparedGenerationGuide(movie, purpose: .depth))
    let result = try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(result.inferredTask, "control")
    XCTAssertEqual(result.attachments.first, before.clips[0].attachments.first)
    XCTAssertEqual(result.attachments.last?.controlType, "depth_map")
    XCTAssertEqual(result.attachments.last?.role, .control)
    XCTAssertEqual(result.attachments.last?.description, movie.name)
    XCTAssertEqual(result.generationSelection?.steps, 13)
    XCTAssertEqual(result.generationSelection?.cfg, 2.5)
    XCTAssertEqual(result.seed, 41)
    XCTAssertEqual(store.project.assets, before.assets)
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testImportedSequenceMovieSupportsPreparedGuideWithoutChangingMediaLink() throws {
    let store = try fixture(family: "motion_track", tasks: ["control"])
    let sequence = MediaAsset(name: "Imported image sequence", kind: .sequence, path: "/tmp/imported-sequence.mp4")
    store.project.assets.append(sequence)
    let before = store.project
    XCTAssertTrue(store.usePreparedGenerationGuide(sequence, purpose: .motionTracks))
    let guide = try XCTUnwrap(store.selectedClip?.attachments.last)
    XCTAssertEqual(guide.role, .control)
    XCTAssertEqual(guide.controlType, "motion_track")
    XCTAssertEqual(guide.assetID, sequence.id)
    XCTAssertEqual(store.project.assets, before.assets)
    XCTAssertEqual(store.project.assets.last?.kind, .sequence)
    XCTAssertEqual(store.project.assets.last?.path, sequence.path)
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testCrossViewAddsExplicitWarpAndOriginalSourceRoles() throws {
    let store = try fixture(family: "crossview_ingredients", tasks: ["control"])
    let warp = MediaAsset(name: "Warp", kind: .video, path: "/tmp/warp.mp4")
    let source = MediaAsset(name: "Original", kind: .video, path: "/tmp/source.mp4")
    store.project.assets += [warp, source]
    XCTAssertTrue(store.usePreparedGenerationGuide(warp, purpose: .crossViewWarp))
    XCTAssertTrue(store.usePreparedGenerationGuide(source, purpose: .crossViewSource))
    let guides = try XCTUnwrap(store.selectedClip).attachments.filter { $0.role == .control }
    XCTAssertEqual(guides.map(\.controlType), ["crossview_warp", "crossview_warp"])
    XCTAssertEqual(guides.compactMap(\.referenceRole), ["warp", "source"])
    XCTAssertEqual(guides.map(\.assetID), [warp.id, source.id])
    XCTAssertEqual(store.selectedClip?.generationSelection?.steps, 13)
  }

  @MainActor func testPreparedGuideRejectsWrongFamilyAndImageWithoutMutation() throws {
    let store = try fixture(family: "union", tasks: ["control"])
    let movie = MediaAsset(name: "Tracks", kind: .video, path: "/tmp/tracks.mp4")
    store.project.assets.append(movie)
    let before = store.project
    XCTAssertFalse(store.usePreparedGenerationGuide(movie, purpose: .motionTracks))
    XCTAssertFalse(store.usePreparedGenerationGuide(before.assets[0], purpose: .depth))
    XCTAssertEqual(store.project, before)
    XCTAssertFalse(store.canUndo)
  }

  @MainActor func testFreshFamilyMetadataWinsAndStaleDescriptionUsesPinnedProfile() throws {
    let store = try fixture(family: "union", tasks: ["control"])
    let clip = try XCTUnwrap(store.selectedClip)
    store.generationDescriptions[clip.id] = ["studioEngine": "ltx25", "studioTask": clip.inferredTask,
      "studioProfile": clip.profileID, "studioInput": "stale", "generation": ["referenceFamily": "ingredients"]]
    XCTAssertEqual(store.generationInputReferenceFamily(for: clip), "union")
    store.generationDescriptions[clip.id]?["studioInput"] = store.generationRequestKey(for: clip)
    XCTAssertEqual(store.generationInputReferenceFamily(for: clip), "ingredients")
  }

  @MainActor func testTimedKeyframeSelectsFFLFWithoutResettingAuthoredSettings() throws {
    let store = try fixture(family: "ordinary", tasks: ["t2v", "fflf"])
    let image = try XCTUnwrap(store.project.assets.first)
    let before = store.project
    XCTAssertTrue(store.useGenerationKeyframe(image))
    XCTAssertEqual(store.selectedClip?.inferredTask, "fflf")
    XCTAssertEqual(store.selectedClip?.attachments.last?.role, .keyframe)
    XCTAssertEqual(store.selectedClip?.attachments.last?.assetID, image.id)
    XCTAssertEqual(store.selectedClip?.generationSelection?.steps, 13)
    XCTAssertEqual(store.selectedClip?.generationSelection?.cfg, 2.5)
    XCTAssertEqual(store.selectedClip?.seed, 41)
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testTimedKeyframePreservesControlAndAudioDriverTasks() throws {
    for task in ["control", "a2v"] {
      let store = try fixture(family: nil, tasks: [task, "fflf"], task: task)
      let image = try XCTUnwrap(store.project.assets.first)
      let before = try XCTUnwrap(store.selectedClip?.generationSelection)
      XCTAssertTrue(store.useGenerationKeyframe(image))
      XCTAssertEqual(store.selectedClip?.generationSelection, before)
      XCTAssertEqual(store.selectedClip?.inferredTask, task)
    }
  }

  @MainActor func testTimedKeyframeRejectsImageOnlyRecipeWithoutMutation() throws {
    let store = try fixture(family: "ordinary", tasks: ["i2v"])
    let before = store.project
    XCTAssertFalse(store.useGenerationKeyframe(before.assets[0]))
    XCTAssertEqual(store.project, before)
    XCTAssertFalse(store.canUndo)
  }
}
