import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class LTX25ConfigLibraryTests: XCTestCase {
  @MainActor func testFastDevIsExplicitPersistsAndResetsWithoutChangingRecommendations() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip()
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25ConfigID, LTX25GenerationConfig.distilled.id)
    let config = try XCTUnwrap(store.ltx25Configs.first { $0.id == "builtin-ltx25-dev-fast" })
    for original in [LTX25GenerationConfig.distilled, .dev, .devHQ] {
      XCTAssertNil(original.selection.ltx25DiffusionVAE)
      XCTAssertNil(original.selection.steps)
      XCTAssertNil(original.selection.refinementSteps)
    }
    store.applyLTX25Config(config)
    XCTAssertFalse(store.ltx25ConfigIsModified)
    XCTAssertEqual(store.recommendedLTX25Config(for: try XCTUnwrap(store.selectedClip)), .dev)
    let reopened = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(store.project))
    XCTAssertEqual(reopened.clips.first?.generationSelection, store.selectedClip?.generationSelection)
    store.editClip { $0.generationSelection?.ltx25DiffusionVAE?.queryChunkSize = 512 }
    XCTAssertTrue(store.ltx25ConfigIsModified)
    store.resetLTX25Config()
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25DiffusionVAE?.queryChunkSize, 8192)
    XCTAssertFalse(store.ltx25ConfigIsModified)
    store.applyLTX25Config(.distilled)
    XCTAssertNil(store.selectedClip?.generationSelection?.ltx25DiffusionVAE)
    store.addClip()
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25ConfigID, LTX25GenerationConfig.distilled.id)
  }

  @MainActor func testSavedControlConfigRejectsAnotherControlFamily() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("union.json")
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "conditioning": ["task": "control"], "components": ["ic_loras": [["/models/union", 1.0]]],
      "config": ["stage1_steps": 8, "stage2_steps": 3]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip(); store.editClip {
      $0.profileID = file.path; $0.generationSelection = GenerationSelection(task: "control", preset: .custom)
      var guide = Attachment(assetID: UUID(), role: .control); guide.controlType = "canny_edges"
      $0.attachments = [guide]
    }
    let saved = await store.saveLTX25Config(name: "My Union")
    XCTAssertTrue(saved)
    let config = try XCTUnwrap(store.savedLTX25Configs.first)
    store.editClip { $0.attachments[0].controlType = "motion_track" }
    let before = store.selectedClip
    store.applyLTX25Config(config)
    XCTAssertEqual(store.selectedClip, before)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testSavingLegacyGuidedConfigPreservesInheritedRecipeSettings() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("legacy.json")
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25", "ffmpeg": "/usr/bin/true",
      "components": ["transformer_path": "/models/dev", "distilled_lora_path": "/models/refinement"],
      "config": ["pipeline_mode": "guided", "stage1_steps": 12, "stage2_steps": 3,
        "video_cfg_scale": 2.5, "audio_cfg_scale": 6.0, "stg_scale": 0.4,
        "stg_blocks": [20], "frame_rate": 24]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime.profilesDirectory = root.path; store.runtime.ffmpegPath = "/usr/bin/true"
    store.addClip()
    for binding in [file.path, "auto"] {
      store.editClip {
        $0.prompt = "Keep this shot."; $0.profileID = binding
        $0.generationSelection = GenerationSelection(task: "t2v", preset: .custom)
        $0.generationSelection?.ltx25Guidance = LTX25GuidanceSettings(mode: .guided, experimentalEnabled: true)
      }
      var request = try store.payload()
      request["runtime"] = ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"]
      let before = try NativeLTXPreparation.compose(request: request)["recipe"] as! [String: Any]
      let savedTrue = await store.saveLTX25Config(name: "Legacy guided")
      XCTAssertTrue(savedTrue)
      XCTAssertEqual(store.selectedClip?.generationSelection?.steps, 12)
      XCTAssertEqual(store.selectedClip?.generationSelection?.cfg, 2.5)
      XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25Guidance?.audioCFG, 6)
      XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25Guidance?.stgBlocks, [20])
      XCTAssertEqual(store.savedLTX25Configs.last?.componentBindingID, file.path)
      request = try store.payload()
      request["runtime"] = ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"]
      let after = try NativeLTXPreparation.compose(request: request)["recipe"] as! [String: Any]
      XCTAssertTrue(NSDictionary(dictionary: before).isEqual(to: after))
    }
  }

  @MainActor func testSavedInheritedSingleStageConfigRejectsFrameContinuityWithoutChangingClip() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("single-stage.json")
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "components": [:], "config": ["stage1_steps": 8, "stage2_steps": 0,
        "stage1_sampler": "euler_ancestral", "ic_lora_single_stage": true]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip(); store.editClip {
      $0.profileID = file.path; $0.generationSelection = GenerationSelection(preset: .custom)
    }
    let savedTrue = await store.saveLTX25Config(name: "Inherited single stage")
    XCTAssertTrue(savedTrue)
    let config = try XCTUnwrap(store.savedLTX25Configs.first)
    store.editClip { $0.continuity = ClipContinuity(mode: "frame", sourceClipID: UUID()) }
    let before = store.selectedClip
    store.applyLTX25Config(config)
    XCTAssertEqual(store.selectedClip, before)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testSavedMSRConfigRejectsPlainPromptClip() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("msr.json")
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "conditioning": ["task": "ref2va"],
      "components": ["msr_lora_path": "/models/msr", "ic_loras": [["/models/msr", 1.0]]],
      "config": ["stage1_steps": 8, "stage2_steps": 3, "ic_lora_single_stage": true]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip(); store.editClip {
      $0.profileID = file.path; $0.generationSelection = GenerationSelection(task: "ref2va", preset: .custom)
    }
    let savedTrue = await store.saveLTX25Config(name: "My MSR")
    XCTAssertTrue(savedTrue)
    let config = try XCTUnwrap(store.savedLTX25Configs.first)
    store.addClip(); let before = store.selectedClip
    store.applyLTX25Config(config)
    XCTAssertEqual(store.selectedClip, before)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testAutomaticBindingChangesInvalidateReadinessEvenWithoutCatalogProfiles() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime.profilesDirectory = root.path; store.addClip()
    let clip = try XCTUnwrap(store.selectedClip), before = store.generationRequestKey(for: clip)
    let file = root.appendingPathComponent("component-only.json")
    try Data(#"{"engine":"ltx25","config":{"stage1_steps":7},"components":{}}"#.utf8).write(to: file)
    let added = store.generationRequestKey(for: clip)
    XCTAssertNotEqual(added, before)
    try Data(#"{"engine":"ltx25","config":{"stage1_steps":7},"components":{"transformer_path":"/new"}}"#.utf8).write(to: file)
    XCTAssertNotEqual(store.generationRequestKey(for: clip), added)
    try FileManager.default.removeItem(at: file)
    XCTAssertEqual(store.generationRequestKey(for: clip), before)
  }

  @MainActor func testApplyingDevConfigToASceneLeaderPreservesTheWholeProject() async throws {
    let store = StudioStore(dataDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), restoreSession: false)
    store.addClip(); let leader = try XCTUnwrap(store.selectedClip)
    var follower = Clip(engine: .ltx25)
    follower.continuity = ClipContinuity(mode: "scene", sourceClipID: leader.id)
    store.project.clips.append(follower)
    let before = store.project
    store.applyLTX25Config(.dev)
    XCTAssertEqual(store.project, before)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testLinkingDevSetupAppliesItsConfigWithoutLeavingADistilledSelection() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip(); store.editClip { $0.prompt = "Keep my shot." }
    let state = ModelSetupState()
    state.begin(try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-guided" }))
    let file = root.appendingPathComponent("dev-binding.json")
    try Data("{}".utf8).write(to: file); state.resultPath = file.path
    state.useRecipeForSelectedClip(store: store)
    XCTAssertEqual(store.selectedClip?.profileID, "auto")
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25ConfigID, "builtin-ltx25-dev")
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25Guidance?.mode, .guided)
    XCTAssertEqual(store.selectedClip?.prompt, "Keep my shot.")
  }

  @MainActor func testSaveReloadApplyAndDeleteKeepTheClipsCapturedSettings() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip()
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25ConfigID, "builtin-ltx25-distilled")
    store.applyLTX25Config(.dev)
    store.editClip { $0.generationSelection?.steps = 18; $0.generationSelection?.cfg = 2.2 }
    XCTAssertTrue(store.ltx25ConfigIsModified)
    let savedTrue = await store.saveLTX25Config(name: "My Dev")
    XCTAssertTrue(savedTrue)
    let custom = try XCTUnwrap(store.savedLTX25Configs.first)
    XCTAssertEqual(custom.name, "My Dev")
    XCTAssertFalse(store.ltx25ConfigIsModified)
    let restored = StudioStore(dataDirectory: root, restoreSession: false)
    XCTAssertEqual(restored.savedLTX25Configs, [custom])
    restored.addClip()
    restored.applyLTX25Config(custom)
    XCTAssertEqual(restored.selectedClip?.generationSelection?.steps, 18)
    XCTAssertEqual(restored.selectedClip?.generationSelection?.cfg, 2.2)
    restored.deleteLTX25Config(custom.id)
    XCTAssertTrue(restored.savedLTX25Configs.isEmpty)
    XCTAssertEqual(restored.selectedClip?.generationSelection?.steps, 18)
    XCTAssertEqual(restored.selectedClip?.generationSelection?.ltx25ConfigName, "My Dev")
    restored.resetLTX25Config()
    XCTAssertEqual(restored.selectedClip?.generationSelection?.steps, nil)
    XCTAssertEqual(restored.selectedClip?.generationSelection?.ltx25Guidance?.mode, .guided)
  }

  @MainActor func testInvalidOrCorruptLibraryNeverOverwritesSavedConfigs() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("ltx25-configs.json")
    try Data("broken library".utf8).write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip()
    let savedFalse = await store.saveLTX25Config(name: "Example")
    XCTAssertFalse(savedFalse)
    XCTAssertEqual(try Data(contentsOf: file), Data("broken library".utf8))
    let blankSaved = await store.saveLTX25Config(name: " ")
    XCTAssertFalse(blankSaved)
  }
}
