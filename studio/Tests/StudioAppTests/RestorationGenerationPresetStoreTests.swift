import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class RestorationGenerationPresetStoreTests: XCTestCase {
  @MainActor func testFirstLaunchLegacyMigrationWaitsForDiscoveryWithoutRequiringAGlobalRootLink() async throws {
    let store = try fixture(), models = store.dataDirectory.appendingPathComponent("models"), flash = models.appendingPathComponent("FlashVSR-Plus")
    try store.persistGenerationLibrary(.init()); store.videoUpscaleRuntime.modelRoot = ""
    store.videoUpscalePresetName = "First launch"; store.saveVideoUpscalePreset()
    let legacy = try XCTUnwrap(store.videoUpscalePresets.first)
    try store.migrateLegacyRestorationPresets()
    XCTAssertNil(store.generationPresets.preset(id: legacy.id.uuidString))
    for file in ["turbo/dit.safetensors", "turbo/lq_proj_in.safetensors", "posi_prompt.pth", "TCDecoder.ckpt"] {
      let url = flash.appendingPathComponent(file); try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true); try Data([0]).write(to: url)
    }
    try store.setGenerationLibraryRoot(models.path); await store.refreshGenerationLibrary(force: true)
    let migrated = try XCTUnwrap(store.generationPresets.preset(id: legacy.id.uuidString))
    XCTAssertNotNil(migrated.snapshot.model?.sourceIdentity)
    try await store.applyRestorationPreset(migrated, expectedTarget: store.restorationPresetTargetToken())
    XCTAssertEqual(store.videoUpscaleRuntime.modelRoot, flash.path)
  }
  @MainActor func testLegacyPresetBeforeDiscoveryResolvesToInstalledModelWithTheSameUUID() async throws {
    let store = try fixture(), installed = store.generationLibraryIndex
    try store.persistGenerationLibrary(.init())
    store.videoUpscalePresetName = "Before discovery"; store.saveVideoUpscalePreset()
    let legacy = try XCTUnwrap(store.videoUpscalePresets.first)
    try store.migrateLegacyRestorationPresets()
    try store.persistGenerationLibrary(installed); try store.migrateLegacyRestorationPresets()
    let migrated = try XCTUnwrap(store.generationPresets.preset(id: legacy.id.uuidString))
    XCTAssertEqual(migrated.snapshot.model?.id, "flash")
    try await store.applyRestorationPreset(migrated, expectedTarget: store.restorationPresetTargetToken())
    let reopened = StudioStore(dataDirectory: store.dataDirectory, restoreSession: false)
    XCTAssertEqual(reopened.generationPresets.preset(id: legacy.id.uuidString)?.snapshot.model?.id, "flash")
  }
  @MainActor func testStandaloneApplyUndoRestoresWorkingRecipeAndAssociationTogether() async throws {
    let store = try fixture(); store.showVideoUpscale = true
    let original = store.videoUpscaleSettings
    var snapshot = try store.captureRestorationPreset(); snapshot.fields["temporalSigma"] = .number(1.5)
    try await store.applyRestorationPreset(.init(name: "New", snapshot: snapshot), expectedTarget: store.restorationPresetTargetToken())
    XCTAssertTrue(store.canUndo); store.undo()
    XCTAssertEqual(store.videoUpscaleSettings, original); XCTAssertNil(store.restorationPresetSelection)
  }
  @MainActor private func fixture() throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.videoUpscaleRuntime.modelRoot = root.path
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "flash", backend: "restoration", family: "flashVSR"), name: "FlashVSR Turbo", workspaces: [.restoration, .finishing], tasks: ["flashVSR"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "flashVSR", components: ["modelRoot": root.path, "variant": "turbo", "decoder": "tiny"])), alias: "fixture")
    try store.persistGenerationLibrary(index); return store
  }
  @MainActor func testApplyPreservesCanvasAndSelectionModifiedStateSurvivesRestart() async throws {
    let store = try fixture(); store.videoUpscaleSettings.width = 1872; store.videoUpscaleSettings.height = 1056; store.videoUpscaleSettings.fit = .fill
    var snapshot = try store.captureRestorationPreset(); snapshot.fields["temporalSigma"] = .number(1.5)
    let preset = StudioGenerationPreset(name: "My restoration", snapshot: snapshot)
    try await store.applyRestorationPreset(preset, expectedTarget: store.restorationPresetTargetToken())
    XCTAssertEqual(store.videoUpscaleSettings.temporalSigma, 1.5); XCTAssertEqual(store.videoUpscaleSettings.width, 1872)
    XCTAssertEqual(store.videoUpscaleSettings.height, 1056); XCTAssertEqual(store.videoUpscaleSettings.fit, .fill)
    store.videoUpscaleSettings.temporalSigma = 2; store.saveVideoUpscalePreferences()
    let restored = StudioStore(dataDirectory: store.dataDirectory, restoreSession: false)
    XCTAssertEqual(restored.restorationPresetSelection?.presetID, preset.id)
    XCTAssertTrue(try XCTUnwrap(restored.restorationPresetSelection).isModified(restored.captureRestorationPreset()))
    let settings = restored.videoUpscaleSettings, runtime = restored.videoUpscaleRuntime
    restored.setRestorationPresetSelection(nil)
    XCTAssertEqual(restored.videoUpscaleSettings, settings); XCTAssertEqual(restored.videoUpscaleRuntime, runtime)
  }
  @MainActor func testToolAndSourceChangesRejectOldTarget() async throws {
    let store = try fixture(), token = try store.restorationPresetTargetToken()
    let preset = StudioGenerationPreset(name: "Old", snapshot: try store.captureRestorationPreset())
    store.videoUpscaleSourcePath = "/a-new-source.mov"; let settings = store.videoUpscaleSettings
    do { try await store.applyRestorationPreset(preset, expectedTarget: token); XCTFail("Changed source") } catch {}
    XCTAssertEqual(store.videoUpscaleSettings, settings); XCTAssertNil(store.restorationPresetSelection)
  }
  @MainActor func testLegacyMigrationUsesStableIDsIsIdempotentAndPreservesOriginalPreferences() throws {
    let store = try fixture(); store.videoUpscalePresetName = "Old full-size recipe"; store.videoUpscaleSettings.width = 1920; store.videoUpscaleSettings.height = 1080
    store.saveVideoUpscalePreset()
    let old = try XCTUnwrap(store.videoUpscalePresets.first), file = store.dataDirectory.appendingPathComponent("video-upscale.json")
    let original = try Data(contentsOf: file)
    try store.migrateLegacyRestorationPresets(); try store.migrateLegacyRestorationPresets()
    let migrated = try XCTUnwrap(store.generationPresets.preset(id: old.id.uuidString))
    XCTAssertEqual(migrated.name, old.name); XCTAssertNil(migrated.snapshot.fields["width"]); XCTAssertNil(migrated.snapshot.fields["height"])
    XCTAssertEqual(store.generationPresets.presets.filter { $0.id == old.id.uuidString }.count, 1)
    XCTAssertEqual(try Data(contentsOf: file), original)
    let restored = StudioStore(dataDirectory: store.dataDirectory, restoreSession: false)
    XCTAssertEqual(restored.generationPresets.preset(id: old.id.uuidString), migrated)
    XCTAssertEqual(restored.videoUpscalePresets.first?.settings.width, 1920)
  }
}
