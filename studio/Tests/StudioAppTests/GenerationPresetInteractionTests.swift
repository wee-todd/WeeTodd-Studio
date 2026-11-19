import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class GenerationPresetInteractionTests: XCTestCase {
  @MainActor func testMissingComponentDoesNotInventAModifiedState() async throws {
    let store = try fixture(), target = GenerationPresetTarget.restoration(.flashVSR)
    _ = try await store.saveNewPreset(name: "Current", for: target, expectedTarget: store.presetTargetToken(for: target))
    try FileManager.default.removeItem(atPath: store.videoUpscaleRuntime.modelRoot)
    await store.refreshGenerationLibrary(force: true)
    let actions = store.presetActions(for: target)
    XCTAssertFalse(actions.isModified); XCTAssertFalse(actions.canUpdate); XCTAssertNotNil(actions.issue)
  }
  @MainActor private func fixture() throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let models = root.appendingPathComponent("models")
    try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    store.videoUpscaleRuntime.modelRoot = models.path
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "flash", backend: "restoration", family: "flashVSR"), name: "FlashVSR", workspaces: [.restoration], tasks: ["flashVSR"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "flashVSR", components: ["modelRoot": models.path, "variant": "turbo", "decoder": "tiny"])), alias: "fixture")
    try store.persistGenerationLibrary(index); return store
  }
  @MainActor func testExactNoPresetCleanDirtyReviewUpdateAndSaveAsNewStates() async throws {
    let store = try fixture(), target = GenerationPresetTarget.restoration(.flashVSR)
    let none = store.presetActions(for: target)
    XCTAssertFalse(none.hasSelection); XCTAssertFalse(none.canUpdate); XCTAssertTrue(none.canSaveNew)
    let first = try await store.saveNewPreset(name: "Mine", for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertEqual(store.restorationPresetSelection?.presetID, first.id)
    XCTAssertFalse(store.presetActions(for: target).isModified); XCTAssertFalse(store.presetActions(for: target).canUpdate)
    store.videoUpscaleSettings.temporalSigma = 1.5
    XCTAssertTrue(store.presetActions(for: target).canUpdate); XCTAssertEqual(store.presetActions(for: target).changeCount, 1)
    let review = try store.beginPresetUpdate(for: target)
    XCTAssertEqual(review.changes.count, 1); XCTAssertEqual(review.changes.first?.saved, "1"); XCTAssertEqual(review.changes.first?.current, "1.5")
    XCTAssertEqual(store.generationPresets.preset(id: first.id), first) // Opening and cancelling a review writes nothing.
    try await store.confirmPresetUpdate(review)
    let updated = try XCTUnwrap(store.generationPresets.preset(id: first.id))
    XCTAssertEqual(updated.revision, 2); XCTAssertFalse(store.presetActions(for: target).isModified)
    let copy = try await store.saveNewPreset(name: "Copy", for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertNotEqual(copy.id, updated.id); XCTAssertEqual(store.generationPresets.preset(id: first.id), updated)
    XCTAssertEqual(store.restorationPresetSelection?.presetID, copy.id)
    let settings = store.videoUpscaleSettings
    try store.clearPresetSelection(for: target)
    XCTAssertFalse(store.presetActions(for: target).hasSelection); XCTAssertEqual(store.videoUpscaleSettings, settings)
  }
  @MainActor func testChangedDraftPresetRevisionAndDisconnectedComponentsRejectReviewedOverwrite() async throws {
    let store = try fixture(), target = GenerationPresetTarget.restoration(.flashVSR)
    let first = try await store.saveNewPreset(name: "Mine", for: target, expectedTarget: store.presetTargetToken(for: target))
    store.videoUpscaleSettings.temporalSigma = 1.5
    let review = try store.beginPresetUpdate(for: target)
    store.videoUpscaleSettings.temporalSigma = 2
    do { try await store.confirmPresetUpdate(review); XCTFail("Unreviewed draft change") } catch {}
    XCTAssertEqual(store.generationPresets.preset(id: first.id), first)
    let next = try store.beginPresetUpdate(for: target)
    var library = store.generationPresets
    try library.update(id: first.id, expectedRevision: 1, snapshot: first.snapshot); try store.persistGenerationPresets(library)
    do { try await store.confirmPresetUpdate(next); XCTFail("Changed preset revision") } catch {}
    XCTAssertEqual(store.generationPresets.preset(id: first.id)?.revision, 2)
    let latest = try store.beginPresetUpdate(for: target)
    try FileManager.default.removeItem(atPath: store.videoUpscaleRuntime.modelRoot)
    do { try await store.confirmPresetUpdate(latest); XCTFail("Missing components") } catch {}
    XCTAssertEqual(store.generationPresets.preset(id: first.id)?.revision, 2)
    XCTAssertEqual(store.videoUpscaleSettings.temporalSigma, 2)
  }
  @MainActor func testExternalPresetLibraryChangeCannotBeOverwrittenFromCachedState() async throws {
    let store = try fixture(), target = GenerationPresetTarget.restoration(.flashVSR)
    let first = try await store.saveNewPreset(name: "Mine", for: target, expectedTarget: store.presetTargetToken(for: target))
    store.videoUpscaleSettings.temporalSigma = 1.5; let review = try store.beginPresetUpdate(for: target)
    var external = try GenerationPresetLibrary.read(from: store.generationPresetsFile)
    _ = try external.create(name: "Another app's preset", snapshot: first.snapshot); try external.write(to: store.generationPresetsFile)
    do { try await store.confirmPresetUpdate(review); XCTFail("External write must be preserved") } catch {}
    XCTAssertNotNil(try GenerationPresetLibrary.read(from: store.generationPresetsFile).presets.first { $0.name == "Another app's preset" })
    XCTAssertEqual(store.generationPresets.preset(id: first.id)?.revision, 1)
  }
  @MainActor func testCancelledSaveCannotCreateOrSelectAPreset() async throws {
    let store = try fixture(), target = GenerationPresetTarget.restoration(.flashVSR)
    let token = try store.presetTargetToken(for: target)
    let operation = Task { try await store.saveNewPreset(name: "Cancelled", for: target, expectedTarget: token) }
    operation.cancel()
    do { _ = try await operation.value; XCTFail("Cancelled save") } catch {}
    XCTAssertTrue(store.generationPresets.presets.isEmpty); XCTAssertNil(store.restorationPresetSelection)
  }
}
