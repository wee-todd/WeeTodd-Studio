import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class FinishingGenerationPresetStoreTests: XCTestCase {
  @MainActor private func fixture() throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.videoUpscaleRuntime.modelRoot = root.path
    var clip = Clip(); clip.prompt = "My generation"; clip.videoProcessing = VideoProcessingRecipe(); clip.sourcePath = "/source.mov"
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "flash", backend: "restoration", family: "flashVSR"), name: "FlashVSR", workspaces: [.finishing, .restoration], tasks: ["flashVSR"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "flashVSR", components: ["modelRoot": root.path, "variant": "turbo", "decoder": "tiny"])), alias: "fixture")
    try store.persistGenerationLibrary(index); return store
  }
  @MainActor func testApplyChangesFinishingOnlyAsOneUndoableProjectTransaction() async throws {
    let store = try fixture(), original = store.project
    var snapshot = try store.captureFinishingPreset()
    guard case .object(var stages) = snapshot.fields["stages"] else { return XCTFail("Missing stages") }
    var stage = try XCTUnwrap(stages["0"]).decoded(GenerationPresetSnapshot.self)
    stage.fields["temporalSigma"] = .number(1.5); stages["0"] = try .encoded(stage); snapshot.fields["stages"] = .object(stages)
    let preset = StudioGenerationPreset(name: "Finish", snapshot: snapshot)
    try await store.applyFinishingPreset(preset, expectedTarget: store.finishingPresetTargetToken())
    XCTAssertEqual(store.selectedClip?.videoProcessing?.steps[0].settings.temporalSigma, 1.5)
    XCTAssertEqual(store.selectedClip?.generationFingerprint, original.clips[0].generationFingerprint)
    XCTAssertEqual(store.selectedClip?.sourcePath, original.clips[0].sourcePath); XCTAssertEqual(store.selectedClip?.prompt, original.clips[0].prompt)
    XCTAssertEqual(store.finishingPresetSelection?.presetID, preset.id)
    store.undo(); XCTAssertEqual(store.project, original)
  }
  @MainActor func testAssociationOnlyAndApplyingIdenticalEffectiveSettingsKeepPreparedWork() async throws {
    let store = try fixture(), original = try XCTUnwrap(store.selectedClip)
    let preset = StudioGenerationPreset(name: "Finish", snapshot: try store.captureFinishingPreset())
    store.preparedRecipe = "/prepared.json"; store.preparedFingerprint = "ready"
    store.setFinishingPresetSelection(.init(preset))
    XCTAssertEqual(store.preparedRecipe, "/prepared.json"); XCTAssertEqual(store.selectedClip, original)
    store.setFinishingPresetSelection(nil)
    try await store.applyFinishingPreset(preset, expectedTarget: store.finishingPresetTargetToken())
    XCTAssertEqual(store.preparedRecipe, "/prepared.json"); XCTAssertEqual(store.preparedFingerprint, "ready")
  }
  @MainActor func testChangedRecipeRefusesLateApplyAndModelBindingEditsInvalidateAdmissionTarget() async throws {
    let store = try fixture(), token = try store.finishingPresetTargetToken()
    let preset = StudioGenerationPreset(name: "Finish", snapshot: try store.captureFinishingPreset())
    let destination = store.videoProcessingDestination(for: try XCTUnwrap(store.selectedClip))
    store.project.generationLibraryBindings = ["finishing:" + store.selectedClipID!.uuidString + ":restore": .init(id: "different", backend: "restoration", family: "flashVSR")]
    XCTAssertFalse(store.matchesVideoProcessingModelBindings(destination))
    do { try await store.applyFinishingPreset(preset, expectedTarget: token); XCTFail("Changed binding") } catch {}
    XCTAssertEqual(store.selectedClip?.videoProcessing?.steps[0].settings.temporalSigma, 1)
  }
}
