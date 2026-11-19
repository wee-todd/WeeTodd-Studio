import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class ImageGenerationPresetStoreTests: XCTestCase {
  @MainActor func testUndoAssociationOnlyReapplicationKeepsThePreparedEstimate() async throws {
    let store = try store()
    let first = StudioGenerationPreset(name: "Current", snapshot: try store.captureImagePreset())
    try await store.applyImagePreset(first, expectedTarget: store.imagePresetTargetToken())
    let current = StudioGenerationPreset(name: "Same settings", snapshot: try store.captureImagePreset())
    try await store.applyImagePreset(current, expectedTarget: store.imagePresetTargetToken())
    store.imageEstimate = ["prepared": true]
    try store.undoImagePreset()
    XCTAssertEqual(store.imageEstimate?["prepared"] as? Bool, true)
  }
  @MainActor private func store() throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false); store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Project text"; draft.width = 1872; draft.height = 1056
    draft.mossImage?.transformerDirectory = root.appendingPathComponent("transformer").path
    draft.mossImage?.sharedDirectory = root.appendingPathComponent("shared").path
    for folder in ["transformer", "shared"] { try FileManager.default.createDirectory(at: root.appendingPathComponent(folder), withIntermediateDirectories: true) }
    let ref = GenerationModelRef(id: "moss", backend: "nativeMOSS", family: "moss_image", variant: "sft", precision: "bf16")
    let entry = GenerationModelEntry(reference: ref, name: "MOSS", workspaces: [.image], tasks: ["image"], availability: .installed,
      binding: .nativeMOSS(transformer: draft.mossImage!.transformerDirectory, shared: draft.mossImage!.sharedDirectory, library: root.path))
    var index = GenerationLibraryIndex(); try index.registerModel(entry, alias: "fixture")
    try store.persistGenerationLibrary(index); store.imageDraft = draft
    return store
  }
  @MainActor func testApplyingIsOneGuardedUndoAndAssociationSurvivesModifiedRestart() async throws {
    let store = try store(), original = try XCTUnwrap(store.imageDraft)
    var snapshot = try store.captureImagePreset(); snapshot.fields["steps"] = .integer(30)
    let preset = StudioGenerationPreset(name: "Test", snapshot: snapshot)
    let target = try store.imagePresetTargetToken()
    try await store.applyImagePreset(preset, expectedTarget: target)
    XCTAssertEqual(store.imageDraft?.steps, 30); XCTAssertEqual(store.imageDraft?.prompt, original.prompt)
    XCTAssertEqual(store.imageDraft?.width, original.width); XCTAssertEqual(store.imagePresetSelection?.presetID, preset.id)
    XCTAssertFalse(try XCTUnwrap(store.imagePresetSelection).isModified(store.captureImagePreset()))
    store.imageDraft?.steps = 40
    let restored = StudioStore(dataDirectory: store.dataDirectory, restoreSession: false)
    restored.restoreImageWorkspaces()
    XCTAssertEqual(restored.imageDraft?.steps, 40); XCTAssertEqual(restored.imagePresetSelection?.presetID, preset.id)
    XCTAssertTrue(try XCTUnwrap(restored.imagePresetSelection).isModified(restored.captureImagePreset()))
    XCTAssertThrowsError(try store.undoImagePreset()) // Subsequent edits cannot be overwritten by a stale undo.
  }
  @MainActor func testLateApplyAndAssociationOnlyActionsNeverChangeTheDraft() async throws {
    let store = try store(), token = try store.imagePresetTargetToken()
    let preset = StudioGenerationPreset(name: "Test", snapshot: try store.captureImagePreset())
    store.imageDraft?.prompt = "A newer edit"
    let current = store.imageDraft
    do { try await store.applyImagePreset(preset, expectedTarget: token); XCTFail("Stale apply must reject") } catch {}
    XCTAssertEqual(store.imageDraft, current)
    store.setImagePresetSelection(.init(preset)); store.setImagePresetSelection(nil)
    XCTAssertEqual(store.imageDraft, current); XCTAssertNil(store.imagePresetSelection)
  }
  @MainActor func testUndoRestoresTheExactDraftAndPriorAssociation() async throws {
    let store = try store(), original = store.imageDraft
    var snapshot = try store.captureImagePreset(); snapshot.fields["steps"] = .integer(20)
    try await store.applyImagePreset(.init(name: "Twenty", snapshot: snapshot), expectedTarget: store.imagePresetTargetToken())
    try store.undoImagePreset()
    XCTAssertEqual(store.imageDraft, original); XCTAssertNil(store.imagePresetSelection)
  }
}
