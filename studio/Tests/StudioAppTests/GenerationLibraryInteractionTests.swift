import Foundation
import XCTest
import StudioCore
import ImageLayout
@testable import WeeToddStudio

final class GenerationLibraryInteractionTests: XCTestCase {
  @MainActor private func fixture() throws -> (StudioStore, GenerationModelRef, GenerationLoRARef) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for folder in ["old", "new", "shared"] { try FileManager.default.createDirectory(at: root.appendingPathComponent(folder), withIntermediateDirectories: true) }
    for relative in ["old/config.json", "new/config.json", "shared/text_encoder/config.json", "shared/vae/config.json", "shared/tokenizer/tokenizer.json", "shared/tokenizer/tokenizer_config.json"] {
      let file = root.appendingPathComponent(relative)
      try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data("{}".utf8).write(to: file)
    }
    for name in ["old", "new"] {
      try JSONSerialization.data(withJSONObject: ["dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42, "text_dim": 4096, "latent_channels": 128, "freq_dim": 256, "rope_base": 10000]).write(to: root.appendingPathComponent(name + "/config.json"))
    }
    for relative in ["old/model.safetensors", "new/model.safetensors", "shared/text_encoder/model.safetensors", "shared/vae/model.safetensors"] {
      try Data([0]).write(to: root.appendingPathComponent(relative))
    }
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false); store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.width = 1872; draft.height = 1056; draft.prompt = "Keep these exact words"
    draft.mossImage?.transformerDirectory = root.appendingPathComponent("old").path; draft.mossImage?.sharedDirectory = root.appendingPathComponent("shared").path
    var index = GenerationLibraryIndex()
    for name in ["old", "new"] {
      let ref = GenerationModelRef(id: name, backend: "nativeMOSS", family: "moss_image", variant: "sft", precision: "bf16")
      try index.registerModel(.init(reference: ref, name: name, workspaces: [.image], tasks: ["image"], availability: .installed,
        binding: .nativeMOSS(transformer: root.appendingPathComponent(name).path, shared: root.appendingPathComponent("shared").path, library: root.path)), alias: name)
    }
    let ref = GenerationLoRARef(id: "style", backend: "native", family: "moss_image")
    let file = root.appendingPathComponent("style.safetensors"); try Data([0]).write(to: file)
    try index.registerLoRA(.init(reference: ref, name: "My style", compatibleFamilies: ["moss_image"], availability: .installed,
      binding: .nativeFile(path: file.path)), alias: "style")
    try store.persistGenerationLibrary(index); store.imageDraft = draft
    return (store, index.snapshot.modelEntries[1].reference, ref)
  }
  @MainActor func testModelChoicePreservesPresetAssociationAndCreativeStateWithoutGenerating() async throws {
    let (store, model, _) = try fixture(), target = GenerationPresetTarget.image
    let preset = try await store.saveNewPreset(name: "Original", for: target, expectedTarget: store.presetTargetToken(for: target))
    let original = try XCTUnwrap(store.imageDraft)
    try await store.useGenerationLibraryModel(model, for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory.hasSuffix("/new"), true)
    XCTAssertEqual(store.imageDraft?.prompt, original.prompt); XCTAssertEqual(store.imageDraft?.width, original.width); XCTAssertEqual(store.imageDraft?.height, original.height)
    XCTAssertEqual(store.imagePresetSelection?.presetID, preset.id); XCTAssertTrue(store.presetActions(for: target).isModified)
    XCTAssertEqual(store.generationPresets.preset(id: preset.id), preset); XCTAssertFalse(store.bridge.busy)
  }
  @MainActor func testLibraryLoRAAddDoesNotSelectAPresetOrInsertPositivePromptText() async throws {
    let (store, _, lora) = try fixture(), target = GenerationPresetTarget.image
    let prompt = store.imageDraft?.prompt
    try await store.addGenerationLibraryLoRA(lora, for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertEqual(store.imageDraft?.mossImage?.loras?.count, 1); XCTAssertEqual(store.imageDraft?.prompt, prompt)
    XCTAssertNil(store.imagePresetSelection); XCTAssertFalse(store.bridge.busy)
  }
  @MainActor func testModelPickerCanLeaveMOSSWithItsCanvasAndLayoutThenUndoAndReturn() async throws {
    let (store, _, _) = try fixture(), target = GenerationPresetTarget.image
    store.imageDraft?.mossLayout = .init(width: 1872, height: 1056, caption: "Saved scene", regions: [
      .init(type: .text, caption: "Red sign", text: "EDIT ME", bounds: .init(x: 0.1, y: 0.2, width: 0.4, height: 0.2))])
    store.imageDraft?.mossLayout?.guideImagePath = "/editor-guide.png"
    store.imageDraft?.mossImage?.regionalLayout = true
    let original = try XCTUnwrap(store.imageDraft)
    let moss = try ImagePresetRecipe.model(for: original, library: store.generationLibrary).reference
    let root = store.dataDirectory.deletingLastPathComponent()
    let profile = root.appendingPathComponent("h3.json"), vae = root.appendingPathComponent("vae.safetensors")
    try Data("{}".utf8).write(to: profile); try Data([0]).write(to: vae)
    let h3 = GenerationModelRef(id: "h3-image", backend: "nativeH3", family: "h3")
    var index = store.generationLibraryIndex
    try index.registerModel(.init(reference: h3, name: "H3 image", workspaces: [.image], tasks: ["image"], availability: .installed,
      binding: .nativeH3Image(profileID: profile.path, imageVAE: vae.path)), alias: "h3-image")
    try store.persistGenerationLibrary(index)
    let preset = try await store.saveNewPreset(name: "Layout settings", for: target, expectedTarget: store.presetTargetToken(for: target))
    store.imageEstimate = ["eligibility": "allowed"]
    try await store.useGenerationLibraryModel(h3, for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertEqual(store.imageDraft?.executionProvider, .nativeH3)
    XCTAssertEqual(store.imageDraft?.mossLayout, original.mossLayout)
    XCTAssertEqual(store.imageDraft?.prompt, original.prompt)
    XCTAssertEqual(store.imageDraft?.width, 1872); XCTAssertEqual(store.imageDraft?.height, 1056)
    XCTAssertNil(store.imageEstimate) // An old model's admitted request cannot be reused.
    XCTAssertThrowsError(try ImagePresetRecipe.validate(XCTUnwrap(store.imageDraft))) // H3 still requires its 32-pixel grid before generation.
    XCTAssertThrowsError(try NativeH3ImagePreparation.recipe(draft: XCTUnwrap(store.imageDraft), requestID: UUID(), seed: 1))
    XCTAssertEqual(store.imagePresetSelection?.presetID, preset.id)
    let reopened = StudioStore(dataDirectory: store.dataDirectory, restoreSession: false)
    reopened.restoreImageWorkspaces()
    XCTAssertEqual(reopened.imageDraft?.executionProvider, .nativeH3)
    XCTAssertEqual(reopened.imageDraft?.mossLayout, original.mossLayout)
    try store.undoImagePreset()
    XCTAssertEqual(store.imageDraft, original)
    try await store.useGenerationLibraryModel(h3, for: target, expectedTarget: store.presetTargetToken(for: target))
    try await store.useGenerationLibraryModel(moss, for: target, expectedTarget: store.presetTargetToken(for: target))
    XCTAssertEqual(store.imageDraft?.executionProvider, .nativeMOSS)
    XCTAssertEqual(store.imageDraft?.mossLayout, original.mossLayout)
    XCTAssertEqual(store.imageDraft?.mossImage?.regionalLayout, true)
    XCTAssertEqual(store.imageDraft?.width, 1872); XCTAssertEqual(store.imageDraft?.height, 1056)
    XCTAssertFalse(store.bridge.busy)
  }
  @MainActor func testStaleModelUseCannotOverwriteANewerImageWorkspaceEdit() async throws {
    let (store, model, _) = try fixture(), target = GenerationPresetTarget.image, token = try store.presetTargetToken(for: .image)
    store.imageDraft?.prompt = "A newer edit"; let draft = store.imageDraft
    do { try await store.useGenerationLibraryModel(model, for: target, expectedTarget: token); XCTFail("Changed destination") } catch {}
    XCTAssertEqual(store.imageDraft, draft)
  }
  @MainActor func testCancellationDuringLibraryRefreshCannotApplyAModel() async throws {
    let (store, model, _) = try fixture(), target = GenerationPresetTarget.image
    let token = try store.presetTargetToken(for: target), draft = store.imageDraft
    let operation = Task { try await store.useGenerationLibraryModel(model, for: target, expectedTarget: token) }
    for _ in 0..<100 where !store.generationLibraryBusy { await Task.yield() }
    XCTAssertTrue(store.generationLibraryBusy)
    operation.cancel()
    do { try await operation.value; XCTFail("Cancelled model selection") } catch {}
    XCTAssertEqual(store.imageDraft, draft); XCTAssertNil(store.imagePresetSelection)
  }
}
