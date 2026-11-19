import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class GenerationLibraryStoreTests: XCTestCase {
  func testArbitraryReadableFileCannotBecomeTheRippleAuthorAdapter() throws {
    let root = try directory(), profile = root.appendingPathComponent("ltx.json"), adapter = root.appendingPathComponent("ripple.safetensors")
    try Data([0]).write(to: adapter)
    var components: [String: Any] = [:]
    for key in ["transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path"] { let file = root.appendingPathComponent(key); try Data([0]).write(to: file); components[key] = file.path }
    let raw: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25", "components": components, "config": ["pipeline_mode": "distilled", "stage1_steps": 8, "stage2_steps": 3, "frame_rate": 24]]
    try JSONSerialization.data(withJSONObject: raw).write(to: profile)
    let generation: [String: Any] = ["pipelineMode": "distilled", "supportedTasks": ["t2v"], "presets": [], "controls": ["stepsEditable": false, "refinementStepsEditable": false, "cfgEditable": false, "shiftEditable": false, "stepsExplanation": "", "cfgExplanation": "", "shiftExplanation": ""]]
    let descriptor = try JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: generation))
    var settings = RuntimeSettings(root: "", pythonPath: "", profilesDirectory: ""); settings.rippleAdapterPath = adapter.path
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "cached-ripple", backend: "ripple", family: "ltx25", sourceIdentity: "formerly-validated"), name: "Previously admitted Ripple", workspaces: [.ripple], tasks: ["ripple"], availability: .installed,
      binding: .ripple(profileID: profile.path, adapterPath: adapter.path), identityFiles: [profile.path, adapter.path]), alias: "cached-ripple")
    try GenerationLibraryInventory.refreshSpecialized(index: &index, settings: settings, profiles: [.init(id: profile.path, name: "Plain", engine: "ltx25", task: "t2v", generation: descriptor)], flash: nil, h3: nil)
    XCTAssertFalse(index.snapshot.modelEntries.contains { $0.reference.backend == "ripple" && $0.availability == .installed })
  }
  func testAddAndUseCannotReplaceTheInspectedModelWithANewerSameID() throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"); try moss(in: models)
    let inspected = try GenerationLibraryModelImport.inspect(models, profilesDirectory: root.path, preferredVariant: "sft", shared: models.appendingPathComponent("WeeTodd/MOSS-Image/Shared").path, libraryRoot: models.path)
    var changed = inspected.entry; changed.reference.sourceIdentity = "replacement"
    let library = GenerationLibrarySnapshot(models: [changed])
    XCTAssertThrowsError(try GenerationLibraryModelImport.selection(inspected: inspected, linked: inspected.entry.reference, library: library, image: true))
  }
  @MainActor func testRejectedCachedQwenIsNeverPromotedByPathExistence() async throws {
    let root = try directory(), manifest = root.appendingPathComponent("manifest.json")
    try Data("{}".utf8).write(to: manifest)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    var index = GenerationLibraryIndex()
    let ref = GenerationModelRef(id: "qwen", backend: "nativeMLX", family: "qwen_image", precision: "q8", sourceIdentity: "old")
    try index.registerModel(.init(reference: ref, name: "Qwen", workspaces: [.image], tasks: ["image"], availability: .installed, binding: .nativeQwen(manifest: manifest.path), identityFiles: [manifest.path]), alias: "qwen|" + manifest.path)
    try store.persistGenerationLibrary(index); await store.refreshGenerationLibrary(force: true)
    XCTAssertNotEqual(store.generationLibrary.modelEntries.first { $0.id == ref.id }?.availability, .installed)
  }
  func testIncompleteH3LibraryCannotCreateInstalledRestorationPreset() throws {
    let root = try directory(), profile = root.appendingPathComponent("h3.json"); try Data("{}".utf8).write(to: profile)
    var settings = H3RegenerateSettings(); settings.baseProfilePath = profile.path; settings.modelLibraryRoot = root.path
    var index = GenerationLibraryIndex()
    try GenerationLibraryInventory.refreshSpecialized(index: &index, settings: .init(root: "", pythonPath: "", profilesDirectory: ""), profiles: [], flash: nil, h3: settings)
    XCTAssertFalse(index.snapshot.modelEntries.contains { $0.reference.family == "h3Regenerate" && $0.availability == .installed })
  }
  func testImportRecognizesBaseInsteadOfUsingTheCurrentSFTSelection() throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"); try moss(in: models)
    let base = models.appendingPathComponent("WeeTodd/MOSS-Image/Base/transformer")
    try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: ["dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42, "text_dim": 4096, "latent_channels": 128, "freq_dim": 256, "rope_base": 10000]).write(to: base.appendingPathComponent("config.json"))
    try Data([0]).write(to: base.appendingPathComponent("model.safetensors"))
    let result = try GenerationLibraryModelImport.inspect(base, profilesDirectory: root.path, preferredVariant: "sft",
      shared: models.appendingPathComponent("WeeTodd/MOSS-Image/Shared").path, libraryRoot: models.path)
    XCTAssertEqual(result.entry.reference.variant, "base")
  }
  @MainActor func testAutomaticComponentsAndLegacyLTXConfigsAreAdoptedWithoutDeletingTheOriginal() async throws {
    let root = try directory(), profile = root.appendingPathComponent("ltx.json")
    try Data("{}".utf8).write(to: profile)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    store.profiles = [.init(id: profile.path, name: "LTX", engine: "ltx25", task: "t2v")]
    let config = try LTX25GenerationConfig(name: "My saved recipe", selection: LTX25GenerationConfig.distilled.selection)
    store.savedLTX25Configs = [config]
    await store.refreshGenerationLibrary(force: true)
    XCTAssertTrue(store.generationLibrary.modelEntries.contains { if case .nativeVideo("auto") = $0.binding { return $0.reference.backend == "ltx25" }; return false })
    XCTAssertTrue(store.generationPresets.presets.contains { $0.name.contains("My saved recipe") })
    XCTAssertEqual(store.savedLTX25Configs, [config])
    let count = store.generationPresets.presets.count
    await store.refreshGenerationLibrary(force: true)
    XCTAssertEqual(store.generationPresets.presets.count, count)
  }
  private func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }; return root
  }
  private func moss(in root: URL) throws {
    for name in ["WeeTodd/MOSS-Image/SFT/transformer/config.json", "WeeTodd/MOSS-Image/Shared/text_encoder/config.json",
      "WeeTodd/MOSS-Image/Shared/vae/config.json", "WeeTodd/MOSS-Image/Shared/tokenizer/tokenizer.json",
      "WeeTodd/MOSS-Image/Shared/tokenizer/tokenizer_config.json"] {
      let file = root.appendingPathComponent(name)
      try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data("{}".utf8).write(to: file)
    }
    try JSONSerialization.data(withJSONObject: ["dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42, "text_dim": 4096, "latent_channels": 128, "freq_dim": 256, "rope_base": 10000]).write(to: root.appendingPathComponent("WeeTodd/MOSS-Image/SFT/transformer/config.json"))
    try Data([0]).write(to: root.appendingPathComponent("WeeTodd/MOSS-Image/SFT/transformer/model.safetensors"))
    try Data([0]).write(to: root.appendingPathComponent("WeeTodd/MOSS-Image/Shared/text_encoder/model.safetensors"))
    try Data([0]).write(to: root.appendingPathComponent("WeeTodd/MOSS-Image/Shared/vae/model.safetensors"))
    let folder = root.appendingPathComponent("WeeTodd/LoRAs/MOSS-Image")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let header: [String: Any] = ["__metadata__": ["base_model": "MOSS-Image"],
      "layers.0.proj.q.lora_A.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
      "layers.0.proj.q.lora_B.weight": ["dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]]]
    let bytes = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(bytes); data.append(Data(repeating: 0, count: 8))
    try data.write(to: folder.appendingPathComponent("style.safetensors"))
  }

  @MainActor func testKnownModelAndLoRAFoldersAppearWithoutImportingOrChangingTheDraft() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"), app = root.appendingPathComponent("app")
    try moss(in: models)
    let store = StudioStore(dataDirectory: app, restoreSession: false)
    store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Keep this prompt"; draft.width = 1872; draft.height = 1056
    store.imageDraft = draft
    try store.setGenerationLibraryRoot(models.path)
    await store.refreshGenerationLibrary(force: true)
    let model = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.reference.backend == "nativeMOSS" && $0.reference.variant == "sft" && $0.reference.precision == "bf16" })
    XCTAssertEqual(model.availability, .installed)
    XCTAssertTrue(store.generationPresets.presets.contains { $0.origin == .supplied && $0.snapshot.model == model.reference })
    XCTAssertTrue(store.generationLibrary.loraEntries.contains { $0.reference.family == "moss_image" && $0.name == "style" })
    XCTAssertEqual(store.imageDraft, draft)
    XCTAssertTrue(store.globalAssets.isEmpty)
    XCTAssertFalse(store.bridge.busy)
    XCTAssertTrue(FileManager.default.fileExists(atPath: app.appendingPathComponent("generation-library.json").path))
  }

  @MainActor func testLibraryRootAndLogicalSelectionsSurviveARelaunchAndDisconnectedFolder() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"), app = root.appendingPathComponent("app")
    try moss(in: models)
    let first = StudioStore(dataDirectory: app, restoreSession: false)
    try first.setGenerationLibraryRoot(models.path); await first.refreshGenerationLibrary(force: true)
    let model = try XCTUnwrap(first.generationLibrary.modelEntries.first { $0.reference.backend == "nativeMOSS" && $0.reference.variant == "sft" })
    let supplied = try XCTUnwrap(first.generationPresets.presets.first { $0.snapshot.model == model.reference })
    let offline = root.appendingPathComponent("temporarily-offline")
    try FileManager.default.moveItem(at: models, to: offline)
    let second = StudioStore(dataDirectory: app, restoreSession: false)
    XCTAssertEqual(second.generationLibraryRootDirectory, models.path)
    await second.refreshGenerationLibrary(force: true)
    let retained = try XCTUnwrap(second.generationLibrary.modelEntries.first { $0.id == model.id })
    XCTAssertEqual(retained.reference, model.reference)
    XCTAssertEqual(second.generationPresets.preset(id: supplied.id)?.name, supplied.name)
    XCTAssertNotEqual(retained.availability, .installed)
    XCTAssertThrowsError(try second.generationLibrary.resolve(model.reference, in: .init(workspace: .image, task: "image")))
  }
  @MainActor func testCorruptLoRAReplacementCannotKeepThePreviousReadyReference() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"); try moss(in: models)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    try store.setGenerationLibraryRoot(models.path); await store.refreshGenerationLibrary(force: true)
    let model = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.reference.backend == "nativeMOSS" })
    let lora = try XCTUnwrap(store.generationLibrary.loraEntries.first { $0.name == "style" })
    try Data([0]).write(to: models.appendingPathComponent("WeeTodd/LoRAs/MOSS-Image/style.safetensors"))
    await store.refreshGenerationLibrary(force: true)
    XCTAssertNotEqual(store.generationLibrary.loraEntries.first { $0.id == lora.id }?.availability, .installed)
    XCTAssertThrowsError(try store.generationLibrary.resolve(lora.reference, for: model.reference, in: .init(workspace: .image, task: "image")))
  }

  @MainActor func testDiscoveryHonorsExplicitFamilyFolderExclusions() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models")
    try moss(in: models)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    try store.setGenerationLibraryRoot(models.path)
    var excluded = LoRAFolder(path: models.appendingPathComponent("WeeTodd/LoRAs/MOSS-Image").path); excluded.enabled = false
    store.runtime.loraFolders = [excluded]
    await store.refreshGenerationLibrary(force: true)
    XCTAssertFalse(store.generationLibrary.loraEntries.contains { $0.reference.family == "moss_image" })
  }
  @MainActor func testLateLoRAImportRegistersTheFileWithoutApplyingToANewerDraft() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models")
    try moss(in: models)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    try store.setGenerationLibraryRoot(models.path); await store.refreshGenerationLibrary(force: true)
    let entry = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.reference.backend == "nativeMOSS" })
    guard case .nativeMOSS(let transformer, let shared, let family) = entry.binding else { return XCTFail("Missing MOSS binding") }
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: store.project.id)); draft.selectProvider(.nativeMOSS)
    draft.mossImage?.transformerDirectory = transformer; draft.mossImage?.sharedDirectory = shared; draft.mossImage?.libraryDirectory = family
    store.restoringImageWorkspace = false; store.imageDraft = draft
    let token = try store.presetTargetToken(for: .image)
    let imported = root.appendingPathComponent("imported.safetensors")
    try FileManager.default.copyItem(at: models.appendingPathComponent("WeeTodd/LoRAs/MOSS-Image/style.safetensors"), to: imported)
    store.imageDraft?.prompt = "A newer edit"; let current = store.imageDraft
    let result = try await store.linkGenerationLibraryItem(imported,
      intent: .lora(model: entry.reference, target: .image, expectedTarget: token, addAndUse: true))
    XCTAssertFalse(result.applied)
    XCTAssertTrue(store.generationLibrary.loraEntries.contains { if case .nativeFile(let path, _, _, _) = $0.binding { return path == imported.path }; return false })
    XCTAssertEqual(store.imageDraft, current); XCTAssertTrue(store.globalAssets.isEmpty); XCTAssertFalse(store.bridge.busy)
  }
  func testModelFolderImportChecksArchitectureBeforeRegistering() throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models")
    try moss(in: models)
    let transformer = root.appendingPathComponent("unrelated-transformer")
    try FileManager.default.createDirectory(at: transformer, withIntermediateDirectories: true)
    try Data("{\"dim\":3072,\"num_layers\":32,\"latent_channels\":128}".utf8).write(to: transformer.appendingPathComponent("config.json"))
    try Data([0]).write(to: transformer.appendingPathComponent("model.safetensors"))
    XCTAssertThrowsError(try GenerationLibraryModelImport.inspect(transformer, profilesDirectory: root.appendingPathComponent("profiles").path,
      preferredVariant: "sft", shared: models.appendingPathComponent("WeeTodd/MOSS-Image/Shared").path, libraryRoot: models.path))
  }
  func testPreparedQwenImportRejectsAnUnqualifiedRevision() throws {
    let root = try directory(), manifest = root.appendingPathComponent("manifest.json")
    try JSONSerialization.data(withJSONObject: ["format": "weetodd-qwen-image21-v1", "modelID": "Qwen/Qwen-Image-2.1",
      "bits": 8, "groupSize": 64, "sourceRevision": "unknown"]).write(to: manifest)
    XCTAssertThrowsError(try GenerationLibraryModelImport.inspect(manifest, profilesDirectory: root.path, preferredVariant: "sft", shared: "", libraryRoot: ""))
  }
  @MainActor func testSharedMOSSWeightReplacementInvalidatesTheOldModelReference() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models")
    try moss(in: models)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    try store.setGenerationLibraryRoot(models.path); await store.refreshGenerationLibrary(force: true)
    let before = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.reference.backend == "nativeMOSS" }).reference
    let file = models.appendingPathComponent("WeeTodd/MOSS-Image/Shared/vae/model.safetensors")
    let handle = try FileHandle(forWritingTo: file); try handle.write(contentsOf: Data([1])); try handle.close()
    await store.refreshGenerationLibrary(force: true)
    let after = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.id == before.id }).reference
    XCTAssertNotEqual(after.sourceIdentity, before.sourceIdentity)
    XCTAssertThrowsError(try store.generationLibrary.resolve(before, in: .init(workspace: .image, task: "image")))
  }
  @MainActor func testRestorationInventoryUsesInstalledFlashComponentsAndRetainsOfflineIdentity() async throws {
    let root = try directory(), models = root.appendingPathComponent("ai-models"), model = models.appendingPathComponent("FlashVSR-Plus/upstream/FlashVSR_plus")
    for file in ["turbo/dit.safetensors", "turbo/lq_proj_in.safetensors", "posi_prompt.pth", "TCDecoder.ckpt"] {
      let url = model.appendingPathComponent(file)
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data([0]).write(to: url)
    }
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    try store.setGenerationLibraryRoot(models.path)
    XCTAssertNoThrow(try GenerationLibraryModelImport.inspect(model, profilesDirectory: root.path, preferredVariant: "sft", shared: "", libraryRoot: ""))
    await store.refreshGenerationLibrary(force: true)
    let entry = try XCTUnwrap(store.generationLibrary.models(for: .init(workspace: .restoration, task: "flashVSR")).first)
    XCTAssertEqual(entry.availability, .installed)
    XCTAssertTrue(entry.workspaces.contains(.finishing))
    try FileManager.default.removeItem(at: model.appendingPathComponent("TCDecoder.ckpt"))
    await store.refreshGenerationLibrary(force: true)
    let retained = try XCTUnwrap(store.generationLibrary.modelEntries.first { $0.id == entry.id })
    XCTAssertEqual(retained.reference, entry.reference)
    XCTAssertNotEqual(retained.availability, .installed)
  }
}
