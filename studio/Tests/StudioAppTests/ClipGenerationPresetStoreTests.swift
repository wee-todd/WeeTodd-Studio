import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class ClipGenerationPresetStoreTests: XCTestCase {
  @MainActor func testNativePresetsCanConfigureAnEmptyPromptWithoutMakingItGenerationReady() async throws {
    for engine in [Engine.h3, .ltx25] {
      let store = try fixture(), root = store.dataDirectory
      let model = root.appendingPathComponent(engine.rawValue + ".json")
      let components: [String: Any] = engine == .h3
        ? ["task": "t2va", "transformer": "/model/transformer.safetensors", "text_encoder": "/model/qwen", "tokenizer": "/model/tokenizer.json", "video_vae": "/model/video.safetensors", "audio_vae": "/model/audio.safetensors", "loras": []]
        : ["transformer_path": "/model/transformer", "loras": []]
      let config: [String: Any] = engine == .h3
        ? ["width": 768, "height": 448, "duration_seconds": 5.0, "steps": 5, "seed": 1, "drop_adaln": true, "sampling_method": "euler", "transformer_backend": "mlx"]
        : ["pipeline_mode": "distilled", "stage1_steps": 8, "stage2_steps": 3, "frame_rate": 24, "width": 768, "height": 448, "seed": 1, "duration_seconds": 5]
      try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue, "components": components, "config": config,
        "conditioning": ["version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated"]]).write(to: model)
      var clip = Clip(engine: engine); clip.prompt = ""; clip.profileID = model.path; clip.generationSelection = .init(task: "t2v")
      store.project.clips = [clip]; store.selectedClipID = clip.id; store.runtime.profilesDirectory = root.path; store.runtime.ffmpegPath = "/usr/bin/true"
      var index = GenerationLibraryIndex()
      try index.registerModel(.init(reference: .init(id: engine.rawValue, backend: engine.rawValue, family: engine.rawValue), name: engine.rawValue, workspaces: [.clip], tasks: ["t2v"], availability: .installed, binding: .nativeVideo(profileID: model.path)), alias: "fixture")
      try store.persistGenerationLibrary(index)
      let preset = StudioGenerationPreset(name: "Before writing", snapshot: try store.captureClipPreset())
      try await store.applyClipPreset(preset, expectedTarget: store.clipPresetTargetToken())
      XCTAssertEqual(store.selectedClip?.prompt, ""); XCTAssertEqual(store.clipPresetSelection?.presetID, preset.id)
      let request: [String: Any] = ["project": try store.project.generationInputProject.object(), "clipID": clip.id.uuidString, "runtime": try JSONSerialization.jsonObject(with: JSONEncoder().encode(store.runtime))]
      let readiness = engine == .h3 ? try NativeH3Preparation.describe(request: request) : try NativeLTXPreparation.describe(request: request)
      XCTAssertFalse((readiness["readinessErrors"] as? [String] ?? []).isEmpty, "Generation still requires the user's creative prompt")
    }
  }
  @MainActor func testIdenticalSourceMoviePresetKeepsPreparedWorkAndEffectiveBinding() async throws {
    let store = try fixture(), root = store.dataDirectory.path
    var clip = Clip(engine: .ltx25); clip.generationSelection = .init(task: "video_upscale"); clip.generationSelection?.movieUpscaleMethod = .relayVSR
    store.project.clips = [clip]; store.selectedClipID = clip.id; store.runtime.relayVSRModelDirectory = root
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "relay", backend: "restoration", family: "relayVSR", variant: "x2"), name: "Relay", workspaces: [.clip, .restoration], tasks: ["video_upscale", "relayVSR"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "relayVSR", components: ["modelDirectory": root, "scale": "2"])), alias: "fixture")
    try store.persistGenerationLibrary(index)
    let preset = StudioGenerationPreset(name: "Current source", snapshot: try store.captureClipPreset())
    let key = store.generationRequestKey(for: clip)
    store.preparedRecipe = "/prepared.json"; store.preparedFingerprint = "ready"
    try await store.applyClipPreset(preset, expectedTarget: store.clipPresetTargetToken())
    XCTAssertNil(store.project.generationLibraryBindings)
    XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), key)
    XCTAssertEqual(store.preparedRecipe, "/prepared.json"); XCTAssertEqual(store.preparedFingerprint, "ready")
    XCTAssertFalse(store.presetActions(for: .clip(clip.id)).isModified)
  }
  @MainActor func testAutomaticPresetRemainsCleanWhenItsResolvedProfileIsCached() throws {
    let store = try fixture(); store.project.clips[0].profileID = "auto"
    let model = GenerationModelRef(id: "auto-h3", backend: "h3", family: "h3")
    var index = store.generationLibraryIndex
    try index.registerModel(.init(reference: model, name: "Automatic H3", workspaces: [.clip], tasks: ["t2v"], availability: .installed,
      binding: .nativeVideo(profileID: "auto")), alias: "automatic")
    try store.persistGenerationLibrary(index)
    let clip = try XCTUnwrap(store.selectedClip)
    let snapshot = try ClipPresetRecipe.capture(from: clip, project: store.project, resolvedLibrary: store.generationLibrary)
    let preset = StudioGenerationPreset(name: "Automatic", snapshot: snapshot)
    store.setClipPresetSelection(.init(preset))
    store.generationDescriptions[clip.id] = ["studioInput": store.generationRequestKey(for: clip), "profileID": store.generationLibrary.modelEntries.first { $0.id == "h3" }!.binding!.localPaths[0]]
    XCTAssertFalse(store.presetActions(for: .clip(clip.id)).isModified)
  }
  @MainActor func testContinuousSceneAppliesPresetToEveryCapturedMemberInOneUndo() async throws {
    let store = try fixture()
    var first = store.project.clips[0]; first.engine = .ltx25; first.generationSelection = .init(task: "t2v"); first.seed = 1
    var second = first; second.id = UUID(); second.prompt = "Keep the next shot"; second.seed = 2
    second.continuity = .init(mode: "scene", sourceClipID: first.id)
    store.project.clips = [first, second]; store.selectedClipID = second.id
    var index = GenerationLibraryIndex()
    let model = GenerationModelRef(id: "scene-ltx", backend: "ltx25", family: "ltx25")
    try index.registerModel(.init(reference: model, name: "LTX scene", workspaces: [.clip], tasks: ["t2v"], availability: .installed,
      binding: .nativeVideo(profileID: first.profileID)), alias: "scene")
    try store.persistGenerationLibrary(index)
    let original = store.project
    var snapshot = try store.captureClipPreset(); snapshot.fields["seed"] = .integer(42)
    let preset = StudioGenerationPreset(name: "Scene", snapshot: snapshot)
    try await store.applyClipPreset(preset, expectedTarget: store.clipPresetTargetToken())
    XCTAssertEqual(store.project.clips.map(\.seed), [42, 42]); XCTAssertEqual(store.project.clips[1].prompt, second.prompt)
    XCTAssertEqual(store.project.clips[1].continuity, second.continuity)
    XCTAssertEqual(store.project.generationPresetSelections?.count, 2)
    store.undo(); XCTAssertEqual(store.project, original)
  }
  @MainActor func testSourceMovieLibraryChoiceReachesDescribeWithoutChangingGlobalRuntime() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var received: [String] = []
    let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, runtime, _, _ in
      if command == "relay-native-describe" { received.append(runtime.relayVSRModelDirectory ?? ""); return ["readinessErrors": []] }
      throw StudioError.invalid("No inference")
    })
    var clip = Clip(engine: .ltx25); clip.generationSelection = .init(task: "video_upscale"); clip.generationSelection?.movieUpscaleMethod = .relayVSR
    store.project.clips = [clip]; store.selectedClipID = clip.id; store.runtime.relayVSRModelDirectory = "/global/relay"
    let model = GenerationModelRef(id: "relay", backend: "restoration", family: "relayVSR", variant: "x2")
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: model, name: "Relay", workspaces: [.clip, .restoration], tasks: ["video_upscale", "relayVSR"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "relayVSR", components: ["modelDirectory": root.path, "scale": "2"])), alias: "fixture")
    try store.persistGenerationLibrary(index)
    store.project.generationLibraryBindings = ["clip-model:" + clip.id.uuidString: model]
    await store.describeGeneration()
    XCTAssertEqual(received, [root.path]); XCTAssertEqual(store.runtime.relayVSRModelDirectory, "/global/relay")
    XCTAssertEqual(store.selectedClip, clip)
  }
  @MainActor private func fixture() throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let profile = root.appendingPathComponent("h3.json"); try Data("{}".utf8).write(to: profile)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var clip = Clip(name: "My scene", engine: .h3); clip.profileID = profile.path
    clip.prompt = "Keep the source actors"; clip.sourcePath = "/existing.mov"; clip.duration = 5
    clip.generationSelection = .init(task: "t2v"); clip.generationSelection?.steps = 8
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var index = GenerationLibraryIndex()
    try index.registerModel(.init(reference: .init(id: "h3", backend: "h3", family: "h3"), name: "H3", workspaces: [.clip], tasks: ["t2v"],
      availability: .installed, binding: .nativeVideo(profileID: profile.path)), alias: "fixture")
    try store.persistGenerationLibrary(index); return store
  }
  @MainActor func testApplyAndUndoAreAtomicAndDoNotMutateTheSavedSnapshot() async throws {
    let store = try fixture(), original = store.project
    var snapshot = try store.captureClipPreset()
    guard case .object(var selection) = snapshot.fields["selection"] else { return XCTFail("Missing native controls") }
    selection["steps"] = .integer(6); snapshot.fields["selection"] = .object(selection)
    let preset = StudioGenerationPreset(name: "Six", snapshot: snapshot)
    try await store.applyClipPreset(preset, expectedTarget: store.clipPresetTargetToken())
    XCTAssertEqual(store.selectedClip?.generationSelection?.steps, 6)
    XCTAssertEqual(store.selectedClip?.prompt, original.clips[0].prompt); XCTAssertEqual(store.selectedClip?.sourcePath, original.clips[0].sourcePath)
    XCTAssertEqual(store.clipPresetSelection?.presetID, preset.id); XCTAssertEqual(preset.snapshot, snapshot)
    store.undo(); XCTAssertEqual(store.project, original)
    store.redo(); XCTAssertEqual(store.clipPresetSelection?.presetID, preset.id)
  }
  @MainActor func testAssociationSaveClearUndoAndRedoKeepPreparedIdentityAndWorkerPayload() throws {
    let store = try fixture(), clip = try XCTUnwrap(store.selectedClip)
    let preset = StudioGenerationPreset(name: "Current", snapshot: try store.captureClipPreset())
    store.preparedRecipe = "/prepared.json"; store.preparedFingerprint = "ready"; store.preparedPrompt = "Prepared"
    let payload = try store.payload(), key = store.generationRequestKey(for: clip), fingerprint = clip.generationFingerprint
    store.setClipPresetSelection(.init(preset))
    XCTAssertEqual(store.preparedRecipe, "/prepared.json"); XCTAssertEqual(store.preparedFingerprint, "ready")
    XCTAssertEqual(store.generationRequestKey(for: clip), key); XCTAssertEqual(store.selectedClip?.generationFingerprint, fingerprint)
    XCTAssertEqual(try JSONSerialization.data(withJSONObject: store.payload(), options: .sortedKeys),
      try JSONSerialization.data(withJSONObject: payload, options: .sortedKeys))
    store.undo(); XCTAssertNil(store.clipPresetSelection); XCTAssertEqual(store.preparedRecipe, "/prepared.json")
    store.redo(); XCTAssertEqual(store.clipPresetSelection?.presetID, preset.id); XCTAssertEqual(store.preparedRecipe, "/prepared.json")
    store.setClipPresetSelection(nil); XCTAssertEqual(store.preparedRecipe, "/prepared.json")
  }
  @MainActor func testChangedClipAndReplacedProjectRejectLateApply() async throws {
    let store = try fixture(), token = try store.clipPresetTargetToken()
    let preset = StudioGenerationPreset(name: "Current", snapshot: try store.captureClipPreset())
    store.project.clips[0].prompt = "New edit"; let changed = store.project
    do { try await store.applyClipPreset(preset, expectedTarget: token); XCTFail("Stale target") } catch {}
    XCTAssertEqual(store.project, changed)
    store.project = StudioProject()
    do { try await store.applyClipPreset(preset, expectedTarget: token); XCTFail("Replaced project") } catch {}
    XCTAssertTrue(store.project.clips.isEmpty)
  }
}
