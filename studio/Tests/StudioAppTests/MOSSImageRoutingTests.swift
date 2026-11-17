import Foundation
import ImageLayout
import StudioCore
import XCTest
@testable import WeeToddStudio

final class MOSSImageRoutingTests: XCTestCase {
  @MainActor func testCancelledMOSSRenderLeavesDraftWithoutFailureAlertOrAsset() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, payload, _ in
      if command == "moss-native-render" { throw CancellationError() }
      let file = try XCTUnwrap(payload["recipePath"] as? String)
      let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: file))) as? [String: Any])
      let configuration = try XCTUnwrap(root["configuration"] as? [String: Any])
      return ["schema": "weetodd-moss-image-result-v1", "eligibility": "allowed", "fingerprint": "captured",
        "requestID": root["requestID"]!, "seed": configuration["seed"]!, "width": configuration["width"]!, "height": configuration["height"]!]
    })
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Keep this draft"; draft.seed = 42
    draft.mossImage?.transformerDirectory = "/models/transformer"; draft.mossImage?.sharedDirectory = "/models/shared"
    store.imageDraft = draft
    await store.generateMOSSImageAsset()
    XCTAssertNil(store.error); XCTAssertEqual(store.notice, "MOSS image generation cancelled.")
    XCTAssertEqual(store.imageDraft, draft); XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertNil(store.preparedMOSSImage); XCTAssertNil(store.imageEstimate)
  }
  @MainActor func testNativeMOSSCancellationIsNotReportedAsFailure() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let worker = directory.appendingPathComponent("worker.sh"), marker = directory.appendingPathComponent("started")
    let script = "#!/bin/sh\n/usr/bin/touch '\(marker.path)'\nexec /bin/sleep 30\n"
    try Data(script.utf8).write(to: worker); try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    let recipe = directory.appendingPathComponent("recipe.json")
    try Data(#"{"schema":"weetodd-moss-image-request-v1","prompt":"native"}"#.utf8).write(to: recipe)
    var runtime = RuntimeSettings(root: "/no-repository", pythonPath: "/no-python", profilesDirectory: directory.path)
    runtime.mossWorkerPath = worker.path
    let bridge = Bridge()
    let job = Task { try await bridge.invoke("moss-native-render", runtime: runtime,
      payload: ["recipePath": recipe.path], output: directory.appendingPathComponent("take")) }
    for _ in 0..<200 {
      if FileManager.default.fileExists(atPath: marker.path) { break }
      try await Task.sleep(for: .milliseconds(10))
    }
    XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
    bridge.cancel()
    do { _ = try await job.value; XCTFail("A cancelled job must not return a result") }
    catch { XCTAssertTrue(error is CancellationError, "\(error)") }
    XCTAssertFalse(bridge.busy)
  }
  @MainActor func testIsolatedUIFixtureRoundtripWhenProvided() throws {
    guard let output = ProcessInfo.processInfo.environment["WEETODD_MOSS_UI_DATA"],
      let request = ProcessInfo.processInfo.environment["WEETODD_MOSS_UI_REQUEST"],
      let worker = ProcessInfo.processInfo.environment["WEETODD_MOSS_UI_WORKER"] else {
      throw XCTSkip("Opt-in isolated app fixture and persistence roundtrip")
    }
    let directory = URL(fileURLWithPath: output)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: request))) as? [String: Any])
    let configuration = try XCTUnwrap(root["configuration"] as? [String: Any])
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.name = "MOSS serum canvas qualification"
    draft.prompt = try XCTUnwrap(root["prompt"] as? String)
    draft.width = try XCTUnwrap(configuration["width"] as? Int); draft.height = try XCTUnwrap(configuration["height"] as? Int)
    draft.seed = 42; draft.steps = 25; draft.guidance = 2.5; draft.shift = 5
    draft.mossImage?.sampler = "er_sde"; draft.mossImage?.schedule = "simple"; draft.mossImage?.skipStop = 0.5
    draft.mossImage?.weightCacheGB = 16
    draft.mossImage?.transformerDirectory = try XCTUnwrap(root["transformerDirectory"] as? String)
    draft.mossImage?.sharedDirectory = try XCTUnwrap(root["sharedDirectory"] as? String)
    let sharedDirectory = draft.mossImage!.sharedDirectory
    draft.mossImage?.libraryDirectory = URL(fileURLWithPath: sharedDirectory).deletingLastPathComponent().path
    try ProjectStorage.write(store.project, to: directory.appendingPathComponent("Autosave.weetodd"))
    store.runtime.root = "/no-repository"; store.runtime.pythonPath = "/no-python"; store.runtime.mossWorkerPath = worker
    try JSONEncoder().encode(store.runtime).write(to: directory.appendingPathComponent("runtime.json"))
    store.restoringImageWorkspace = false
    store.imageDraft = draft
    let reopened = StudioStore(dataDirectory: directory, restoreSession: true)
    XCTAssertEqual(reopened.imageDraft, draft); XCTAssertEqual(reopened.runtime.pythonPath, "/no-python")
  }
  @MainActor func testNewReferenceDraftRetainsCompleteMOSSRecipe() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    var previous = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    previous.selectProvider(.nativeMOSS); previous.steps = 25; previous.guidance = 2.5; previous.shift = 4
    previous.negativePrompt = "blurry"; previous.mossImage?.sampler = "er_sde"; previous.mossImage?.schedule = "simple"
    store.imageWorkspaceLibrary.referenceProvider = .nativeMOSS
    store.imageWorkspaceLibrary.referenceMOSSRecipe = .init(draft: previous)
    store.imageWorkspaceLibrary.referenceMOSSImage = previous.mossImage
    var context = ReferenceSheetContext(subjectKey: "new", name: "Person", kind: .character, description: "A person")
    context.template = .portrait
    let draft = store.makeReferenceImageDraft(context, previousDraft: nil)
    XCTAssertEqual(draft.guidance, 2.5); XCTAssertEqual(draft.shift, 4); XCTAssertEqual(draft.steps, 25)
    XCTAssertEqual(draft.negativePrompt, "blurry"); XCTAssertEqual(draft.mossImage?.sampler, "er_sde")
  }
  @MainActor func testNativeMOSSConversionDispatchNeverUsesPython() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let worker = directory.appendingPathComponent("worker.sh"), captured = directory.appendingPathComponent("arguments.txt")
    let script = "#!/bin/sh\n/usr/bin/printf '%s\\n' \"$@\" > '\(captured.path)'\n/usr/bin/printf '%s\\n' '{\"status\":\"ok\",\"result\":{\"bits\":8}}'\n"
    try Data(script.utf8).write(to: worker); try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    var runtime = RuntimeSettings(root: "/no-repository", pythonPath: "/no-python", profilesDirectory: directory.path); runtime.mossWorkerPath = worker.path
    _ = try await Bridge().invoke("moss-native-convert", runtime: runtime,
      payload: ["source": "/models/official", "bits": 8, "component": "transformer"], output: directory.appendingPathComponent("converted"))
    let arguments = try String(contentsOf: captured, encoding: .utf8)
    XCTAssertTrue(arguments.hasPrefix("convert-model\n--input\n/models/official\n")); XCTAssertTrue(arguments.contains("--bits\n8"))
    XCTAssertFalse(arguments.contains("studio_bridge.py"))
  }
  @MainActor func testReopeningTakeCreatesIndependentEditableLayoutAndExactRecipe() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    var source = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    source.selectProvider(.nativeMOSS); source.prompt = "Saved scene"; source.width = 896; source.height = 1152
    source.mossImage?.transformerDirectory = "/models/transformer"; source.mossImage?.sharedDirectory = "/models/shared"
    source.mossLayout = .init(width: 896, height: 1152, caption: source.prompt)
    source.steps = 25; source.guidance = 2.5; source.mossImage?.sampler = "er_sde"; source.mossImage?.schedule = "simple"
    var fast = MediaAsset(name: "Fast", kind: .lora, path: "/models/fast.safetensors", scope: .global)
    fast.loraModel = .mossImage; fast.loraProfile = "moss_sft_4step_v2"
    try source.applyMOSSLoRAs([.init(asset: fast)])
    let request = try NativeMOSSImagePreparation.request(draft: source, requestID: UUID(), seed: 77)
    var generation = ImageGeneration(provider: "nativeMOSS", requestFingerprint: "original", modelID: source.modelID, prompt: source.prompt)
    generation.provider = ImageExecutionProvider.nativeMOSS.rawValue
    generation.nativeRequest = try JSONDecoder().decode([String: JSONValue].self, from: request); generation.mossLayout = source.mossLayout
    generation.mossStockRecipe = source.mossImage?.stockRecipe
    var asset = MediaAsset(name: "Saved take", kind: .image, path: "/takes/image.png", scope: .project); asset.generation = generation
    store.reopenMOSSLayout(asset)
    XCTAssertEqual(store.imageDraft?.seed, 77); XCTAssertEqual(store.imageDraft?.mossLayout, source.mossLayout)
    store.imageDraft?.mossLayout?.summary.caption = "New scene"
    XCTAssertEqual(asset.generation?.mossLayout?.summary.caption, "Saved scene")
    XCTAssertEqual(asset.generation?.requestFingerprint, "original")
    let reopenedID = try XCTUnwrap(store.imageDraft?.mossImage?.loras?.first?.id)
    try store.imageDraft?.setMOSSAdapterEnabled(id: reopenedID, enabled: false)
    XCTAssertEqual(store.imageDraft?.steps, 25); XCTAssertEqual(store.imageDraft?.guidance, 2.5)
    XCTAssertEqual(store.imageDraft?.mossImage?.sampler, "er_sde")
    asset.generation?.mossStockRecipe = nil
    store.reopenMOSSLayout(asset)
    let legacyID = try XCTUnwrap(store.imageDraft?.mossImage?.loras?.first?.id)
    try store.imageDraft?.setMOSSAdapterEnabled(id: legacyID, enabled: false)
    XCTAssertEqual(store.imageDraft?.steps, 50); XCTAssertEqual(store.imageDraft?.guidance, 3)
    XCTAssertEqual(store.imageDraft?.mossImage?.sampler, "euler")
  }
  @MainActor func testPinnedFastAdapterImportRetainsItsRecipeWhenProvided() async throws {
    guard let source = ProcessInfo.processInfo.environment["WEETODD_MOSS_FAST_ADAPTER"] else { throw XCTSkip("Opt-in pinned full adapter identity check") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.runtime.pythonPath = "/no-python"
    await store.importURLs([URL(fileURLWithPath: source)], scope: .global, loraModel: .mossImage)
    let asset = try XCTUnwrap(store.globalAssets.first)
    XCTAssertEqual(asset.loraModel, .mossImage); XCTAssertEqual(asset.loraProfile, "moss_sft_4step_v2")
  }
  @MainActor func testMOSSProcessDispatchWorksWithoutPython() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let worker = directory.appendingPathComponent("worker.sh"), captured = directory.appendingPathComponent("captured.json")
    let script = "#!/bin/sh\n/bin/cp \"$3\" '\(captured.path)'\n/usr/bin/printf '%s\\n' '{\"status\":\"ok\",\"result\":{\"eligibility\":\"allowed\"}}'\n"
    try Data(script.utf8).write(to: worker)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    let recipe = directory.appendingPathComponent("recipe.json")
    try Data(#"{"schema":"weetodd-moss-image-request-v1","prompt":"native"}"#.utf8).write(to: recipe)
    var runtime = RuntimeSettings(root: "/no-repository", pythonPath: "/no-python", profilesDirectory: directory.path)
    runtime.mossWorkerPath = worker.path
    let output = directory.appendingPathComponent("take")
    let result = try await Bridge().invoke("moss-native-preflight", runtime: runtime, payload: ["recipePath": recipe.path], output: output)
    XCTAssertEqual(result["eligibility"] as? String, "allowed")
    let sent = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: captured)) as? [String: Any])
    XCTAssertEqual(sent["schema"] as? String, "weetodd-moss-image-request-v1")
    XCTAssertEqual(sent["outputPath"] as? String, output.appendingPathComponent("image.png").path)
    XCTAssertNil(sent["runtime"])
  }

  @MainActor func testMOSSPreflightCapturesExactDraftAndInvalidatesAfterEdit() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    var commands: [String] = []
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, payload, _ in
      commands.append(command)
      let file = try XCTUnwrap(payload["recipePath"] as? String)
      let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: file))) as? [String: Any])
      let configuration = try XCTUnwrap(root["configuration"] as? [String: Any])
      return ["schema": "weetodd-moss-image-result-v1", "eligibility": "allowed", "fingerprint": "captured",
        "requestID": root["requestID"]!, "seed": configuration["seed"]!, "width": configuration["width"]!, "height": configuration["height"]!]
    })
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.prompt = "First prompt"; draft.seed = 42
    draft.mossImage?.transformerDirectory = "/models/transformer"; draft.mossImage?.sharedDirectory = "/models/shared"
    store.imageDraft = draft
    await store.prepareImageGeneration()
    XCTAssertEqual(commands, ["moss-native-preflight"])
    XCTAssertEqual(store.preparedMOSSImage?.draft.prompt, "First prompt")
    store.imageDraft?.prompt = "Changed prompt"
    XCTAssertNotEqual(store.preparedMOSSImage?.draft, store.imageDraft)
  }
}
