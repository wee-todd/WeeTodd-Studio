import Foundation
import ImageLayout
import XCTest
@testable import StudioCore

final class MOSSImageTests: XCTestCase {
  func testRegionalControlCannotBeSilentlyIgnoredByAnOlderWorker() throws {
    XCTAssertThrowsError(try NativeMOSSImageResult.validateRegionalControl(["normalizedRequest":["configuration":[:]]], required: true))
    var result: [String:Any] = ["normalizedRequest":["configuration":["regionalLayout":true]],
      "releasedStages":["conditioning","transformer","decoding"]]
    XCTAssertThrowsError(try NativeMOSSImageResult.validateCleanup(result))
    result["conditioningMode"] = "regional-attention-v2"; result["regionalContexts"] = [["group":["role":"background"]]]
    XCTAssertNoThrow(try NativeMOSSImageResult.validateCleanup(result))
  }
  func testCollectedMOSSReferencesResolveWhenTheProjectMoves() throws {
    var project = StudioProject()
    var asset = MediaAsset(name: "Take", kind: .image, path: "/cache/image.png", scope: .project)
    var layout = ImageLayoutDraft(width: 1024, height: 1024, caption: "Scene")
    layout.guideImagePath = "/cache/source.png"
    layout.sourceReconstruction = .init(path: "/cache/source.png", sha256: String(repeating: "b", count: 64))
    var generation = ImageGeneration(provider: "nativeMOSS", requestFingerprint: "original", modelID: "SFT", prompt: "Scene")
    generation.mossLayout = layout; asset.generation = generation; project.assets = [asset]
    var collected: [String] = []
    try ProjectStorage.collectMOSSLayoutReferences(&project) { value in collected.append(value); return "Media/source.png" }
    XCTAssertEqual(project.assets[0].generation?.mossLayout?.sourceReconstruction?.path, "Media/source.png")
    XCTAssertTrue(collected.contains("/cache/source.png"))
    ProjectStorage.mapPaths(&project) { $0.hasPrefix("Media/") ? "/moved/" + $0 : $0 }
    let source = try XCTUnwrap(project.assets[0].generation?.mossLayout?.sourceReconstruction)
    XCTAssertEqual(source.path, "/moved/Media/source.png"); XCTAssertEqual(source.sha256, layout.sourceReconstruction?.sha256)
    XCTAssertEqual(project.assets[0].generation?.mossLayout?.guideImagePath, source.path)
    XCTAssertNoThrow(try source.validate())
    XCTAssertEqual(project.assets[0].generation?.requestFingerprint, "original")
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    ProjectStorage.mapPaths(&project) { $0.replacingOccurrences(of: "/moved/", with: "") }
    let file = directory.appendingPathComponent("Moved.weetodd")
    try ProjectStorage.write(project, to: file)
    let reopened = try ProjectStorage.read(file)
    XCTAssertEqual(reopened.assets[0].generation?.mossLayout?.sourceReconstruction?.path, directory.appendingPathComponent("Media/source.png").path)
    XCTAssertEqual(reopened.assets[0].generation?.mossLayout?.sourceReconstruction?.sha256, source.sha256)
  }
  func testSourceStageCleanupCannotBeMistakenForVAEControlOrTextGeneration() throws {
    let reference: [String: Any] = ["path": "/source.png", "sha256": String(repeating: "a", count: 64), "noiseLevel": 0.08]
    var result: [String: Any] = ["normalizedRequest": ["sourceImage": reference],
      "releasedStages": ["encoding", "conditioning", "transformer", "decoding"],
      "generationMode": "source-reconstruction", "modelEvaluations": 8,
      "sourceReconstruction": ["source": reference]]
    XCTAssertNoThrow(try NativeMOSSImageResult.validateCleanup(result))
    result["releasedStages"] = ["conditioning", "transformer", "decoding"]
    XCTAssertThrowsError(try NativeMOSSImageResult.validateCleanup(result))
    result["releasedStages"] = ["encoding", "decoding"]; result["modelEvaluations"] = 0
    XCTAssertThrowsError(try NativeMOSSImageResult.validateCleanup(result))
    XCTAssertNoThrow(try NativeMOSSImageResult.validateCleanup(["normalizedRequest": [:], "releasedStages": ["conditioning", "transformer", "decoding"]]))
  }
  func testFaithfulRecipeIsExplicitAndDisablesSpecializedAdapters() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS)
    var asset = MediaAsset(name: "Fast", kind: .lora, path: "/models/fast.safetensors", scope: .global)
    asset.loraModel = .mossImage; asset.loraProfile = "moss_sft_4step_v2"
    try draft.applyMOSSLoRAs([.init(asset: asset)])
    draft.applyFaithfulMOSSReconstructionRecipe()
    XCTAssertEqual(draft.steps, 8); XCTAssertEqual(draft.guidance, 1)
    XCTAssertEqual(draft.mossImage?.sampler, "euler"); XCTAssertNil(draft.mossImage?.stockRecipe)
    XCTAssertFalse(draft.mossImage!.loras![0].isEnabled)
    XCTAssertEqual(draft.mossImage?.variant, "sft"); XCTAssertNil(draft.mossImage?.quantizationBits)
  }
  func testExplicitSourceTransportHistoryAndSavedTakeBinding() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Scene"
    draft.mossImage?.transformerDirectory = "/models/SFT/transformer"; draft.mossImage?.sharedDirectory = "/models/Shared"
    var layout = ImageLayoutDraft(width: 1024, height: 1024, caption: "Scene")
    layout.guideImagePath = "/guide.png"
    let compiled = try MOSSLayoutCompiler.compile(layout)
    let before = layout
    layout.sourceReconstruction = .init(path: "/source.png", sha256: String(repeating: "b", count: 64))
    XCTAssertEqual(try MOSSLayoutCompiler.compile(layout), compiled)
    XCTAssertEqual(try MOSSLayoutInterchange.exportOfficialDraft(layout), try MOSSLayoutInterchange.exportOfficialDraft(before))
    var history = ImageLayoutHistory(); history.record(before: before, after: layout)
    XCTAssertEqual(history.undo(current: layout), before)
    XCTAssertEqual(history.redo(current: before), layout)
    draft.mossLayout = layout
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 42)) as? [String: Any])
    let source = try XCTUnwrap(root["sourceImage"] as? [String: Any])
    XCTAssertEqual(source["sha256"] as? String, layout.sourceReconstruction?.sha256)
    let sent = try XCTUnwrap(root["layout"] as? [String: Any])
    XCTAssertNil(sent["sourceReconstruction"]); XCTAssertNil(sent["guideImagePath"])
    let admitted = try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONSerialization.data(withJSONObject: sent))
    XCTAssertThrowsError(try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft))
    let saved = try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft, admittedSource: layout.sourceReconstruction)
    XCTAssertEqual(saved.sourceReconstruction, layout.sourceReconstruction)
    draft.mossLayout?.sourceReconstruction?.enabled = false
    let disabled = try XCTUnwrap(JSONSerialization.jsonObject(with: NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 42)) as? [String: Any])
    XCTAssertNil(disabled["sourceImage"])
    XCTAssertThrowsError(try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft, admittedSource: layout.sourceReconstruction))
    XCTAssertNoThrow(try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft))
    draft.mossLayout?.sourceReconstruction?.noiseLevel = .nan
    XCTAssertThrowsError(try NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 42))
  }
  func testVariantChangeInvalidatesOldTransformerLink() throws {
    var settings = MOSSImageSettings(); settings.transformerDirectory = "/models/SFT/transformer"
    try settings.selectVariant("base")
    XCTAssertEqual(settings.variant, "base"); XCTAssertEqual(settings.transformerDirectory, "")
    XCTAssertThrowsError(try settings.selectVariant("unknown"))
  }
  func testFastAdapterProfileRestoresPreviousStockRecipeAndGroupsRetainMOSSFamily() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS); draft.steps = 25; draft.guidance = 2.5
    draft.mossImage?.sampler = "er_sde"; draft.mossImage?.schedule = "simple"; draft.mossImage?.skipStop = 0.5
    var asset = MediaAsset(name: "Fast", kind: .lora, path: "/models/fast.safetensors", scope: .global)
    asset.loraModel = .mossImage; asset.loraProfile = "moss_sft_4step_v2"
    try draft.applyMOSSLoRAs([.init(asset: asset)])
    XCTAssertEqual(draft.steps, 4); XCTAssertEqual(draft.guidance, 1); XCTAssertEqual(draft.mossImage?.sampler, "euler")
    try draft.setMOSSAdapterEnabled(id: draft.mossImage!.loras![0].id, enabled: false)
    XCTAssertEqual(draft.steps, 25); XCTAssertEqual(draft.guidance, 2.5); XCTAssertEqual(draft.mossImage?.sampler, "er_sde")
    var group = LoRAGroup(name: "MOSS", engine: .drawThings, members: draft.mossImage!.loras!)
    group.imageProvider = .nativeMOSS; try group.validate()
    XCTAssertFalse(group.supports(.drawThings))
    XCTAssertThrowsError(try draft.applyMOSSLoRAs(draft.mossImage!.loras!))
  }
  func testSelectingMOSSUsesItsOwnRecipeAndRetainsOtherBackendSettings() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.modelID = "existing-image-model"; draft.steps = 7; draft.guidance = 4
    draft.selectProvider(.nativeMOSS)
    XCTAssertEqual(draft.modelID, MOSSImageSettings.sftModelID)
    XCTAssertEqual(draft.steps, 50); XCTAssertEqual(draft.guidance, 3)
    XCTAssertNotNil(draft.mossImage)
    draft.selectProvider(.drawThings)
    XCTAssertEqual(draft.modelID, "existing-image-model"); XCTAssertEqual(draft.steps, 7)
    draft.selectProvider(.nativeMOSS); XCTAssertEqual(draft.steps, 50)
  }
  func testNativeRequestUsesCurrentCanvasDimensionsAndLeavesGuideOutOfConditioning() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS)
    draft.mossImage?.transformerDirectory = "/tmp/transformer"
    draft.mossImage?.sharedDirectory = "/tmp/shared"
    draft.width = 896; draft.height = 1152; draft.prompt = "Serum ad"
    var layout = ImageLayoutDraft(width: 1024, height: 1024, caption: "Older caption", regions: [
      .init(type: .text, caption: "Thin letters", text: "GLOW SERUM", bounds: .init(x: 0.14, y: 0.07, width: 0.72, height: 0.1))])
    layout.guideImagePath = "/tmp/editor-guide.jpg"; layout.guideOpacity = 0.5; draft.mossLayout = layout
    let bytes = try NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 55)
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
    let settings = try XCTUnwrap(root["configuration"] as? [String: Any])
    XCTAssertEqual(settings["seed"] as? Int, 55)
    XCTAssertEqual(settings["width"] as? Int, 896)
    let sent = try XCTUnwrap(root["layout"] as? [String: Any])
    XCTAssertNil(sent["guideImagePath"])
    let admitted = try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONSerialization.data(withJSONObject: sent))
    XCTAssertEqual(admitted.summary.caption, "Serum ad")
    XCTAssertEqual(admitted.width, 896)
    XCTAssertTrue(try MOSSLayoutCompiler.compile(admitted).effectivePrompt.contains("[-496,-323,-380,323]"))
    XCTAssertEqual(draft.mossLayout?.guideImagePath, "/tmp/editor-guide.jpg")
  }
  func testMOSSRejectsImageConditioningAndWrongModelSelection() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS); draft.prompt = "test"
    draft.mossImage?.transformerDirectory = "/tmp/transformer"; draft.mossImage?.sharedDirectory = "/tmp/shared"
    draft.canvas = .init(path: "/tmp/source.png")
    XCTAssertNotNil(draft.imageInputIssue)
    XCTAssertThrowsError(try NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 0))
    draft.canvas = nil; draft.modelID = "other-model"
    XCTAssertThrowsError(try NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 0))
  }
  func testSavedTakeRetainsEditorGuideWithoutAddingItToInference() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Current scene"
    var authored = ImageLayoutDraft(width: 1024, height: 1024, caption: "Earlier scene")
    authored.guideImagePath = "/cache/source.png"; authored.guideOpacity = 0.4
    authored.analysis = .init(requestID: "analysis", sourceSHA256: String(repeating: "a", count: 64),
      sourceWidth: 800, sourceHeight: 453, sourceName: "source.jpg", modelIdentity: "qwen", recipe: "v1", warnings: [])
    draft.mossLayout = authored; draft.width = 1872; draft.height = 1056
    var admitted = authored; admitted.guideImagePath = nil; admitted.guideOpacity = nil
    admitted.width = draft.width; admitted.height = draft.height; admitted.summary.caption = draft.prompt
    let saved = try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft)
    XCTAssertEqual(saved.guideImagePath, authored.guideImagePath)
    XCTAssertEqual(saved.analysis, authored.analysis)
    XCTAssertEqual(try MOSSLayoutCompiler.compile(saved), try MOSSLayoutCompiler.compile(admitted))
    XCTAssertEqual(try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONEncoder().encode(saved)), saved)
    admitted.summary.caption = "A different result"
    XCTAssertThrowsError(try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft))
  }
  func testInvalidLayerExpressionIsNotSilentlyIgnored() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeMOSS); draft.prompt = "test"
    draft.mossImage?.transformerDirectory = "/tmp/transformer"; draft.mossImage?.sharedDirectory = "/tmp/shared"
    draft.mossImage?.skipLayerExpression = "22,garbage"
    XCTAssertThrowsError(try NativeMOSSImagePreparation.request(draft: draft, requestID: UUID(), seed: 0))
  }
  func testModelLibraryDiscoveryUsesRequestedVariantAndSharedComponents() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    for name in ["Shared/text_encoder", "Shared/vae", "Shared/tokenizer", "SFT/transformer", "Base/transformer"] {
      let folder = directory.appendingPathComponent(name)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      try Data("{}".utf8).write(to: folder.appendingPathComponent(name.hasSuffix("tokenizer") ? "tokenizer.json" : "config.json"))
    }
    let sft = try XCTUnwrap(MOSSModelSetup.discover(root: directory, variant: "sft"))
    let base = try XCTUnwrap(MOSSModelSetup.discover(root: directory, variant: "base"))
    XCTAssertEqual(sft.sharedDirectory, base.sharedDirectory)
    XCTAssertTrue(base.transformerDirectory.hasSuffix("Base/transformer"))
    XCTAssertTrue(sft.transformerDirectory.hasSuffix("SFT/transformer"))
  }
  func testLegacyDraftAndLayoutSessionRoundtrip() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    let oldBytes = try JSONEncoder().encode(draft)
    XCTAssertEqual(try JSONDecoder().decode(DrawThingsImageDraft.self, from: oldBytes), draft)
    draft.mossLayout = .init(width: 1024, height: 1024, caption: "Saved layout", regions: [
      .init(type: .object, caption: "A cup", bounds: .init(x: 0.25, y: 0.25, width: 0.5, height: 0.5))])
    var library = ImageWorkspaceLibrary(); library.record(draft, preview: nil)
    let loaded = try JSONDecoder().decode(ImageWorkspaceLibrary.self, from: JSONEncoder().encode(library))
    XCTAssertEqual(loaded.sessions[draft.storageKey]?.draft.mossLayout, draft.mossLayout)
  }
  func testMOSSResultRejectsStaleFingerprintOrWrongGeometry() throws {
    let id = UUID()
    let result: [String: Any] = ["schema": "weetodd-moss-image-result-v1", "requestID": id.uuidString,
      "eligibility": "allowed", "fingerprint": "expected", "seed": 55, "width": 896, "height": 1152]
    try NativeMOSSImageResult.validateIdentity(result, requestID: id, fingerprint: "expected", seed: 55, width: 896, height: 1152)
    XCTAssertThrowsError(try NativeMOSSImageResult.validateIdentity(result, requestID: id, fingerprint: "stale", seed: 55, width: 896, height: 1152))
    XCTAssertThrowsError(try NativeMOSSImageResult.validateIdentity(result, requestID: id, fingerprint: "expected", seed: 55, width: 1024, height: 1024))
  }
}
