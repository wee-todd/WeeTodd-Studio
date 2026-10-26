import AppKit
import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class H3ImageInteractionTests: XCTestCase {
  @MainActor func testImageLoRAApplicationUsesVisibleTriggersAndFrozenGroupsWithoutChangingMovie() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.addClip();let original=store.project
    var image=DrawThingsImageDraft(destination:.init(scope:.project,projectID:store.project.id))
    image.selectProvider(.nativeH3);image.prompt="A robot."
    store.imageDraft=image
    var asset=MediaAsset(name:"Weapon",kind:.lora,path:"/tmp/weapon.safetensors",scope:.global)
    asset.loraModel = .h3
    let key=LoRAPromptKey.native(asset.path)
    store.loraPromptLibrary[key] = .init(triggers:["BUNNY"],notes:"Action guidance")
    store.applyH3ImageLoRAMembers([LoRAMember(asset:asset)])
    XCTAssertEqual(store.imageDraft?.prompt,"BUNNY\nA robot.")
    XCTAssertEqual(store.imageDraft?.h3Image?.loras.first?.asset.loraPrompt?.notes,"Action guidance")
    XCTAssertEqual(store.project,original)
    let before=store.imageDraft
    store.applyH3ImageLoRAMembers([LoRAMember(asset:asset)])
    XCTAssertEqual(store.imageDraft,before)
    let group=try XCTUnwrap(store.currentLoRAGroup(imageWorkspace:true))
    store.loraPromptLibrary[key] = .init(triggers:["changed"])
    store.imageDraft?.prompt="A new robot."
    store.applyLoRAGroup(group,mode:.replace,imageWorkspace:true)
    XCTAssertEqual(store.imageDraft?.prompt,"BUNNY\nA new robot.")
  }
  @MainActor func testH3ProviderUsesInstalledProfileWithoutDiscardingOtherBackends() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.profiles=[ModelProfile(id:"/tmp/model.json",name:"H3 model",engine:"h3",task:"t2v")]
    store.imageDraft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:store.project.id))
    store.imageDraft?.modelID="dt.ckpt";store.imageDraft?.prompt="A robot."
    store.selectImageProvider(.nativeH3)
    XCTAssertEqual(store.imageDraft?.modelID,"/tmp/model.json")
    XCTAssertEqual(store.imageDraft?.steps,20)
    XCTAssertEqual(store.imageDraft?.h3Image?.decoderMode,.qualityFiveFrames)
    store.selectImageProvider(.drawThings)
    XCTAssertEqual(store.imageDraft?.modelID,"dt.ckpt")
    XCTAssertEqual(store.imageDraft?.prompt,"A robot.")
  }
  @MainActor func testImageDecoderLinkPersistsWithoutRewritingAnExistingDraft() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:store.project.id))
    draft.selectProvider(.nativeH3);draft.h3Image?.imageVAEPath="/tmp/frozen.safetensors"
    store.imageDraft=draft
    try store.saveH3ImageVAEPath("/tmp/new.safetensors",applyToDraft:false)
    XCTAssertEqual(store.imageDraft,draft)
    let restored=StudioStore(dataDirectory:root,restoreSession:false)
    restored.loadH3ImageModels()
    XCTAssertEqual(restored.h3ImageVAEPath,"/tmp/new.safetensors")
  }

  @MainActor func testNativeImageCompletionKeepsResolvedSeedAndProvenance() async throws {
    let (store, calls) = try nativeFixture(replaceDocumentDuringRender: false)
    let clips = store.project.clips
    await store.generateImageAsset()
    XCTAssertNil(store.error)
    XCTAssertEqual(calls(), ["h3-native-preflight", "h3-native-render"])
    XCTAssertEqual(store.project.clips, clips)
    let asset = try XCTUnwrap(store.project.assets.last)
    XCTAssertEqual(asset.generation?.provider, "nativeH3")
    XCTAssertNotNil(asset.generation?.nativeRequest)
    XCTAssertEqual(store.imagePreviewPath, asset.path)
    XCTAssertEqual(store.imageDraft?.seed, -1)
    XCTAssertNil(store.preparedH3Image)
    await store.generateImageAsset()
    XCTAssertEqual(calls().count, 4, "Each subsequent generation checks a fresh captured seed")
    XCTAssertEqual(store.project.assets.count, 2)
  }

  @MainActor func testLateNativeImageIsRetainedWithoutAttachingToReopenedMovie() async throws {
    let (store, _) = try nativeFixture(replaceDocumentDuringRender: true)
    await store.generateImageAsset()
    XCTAssertNil(store.error)
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.notice.contains("destination movie changed"))
  }

  @MainActor private func nativeFixture(replaceDocumentDuringRender: Bool) throws -> (StudioStore, () -> [String]) {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder, withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:folder) }
    let profile = folder.appendingPathComponent("profile.json")
    let root: [String:Any] = ["format":"weetodd-headless-v2", "engine":"h3", "prompt":"A robot.",
      "components":["task":"ref2va", "transformer":"/models/selected.safetensors",
        "video_vae":"/models/video.safetensors", "audio_vae":"/models/audio.safetensors",
        "text_encoder":"/models/qwen", "tokenizer":"/models/tokenizer.json"],
      "config":["width":32,"height":32,"duration_seconds":5,"steps":5,"seed":42]]
    try JSONSerialization.data(withJSONObject:root).write(to:profile)
    var calls: [String] = [], active: StudioStore?
    let store = StudioStore(dataDirectory:folder, restoreSession:false) { command, _, body, output in
      calls.append(command)
      let recipeURL = URL(fileURLWithPath:try XCTUnwrap(body["recipePath"] as? String))
      let recipe = try JSONSerialization.jsonObject(with:Data(contentsOf:recipeURL)) as! [String:Any]
      let still = recipe["still_image"] as! [String:Any], config = recipe["config"] as! [String:Any]
      let fingerprint = "sha256:" + String(repeating:"a",count:64)
      var result: [String:Any] = ["nativeRuntime":"swift-mlx", "jobID":UUID().uuidString,
        "requestID":still["request_id"]!, "fingerprint":fingerprint, "eligibility":"allowed",
        "normalizedRequest":["engine":"h3", "configuration":["seed":config["seed"]!,"width":32,"height":32]]]
      if command == "h3-native-render" {
        XCTAssertEqual(still["expected_fingerprint"] as? String,fingerprint)
        let directory = try XCTUnwrap(output)
        try FileManager.default.createDirectory(at:directory, withIntermediateDirectories:true)
        let png = directory.appendingPathComponent("image.png")
        let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:32,pixelsHigh:32,
          bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,
          bytesPerRow:128,bitsPerPixel:32))
        try XCTUnwrap(rep.representation(using:.png,properties:[:])).write(to:png)
        result["asset"] = ["path":png.path,"width":32,"height":32]
        if replaceDocumentDuringRender, let active {
          try active.replaceDocument(active.project,url:nil,isDirty:false)
        }
      } else { XCTAssertEqual(command,"h3-native-preflight") }
      return result
    }
    active = store
    var draft = DrawThingsImageDraft(destination:.init(scope:.project,projectID:store.project.id))
    draft.selectProvider(.nativeH3); draft.modelID=profile.path; draft.width=32; draft.height=32
    draft.prompt="A robot."; draft.seed = -1; draft.steps=4
    store.imageDraft=draft
    return (store, { calls })
  }

}
