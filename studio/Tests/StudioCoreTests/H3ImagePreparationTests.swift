import Foundation
import XCTest
@testable import StudioCore

final class H3ImagePreparationTests: XCTestCase {
  func testDirect1080DraftKeepsRequestedImageCanvasAndRejectsLargerBudgets() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(.nativeH3); draft.modelID = try profile(in: folder).path
    draft.prompt = "A copper robot."; draft.width = 1920; draft.height = 1088
    let root = try JSONSerialization.jsonObject(with: NativeH3ImagePreparation.recipe(
      draft: draft, requestID: UUID(), seed: 42)) as! [String:Any]
    let config = root["config"] as! [String:Any]
    XCTAssertEqual(config["width"] as? Int, 1920)
    XCTAssertEqual(config["height"] as? Int, 1088)
    draft.height = 1120
    XCTAssertThrowsError(try NativeH3ImagePreparation.recipe(draft: draft, requestID: UUID(), seed: 42))
    draft.width = 1952; draft.height = 1024
    XCTAssertThrowsError(try NativeH3ImagePreparation.recipe(draft: draft, requestID: UUID(), seed: 42))
  }
  func testH3ImageDraftRestoresWithoutBecomingDrawThings() throws {
    let draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    var object=try JSONSerialization.jsonObject(with:JSONEncoder().encode(draft)) as! [String:Any]
    object["provider"]="nativeH3"
    object["h3Image"]=["decoderMode":"quality_five_frames","frameSelection":"middle","imageVAEPath":"","loras":[]]
    let restored=try JSONDecoder().decode(DrawThingsImageDraft.self,from:JSONSerialization.data(withJSONObject:object))
    XCTAssertEqual(restored.executionProvider.rawValue,"nativeH3")
  }
  func profile(in folder: URL, task: String = "ref2va") throws -> URL {
    let url=folder.appendingPathComponent("profile.json")
    let root:[String:Any]=["format":"weetodd-headless-v2","engine":"h3","prompt":"Saved prompt",
      "components":["task":task,"transformer":"/tmp/selected.safetensors","text_encoder":"/tmp/qwen",
        "tokenizer":"/tmp/tokenizer.json","video_vae":"/tmp/original.safetensors","audio_vae":"/tmp/audio.safetensors"],
      "config":["width":768,"height":768,"duration_seconds":5.0,"steps":21,"seed":1],
      "conditioning":["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]]
    try JSONSerialization.data(withJSONObject:root).write(to:url)
    return url
  }
  func testPreparedRecipeCapturesActualSeedAndConvertsEvaluationsToSchedulePoints() throws {
    let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:folder) }
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    draft.selectProvider(.nativeH3);draft.modelID=try profile(in:folder).path
    draft.prompt="A copper robot.";draft.steps=20;draft.seed = -1
    let bytes=try NativeH3ImagePreparation.recipe(draft:draft,requestID:UUID(),seed:42)
    let root=try JSONSerialization.jsonObject(with:bytes) as! [String:Any]
    let config=root["config"] as! [String:Any]
    XCTAssertEqual(config["steps"] as? Int,21)
    XCTAssertEqual(config["seed"] as? Int,42)
    XCTAssertEqual((root["components"] as? [String:Any])?["transformer"] as? String,"/tmp/selected.safetensors")
    XCTAssertEqual((root["still_image"] as? [String:Any])?["mode"] as? String,"quality_five_frames")
    XCTAssertEqual(draft.seed,-1)
    draft.guidance=2
    XCTAssertThrowsError(try NativeH3ImagePreparation.recipe(draft:draft,requestID:UUID(),seed:42))
  }
  func testProviderSwitchingRestoresSettingsAndKeepsAuthoredInputs() throws {
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    draft.prompt="My picture";draft.modelID="dt.ckpt";draft.width=640
    draft.canvas=ImageWorkspaceInput(path:"/tmp/canvas.png")
    draft.selectProvider(.nativeH3);draft.width=768;draft.steps=20
    draft.h3Image?.imageVAEPath="/tmp/image.safetensors"
    draft.selectProvider(.nativeMLX);draft.width=1024;draft.steps=40
    draft.selectProvider(.nativeH3)
    XCTAssertEqual(draft.width,768);XCTAssertEqual(draft.steps,20)
    XCTAssertEqual(draft.h3Image?.imageVAEPath,"/tmp/image.safetensors")
    draft.selectProvider(.drawThings)
    XCTAssertEqual(draft.modelID,"dt.ckpt");XCTAssertEqual(draft.width,640)
    XCTAssertEqual(draft.prompt,"My picture");XCTAssertEqual(draft.canvas?.path,"/tmp/canvas.png")
  }

  func testProcessorDirectoryResolvesToItsTokenizerJSON() throws {
    let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:folder) }
    let url=try profile(in:folder)
    let processor=folder.appendingPathComponent("processor")
    try FileManager.default.createDirectory(at:processor,withIntermediateDirectories:true)
    try Data("{}".utf8).write(to:processor.appendingPathComponent("tokenizer.json"))
    var root=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    root["components"]=(root["components"] as! [String:Any]).merging(["tokenizer":processor.path]) {_,new in new}
    try JSONSerialization.data(withJSONObject:root).write(to:url)
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    draft.selectProvider(.nativeH3);draft.modelID=url.path;draft.prompt="A robot."
    let result=try JSONSerialization.jsonObject(with:NativeH3ImagePreparation.recipe(draft:draft,requestID:UUID(),seed:42)) as! [String:Any]
    XCTAssertEqual((result["components"] as? [String:Any])?["tokenizer"] as? String,processor.appendingPathComponent("tokenizer.json").path)
  }
  func testInstalledImageRecipeExportWhenRequested() throws {
    let env=ProcessInfo.processInfo.environment
    guard let profile=env["WEETODD_H3_IMAGE_PROFILE"],let output=env["WEETODD_H3_IMAGE_RECIPE_OUTPUT"] else {
      throw XCTSkip("Set an installed H3 profile and output recipe for qualification.")
    }
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    draft.selectProvider(.nativeH3);draft.modelID=profile;draft.width=512;draft.height=512
    draft.steps=4;draft.seed=745693
    draft.prompt="A cinematic editorial photograph of a small copper and turquoise robot sitting beside a ceramic coffee cup on a wooden workshop bench, warm window light from the left, detailed brushed metal and tiny engraved control labels, shallow depth of field, a quiet carefully composed still life."
    if let vae=env["WEETODD_H3_IMAGE_SINGLE_VAE"] {
      draft.h3Image?.decoderMode = .singleLatent;draft.h3Image?.imageVAEPath=vae
    }
    if let reference=env["WEETODD_H3_IMAGE_REFERENCE"] {
      draft.canvas=ImageWorkspaceInput(path:reference)
      draft.prompt="Use <Picture 1> as the reference. Keep the same small copper and turquoise robot, wooden workshop bench, camera angle and warm window light. Replace the ceramic coffee cup with a small red enamel teapot. A carefully composed cinematic editorial photograph."
    }
    let data=try NativeH3ImagePreparation.recipe(draft:draft,requestID:UUID(),seed:745693)
    try data.write(to:URL(fileURLWithPath:output),options:.atomic)
  }

}
