import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeH3VDNSetupTests:XCTestCase {
  @MainActor func testInstalledVDNGuidedSetupCreatesFrozenStudioQualificationWithoutPython() async throws {
    let env=ProcessInfo.processInfo.environment
    guard let source=env["WEETODD_VDN_STUDIO_SOURCE"],let worker=env["WEETODD_VDN_STUDIO_WORKER"],
      let output=env["WEETODD_VDN_STUDIO_OUTPUT"] else { throw XCTSkip("Opt-in installed VDN setup and immutable lifecycle preparation; no generation.") }
    let original=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:source))) as! [String:Any]
    let components=original["components"] as! [String:Any],config=original["config"] as! [String:Any]
    let vdn=original["vdn"] as! [String:Any],adapters=(original["loras"] as! [String:Any])["adapters"] as! [[String:Any]]
    let root=URL(fileURLWithPath:output)
    guard !FileManager.default.fileExists(atPath:root.path) else { throw StudioError.invalid("Use a fresh VDN setup qualification directory.") }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.runtime=RuntimeSettings(root:"/unavailable",pythonPath:"/unavailable/python",profilesDirectory:root.appendingPathComponent("Profiles").path)
    store.runtime.nativeH3Enabled=true;store.runtime.h3WorkerPath=worker;store.runtime.ffmpegPath=original["ffmpeg"] as? String ?? "/opt/homebrew/bin/ffmpeg"
    let state=ModelSetupState();let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id=="swift-h3-vdn8" });state.begin(preset)
    for field in preset.components {
      let value=field.key=="vdn_stage" ? vdn["checkpoint"] : field.key=="vdn_transformer" ? components["transformer"] : field.key=="vdn_input_grid" ? adapters[1]["adaln_input_grid"] : components[field.key]
      state.selection.components[field.key]=try XCTUnwrap(value as? String)
    }
    let scan=try NativeModelSetup.scan(presetID:preset.id,roots:[state.selection.components["vdn_stage"]!,state.selection.components["vdn_transformer"]!,state.selection.components["vdn_input_grid"]!])
    for key in ["vdn_stage","vdn_transformer","vdn_input_grid"] { XCTAssertTrue(scan.candidates[key]?.contains(state.selection.components[key]!)==true,"\(key) discovery failed") }
    state.memoryMode = .lowerMemory
    await state.createRecipe(store:store)
    guard state.error.isEmpty else { throw StudioError.invalid(state.error) }
    let profile=try XCTUnwrap(store.profiles.first { $0.id==state.resultPath })
    XCTAssertEqual(profile.generation?.supportedTasks,["t2v"]);XCTAssertEqual(profile.generation?.vdn,true)
    let created=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:state.resultPath))) as! [String:Any]
    for key in ["vdn","loras"] { XCTAssertTrue(NSDictionary(dictionary:created[key] as! [String:Any]).isEqual(to:original[key] as! [String:Any])) }
    // Freeze the reviewed worker workload before the expensive Studio generation.
    let frozen=root.appendingPathComponent("frozen-recipe.json")
    try JSONSerialization.data(withJSONObject:original,options:[.prettyPrinted,.sortedKeys]).write(to:frozen)
    var clip=Clip(engine:.h3);clip.generationSelection = .init(task:"t2v");clip.profileID=frozen.path
    clip.prompt=original["prompt"] as! String;clip.duration=config["duration_seconds"] as! Double
    clip.seed=config["seed"] as! Int;clip.generationWidth=config["width"] as! Int;clip.generationHeight=config["height"] as! Int
    var project=StudioProject();project.clips=[clip]
    let request:[String:Any]=["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),"globalAssets":[],"clipID":clip.id.uuidString]
    let editor=root.appendingPathComponent("editor-request.json");try JSONSerialization.data(withJSONObject:request).write(to:editor)
    let manifest:[String:String]=["engine":"h3","task":"t2va","recipe":frozen.path,"editorRequest":editor.path,
      "worker":worker,"workerSHA256":try NativeHeadlessJob.fileHash(URL(fileURLWithPath:worker)),
      "ffmpeg":store.runtime.ffmpegPath,"output":root.appendingPathComponent("Studio-lifecycle").path]
    try JSONSerialization.data(withJSONObject:manifest,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent("lifecycle-manifest.json"))
    XCTAssertFalse(store.bridge.busy);XCTAssertFalse(state.resultPath.isEmpty)
  }
  @MainActor func testSelectingVDNSetupReplacesIncompatibleSamplerOverridesWithoutDeletingMedia() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let recipe=root.appendingPathComponent("vdn.json");try Data("{}".utf8).write(to:recipe)
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    var clip=Clip(engine:.h3);clip.prompt="Keep my scene";clip.generationSelection = .init(task:"t2v")
    clip.generationSelection!.steps=4;clip.generationSelection!.h3SamplingMethod = .resMultistep
    clip.attachments=[Attachment(assetID:UUID(),role:.lora)]
    store.project.clips=[clip];store.selectedClipID=clip.id
    let state=ModelSetupState();state.begin(try XCTUnwrap(NativeModelSetup.catalog().first { $0.id=="swift-h3-vdn8" }))
    state.resultPath=recipe.path;state.useRecipeForSelectedClip(store:store)
    let updated=try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(updated.profileID,recipe.path);XCTAssertEqual(updated.generationSelection?.steps,8)
    XCTAssertEqual(updated.generationSelection?.h3SamplingMethod,.euler)
    XCTAssertEqual(updated.attachments,clip.attachments);XCTAssertEqual(updated.prompt,clip.prompt)
  }
}
