import Foundation
import XCTest
@testable import StudioCore

final class NativeH3VDNTests: XCTestCase {
  private func fixture() throws -> (URL, [String:Any], StudioProject, [String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let repo="/models/vdn",stage=repo+"/stage-dmd-step-250"
    let paths=[stage+"/adapters/default/adapter_model.safetensors",stage+"/adapters/turbo/adapter_model.safetensors"]
    let adapters:[[String:Any]]=paths.enumerated().map { index,path in
      ["path":path,"strength":1.0,"profile":index==0 ? "standard":"turbo",
       "qkv_layout":"contiguous_qkv","start_after_evaluations":0,
       "adaln_input_grid":index==0 ? NSNull():"/models/silu-grid.safetensors"]
    }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"h3","candidate":"VDN8",
      "prompt":"Frozen prompt","components":["task":"t2va","transformer":"/models/base",
        "text_encoder":"/models/qwen","tokenizer":"/models/tokenizer.json","video_vae":"/models/video","audio_vae":"/models/audio"],
      "config":["width":672,"height":384,"duration_seconds":5.0,"steps":9,"seed":42,"drop_adaln":true,"sampling_method":"euler","memory_mode":"low_memory_bf16"],
      "conditioning":["version":1,"task":"t2v","inputs":[],"audio_policy":"generated"],
      "vdn":["repository":repo,"checkpoint":stage,"model_spec":stage+"/model_spec.json",
        "linear_branch":stage+"/linear_branch/model.safetensors","default_adapter":paths[0],
        "turbo_adapter":paths[1],"schedule_points":9,"stage":"stage-dmd-step-250","inference_backend":"verified"],
      "loras":["version":1,"adapters":adapters]]
    try save(recipe,root)
    var clip=Clip(engine:.h3);clip.profileID=root.appendingPathComponent("vdn.json").path
    clip.prompt="Frozen prompt";clip.duration=5;clip.seed=42;clip.generationWidth=672;clip.generationHeight=384
    var project=StudioProject();project.clips=[clip]
    return(root,recipe,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  private func save(_ recipe:[String:Any],_ root:URL) throws {
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("vdn.json"))
  }
  private func request(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
     "runtime":runtime,"clipID":project.clips[0].id.uuidString,"globalAssets":[]]
  }
  func testFastH3ProfilesExposeFrozenFourEvaluationsAndRejectOverrides() throws {
    let(root,base,original,runtime)=try fixture()
    for variant in ["dense-v1","vsa-v1"] {
      var recipe = base;recipe.removeValue(forKey:"vdn");recipe.removeValue(forKey:"loras")
      recipe["fasth3"] = ["variant":variant]
      var config = recipe["config"] as! [String:Any];config["steps"] = 5;recipe["config"] = config
      try save(recipe,root)
      let catalog = try NativeH3Preparation.catalog(directory:root.path)
      let generation = try XCTUnwrap(catalog.first?["generation"] as? [String:Any])
      XCTAssertEqual(generation["fasth3"] as? Bool,true)
      let controls = try XCTUnwrap(generation["controls"] as? [String:Any])
      XCTAssertEqual(controls["evaluations"] as? Int,4);XCTAssertEqual(controls["stepsEditable"] as? Bool,false)
      let result = try NativeH3Preparation.compose(request:request(original,runtime))
      XCTAssertEqual(((result["recipe"] as? [String:Any])?["fasth3"] as? [String:String])?["variant"],variant)
      var changed = original;changed.clips[0].generationSelection = .init(task:"t2v")
      changed.clips[0].generationSelection!.steps = 8
      XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(changed,runtime)))
      changed = original;changed.clips[0].profileID = "auto"
      XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(changed,runtime)))
    }
  }

  func testCatalogAdmitsVDNWithFrozenControlsAndCompleteSourceIdentity() throws {
    let(root,recipe,project,runtime)=try fixture()
    let catalog=try NativeH3Preparation.catalog(directory:root.path)
    XCTAssertEqual(catalog.count,1)
    let generation=try XCTUnwrap(catalog.first?["generation"] as? [String:Any])
    XCTAssertEqual(generation["supportedTasks"] as? [String],["t2v"])
    XCTAssertEqual(generation["vdn"] as? Bool,true)
    let controls=try XCTUnwrap(generation["controls"] as? [String:Any])
    XCTAssertEqual(controls["evaluations"] as? Int,8);XCTAssertEqual(controls["stepsEditable"] as? Bool,false)
    let result=try NativeH3Preparation.compose(request:request(project,runtime))
    let composed=try XCTUnwrap(result["recipe"] as? [String:Any])
    for key in ["vdn","loras"] {
      XCTAssertTrue(NSDictionary(dictionary:try XCTUnwrap(composed[key] as? [String:Any])).isEqual(to:try XCTUnwrap(recipe[key] as? [String:Any])))
    }
    let description=try NativeH3Preparation.describe(request:request(project,runtime))
    let sources=try XCTUnwrap(description["sourcePaths"] as? [String])
    for key in ["model_spec","linear_branch","default_adapter","turbo_adapter"] {
      XCTAssertTrue(sources.contains((recipe["vdn"] as! [String:Any])[key] as! String))
    }
    XCTAssertTrue(sources.contains("/models/silu-grid.safetensors"))
  }
  func testVDNRejectsOverridesControlsAndAutomaticSelectionBeforeInference() throws {
    let(_,_,original,runtime)=try fixture()
    var bad:[StudioProject]=[]
    var p=original;p.clips[0].profileID="auto";bad.append(p)
    p=original;p.clips[0].generationSelection = .init(task:"t2v");p.clips[0].generationSelection!.steps=4;bad.append(p)
    p=original;p.clips[0].generationSelection = .init(task:"t2v");p.clips[0].generationSelection!.h3SamplingMethod = .resMultistep;bad.append(p)
    p=original;p.clips[0].attachments=[Attachment(assetID:UUID(),role:.lora)];bad.append(p)
    p=original;p.clips[0].continuity = .init(mode:"independent",saveContext:true);bad.append(p)
    p=original;p.clips[0].generationSelection = .init(task:"t2v");p.clips[0].generationSelection!.h3Joint = .init();bad.append(p)
    for changed in bad { XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(changed,runtime))) }
    p=original;p.clips[0].generationSelection = .init(task:"t2v");p.clips[0].generationSelection!.steps=8
    XCTAssertNoThrow(try NativeH3Preparation.compose(request:request(p,runtime)))
  }
  func testMalformedOrUnqualifiedVDNProfilesStayOutOfCatalog() throws {
    let(root,original,_,_)=try fixture()
    for mutation in 0..<4 {
      var recipe=original
      if mutation==0 { var vdn=recipe["vdn"] as! [String:Any];vdn["stage"]="stage-b-step-2000";recipe["vdn"]=vdn }
      if mutation==1 { var config=recipe["config"] as! [String:Any];config["steps"]=5;recipe["config"]=config }
      if mutation==2 { var stack=recipe["loras"] as! [String:Any];var a=stack["adapters"] as! [[String:Any]];a[1]["strength"]=0.8;stack["adapters"]=a;recipe["loras"]=stack }
      if mutation==3 { recipe["continuation"]=["save_context":true] }
      try save(recipe,root);XCTAssertTrue(try NativeH3Preparation.catalog(directory:root.path).isEmpty)
    }
  }
  func testGuidedVDNSetupUsesLinkedPrunedBaseAndExactReleasedStack() throws {
    let(root,_,_,_)=try fixture()
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id=="swift-h3-vdn8" })
    XCTAssertEqual(Set(preset.components.filter { $0.importOnly == true }.map(\.key)),["vdn_transformer","vdn_stage","vdn_input_grid"])
    XCTAssertTrue(NativeModelSetup.catalog().filter { $0.id != preset.id && $0.id != "swift-h3-vdn50" && !$0.id.hasPrefix("swift-h3-fast-") }.flatMap(\.components).allSatisfy { $0.importOnly != true })
    XCTAssertTrue(NativeModelSetup.catalog().contains { $0.id=="swift-h3-vdn50" })
    var selected:[String:String]=[:]
    for field in preset.components {
      let url=root.appendingPathComponent(field.key=="vdn_stage" ? "stage-dmd-step-250":field.key)
      if field.kind=="directory" { try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true) }
      else { try Data([1]).write(to:url) }
      if field.key=="tokenizer" { try Data("{}".utf8).write(to:url.appendingPathComponent("tokenizer.json")) }
      selected[field.key]=url.path
    }
    let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
    let config=try XCTUnwrap(recipe["config"] as? [String:Any]);XCTAssertEqual(config["steps"] as? Int,9)
    XCTAssertEqual(config["width"] as? Int,672);XCTAssertEqual(config["height"] as? Int,384)
    XCTAssertEqual((recipe["components"] as? [String:Any])?["transformer"] as? String,selected["vdn_transformer"])
    let stack=try XCTUnwrap((recipe["loras"] as? [String:Any])?["adapters"] as? [[String:Any]])
    XCTAssertEqual(stack.compactMap { $0["strength"] as? Double },[1,1])
    XCTAssertEqual(stack[1]["adaln_input_grid"] as? String,selected["vdn_input_grid"])
    _=try NativeH3VDNProfile.packet(recipe)
    selected["vdn_stage"]=root.path
    XCTAssertThrowsError(try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory))
  }
  func testFiftyStepProfileHasSingleAdapterAndFiftyFrozenEvaluations() throws {
    let(root,original,originalProject,runtime)=try fixture()
    var recipe=original
    let stage="/models/vdn/stage-b-step-2000"
    var fields=recipe["vdn"] as! [String:Any]
    fields["stage"]="stage-b-step-2000";fields["checkpoint"]=stage
    fields["model_spec"]=stage+"/model_spec.json";fields["linear_branch"]=stage+"/linear_branch/model.safetensors"
    fields["default_adapter"]=stage+"/adapters/default/adapter_model.safetensors"
    fields["turbo_adapter"]=NSNull();fields["schedule_points"]=51;recipe["vdn"]=fields
    var config=recipe["config"] as! [String:Any];config["steps"]=51;recipe["config"]=config
    recipe["loras"]=["version":1,"adapters":[["path":fields["default_adapter"]!,"strength":1,
      "profile":"standard","qkv_layout":"contiguous_qkv","start_after_evaluations":0]]]
    try save(recipe,root)
    let catalog=try NativeH3Preparation.catalog(directory:root.path)
    XCTAssertEqual(catalog.count,1)
    XCTAssertEqual(catalog.first?["name"] as? String,"MiniMax H3 · VDN 50-step · Swift")
    let generation=try XCTUnwrap(catalog.first?["generation"] as? [String:Any])
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,50)
    var project=originalProject
    project.clips[0].generationSelection = .init(task:"t2v");project.clips[0].generationSelection!.steps=50
    XCTAssertNoThrow(try NativeH3Preparation.compose(request:request(project,runtime)))
    project.clips[0].generationSelection!.steps=8
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
    let sources=NativeH3VDNProfile.sources(recipe)
    XCTAssertFalse(sources.contains { $0.contains("turbo") })
    XCTAssertNotNil(NativeModelSetup.catalog().first { $0.id=="swift-h3-vdn50" })
  }

}
