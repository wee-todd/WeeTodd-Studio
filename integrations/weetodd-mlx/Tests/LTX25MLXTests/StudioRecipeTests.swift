import XCTest
@testable import LTX25MLX

final class StudioRecipeTests: XCTestCase {
  func testExtensionRecipeKeepsFrozenSourceAndOneCausalPublicationWindow() throws {
    var recipe=fixture()
    var config=recipe["config"] as! [String:Any]
    config["width"]=768;config["height"]=448;config["duration_seconds"]=4
    recipe["config"]=config
    let source:[String:Any]=["id":"source","kind":"video","role":"reference",
      "path":"/source.mp4","sha256":String(repeating:"a",count:64),"strength":1]
    recipe["conditioning"]=["version":1,"task":"extension",
      "audio_policy":"source_reencoded_and_generated_extension","inputs":[source],
      "extension":["direction":"after","context_frames":25,"additional_frames":96]]
    let data=try JSONSerialization.data(withJSONObject:recipe)
    let compiled=try MLXStudioExtensionRecipe.compile(data:data,outputDirectory:"/output")
    XCTAssertEqual(compiled.request.frames,121)
    XCTAssertEqual(compiled.window.outputRange,25..<121)
    XCTAssertEqual(compiled.source.path,"/source.mp4")
    XCTAssertEqual(compiled.sourceSHA256,String(repeating:"a",count:64))
    var wrong=recipe
    var conditioning=wrong["conditioning"] as! [String:Any]
    var reference=source;reference["sha256"]="invalid";conditioning["inputs"]=[reference]
    wrong["conditioning"]=conditioning
    XCTAssertThrowsError(try MLXStudioExtensionRecipe.compile(
      data:JSONSerialization.data(withJSONObject:wrong),outputDirectory:"/output"))
    conditioning=recipe["conditioning"] as! [String:Any]
    conditioning["extension"]=["direction":"before","context_frames":25,"additional_frames":96]
    wrong["conditioning"]=conditioning
    XCTAssertThrowsError(try MLXStudioExtensionRecipe.compile(
      data:JSONSerialization.data(withJSONObject:wrong),outputDirectory:"/output"))
  }
  func testAudioDrivenRecipeRetainsExactSourceIntervalAndRejectsExtraInputs() throws {
    var recipe=fixture()
    let source:[String:Any]=["id":"voice","kind":"audio","role":"audio_driver",
      "path":"/source.wav","strength":1,"source_start_seconds":1.25,"source_duration_seconds":2.0]
    recipe["conditioning"]=["version":1,"task":"a2v","audio_policy":"source","inputs":[source]]
    let request=try compile(recipe)
    XCTAssertEqual(request.version,4)
    XCTAssertEqual(request.audioReference?.sourceStartSeconds,1.25)
    XCTAssertEqual(request.audioReference?.sourceDurationSeconds,2.0)
    XCTAssertEqual(request.referenceImages.count,0)
    var bad=source;bad["strength"]=0.5
    recipe["conditioning"]=["version":1,"task":"a2v","inputs":[bad]]
    XCTAssertThrowsError(try compile(recipe))
    bad=source;bad.removeValue(forKey:"source_start_seconds")
    recipe["conditioning"]=["version":1,"task":"a2v","inputs":[bad]]
    XCTAssertThrowsError(try compile(recipe))
    recipe["conditioning"]=["version":1,"task":"a2v","inputs":[source,source]]
    XCTAssertThrowsError(try compile(recipe))
    let first:[String:Any]=["id":"first","kind":"image","role":"keyframe",
      "path":"/first.png","strength":1,"frame_index":0]
    recipe["conditioning"]=["version":1,"task":"a2v","audio_policy":"source",
      "inputs":[source,first]]
    let withFirst=try compile(recipe)
    XCTAssertEqual(withFirst.task,"a2v")
    XCTAssertEqual(withFirst.referenceImages.map(\.role),["first"])
    var last=first;last["frame_index"]="last";recipe["conditioning"]=["version":1,"task":"a2v",
      "inputs":[source,last]]
    XCTAssertThrowsError(try compile(recipe))
  }
  func fixture() -> [String:Any] {
    ["engine":"ltx25", "format":"weetodd-headless-v2", "prompt":"A cup moves.",
     "components":["transformer_path":"/models/transformer", "text_encoder_path":"/models/text",
       "video_vae_path":"/models/video", "audio_vae_path":"/models/audio", "spatial_upscaler_path":"/models/upscale"],
     "config":["pipeline_mode":"distilled", "width":1344,"height":768,"duration_seconds":88.0/24,
       "frame_rate":24,"seed":43,"stage1_steps":8,"stage2_steps":3],
     "conditioning":["version":1,"task":"fflf","inputs":[
       ["id":"first","kind":"image","role":"keyframe","path":"/first.png","frame_index":0,"strength":1],
       ["id":"last","kind":"image","role":"keyframe","path":"/last.png","frame_index":"last","strength":0.8]]]]
  }
  func compile(_ object:[String:Any]) throws -> MLXDistilledRequest {
    try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output")
  }
  func testPreservesMatchedGeometrySeedReferencesAndNoise() throws {
    let request=try compile(fixture())
    XCTAssertEqual(request.frames,89);XCTAssertEqual(request.seed,43)
    XCTAssertEqual(request.width,1344);XCTAssertEqual(request.height,768)
    XCTAssertEqual(request.noisePolicy.rawValue,"mlx_threefry_bf16_v1")
    XCTAssertEqual(request.referenceImages.map(\.role),["first","last"])
    XCTAssertEqual(request.referenceImages.last?.strength,0.8)
    XCTAssertEqual(request.referenceImages.first?.crf,33)
    XCTAssertEqual(request.connectorCheckpoint,"/models/transformer/pages/fixed.safetensors")
  }
  func testRejectsUnsupportedControlsInsteadOfDroppingThem() throws {
    for (key,value) in [("stage1_steps",4 as Any),("video_cfg_scale",3),("dfr_enabled",true),
      ("duration_mode","automatic"),("stage1_sampler","euler"),("new_control",true)] {
      var recipe=fixture(),config=recipe["config"] as! [String:Any];config[key]=value;recipe["config"]=config
      XCTAssertThrowsError(try compile(recipe),key)
    }
    var recipe=fixture();recipe["scene"]=["clips":[]]
    XCTAssertThrowsError(try compile(recipe))
    recipe=fixture();var config=recipe["config"] as! [String:Any];config["seed"]=true;recipe["config"]=config
    XCTAssertThrowsError(try compile(recipe))
  }
  func testStudioDFRRecipeRoutesSpatialAndTemporalWithExactAdapter() throws {
    var recipe=fixture()
    var config=recipe["config"] as! [String:Any]
    config["dfr_enabled"]=true
    config["dfr_detailing_lora_path"]="/models/detail.safetensors"
    config["dfr_detailing_lora_strength"]=0.5
    recipe["config"]=config
    let spatial=try compile(recipe)
    XCTAssertEqual(spatial.version,8)
    XCTAssertEqual(spatial.task,"dfr")
    XCTAssertEqual(spatial.dfr?.adapterPath,"/models/detail.safetensors")
    XCTAssertEqual(spatial.dfr?.adapterStrength,0.5)
    XCTAssertEqual(spatial.referenceImages.map(\.role),["first","last"])
    XCTAssertTrue(spatial.stageOneLoras.isEmpty)
    config["dfr_temporal_rounds"]=2
    config["dfr_temporal_upsampler_path"]="/models/temporal.safetensors"
    recipe["config"]=config
    let temporal=try compile(recipe)
    XCTAssertEqual(temporal.version,9)
    XCTAssertEqual(temporal.dfr?.temporalRounds,2)
    XCTAssertEqual(temporal.dfr?.temporalUpscalerPath,"/models/temporal.safetensors")
    config["dfr_temporal_upsampler_path"]=""
    recipe["config"]=config
    XCTAssertThrowsError(try compile(recipe))
    config["dfr_temporal_upsampler_path"]="/models/temporal.safetensors"
    config["dfr_prebaked_transformer_path"]="/models/prebaked"
    recipe["config"]=config
    XCTAssertThrowsError(try compile(recipe))
    config["dfr_prebaked_transformer_path"]=""
    recipe["config"]=config
    var components=recipe["components"] as! [String:Any]
    components["loras"]=[["/models/other.safetensors",0.5]]
    recipe["components"]=components
    XCTAssertThrowsError(try compile(recipe))
  }
  func testRejectsInactiveDistilledNegativePromptBeforeModelLoad() throws {
    var recipe = fixture()
    var config = recipe["config"] as! [String: Any]
    config["negative_prompt"] = "no ghosting"
    recipe["config"] = config
    XCTAssertThrowsError(try compile(recipe))
  }
  func testLoRAOrderAndStrengthSurviveBothStages() throws {
    var recipe=fixture()
    var components=recipe["components"] as! [String:Any]
    components["loras"]=[["/ltx23-style.safetensors",0.8],["/character.safetensors",0.2]]
    recipe["components"]=components
    let request=try compile(recipe)
    XCTAssertEqual(request.stageOneLoras.map(\.path),["/ltx23-style.safetensors","/character.safetensors"])
    XCTAssertEqual(request.stageTwoLoras.map(\.strength),[0.8,0.2])
    components["loras"]=[["/style",1,"alpha override"]];recipe["components"]=components
    XCTAssertThrowsError(try compile(recipe))
  }
  func testRejectsMiddleReferenceAndRoundTiesLikePython() throws {
    var recipe=fixture();var c=recipe["conditioning"] as! [String:Any]
    var inputs=c["inputs"] as! [[String:Any]];inputs[1]["frame_index"]=40;c["inputs"]=inputs;recipe["conditioning"]=c
    XCTAssertThrowsError(try compile(recipe))
    recipe=fixture();var config=recipe["config"] as! [String:Any];config["duration_seconds"]=20.0/24;recipe["config"]=config
    XCTAssertEqual(try compile(recipe).frames,17)
  }
  func testStudioSingleAnchorAndReversedAttachmentOrder() throws {
    var recipe=fixture(),c=fixture()["conditioning"] as! [String:Any]
    let inputs=c["inputs"] as! [[String:Any]]
    c["inputs"]=[inputs[0]];recipe["conditioning"]=c
    XCTAssertEqual(try compile(recipe).task,"i2v")
    c["inputs"]=Array(inputs.reversed());recipe["conditioning"]=c
    XCTAssertEqual(try compile(recipe).referenceImages.map(\.path),["/first.png","/last.png"])
    c["inputs"]=[inputs[0],inputs[0]];recipe["conditioning"]=c
    XCTAssertThrowsError(try compile(recipe))
  }
  func testSharedConditioningContractRejectsMismatchedTaskPolicyAndIdentity() throws {
    var recipe=fixture(), c=fixture()["conditioning"] as! [String:Any]
    c["audio_policy"]="generated";recipe["conditioning"]=c
    XCTAssertEqual(try compile(recipe).task,"fflf")
    c["audio_policy"]="source";recipe["conditioning"]=c
    XCTAssertThrowsError(try compile(recipe))
    c["audio_policy"]="generated";c["task"]="i2v";recipe["conditioning"]=c
    XCTAssertThrowsError(try compile(recipe))
    c["task"]="fflf"
    var inputs=c["inputs"] as! [[String:Any]];inputs[1]["id"]="first";c["inputs"]=inputs
    recipe["conditioning"]=c
    XCTAssertThrowsError(try compile(recipe))
  }
}
