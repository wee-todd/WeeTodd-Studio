import XCTest
import Foundation
import Darwin
import LTX25MLX

final class MLXDistilledRequestTests:XCTestCase {
  private func ordinaryPolicyRequest(version:Int) throws -> [String:Any] {
    var value=ordinary();value["version"]=version
    if version<14 { value.removeValue(forKey:"generated_keyframes") }
    if version<13 { value.removeValue(forKey:"automatic_duration") }
    if version == 12 || version == 14 {
      let policy=try MLXGuidedSampling(mode:.guidedHQ,steps:15,negativePrompt:"blur",
        stg:0,videoRescale:0.45,audioRescale:1,stgBlocks:[],
        distilledAdapterPath:"/models/refinement.safetensors")
      value["guided_sampling"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(policy))
    }
    if version>=13 {
      let policy=MLXAutomaticDurationPolicy(headCheckpointPath:"/models/duration.safetensors",
        minimumSeconds:1,maximumSeconds:3)
      value["automatic_duration"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(policy))
    }
    if version == 14 {
      value["task"]="i2v"
      value["reference_images"]=[
        ["role":"first","path":"/first.png","strength":1,"crf":0],
        ["role":"keyframe","frame_index":7,"path":"/timed.png","strength":0.8,"crf":33]]
    }
    return value
  }
  func testOrdinaryVersionsRejectIngredientsOnlySamplerAtDecode() throws {
    for version in [12,13,14] {
      var value=try ordinaryPolicyRequest(version:version)
      XCTAssertEqual(try decode(value).ingredientsSampling,.deterministic)
      value["ingredients_sampling"]="euler_ancestral_cfg_pp_float32_v1"
      XCTAssertThrowsError(try decode(value),"version \(version) must reject before any weighted provider") { error in
        XCTAssertTrue(String(describing:error).contains("Ingredients sampler"))
      }
    }
  }
  func testDeterministicPlaceholderPreservesGuidedAutomaticTimedAndGeneratedContract() throws {
    let request=try decode(ordinaryPolicyRequest(version:14))
    let replay=try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request))
    XCTAssertEqual(replay.version,14);XCTAssertEqual(replay.ingredientsSampling,.deterministic)
    XCTAssertEqual(replay.guidedSampling?.mode,.guidedHQ);XCTAssertEqual(replay.guidedSampling?.steps,15)
    XCTAssertEqual(replay.guidedSampling?.negativePrompt,"blur")
    XCTAssertEqual(replay.automaticDuration?.minimumSeconds,1)
    XCTAssertEqual(replay.automaticDuration?.maximumSeconds,3)
    XCTAssertEqual(replay.referenceFrames,[0,7]);XCTAssertEqual(replay.generatedKeyframes,2)
    let resolved=try replay.replacingFrames(25)
    XCTAssertEqual(resolved.referenceFrames,[0,7]);XCTAssertEqual(resolved.generatedKeyframes,2)
    XCTAssertEqual(resolved.frames,25)
  }
  func ordinary() -> [String:Any] {
    var value=base();value["version"]=14;value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["generated_keyframes"]=2
    for name in ["audio_reference","union_control_guide","ingredients_sheet","msr","dfr","ic_control",
      "guided_sampling","automatic_duration"] { value[name]=NSNull() }
    value["ingredients_sampling"]="deterministic_bf16_v1"
    return value
  }
  func testOrdinaryRequestKeepsTimedImageOrderingAndOldRequestsRejectTimedFields() throws {
    var value=ordinary();value["task"]="i2v"
    let refs:[[String:Any]]=[
      ["role":"keyframe","frame_index":23,"path":"/later.png","strength":0.8,"crf":33],
      ["role":"first","path":"/first.png","strength":1,"crf":0],
      ["role":"keyframe","frame_index":7,"path":"/earlier.png","strength":0.9,"crf":0]]
    value["reference_images"]=refs
    let request=try decode(value)
    XCTAssertEqual(request.referenceFrames,[23,0,7])
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request)).referenceFrames,[23,0,7])
    value["version"]=3
    for key in ["generated_keyframes","audio_reference","union_control_guide","ingredients_sheet","msr","dfr",
      "ic_control","ingredients_sampling","guided_sampling","automatic_duration"] { value.removeValue(forKey:key) }
    XCTAssertThrowsError(try decode(value))
  }
  func testOrdinaryRequestRejectsInvalidCountsBooleanFramesDuplicateAndUnknownFields() throws {
    for invalid:Any in [-1,9,true,NSNull()] {
      var value=ordinary();value["generated_keyframes"]=invalid;XCTAssertThrowsError(try decode(value))
    }
    var value=ordinary();value.removeValue(forKey:"generated_keyframes");XCTAssertThrowsError(try decode(value))
    value=ordinary();value["task"]="i2v"
    var image:[String:Any]=["role":"keyframe","frame_index":7,"path":"/a.png","strength":1,"crf":0]
    value["reference_images"]=[image,image];XCTAssertThrowsError(try decode(value))
    for invalid:Any in [-1,33,true] {
      image["frame_index"]=invalid;value["reference_images"]=[image];XCTAssertThrowsError(try decode(value))
    }
    image["frame_index"]=7;image["ignored"]=1;value["reference_images"]=[image]
    XCTAssertThrowsError(try decode(value))
  }
  func testAuthoredIngredientsHasExplicitVersionAndNeverReinterpretsLegacyRequests() throws {
    var value=base();value["version"]=6;value["task"]="ingredients"
    value["frames"]=121;value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull();value["union_control_guide"]=NSNull()
    value["ingredients_sheet"]=["path":"/sheet.png","source_sha256":String(repeating:"b",count:64),
      "adapter_path":"/ingredients.safetensors","adapter_strength":1.4,"reference_strength":1.0]
    let legacy=try decode(value)
    XCTAssertEqual(legacy.ingredientsSampling,.deterministic)
    let encoded=try JSONSerialization.jsonObject(with:JSONEncoder().encode(legacy)) as! [String:Any]
    XCTAssertNil(encoded["ingredients_sampling"])
    value["version"]=11;value["msr"]=NSNull();value["dfr"]=NSNull();value["ic_control"]=NSNull()
    XCTAssertThrowsError(try decode(value))
    value["ingredients_sampling"]="euler_ancestral_cfg_pp_float32_v1"
    let authored=try decode(value)
    XCTAssertEqual(authored.ingredientsSampling,.ancestralCFGPP)
    XCTAssertEqual(authored.ingredientsSampling.transformerEvaluations,16)
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,
      from:JSONEncoder().encode(authored)).ingredientsSampling,.ancestralCFGPP)
    var sheet=value["ingredients_sheet"] as! [String:Any]
    sheet["reference_strength"]=0.75;value["ingredients_sheet"]=sheet
    XCTAssertThrowsError(try decode(value))
    sheet["reference_strength"]=1.0;value["ingredients_sheet"]=sheet
    value["ingredients_sampling"]="deterministic_bf16_v1";XCTAssertThrowsError(try decode(value))
    value["ingredients_sampling"]="euler_ancestral_cfg_pp_float32_v1"
    value["task"]="t2v";XCTAssertThrowsError(try decode(value))
    value["task"]="ingredients";value["version"]=6;XCTAssertThrowsError(try decode(value))
  }
  func testDFRRequestRequiresDedicatedAdapterAndExactSeamCanvas() throws {
    var value=base();value["version"]=8;value["task"]="dfr"
    value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull()
    value["union_control_guide"]=NSNull();value["ingredients_sheet"]=NSNull()
    value["msr"]=NSNull();value["frames"]=49
    value["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":0.5]
    let request=try decode(value)
    XCTAssertEqual(request.dfr?.adapterStrength,0.5)
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,
      from:JSONEncoder().encode(request)).dfr?.adapterPath,"/pixel-spatial.safetensors")
    let first:[String:Any]=["role":"first","path":"/first.png","strength":1,"crf":33]
    let last:[String:Any]=["role":"last","path":"/last.png","strength":0.8,"crf":0]
    value["reference_images"]=[first,last]
    XCTAssertEqual(try decode(value).referenceImages.map(\.role),["first","last"])
    value["reference_images"]=[first]
    XCTAssertEqual(try decode(value).referenceImages.map(\.role),["first"])
    value["reference_images"]=[last]
    XCTAssertThrowsError(try decode(value))
    value["reference_images"]=[]
    value["frames"]=41;XCTAssertEqual(try decode(value).recipe().high.frames,49)
    value["frames"]=49;value["stage_two_loras"]=[["path":"/ordinary.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(value))
    value["stage_two_loras"]=[];value["dfr"]=NSNull();XCTAssertThrowsError(try decode(value))
    value["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":0.5]
    value["task"]="t2v";XCTAssertThrowsError(try decode(value))
  }
  func testTemporalDFRRequiresVersionNineAndMatchedCheckpoint() throws {
    var value=base();value["version"]=9;value["task"]="dfr"
    value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull()
    value["union_control_guide"]=NSNull();value["ingredients_sheet"]=NSNull()
    value["msr"]=NSNull();value["frames"]=41;value["fps"]=24
    value["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":1,
      "temporal_upscaler_path":"/temporal.safetensors","temporal_rounds":1]
    let request=try decode(value)
    XCTAssertEqual(request.dfr?.temporalRounds,1)
    XCTAssertEqual(try MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:1),81)
    XCTAssertEqual(try decode(JSONSerialization.jsonObject(with:JSONEncoder().encode(request)) as! [String:Any]).dfr?.temporalRounds,1)
    value["version"]=8;XCTAssertThrowsError(try decode(value))
    value["version"]=9;value["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":1]
    XCTAssertThrowsError(try decode(value))
    value["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":1,
      "temporal_upscaler_path":"/temporal.safetensors","temporal_rounds":2]
    value["fps"]=48;XCTAssertThrowsError(try decode(value))
  }
  func testMSRRequestKeepsOrderedOneToFiveReferencesAndRejectsOtherTasks() throws {
    var value=base();value["version"]=7;value["task"]="msr"
    value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull()
    value["union_control_guide"]=NSNull();value["ingredients_sheet"]=NSNull()
    let one:[String:Any]=["path":"/first.png","source_sha256":String(repeating:"a",count:64),
      "role":"subject","priority":"auto","size_policy":"quality","reference_frames":"25",
      "strength":1.0,"attention_strength":0.7]
    value["msr"]=["adapter_path":"/msr.safetensors","adapter_strength":1.0,
      "references":[one]]
    let request=try decode(value)
    XCTAssertEqual(request.msr?.references.map(\.role),["subject"])
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,
      from:JSONEncoder().encode(request)).msr?.references.count,1)
    value["msr"]=["adapter_path":"/msr.safetensors","adapter_strength":1.0,
      "references":Array(repeating:one,count:6)]
    XCTAssertThrowsError(try decode(value))
    value["msr"]=["adapter_path":"/msr.safetensors","adapter_strength":1.0,
      "references":[one]]
    value["stage_one_loras"]=[["path":"/ordinary.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(value))
    value["stage_one_loras"]=[];value["task"]="t2v"
    XCTAssertThrowsError(try decode(value))
  }
  func testIngredientsRequestRequiresFullLengthFrozenSheetAndSingleStage() throws {
    var value=base();value["version"]=6;value["task"]="ingredients"
    value["frames"]=121;value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull()
    value["union_control_guide"]=NSNull()
    value["ingredients_sheet"]=["path":"/sheet.png","source_sha256":String(repeating:"b",count:64),
      "adapter_path":"/ingredients.safetensors","adapter_strength":1.2,"reference_strength":1.0]
    let request=try decode(value)
    XCTAssertEqual(request.ingredientsSheet?.path,"/sheet.png")
    XCTAssertEqual(try decode(JSONSerialization.jsonObject(with:JSONEncoder().encode(request)) as! [String:Any]).task,
      "ingredients")
    value["frames"]=113;XCTAssertThrowsError(try decode(value))
    value["frames"]=121;value["stage_two_loras"]=[["path":"/ordinary.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(value))
    value["stage_two_loras"]=[];value["stage_one_loras"]=[["path":"/ingredients.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(value))
    value["stage_one_loras"]=[];value["task"]="t2v";XCTAssertThrowsError(try decode(value))
  }
  func testUnionControlRequestHasDedicatedGuideAndCleanSecondStage() throws {
    var value=base();value["version"]=5;value["task"]="union_control"
    value["noise_policy"]="mlx_threefry_bf16_v1"
    value["reference_images"]=[];value["audio_reference"]=NSNull()
    value["union_control_guide"]=["path":"/guide.rgb","source_sha256":String(repeating:"a",count:64),
      "adapter_path":"/union.safetensors",
      "adapter_strength":1.0,"reference_strength":1.0]
    let request=try decode(value)
    XCTAssertEqual(request.unionControlGuide?.path,"/guide.rgb")
    XCTAssertEqual(request.stageTwoLoras.count,0)
    XCTAssertEqual(try decode(JSONSerialization.jsonObject(with:JSONEncoder().encode(request)) as! [String:Any]).task,
      "union_control")
    value["task"]="t2v";XCTAssertThrowsError(try decode(value))
    value["task"]="union_control";value["audio_reference"]=["path":"/audio.wav",
      "source_start_seconds":0,"source_duration_seconds":1]
    XCTAssertThrowsError(try decode(value))
    value["audio_reference"]=NSNull()
    value["stage_one_loras"]=[["path":"/union.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(value))
    value["stage_one_loras"]=[]
    value["union_control_guide"]=["path":"/guide.rgb","source_sha256":String(repeating:"a",count:64),
      "adapter_path":"/union.safetensors",
      "adapter_strength":0,"reference_strength":1]
    XCTAssertThrowsError(try decode(value))
    value["union_control_guide"]=["path":"/guide.rgb","source_sha256":"not-a-digest",
      "adapter_path":"/union.safetensors",
      "adapter_strength":1,"reference_strength":1]
    XCTAssertThrowsError(try decode(value))
  }
  func testVersionFourAllowsAudioSourceWithOptionalFirstImageOnly() throws {
    var value=base();value["version"]=4;value["task"]="a2v"
    value["reference_images"]=[];value["noise_policy"]="mlx_threefry_bf16_v1"
    let source:[String:Any]=["path":"/audio/source.wav","source_start_seconds":1.25,
      "source_duration_seconds":2.0]
    value["audio_reference"]=source
    let request=try decode(value)
    XCTAssertEqual(request.audioReference?.sourceStartSeconds,1.25)
    XCTAssertEqual(request.audioReference?.sourceDurationSeconds,2.0)
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request)).task,"a2v")
    value["audio_reference"]=NSNull();XCTAssertThrowsError(try decode(value))
    value["audio_reference"]=source;value["task"]="t2v";XCTAssertThrowsError(try decode(value))
    value["task"]="a2v";value["reference_images"]=[["role":"first","path":"/first.png","strength":1,"crf":33]]
    XCTAssertEqual(try decode(value).referenceImages.map(\.role),["first"])
    value["reference_images"]=[["role":"last","path":"/last.png","strength":1,"crf":33]]
    XCTAssertThrowsError(try decode(value))
    value["reference_images"]=[];var bad=source;bad["source_start_seconds"] = -1
    value["audio_reference"]=bad;XCTAssertThrowsError(try decode(value))
  }
  func base() -> [String:Any] {
    ["version":1,"engine":"ltx25","task":"t2v","gemma_root":"/models/gemma",
     "transformer_root":"/models/transformer","connector_checkpoint":"/models/connector",
     "video_checkpoint":"/models/video","audio_checkpoint":"/models/audio",
     "spatial_upscaler_checkpoint":"/models/upscaler","prompt":"A red fox.",
     "width":512,"height":256,"frames":33,"fps":24,"seed":42,
     "output_directory":"/output/run","stage_one_loras":[],"stage_two_loras":[]]
  }
  func decode(_ value:[String:Any]) throws -> MLXDistilledRequest {
    try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:value))
  }
  func testVersionThreeRequiresKnownExplicitNoisePolicy() throws {
    var value=base();value["version"]=3;value["reference_images"]=[]
    XCTAssertThrowsError(try decode(value))
    value["noise_policy"]="mlx_threefry_bf16_v1"
    let request=try decode(value)
    XCTAssertEqual(request.noisePolicy,.releasedMLX)
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request)).noisePolicy,.releasedMLX)
    value["noise_policy"]="unknown";XCTAssertThrowsError(try decode(value))
    XCTAssertEqual(try decode(base()).noisePolicy,.native)
    value=base();value["noise_policy"]="native_box_muller_v1";XCTAssertThrowsError(try decode(value))
  }
  func testStrictVersionedRequestRetainsPerStageAdapterOrder() throws {
    var value=base()
    value["stage_one_loras"]=[["path":"/models/a","strength":0.8],["path":"/models/b","strength":0.4,"enabled":false]]
    let request=try decode(value)
    XCTAssertEqual(request.stageOneLoras.map(\.path),["/models/a","/models/b"])
    XCTAssertEqual(request.stageTwoLoras.count,0)
    XCTAssertEqual(try request.recipe().first.steps.count,8)
    let roundtrip=try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request))
    XCTAssertEqual(roundtrip.stageOneLoras.map(\.enabled),[true,false])
  }
  func testUnknownControlsTasksInvalidPathsAndGeometryAreRejected() throws {
    for (key,value):(String,Any) in [("version",2),("engine","h3"),("task","fflf"),
      ("width",1343),("frames",32),("fps",0),("gemma_root","relative"),
      ("seed",-1),("guidance",7),("reference_images",["/a"])] {
      var invalid=base(); invalid[key]=value; XCTAssertThrowsError(try decode(invalid),key)
    }
    var missing=base(); missing.removeValue(forKey:"stage_two_loras")
    XCTAssertThrowsError(try decode(missing))
  }
  func testRequestFileAdmissionRejectsFIFOAndOversizedFiles() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let data=try JSONSerialization.data(withJSONObject:base())
    let regular=root.appendingPathComponent("request.json")
    try data.write(to:regular)
    XCTAssertEqual(try MLXDistilledRequest.load(regular).width,512)
    XCTAssertThrowsError(try MLXDistilledRequest.load(root))
    let large=root.appendingPathComponent("large.json")
    try Data(repeating:32,count:1024*1024+1).write(to:large)
    XCTAssertThrowsError(try MLXDistilledRequest.load(large))
    let fifo=root.appendingPathComponent("request.fifo")
    XCTAssertEqual(mkfifo(fifo.path,0o600),0)
    // Even with a writer and valid JSON available, a FIFO is not an admitted
    // request. The descriptor check must reject it without attempting a read.
    let descriptor=Darwin.open(fifo.path,O_RDWR | O_NONBLOCK)
    XCTAssertGreaterThanOrEqual(descriptor,0)
    defer { Darwin.close(descriptor) }
    let written=data.withUnsafeBytes { Darwin.write(descriptor,$0.baseAddress,data.count) }
    XCTAssertEqual(written,data.count)
    XCTAssertThrowsError(try MLXDistilledRequest.load(fifo))
  }

  func testVersionTwoRequiresExplicitOrderedEndpointReferences() throws {
    var value=base();value["version"]=2;value["task"]="fflf"
    let first:[String:Any]=["role":"first","path":"/images/first.png","strength":1,"crf":33]
    let last:[String:Any]=["role":"last","path":"/images/last.png","strength":0.8,"crf":0]
    value["reference_images"]=[first,last]
    let request=try decode(value)
    XCTAssertEqual(request.referenceImages.map(\.role),["first","last"])
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request)).referenceImages.count,2)
    for invalid:[Any] in [[last,first],[first,first],[first],[]] {
      value["reference_images"]=invalid;XCTAssertThrowsError(try decode(value))
    }
    value["task"]="i2v";value["reference_images"]=[first];XCTAssertNoThrow(try decode(value))
    var bad=first;bad["crf"]=52;value["reference_images"]=[bad];XCTAssertThrowsError(try decode(value))
    bad=first;bad["strength"] = -0.1;value["reference_images"]=[bad];XCTAssertThrowsError(try decode(value))
  }

}
