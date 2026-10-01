import XCTest
import Foundation
import Darwin
import LTX25MLX

final class MLXDistilledRequestTests:XCTestCase {
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
