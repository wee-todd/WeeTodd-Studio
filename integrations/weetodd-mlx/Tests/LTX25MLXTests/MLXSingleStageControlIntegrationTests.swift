import XCTest
import Foundation
import LTX25Engine
import AdapterRuntime
@testable import LTX25MLX

final class MLXSingleStageControlIntegrationTests:XCTestCase {
  private func object(task:String="ic_control",method:String="euler_ancestral_cfg_pp") -> [String:Any] {
    var v=MLXDistilledRequestTests().ordinary()
    v["version"]=15;v["width"]=512;v["height"]=256;v["frames"]=49;v["fps"]=24
    v["task"]=task;v["spatial_upscaler_checkpoint"]=""
    v["single_stage_sampling"]=["method":method,"negative_schedule":method == "euler_ancestral_cfg_pp" ? "balanced":"full","negative_prompt":""]
    v["reference_images"]=[["role":"keyframe","frame_index":7,"path":"/timed.png","strength":0.75,"crf":33],
      ["role":"first","path":"/first.png","strength":1,"crf":33]]
    v["stage_one_loras"]=[["path":"/style.safetensors","strength":0.8]]
    if task == "ic_control" { v["ic_control"]=icObject() }
    if task == "union_control" { v["union_control_guide"]=unionObject() }
    return v
  }
  private func icObject(cross:Bool=false) -> [String:Any] {
    let family=cross ? "crossview_warp":"motion_track",roles=cross ? ["warp","source"]:["control"]
    return ["family":family,"adapters":[["path":"/control.safetensors","family":family,"strength":1.1]],
      "guides":roles.enumerated().map { ["path":"/guide-\($0.offset).rgb","source_sha256":String(repeating:"a",count:64),"role":$0.element,"strength":0.5] as [String:Any] },
      "publication_audio":cross ? ["path":"/source.wav","source_sha256":String(repeating:"b",count:64),"source_start_seconds":0,"source_duration_seconds":3] as Any:NSNull()]
  }
  private func unionObject() -> [String:Any] {
    ["path":"/union.rgb","source_sha256":String(repeating:"c",count:64),"adapter_path":"/union.safetensors","adapter_strength":1.2,"reference_strength":0.25]
  }
  private func decode(_ v:[String:Any]) throws -> MLXDistilledRequest {
    try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:v))
  }
  func testVersionFifteenCombinesBothGuideFamiliesWithOrderedAnchorsSlotsAndGenericAdapter() throws {
    for task in ["ic_control","union_control"] {
      let r=try decode(object(task:task)),layout=try XCTUnwrap(r.singleStageControlLayout())
      XCTAssertEqual(r.version,15);XCTAssertEqual(r.referenceFrames,[7,0]);XCTAssertEqual(layout.generatedFrames,[16,32])
      XCTAssertEqual(layout.guideTokens,224);XCTAssertEqual(layout.videoTokens,1504)
      XCTAssertEqual(layout.slotTokens,256);XCTAssertTrue(layout.requiresPerTokenVideo)
      XCTAssertEqual(layout.videoAttentionGroups,[])
      XCTAssertEqual(try r.recipe().low.width,512);XCTAssertEqual(try r.recipe().low.height,256)
      let replay=try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(r))
      XCTAssertEqual(replay.referenceFrames,[7,0]);XCTAssertEqual(replay.icControl?.family,r.icControl?.family)
      XCTAssertEqual(replay.singleStageSampling?.transformerEvaluations,12)
    }
  }
  func testCrossViewPublicationAudioDoesNotBecomeFrozenDiffusionAudio() throws {
    var v=object();v["ic_control"]=icObject(cross:true)
    let r=try decode(v)
    XCTAssertNil(r.audioReference);XCTAssertNotNil(r.icControl?.publicationAudio)
    XCTAssertEqual(r.singleStageSampling?.method,.cfgpp)
    XCTAssertEqual(try r.singleStageControlLayout()?.guideTokens,1792)
  }
  func testMutuallyExclusiveSpecializationsDuplicateAdaptersAndFrozenA2VReject() throws {
    var both=object();both["union_control_guide"]=unionObject();XCTAssertThrowsError(try decode(both))
    var duplicate=object();duplicate["stage_one_loras"]=[["path":"/control.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(duplicate))
    var stageTwo=object();stageTwo["stage_two_loras"]=[["path":"/other.safetensors","strength":1]]
    XCTAssertThrowsError(try decode(stageTwo))
    var source=object(task:"a2v");source["audio_reference"]=["path":"/source.wav","source_start_seconds":0,"source_duration_seconds":3]
    XCTAssertThrowsError(try decode(source))
    source["single_stage_sampling"]=["method":"euler","negative_schedule":"full","negative_prompt":""]
    XCTAssertNoThrow(try decode(source))
    for key in ["ingredients_sheet","msr","dfr"] {
      var invalid=object();invalid[key]=[:] as [String:String];XCTAssertThrowsError(try decode(invalid))
    }
  }
  func testDedicatedAdapterResolutionPreservesOrderAndStrengthExactlyOnceWithoutLoadingHeaders() throws {
    let r=try decode(object())
    let generic=[LoRAAdapter(path:"/style.safetensors",strength:0.8),LoRAAdapter(path:"/second.safetensors",strength:-0.2)]
    let stack=try MLXSingleStageSamplingRunner.resolveAdapters(generic:generic,icControl:r.icControl,unionControlGuide:nil)
    XCTAssertEqual(stack.map(\.path),["/style.safetensors","/second.safetensors","/control.safetensors"])
    XCTAssertEqual(stack.map(\.strength),[0.8,-0.2,1.1])
    XCTAssertThrowsError(try MLXSingleStageSamplingRunner.resolveAdapters(generic:[LoRAAdapter(path:"/control.safetensors",strength:1)],icControl:r.icControl,unionControlGuide:nil))
  }
  func testMemoryAdmissionAndCFGPPReserveIncludeAllCombinedRows() throws {
    let r=try decode(decoderRequestBase(object())),g=try r.recipe().high,layout=try XCTUnwrap(r.singleStageControlLayout())
    let c=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:g.audioFrames,textTokens:1024)
    let expected=try MLXAVBlock.estimatedActivationBytes(configuration:c,perTokenVideo:true)+layout.cfgppReserveBytes(audioTokens:g.audioFrames)
    let gib=1024*1024*1024
    let plan=try MLXStudioMemoryPlan(request:r,physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    XCTAssertEqual(plan.transformerActivationBytes,expected)
    XCTAssertNoThrow(try MLXMediaPipeline.admit(r,videoActivationBytes:plan.videoActivationBytes,transformerActivationBytes:expected,videoBackend:.mlx,audioBackend:.mlx))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(r,videoActivationBytes:plan.videoActivationBytes,transformerActivationBytes:expected-1,videoBackend:.mlx,audioBackend:.mlx))
    var plain=object(task:"i2v");plain["ic_control"]=NSNull();let p=try decode(decoderRequestBase(plain))
    let plainPlan=try MLXStudioMemoryPlan(request:p,physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    XCTAssertGreaterThan(plan.transformerActivationBytes,plainPlan.transformerActivationBytes)
  }
  func testVersionTenControlAndVersionFourteenOrdinaryEncodedFieldsRemainUnchanged() throws {
    let ordinary=MLXDistilledRequestTests().ordinary(),r=try decode(ordinary)
    XCTAssertNil(try r.singleStageControlLayout())
    let encoder=JSONEncoder();encoder.outputFormatting = .sortedKeys
    let first=try encoder.encode(r),second=try encoder.encode(JSONDecoder().decode(MLXDistilledRequest.self,from:first))
    XCTAssertEqual(first,second)
    var legacy=object();legacy["version"]=10;legacy["reference_images"]=[];legacy["stage_one_loras"]=[]
    for key in ["ingredients_sampling","guided_sampling","automatic_duration","generated_keyframes","single_stage_sampling"] { legacy.removeValue(forKey:key) }
    legacy["spatial_upscaler_checkpoint"]="/upscaler.safetensors"
    let control=try decode(legacy);XCTAssertNil(try control.singleStageControlLayout())
    XCTAssertEqual(control.icControl?.family,"motion_track");XCTAssertEqual(control.referenceFrames,[])
    let encodedControl=try encoder.encode(control)
    let value=try XCTUnwrap(JSONSerialization.jsonObject(with:encodedControl) as? [String:Any])
    let encodedIC=try XCTUnwrap(value["ic_control"] as? [String:Any])
    XCTAssertTrue(encodedIC["publication_audio"] is NSNull)
    let replayControl=try JSONDecoder().decode(MLXDistilledRequest.self,from:encodedControl)
    XCTAssertEqual(replayControl.version,10)
    XCTAssertEqual(try encoder.encode(replayControl),encodedControl)
    legacy["reference_images"]=[["role":"first","path":"/first.png","strength":1,"crf":33]]
    XCTAssertThrowsError(try decode(legacy))
  }
}
