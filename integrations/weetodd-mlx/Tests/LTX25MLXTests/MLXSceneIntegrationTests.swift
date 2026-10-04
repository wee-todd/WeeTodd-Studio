import Foundation
import XCTest
import LTX25Engine
@testable import LTX25MLX

final class MLXSceneIntegrationTests:XCTestCase {
  private func fixture(images:[Int]=[]) -> [String:Any] {
    ["format":"weetodd-headless-v2","engine":"ltx25","prompt":"Two shots.",
      "components":["transformer_path":"/models/transformer","text_encoder_path":"/models/text",
        "video_vae_path":"/models/video","audio_vae_path":"/models/audio","spatial_upscaler_path":"/models/upscale"],
      "config":["pipeline_mode":"distilled","width":512,"height":256,"duration_seconds":5,
        "frame_rate":24,"seed":11,"stage1_steps":8,"stage2_steps":3],
      "conditioning":["version":1,"task":images.isEmpty ? "t2v":"fflf","audio_policy":"generated",
        "inputs":images.map { ["id":"anchor-\($0)","kind":"image","role":"keyframe","path":"/image-\($0).png",
          "frame_index":$0,"strength":0.75] as [String:Any] }],
      "scene":["version":2,"overlap_frames":25,"boundary_image_policy":"balanced","soundscape":"","music":"",
        "segments":[["clip_id":"first","prompt":"A warrior waits.","duration_seconds":2,"seed":11],
          ["clip_id":"second","prompt":"He turns.","duration_seconds":3,"seed":12]]]]
  }
  private func compile(_ object:[String:Any]) throws -> MLXStudioSceneRecipe.Compiled {
    try MLXStudioSceneRecipe.compile(data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output")
  }
  func testBalancedAndStrictRouteGlobalImagesWithoutWideningOrdinaryDecoder() throws {
    var object=fixture(images:[119,49,48,0,47])
    let balanced=try compile(object)
    XCTAssertEqual(balanced.version,2)
    XCTAssertEqual(balanced.requests.map(\.task),["t2v","t2v"])
    XCTAssertTrue(balanced.requests.allSatisfy { $0.referenceImages.isEmpty && !$0.usesOrdinaryKeyframes })
    XCTAssertEqual(try balanced.references(in:0).map { $0.frameIndex! },[0,47,48])
    XCTAssertEqual(try balanced.references(in:1).map { $0.frameIndex! },[25,95])
    XCTAssertEqual(try balanced.references(in:1).map(\.path),["/image-49.png","/image-119.png"])
    XCTAssertEqual(balanced.strictBoundaries,[])
    var scene=object["scene"] as! [String:Any];scene["boundary_image_policy"]="strict";object["scene"]=scene
    let strict=try compile(object)
    XCTAssertEqual(try strict.references(in:1).map { $0.frameIndex! },[23,24,25,95])
    XCTAssertEqual(strict.strictBoundaries,[1])
    XCTAssertThrowsError(try strict.references(in:2))
  }
  func testThirtyTwoSceneReferencesAreSeparateFromEightImageOrdinaryContract() throws {
    let compiled=try compile(fixture(images:Array(0..<32)))
    XCTAssertEqual(try compiled.references(in:0).count,32)
    XCTAssertEqual(compiled.requests[0].referenceImages.count,0)
    XCTAssertEqual(compiled.imageRouting?.retainedConditioningBytes,32*160*131*4)
    XCTAssertThrowsError(try compile(fixture(images:Array(0..<33))))
    var ordinary=fixture(images:Array(0..<9));ordinary.removeValue(forKey:"scene")
    XCTAssertThrowsError(try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:ordinary),outputDirectory:"/output"))
  }
  func testAudioAndImagesPreserveConsecutiveOverlappingIntervalsAndExactBudget() throws {
    var object=fixture(images:[7,119]),condition=object["conditioning"] as! [String:Any]
    condition["task"]="a2v";condition["audio_policy"]="source"
    var inputs=condition["inputs"] as! [[String:Any]]
    inputs.append(["id":"audio","kind":"audio","role":"audio_driver","path":"/song.wav","strength":1,
      "source_start_seconds":5.0,"source_duration_seconds":5.0])
    condition["inputs"]=inputs;object["conditioning"]=condition
    let compiled=try compile(object)
    XCTAssertEqual(compiled.requests.map(\.task),["a2v","a2v"])
    XCTAssertEqual(compiled.requests.map { $0.audioReference!.sourceStartSeconds },[5,6])
    XCTAssertEqual(compiled.requests[1].audioReference?.sourceDurationSeconds,4)
    XCTAssertEqual(try compiled.references(in:1).first?.frameIndex,95)
    let plain=try compile(fixture())
    XCTAssertGreaterThan(try MLXSceneSampler.estimatedTransformerActivationBytes(compiled),
      try MLXSceneSampler.estimatedTransformerActivationBytes(plain))
    let request=compiled.requests[1],recipe=try request.recipe()
    let history=try MLXExtensionGuideLayout(geometry:recipe.high,contextFrames:25,
      videoGuideLatentFrames:3,audioGuideTokens:compiled.plan.joinAudioTokens[0])
    let layout=try MLXSceneKeyframeLayout(geometry:recipe.high,
      anchors:compiled.imageRouting!.windows[1].map(\.layoutAnchor),extensionGuide:history)
    let config=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:layout.audioTokens,textTokens:1024)
    let expected=try MLXAVBlock.estimatedActivationBytes(configuration:config,perTokenVideo:true,perTokenAudio:true)
      + compiled.imageRouting!.retainedConditioningBytes
    XCTAssertEqual(try MLXSceneSampler.estimatedTransformerActivationBytes(compiled),expected)
  }
  func testVersionTwoRejectsUnfrozenFramesFieldsAndUnsupportedMixesBeforeWeights() throws {
    for frame in [true as Any,"last" as Any,120 as Any,-1 as Any,7.25 as Any] {
      var object=fixture(images:[7]),condition=object["conditioning"] as! [String:Any]
      var inputs=condition["inputs"] as! [[String:Any]];inputs[0]["frame_index"]=frame
      condition["inputs"]=inputs;object["conditioning"]=condition
      XCTAssertThrowsError(try compile(object))
    }
    for (key,value) in [("pipeline_mode","guided" as Any),("generated_keyframes",1 as Any),("dfr_enabled",true as Any)] {
      var object=fixture(images:[7]),config=object["config"] as! [String:Any];config[key]=value;object["config"]=config
      XCTAssertThrowsError(try compile(object))
    }
    var object=fixture(images:[7,7]);XCTAssertThrowsError(try compile(object))
    object=fixture(images:[7]);var condition=object["conditioning"] as! [String:Any]
    var inputs=condition["inputs"] as! [[String:Any]];inputs[0]["unexpected"]=true
    condition["inputs"]=inputs;object["conditioning"]=condition
    XCTAssertThrowsError(try compile(object))
    object=fixture(images:[7]);var scene=object["scene"] as! [String:Any]
    var segments=scene["segments"] as! [[String:Any]];segments[1]["image_input"]=inputs[0]
    scene["segments"]=segments;object["scene"]=scene
    XCTAssertThrowsError(try compile(object))
    object=fixture(images:[7]);scene=object["scene"] as! [String:Any];scene["version"]=true;object["scene"]=scene
    XCTAssertThrowsError(try compile(object))
  }
  func testBothSceneVersionsRejectOtherwiseValidSingleStageRequestsBeforeWeights() throws {
    for version in [1,2] {
      for (method,eta) in [("euler",0),("euler_ancestral",1),("euler_ancestral_cfg_pp",1)] {
        var object=fixture(),config=object["config"] as! [String:Any]
        config["ic_lora_single_stage"]=true;config["stage2_steps"]=0
        config["stage1_sampler"]=method;config["stage1_eta"]=eta
        object["config"]=config
        var scene=object["scene"] as! [String:Any];scene["version"]=version;object["scene"]=scene
        var ordinary=object;ordinary.removeValue(forKey:"scene")
        let admitted=try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:ordinary),outputDirectory:"/output")
        XCTAssertNotNil(admitted.singleStageSampling,"The negative fixture must be valid outside a scene.")
        XCTAssertThrowsError(try compile(object)) { error in
          XCTAssertTrue(String(describing:error).contains("two-stage"),"Unexpected rejection: \(error)")
        }
      }
    }
  }
  func testInstalledSceneHistorySourceProvidersAdmitThirtyTwoImagesWithoutMarkerOrPayload() throws {
    let env=ProcessInfo.processInfo.environment
    guard let transformer=env["WEETODD_LTX_SCENE_TRANSFORMER"],let upscaler=env["WEETODD_LTX_SCENE_UPSCALER"],
      let statistics=env["WEETODD_LTX_SCENE_STATISTICS"] else {
      throw XCTSkip("Installed scene transformer/upscaler header qualification is explicit.")
    }
    let recipe=try DistilledTwoStageRecipe(width:512,height:256,frames:97,fps:24,seed:12)
    let runner=try MLXDistilledSamplingRunner(recipe:recipe,transformerRoot:URL(fileURLWithPath:transformer),
      upscalerCheckpoint:URL(fileURLWithPath:upscaler),statisticsCheckpoint:URL(fileURLWithPath:statistics),
      sceneAnchors:(1...32).map { .init(frame:$0,strength:0.75) },
      extensionContextFrames:25,extensionVideoGuideLatentFrames:3,extensionAudioGuideTokens:25,
      noisePolicy:.releasedMLX,maximumActivationBytes:32*1024*1024*1024)
    XCTAssertEqual(runner.admittedKeyframeMarkers,[false,false])
  }
}
