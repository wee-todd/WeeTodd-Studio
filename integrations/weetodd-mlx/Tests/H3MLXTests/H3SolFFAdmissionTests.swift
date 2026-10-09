import Foundation
import XCTest
@testable import H3MLX

final class H3SolFFAdmissionTests: XCTestCase {
  private func recipe() -> [String: Any] {
    ["format":"weetodd-headless-v2", "engine":"h3", "prompt":"A boxer punches a bag.",
     "components":["task":"fl2va", "transformer":"/tmp/fl.safetensors", "text_encoder":"/tmp/qwen", "vision_encoder":"/tmp/vision", "tokenizer":"/tmp/tokenizer", "video_vae":"/tmp/video", "audio_vae":"/tmp/audio"],
     "loras":["version":1, "adapters":[["path":"/tmp/fl-turbo.safetensors", "strength":1.0, "profile":"turbo", "qkv_layout":"contiguous_qkv"]]],
     "config":["width":64, "height":64, "duration_seconds":5, "steps":5, "seed":123, "sampling_method":"euler", "drop_adaln":true, "attention_policy":"sol_experimental"],
     "conditioning":["version":1, "task":"fflf", "audio_policy":"generated", "inputs":[["id":"first", "kind":"image", "role":"first", "frame_index":0, "path":"/tmp/first.png", "sha256":String(repeating:"a",count:64), "strength":1.0]]]]
  }
  func testSolFFRecipeAdmitsBeforeMediaAndKeepsFirstFrameOnly() throws {
    var root=recipe()
    var config=root["config"] as! [String:Any];config["transformer_weight_cache_gb"]=48;root["config"]=config
    XCTAssertEqual(try H3AttentionPolicy.admitRecipe(root,canvasAdmission:.ordinary),.solExperimental)
    var calls=0
    let request=try H3StudioRecipe.compileFL2VA(data:JSONSerialization.data(withJSONObject:root)) { _,_,width,height in
      calls += 1
      return H3StillReference(rgb8:Data(count:width*height*3),width:width,height:height)
    }
    XCTAssertEqual(calls,1)
    XCTAssertEqual(request.anchors.count,1)
    XCTAssertEqual(request.base.requestedSteps,5)
    XCTAssertEqual(request.base.transformerWeightCacheGB,48)
  }
  func testFLStateSupportsItsPartitionWithoutAdmittingReferenceCurveRank() throws {
    try H3SolTaskPolicy.validateTask(task:"fl2va",contextFrames:0,isRefinement:false,ordinaryCanvas:true,mlxBackend:true,hasFast:false,hasVDN:false,hasFun:false,hasMotion:false)
    try H3SolTaskPolicy.validateState(referenceLayout:false,blockCount:50,weightDecoded:true,hasNativeWorker:false,maximumRows:40000,hasFast:false,hasCurveRank:true,hasVDN:false,hasFun:false)
    XCTAssertThrowsError(try H3SolTaskPolicy.validateState(referenceLayout:true,blockCount:50,weightDecoded:true,hasNativeWorker:false,maximumRows:40000,hasFast:false,hasCurveRank:true,hasVDN:false,hasFun:false))
  }
  func testAnchorsTextAndAudioAreOutsideApproximationRange() throws {
    let g=try H3SolGeometry(rows:38937,heads:56,approximationRange:2631..<38937)
    XCTAssertTrue(g.requiresExact(queryBlock:0,keyBlock:100))
    XCTAssertTrue(g.requiresExact(queryBlock:100,keyBlock:40))
    XCTAssertFalse(g.requiresExact(queryBlock:42,keyBlock:100))
  }
}
