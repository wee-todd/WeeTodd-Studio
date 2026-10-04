import Foundation
import XCTest
@testable import H3MLX

final class H3JointRefinementRecipeTests: XCTestCase {
  private func recipe() -> [String: Any] {
    ["format":"weetodd-headless-v2","engine":"h3","prompt":"A boxer holds guard.",
      "components":["task":"t2va","transformer":"/base","text_encoder":"/text","tokenizer":"/tokenizer","video_vae":"/video","audio_vae":"/audio"],
      "config":["width":64,"height":64,"duration_seconds":2.5,"steps":20,"seed":42],
      "conditioning":["version":1,"task":"t2v","inputs":[],"audio_policy":"generated"]]
  }
  private func fields(mode: String = "spatial") -> [String: Any] {
    var result: [String:Any] = ["version":1,"mode":mode,"source_manifest":"/source/manifest.json",
      "source_manifest_sha256":String(repeating:"a",count:64),"strength":0.35,"preserve_audio":true]
    if mode == "spatial" { result["resize_method"]="bilinear" }
    return result
  }
  func testExplicitPublicationAndRefinementAreStrippedBeforeOrdinaryCompilation() throws {
    var root=recipe();root["refinement"]=fields();root["joint_latents"]=["version":1,"save_full":true]
    let prepared=try XCTUnwrap(H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    XCTAssertTrue(prepared.saveFullLatents);XCTAssertEqual(prepared.controls?.strength,0.35)
    XCTAssertEqual(prepared.resizeMethod,.bilinear)
    let request=try H3StudioRecipe.compile(data:prepared.ordinaryRecipe)
    XCTAssertEqual(request.requestedSteps,20);XCTAssertEqual(request.geometry.frames,73)
    XCTAssertEqual(request.geometry.width,64);XCTAssertEqual(request.seed,42)
  }
  func testOldRecipeHasNoInitializedBehaviorAndUnsupportedCombinationRejectsBeforeMedia() throws {
    XCTAssertNil(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:recipe())))
    var root=recipe();root["refinement"]=fields();root["continuation"]=["version":2]
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    root=recipe();var controls=fields();controls["strength"]=true;root["refinement"]=controls
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    controls=fields();controls["resize_method"]="invented";root["refinement"]=controls
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    controls=fields();controls["evaluations"]=4;root["refinement"]=controls
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
  }
  func testMotionInitializedPrimitiveRetainsOwnedPublicScopeAndNoiseSemantics() throws {
    var root=recipe();var controls=fields(mode:"motion");controls["start_video_sigma"]=0.35;controls["evaluations"]=3;root["refinement"]=controls
    let prepared=try XCTUnwrap(H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    XCTAssertEqual(prepared.controls?.evaluations,3);XCTAssertEqual(prepared.controls?.preserveAudio,true)
    var config=root["config"] as! [String:Any];config["steps"]=5;root["config"]=config
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    root=recipe();controls["start_video_sigma"]=0.2;root["refinement"]=controls
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
  }
}
