import Foundation
import XCTest
@testable import H3MLX

final class H3TwoStageRecipeTests: XCTestCase {
  private func recipe() -> [String:Any] {
    ["format":"weetodd-headless-v2","engine":"h3","prompt":"A robot walks.",
      "components":["task":"t2va","transformer":"/base","text_encoder":"/text","tokenizer":"/tokenizer","video_vae":"/video","audio_vae":"/audio"],
      "config":["width":1536,"height":864,"duration_seconds":3.0,"steps":21,"seed":42],
      "conditioning":["version":1,"task":"t2v","inputs":[],"audio_policy":"generated"]]
  }
  private var controls:[String:Any] {["version":1,"draft_scale":2.0,"strength":0.35,"evaluations":3,"resize_method":"bilinear","refinement_adapter_policy":"standard"]}
  func testWrapperPreservesExactFinalAndStripsOnlyOwnedFields() throws {
    var root=recipe();root["two_stage"]=controls;root["joint_latents"]=["version":1,"save_full":true]
    let prepared=try XCTUnwrap(H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    XCTAssertEqual(prepared.plan.base.height,448);XCTAssertEqual(prepared.plan.target.height,864)
    XCTAssertTrue(prepared.saveFullLatents)
    let ordinary=try JSONSerialization.jsonObject(with:prepared.ordinaryRecipe) as! [String:Any]
    XCTAssertNil(ordinary["two_stage"]);XCTAssertNil(ordinary["joint_latents"])
    let request=try H3StudioRecipe.compile(data:prepared.ordinaryRecipe,canvasAdmission:.spatialRefinement)
    XCTAssertEqual(request.geometry.width,1536);XCTAssertEqual(request.requestedSteps,21)
    XCTAssertNil(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:recipe())))
  }
  func testMalformedControlsAndIncompatibleModesFailClosed() throws {
    for (key,value) in [("version",true as Any),("evaluations",true),("strength",true),("draft_scale",0),("evaluations",65),("unknown",1),("refinement_adapter_policy","all")] {
      var root=recipe(),fields=controls;fields[key]=value;root["two_stage"]=fields
      XCTAssertThrowsError(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)),key)
    }
    for key in ["refinement","continuation","motion_fidelity","fasth3","vdn"] {
      var root=recipe();root["two_stage"]=controls;root[key]=["version":1]
      XCTAssertThrowsError(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)),key)
    }
    var root=recipe(),fields=controls;fields["learned_upscaler_path"]="/missing.safetensors";root["two_stage"]=fields
    XCTAssertThrowsError(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
  }
}
