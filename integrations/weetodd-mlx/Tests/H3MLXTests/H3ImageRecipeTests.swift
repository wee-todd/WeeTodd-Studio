import Foundation
import XCTest
@testable import H3MLX

final class H3ImageRecipeTests: XCTestCase {
  func recipe(task: String = "t2va", mode: String = "quality_five_frames") -> [String:Any] {
    ["format":"weetodd-headless-v2", "engine":"h3", "prompt":"A copper robot.",
      "components":["task":task, "transformer":"/tmp/selected.safetensors", "text_encoder":"/tmp/qwen",
        "tokenizer":"/tmp/tokenizer.json", "video_vae":"/tmp/original.safetensors", "audio_vae":"/tmp/audio.safetensors"],
      "conditioning":["version":1, "task":"t2v", "inputs":[], "audio_policy":"generated"],
      "config":["width":768, "height":768, "duration_seconds":5.0, "steps":21, "seed":42],
      "still_image":["version":1, "mode":mode, "frame_selection":"middle"]]
  }
  func compile(_ root: [String:Any]) throws -> H3ImageRecipe.Prepared {
    try H3ImageRecipe.compile(data: JSONSerialization.data(withJSONObject: root)) { _,_,_ in
      .init(rgb8: Data(count:64*64*3), width:64, height:64)
    }
  }
  func testDirect1080StillRequestsAdmitTextAndReferenceWithoutRefinement() throws {
    for mode in ["quality_five_frames", "single_latent"] {
      for reference in [false, true] {
        var root = recipe(task: reference ? "ref2va" : "t2va", mode: mode)
        var config = root["config"] as! [String:Any]
        config["width"] = 1920; config["height"] = 1088
        root["config"] = config
        if mode == "single_latent" {
          root["still_image"] = ["version":1,"mode":mode,"frame_selection":"middle","image_vae":"/tmp/image-only.safetensors"]
        }
        if reference {
          root["conditioning"] = ["version":1,"task":"ref2va","audio_policy":"generated","inputs":[
            ["id":"one","kind":"image","role":"reference","path":"/tmp/ref.png","strength":1.0,"sha256":String(repeating:"a",count:64)]]]
        }
        let prepared = try compile(root)
        XCTAssertEqual(prepared.geometry.width, 1920)
        XCTAssertEqual(prepared.geometry.height, 1088)
        XCTAssertEqual(prepared.geometry.canvasAdmission, .ordinary)
        XCTAssertLessThan(prepared.geometry.videoRows, 40_000)
      }
    }
  }
  func testTextOnlyReferenceCheckpointKeepsItsIdentityAndShortGrid() throws {
    let prepared = try compile(recipe(task:"ref2va"))
    let text = try XCTUnwrap(prepared.text)
    XCTAssertNil(prepared.reference)
    XCTAssertEqual(text.transformer.path, "/tmp/selected.safetensors")
    XCTAssertEqual(text.geometry.frames, 5)
    XCTAssertEqual(text.requestedSteps, 21)
    XCTAssertEqual(prepared.sourceTask, "ref2va")
    XCTAssertEqual(prepared.decoder.path, "/tmp/original.safetensors")
  }
  func testSingleImageDecoderNeverReplacesReferenceEncoder() throws {
    var root=recipe(task:"ref2va",mode:"single_latent")
    root["still_image"]=["version":1,"mode":"single_latent","frame_selection":"middle","image_vae":"/tmp/image-only.safetensors"]
    root["conditioning"]=["version":1,"task":"ref2va","audio_policy":"generated","inputs":[
      ["id":"one","kind":"image","role":"reference","path":"/tmp/ref.png","strength":1.0,"sha256":String(repeating:"a",count:64)]]]
    let prepared=try compile(root)
    let ref=try XCTUnwrap(prepared.reference)
    XCTAssertEqual(ref.geometry.frames,1)
    XCTAssertEqual(ref.videoVAE.path,"/tmp/original.safetensors")
    XCTAssertEqual(prepared.decoder.path,"/tmp/image-only.safetensors")
    root["components"]=(root["components"] as! [String:Any]).merging(["task":"fl2va"]) {_,new in new}
    XCTAssertThrowsError(try compile(root))
  }
  func testTextOnlyReferenceProfilesValidateAndOmitUnusedReferenceNoise() throws {
    for task in ["ref2va", "fl2va"] {
      var root = recipe(task:task)
      var config = root["config"] as! [String:Any]
      config["visual_condition_strength"] = 0.8
      config["audio_condition_strength"] = 0.6
      root["config"] = config
      let prepared = try compile(root)
      XCTAssertEqual(prepared.sourceTask,task)
      XCTAssertEqual(prepared.text?.transformer.path,"/tmp/selected.safetensors")
      config["visual_condition_strength"] = true
      root["config"] = config
      XCTAssertThrowsError(try compile(root),"Malformed profile strengths must not be silently discarded")
    }
  }
  func testVideoControlsAndMalformedImageSettingsFailBeforeResolvingInputs() throws {
    var resolutions=0
    var root=recipe()
    for extra in ["two_stage","continuation","joint_refinement","fasth3","motion_fidelity"] {
      root=recipe(); root[extra]=[:]
      XCTAssertThrowsError(try H3ImageRecipe.compile(data: JSONSerialization.data(withJSONObject:root)) { _,_,_ in
        resolutions += 1
        return .init(rgb8:Data(),width:64,height:64)
      })
    }
    XCTAssertEqual(resolutions,0)
    root=recipe();root["still_image"]=["version":true,"mode":"quality_five_frames","frame_selection":"middle"]
    XCTAssertThrowsError(try compile(root))
    root=recipe(mode:"single_latent")
    XCTAssertThrowsError(try compile(root))
  }
}
