import Foundation
import XCTest
@testable import LTX25MLX

extension StudioRecipeTests {
  func singleStageControlRecipe(_ family: String) -> [String: Any] {
    var recipe = fixture(), config = recipe["config"] as! [String: Any]
    config["width"] = 576; config["height"] = 320
    config["ic_lora_single_stage"] = true; config["stage2_steps"] = 0
    config["stage1_sampler"] = "euler_ancestral_cfg_pp"; config["cfg_pp_schedule"] = "speed"
    config["negative_prompt"] = "blur"; config["generated_keyframes"] = 2
    config["stage1_eta"] = 1
    recipe["config"] = config
    var components = recipe["components"] as! [String: Any]
    components["ic_loras"] = [["/models/control.safetensors", 0.8]]
    components["loras"] = [["/models/style.safetensors", 0.5]]
    components["spatial_upscaler_path"] = ""; recipe["components"] = components
    let anchors = (recipe["conditioning"] as! [String: Any])["inputs"] as! [[String: Any]]
    func guide(_ role: String) -> [String: Any] {
      var item: [String: Any] = ["id": role, "kind": "video", "role": "control",
        "control_type": family == "union" ? "pose_skeleton" : family,
        "path": "/\(role).rgb24", "sha256": String(repeating: "a", count: 64),
        "strength": 0.7, "format": "rgb24"]
      if family == "crossview_warp" { item["reference_role"] = role }
      return item
    }
    var condition: [String: Any] = ["version": 1, "task": "control",
      "audio_policy": family == "crossview_warp" ? "source" : "generated",
      "inputs": (family == "crossview_warp" ? [guide("warp"), guide("source")] : [guide("control")]) + anchors]
    if family == "crossview_warp" {
      condition["publication_audio"] = ["path": "/source.wav", "sha256": String(repeating: "b", count: 64),
        "source_start_seconds": 0, "source_duration_seconds": 3.6]
    }
    recipe["conditioning"] = condition
    return recipe
  }

  func testSingleStageStudioControlsRetainAnchorsSlotsGenericLoRAsAndSchedule() throws {
    for family in ["union", "motion_track", "crossview_warp"] {
      let request = try compile(singleStageControlRecipe(family))
      XCTAssertEqual(request.version, 15)
      XCTAssertEqual(request.task, family == "union" ? "union_control" : "ic_control")
      XCTAssertEqual(request.referenceImages.map(\.role), ["first", "last"])
      XCTAssertEqual(request.generatedKeyframes, 2)
      XCTAssertEqual(request.stageOneLoras.map(\.path), ["/models/style.safetensors"])
      XCTAssertTrue(request.stageTwoLoras.isEmpty)
      XCTAssertEqual(request.singleStageSampling?.negativeSchedule, .speed)
      XCTAssertEqual(request.singleStageSampling?.negativePrompt, "blur")
      XCTAssertNotNil(try request.singleStageControlLayout())
      let decoded = try JSONDecoder().decode(MLXDistilledRequest.self, from: JSONEncoder().encode(request))
      XCTAssertEqual(decoded.referenceImages.map(\.path), ["/first.png", "/last.png"])
    }
  }

  func testSingleStageStudioControlsRejectUnknownMediaDuplicateIDsAndTwoStageAnchors() throws {
    let valid = singleStageControlRecipe("union")
    for bad in ["role", "id", "ignored"] {
      var recipe = valid, condition = recipe["conditioning"] as! [String: Any]
      var inputs = condition["inputs"] as! [[String: Any]]
      if bad == "role" { inputs[1]["role"] = "reference" }
      else if bad == "id" { inputs[1]["id"] = inputs[0]["id"] }
      else { inputs[0][bad] = true }
      condition["inputs"] = inputs; recipe["conditioning"] = condition
      XCTAssertThrowsError(try compile(recipe), bad)
    }
    var old = valid, config = old["config"] as! [String: Any]
    config["ic_lora_single_stage"] = false; config["stage2_steps"] = 3; old["config"] = config
    XCTAssertThrowsError(try compile(old))
  }
}
