import Foundation
import XCTest
@testable import H3MLX

final class H3FunControlRecipeTests: XCTestCase {
  private var recipeMemoryMode = "normal"
  private var recipeVideoDecodePrecision = "float32"
  private func recipe() -> [String: Any] {
    ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "A dancer moves",
      "components": ["task": "t2va", "transformer": "/tmp/h3.safetensors",
        "text_encoder": "/tmp/qwen-pages", "tokenizer": "/tmp/tokenizer.json",
        "video_vae": "/tmp/video.safetensors", "audio_vae": "/tmp/audio.safetensors",
        "fun_controlnet": "/tmp/fun.safetensors"],
      "config": ["width": 32, "height": 32, "duration_seconds": 2.5, "steps": 5, "seed": 17, "memory_mode": recipeMemoryMode,"video_decode_precision":recipeVideoDecodePrecision],
      "conditioning": ["version": 1, "task": "control", "audio_policy": "generated",
        "inputs": [["id": "guide", "kind": "video", "role": "control",
          "path": "/tmp/pose.mp4", "sha256": String(repeating: "a", count: 64),
          "control_type": "pose_skeleton", "strength": 0.75]]]]
  }

  func testControlRecipePreservesGuideStrengthAndExactAlignedGeometry() throws {
    let request = try H3StudioRecipe.compileControl(
      data: JSONSerialization.data(withJSONObject: recipe())) { path, hash, geometry in
      XCTAssertEqual(path, "/tmp/pose.mp4"); XCTAssertEqual(hash, String(repeating: "a", count: 64))
      XCTAssertEqual(geometry.frames, 73)
      return H3VideoReference(rgb8: Data(count: geometry.frames * 32 * 32 * 3),
        frameCount: geometry.frames, width: 32, height: 32)
    }
    XCTAssertEqual(request.videoDecodeMemoryMode?.rawValue, recipeMemoryMode)
    XCTAssertEqual(request.videoDecodePrecision.rawValue,recipeVideoDecodePrecision)
    XCTAssertEqual(request.funControl?.strength, 0.75)
    XCTAssertEqual(request.funControl?.checkpoint.path, "/tmp/fun.safetensors")
    XCTAssertEqual(request.seed, 17)
    XCTAssertEqual(request.geometry.frames, 73)
    XCTAssertThrowsError(try H3FL2VARequest(base: request,
      vision: URL(fileURLWithPath: "/tmp/vision.safetensors"),
      images: [H3StillReference(rgb8: Data(count: 32 * 32 * 3), width: 32, height: 32)],
      anchors: [.first]))
    XCTAssertThrowsError(try H3StudioRecipe.compile(
      data: JSONSerialization.data(withJSONObject: recipe())))
  }

  func testUnsupportedControlsFailBeforeResolvingAnyMedia() throws {
    var recipes: [[String: Any]] = []
    for strength: Any in [-0.01, 1.01, true, "strong"] {
      var value = recipe(); var condition = value["conditioning"] as! [String: Any]
      var inputs = condition["inputs"] as! [[String: Any]]
      inputs[0]["strength"] = strength; condition["inputs"] = inputs
      value["conditioning"] = condition; recipes.append(value)
    }
    for control in ["inpaint", "unprocessed_video"] {
      var value = recipe(); var condition = value["conditioning"] as! [String: Any]
      var inputs = condition["inputs"] as! [[String: Any]]
      inputs[0]["control_type"] = control; condition["inputs"] = inputs
      value["conditioning"] = condition; recipes.append(value)
    }
    var lora = recipe(); var component = lora["components"] as! [String: Any]
    component["loras"] = [["/tmp/turbo.safetensors", "strong"]]; lora["components"] = component; recipes.append(lora)
    var optimized = recipe(); var config = optimized["config"] as! [String: Any]
    config["inference_optimization"] = "fast"; optimized["config"] = config; recipes.append(optimized)
    for value in recipes {
      var resolved = false
      XCTAssertThrowsError(try H3StudioRecipe.compileControl(
        data: JSONSerialization.data(withJSONObject: value)) { _, _, _ in
          resolved = true
          return H3VideoReference(rgb8: Data(), frameCount: 0, width: 0, height: 0)
        })
      XCTAssertFalse(resolved)
    }
  }

  func testGuideCannotChangeAdmittedGeometryOrAddAudio() throws {
    for changed in [true, false] {
      XCTAssertThrowsError(try H3StudioRecipe.compileControl(
        data: JSONSerialization.data(withJSONObject: recipe())) { _, _, geometry in
          H3VideoReference(rgb8: Data(count: geometry.frames * 32 * 32 * 3),
            frameCount: changed ? geometry.frames - 1 : geometry.frames, width: 32, height: 32,
            audio: changed ? nil : H3AudioReference(samples: [], frames: 0))
        })
    }
  }
  func testFP16PrecisionSurvivesFunRequestClone() throws {
    recipeMemoryMode = "low_memory_bf16";recipeVideoDecodePrecision = "float16"
    try testControlRecipePreservesGuideStrengthAndExactAlignedGeometry()
  }
  func testLowerMemoryModeSurvivesFunRequestClone() throws {
    recipeMemoryMode = "low_memory_bf16"
    try testControlRecipePreservesGuideStrengthAndExactAlignedGeometry()
  }
}
