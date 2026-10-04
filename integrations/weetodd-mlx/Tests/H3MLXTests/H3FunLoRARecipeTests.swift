import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FunLoRARecipeTests: XCTestCase {
  private func data(adapter: [String: Any], steps: Int = 20) throws -> Data {
    try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "One dancer follows the guide.",
      "components": ["task": "t2va", "transformer": "/base", "text_encoder": "/qwen", "tokenizer": "/tokenizer",
        "video_vae": "/video", "audio_vae": "/audio", "fun_controlnet": "/control.safetensors"],
      "config": ["width": 32, "height": 32, "duration_seconds": 2.5, "steps": steps, "seed": 3],
      "loras": ["version": 1, "adapters": [adapter]],
      "conditioning": ["version": 1, "task": "control", "audio_policy": "generated", "inputs": [["id": "guide", "kind": "video", "role": "control",
        "path": "/guide.mp4", "sha256": String(repeating: "a", count: 64), "strength": 0.75, "control_type": "canny_edges"]]]])
  }
  private func adapter(_ changes: [String: Any] = [:]) -> [String: Any] {
    ["path": "/adapter.safetensors", "strength": 0.5, "profile": "standard", "qkv_layout": "contiguous_qkv", "start_after_evaluations": 2]
      .merging(changes) { _, new in new }
  }
  func testFunClonePreservesEveryExplicitAdapterControl() throws {
    let request = try H3StudioRecipe.compileControl(data: data(adapter: adapter())) { _, _, geometry in
      .init(rgb8: Data(count: geometry.frames * 32 * 32 * 3), frameCount: geometry.frames, width: 32, height: 32)
    }
    let selected = try XCTUnwrap(request.loRAAdapters.first)
    XCTAssertEqual(selected.profile, .standard); XCTAssertEqual(selected.qkvLayout, .contiguousQKV)
    XCTAssertEqual(selected.startAfterEvaluations, 2); XCTAssertEqual(selected.strength, 0.5)
    XCTAssertEqual(request.requestedSteps, 20); XCTAssertEqual(request.funControl?.strength, 0.75)
  }
  func testIncompatibleTurboScheduleAndMalformedAdaptersRejectBeforeGuideResolution() throws {
    var calls = 0
    for entry in [adapter(["profile": "turbo", "start_after_evaluations": 0]), adapter(["strength": true]), adapter(["qkv_layout": "guessed"])] {
      XCTAssertThrowsError(try H3StudioRecipe.compileControl(data: data(adapter: entry)) { _, _, _ in
        calls += 1; return .init(rgb8: Data(), frameCount: 0, width: 0, height: 0)
      })
    }
    XCTAssertEqual(calls, 0)
  }
  func testLoRAChangesBaseStreamAndControlKeepsItsIndependentStream() throws {
    try Device.withDefaultDevice(.cpu) {
      let input = MLXArray([Float(1), 2], [1, 2, 1]), control = MLXArray.zeros([1, 2, 1])
      let a = MLXArray([Float(1)], [1, 1]), b = MLXArray([Float(0.5)], [1, 1])
      var expected = [Float(1), 2], controlValue = Float(0)
      for index in 0..<50 {
        expected = expected.map { $0 * 1.5 }
        if H3FunControlLayout.v1InjectionLayers.contains(index) { controlValue += 1; expected = expected.map { $0 + controlValue } }
      }
      let actual = try H3FunControlMath.runBlocks(input: input, blockCount: 50, control: control,
        baseBlock: { _, hidden in H3LoRAProjection.apply(base: hidden, input: hidden, a: a, b: b, alpha: 1, strength: 1, reorderQKV: false) },
        controlBlock: { _, current in let next = current + 1; return (next, next) }).asArray(Float.self)
      XCTAssertEqual(actual, expected)
    }
  }
}
