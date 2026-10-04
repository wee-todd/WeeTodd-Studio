import Foundation
import XCTest
@testable import H3MLX

final class H3ReferencePreparationRecipeTests: XCTestCase {
  private let digest = String(repeating: "a", count: 64)
  private func recipe(task: String, inputs: [[String: Any]]) throws -> Data {
    try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "h3",
      "components": ["task": "ref2va", "transformer": "/models/transformer", "text_encoder": "/models/qwen",
        "vision_encoder": "/models/qwen", "tokenizer": "/models/tokenizer.json", "video_vae": "/models/video", "audio_vae": "/models/audio"],
      "config": ["width": 384, "height": 256, "duration_seconds": 2.5, "seed": 42, "steps": 5],
      "prompt": "One subject raises one hand.",
      "conditioning": ["version": 1, "task": task, "audio_policy": "generated", "inputs": inputs]])
  }
  private func image(_ id: String) -> [String: Any] {
    ["id": id, "kind": "image", "role": "reference", "path": "/" + id + ".png",
      "sha256": digest, "image_pixel_budget_percent": 100]
  }
  func testImagePolicyIsResolvedAndInvalidLastPolicyFailsBeforeAnyMedia() throws {
    var calls = 0
    let request = try H3StudioRecipe.compileMediaReferences(data: recipe(task: "ref2va", inputs: [image("one")])) {
      _, kind, _, geometry, controls in
      calls += 1; XCTAssertEqual(kind, "image"); XCTAssertEqual(geometry.width, 384)
      XCTAssertEqual(controls?.imagePixelBudgetPercent, 100)
      return .image(.init(rgb8: Data(repeating: 17, count: 64 * 64 * 3), width: 64, height: 64, pixelBudgetPercent: 100))
    }
    XCTAssertEqual(request.references.count, 1); XCTAssertEqual(calls, 1)
    var bad = image("bad"); bad["image_pixel_budget_percent"] = true; calls = 0
    XCTAssertThrowsError(try H3StudioRecipe.compileMediaReferences(data: recipe(task: "ref2va", inputs: [image("one"), bad])) {
      _, _, _, _, _ in calls += 1; return .image(.init(rgb8: Data(), width: 64, height: 64))
    }); XCTAssertEqual(calls, 0)
    XCTAssertThrowsError(try H3StudioRecipe.compileMediaReferences(data: recipe(task: "ref2va", inputs: [image("one")])) {
      _, _, _, _, _ in .image(.init(rgb8: Data(repeating: 17, count: 64 * 64 * 3), width: 64, height: 64))
    })
  }
  func testMovieSidecarRetainsDensityDecisionAndSourceTimeline() throws {
    let input: [String: Any] = ["id": "movie", "kind": "video", "role": "reference", "path": "/movie.mp4",
      "sha256": digest, "size_policy": "match_output", "temporal_density": "half",
      "soundtrack_path": "/audio.wav", "soundtrack_sha256": digest]
    let pixels = Data(repeating: 17, count: 56 * 64 * 64 * 3)
    let request = try H3StudioRecipe.compileMediaReferences(data: recipe(task: "ref2va", inputs: [input])) {
      _, kind, _, _, controls in
      if kind == "audio" { XCTAssertNil(controls); return .audio(.init(samples: [Float](repeating: 0.1, count: 64_000), frames: 32_000)) }
      let decision = try H3ReferenceTemporalPolicy.resolve(rgb8: pixels, frames: 56,
        width: 64, height: 64, policy: .half)
      return .video(.init(rgb8: pixels, frameCount: 56, width: 64, height: 64,
        controls: controls, temporalDecision: decision))
    }
    guard case .video(let video) = request.references[0] else { return XCTFail("Expected one audiovisual movie reference.") }
    XCTAssertEqual(video.controls?.temporalDensity, .half)
    XCTAssertEqual(video.persistentFrameCount, 22); XCTAssertEqual(video.sourceLatentFrames, 17)
    XCTAssertEqual(video.audio?.frames, 32_000)
  }
  func testA2VImageBudgetAppliesToLaterAnchorWithoutChangingDriverInterval() throws {
    var anchor = image("later"); anchor["role"] = "keyframe"; anchor["frame_index"] = 50
    let driver: [String: Any] = ["id": "driver", "kind": "audio", "role": "audio_driver", "path": "/audio.wav", "sha256": digest,
      "source_start_seconds": 0.125, "source_duration_seconds": 2.5]
    var calls: [String] = []
    let request = try H3StudioRecipe.compileA2V(data: recipe(task: "a2v", inputs: [driver, anchor])) {
      path, _, start, duration, _, controls in
      calls.append(path)
      if duration != 0 { XCTAssertEqual(start, 0.125); XCTAssertNil(controls)
        return .audio(.init(samples: [Float](repeating: 0.1, count: 160_000), frames: 80_000)) }
      XCTAssertEqual(controls?.imagePixelBudgetPercent, 100)
      return .image(.init(rgb8: Data(repeating: 17, count: 64 * 64 * 3), width: 64, height: 64, pixelBudgetPercent: 100))
    }
    XCTAssertEqual(calls, ["/audio.wav", "/later.png"])
    guard case .timedImage(let still, let frame) = request.references[1] else { return XCTFail("Expected later image anchor.") }
    XCTAssertEqual(frame, 50); XCTAssertEqual(still.pixelBudgetPercent, 100)
  }
}
