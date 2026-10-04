import Foundation
import XCTest
@testable import H3MLX

final class H3RefHistoryRecipeTests: XCTestCase {
  private func root(task: String = "ref2va") -> [String: Any] {
    ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "The subject holds guard.",
      "components": ["task": "ref2va", "transformer": "/base", "text_encoder": "/qwen", "vision_encoder": "/vision", "tokenizer": "/tokenizer", "video_vae": "/video", "audio_vae": "/audio"],
      "config": ["width": 64, "height": 32, "duration_seconds": 68.0 / 24, "steps": 5, "seed": 42],
      "conditioning": ["version": 1, "task": task, "audio_policy": "generated", "inputs": [
        ["id": "identity", "kind": "image", "role": "reference", "path": "/identity.png", "sha256": String(repeating: "a", count: 64)],
        ["id": "pose", "kind": "image", "role": "reference", "path": "/pose.png", "sha256": String(repeating: "b", count: 64), "frame_index": "last"]]],
      "continuation": ["version": 4, "context_frames": 22, "source_context": "/context/manifest.json", "source_manifest_sha256": String(repeating: "c", count: 64), "save_context": true]]
  }
  func testVisibleTimedCoordinatesOffsetExactlyOnceAndIdentityRemainsUntimed() throws {
    let prepared = try H3Ref2VAContinuationRecipe.prepare(data: JSONSerialization.data(withJSONObject: root()))
    XCTAssertEqual(prepared.plan.generatedFrames, 90); XCTAssertEqual(prepared.plan.publishedFrames, 68)
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: prepared.sampledRecipe) as? [String: Any])
    XCTAssertNil(object["continuation"])
    let media = try XCTUnwrap((object["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertNil(media[0]["frame_index"]); XCTAssertEqual(media[1]["frame_index"] as? Int, 89)
    XCTAssertEqual((object["config"] as? [String: Any])?["duration_seconds"] as? Double, 3.75)
    XCTAssertThrowsError(try H3Ref2VAContinuationRecipe.prepare(data: prepared.sampledRecipe))
  }
  func testInvalidVersionAndUnsavableWindowAreRejectedWithoutReadingContext() throws {
    var recipe = root(); var fields = try XCTUnwrap(recipe["continuation"] as? [String: Any]); fields["version"] = 3; recipe["continuation"] = fields
    XCTAssertThrowsError(try H3Ref2VAContinuationRecipe.prepare(data: JSONSerialization.data(withJSONObject: recipe)))
    recipe = root(); var config = try XCTUnwrap(recipe["config"] as? [String: Any]); config["duration_seconds"] = 2.625; recipe["config"] = config
    XCTAssertThrowsError(try H3Ref2VAContinuationRecipe.prepare(data: JSONSerialization.data(withJSONObject: recipe)))
  }
  func testA2VUsesVisibleSourceIntervalAndPlacesDriverAfterHistoryWithoutChangingInputOrder() throws {
    var recipe = root(task: "a2v")
    recipe["conditioning"] = ["version": 1, "task": "a2v", "audio_policy": "generated", "inputs": [
      ["id": "driver", "kind": "audio", "role": "audio_driver", "path": "/driver.wav", "sha256": String(repeating: "a", count: 64), "source_start_seconds": 0, "source_duration_seconds": 68.0 / 24],
      ["id": "pose", "kind": "image", "role": "keyframe", "path": "/pose.png", "sha256": String(repeating: "b", count: 64), "frame_index": "last"]]]
    let prepared = try H3Ref2VAContinuationRecipe.prepare(data: JSONSerialization.data(withJSONObject: recipe))
    var paths: [String] = []
    let request = try H3StudioRecipe.compileA2V(data: prepared.sampledRecipe, driverTargetFrame: prepared.plan.overlapFrames,
      visibleDurationSeconds: Double(prepared.plan.publishedFrames) / 24) { path, _, _, duration in
        paths.append(path)
        if duration > 0 { return .audio(.init(samples: [Float](repeating: 0.125, count: 192_000), frames: 96_000)) }
        return .image(.init(rgb8: Data(count: 64 * 64 * 3), width: 64, height: 64))
      }
    XCTAssertEqual(paths, ["/driver.wav", "/pose.png"])
    if case .timedAudio(_, let frame) = request.references[0] { XCTAssertEqual(frame, 22) } else { XCTFail("Driver lost history placement.") }
    if case .timedImage(_, let frame) = request.references[1] { XCTAssertEqual(frame, 89) } else { XCTFail("Anchor lost endpoint placement.") }
  }
}
