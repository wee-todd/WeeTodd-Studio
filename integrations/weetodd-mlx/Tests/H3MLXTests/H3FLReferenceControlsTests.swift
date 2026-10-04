import Foundation
import XCTest
@testable import H3MLX

final class H3FLReferenceControlsTests: XCTestCase {
  private func recipe(config: [String: Any]) throws -> Data {
    var settings: [String: Any] = ["width": 64, "height": 64, "duration_seconds": 2.5, "seed": 42, "steps": 5]
    settings.merge(config) { _, value in value }
    return try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "The robot raises one hand.",
      "components": ["task": "fl2va", "transformer": "/transformer", "text_encoder": "/qwen", "vision_encoder": "/vision", "tokenizer": "/tokenizer", "video_vae": "/video", "audio_vae": "/audio"], "config": settings,
      "conditioning": ["version": 1, "task": "fflf", "audio_policy": "generated", "inputs": [
        ["id": "first", "kind": "image", "role": "first", "path": "/image.png", "sha256": String(repeating: "a", count: 64), "frame_index": 0]]]])
  }
  func testExplicitFLStrengthIsAdmittedBeforeMediaAndAbsenceRetainsLegacyCoefficients() throws {
    var calls = 0
    func compile(_ settings: [String: Any]) throws -> H3FL2VARequest {
      try H3StudioRecipe.compileFL2VA(data: recipe(config: settings)) { _, _, width, height in
        calls += 1; return .init(rgb8: Data(count: width * height * 3), width: width, height: height)
      }
    }
    let legacy = try compile([:])
    XCTAssertNil(legacy.referenceNoise)
    let explicit = try compile(["visual_condition_strength": 0.25, "audio_condition_strength": 1])
    XCTAssertEqual(explicit.referenceNoise, try .init(visual: 0.25, audio: 1))
    XCTAssertEqual(calls, 2)
    XCTAssertThrowsError(try compile(["visual_condition_strength": true]))
    XCTAssertEqual(calls, 2)
  }
  func testCleanHistoryUsesItsOwnClockWhenFLAnchorStrengthIsLower() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 3.75)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1], anchors: [.frame(22)], contextFrames: 22)
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let prefix = 7
    let plan = try H3ReferenceRowSchedule(layout: layout, video: video, audio: audio, visualConditionStrength: 0.25, cleanVideoPrefixRows: prefix)
    for step in video.timesteps.indices {
      for index in layout.conditionVideoIndices.prefix(prefix) { XCTAssertEqual(plan.table[Int(plan.indicesByStep[step][index])], 1) }
      for index in layout.conditionVideoIndices.dropFirst(prefix) {
        XCTAssertEqual(plan.table[Int(plan.indicesByStep[step][index])], max(video.timesteps[step], 0.25))
      }
    }
  }
}
