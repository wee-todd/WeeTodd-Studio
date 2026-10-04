import Foundation
import XCTest
@testable import H3MLX

final class H3TimedReferenceRecipeTests: XCTestCase {
  private static let digest = String(repeating: "a", count: 64)
  private static func data(task: String, inputs: [[String: Any]], config: [String: Any] = [:]) throws -> Data {
    var settings: [String: Any] = ["width": 384, "height": 256, "duration_seconds": 2.5, "seed": 42, "steps": 5]
    settings.merge(config) { _, new in new }
    return try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "h3",
      "components": ["task": "ref2va", "transformer": "/models/transformer", "text_encoder": "/models/qwen",
        "vision_encoder": "/models/qwen", "tokenizer": "/models/tokenizer.json", "video_vae": "/models/video", "audio_vae": "/models/audio"],
      "config": settings, "prompt": "One subject holds still, then raises one hand.",
      "conditioning": ["version": 1, "task": task, "audio_policy": "generated", "inputs": inputs]])
  }
  private static func image(_ id: String, frame: Any?, role: String = "reference") -> [String: Any] {
    var result: [String: Any] = ["id": id, "kind": "image", "role": role, "path": "/media/" + id + ".png", "sha256": digest, "strength": 1]
    if let frame { result["frame_index"] = frame }; return result
  }
  private static func still() -> H3StillReference { .init(rgb8: Data(repeating: 17, count: 64 * 64 * 3), width: 64, height: 64) }

  // Existing compiler API: these are executable RED cases against the prior implementation.
  func testTimedRefImagesAreAdmittedAndInvalidLastInputPrecedesAnyResolver() throws {
    let input = [Self.image("first", frame: nil), Self.image("guide", frame: 25)]
    var calls = 0
    let request = try H3StudioRecipe.compileMediaReferences(data: Self.data(task: "ref2va", inputs: input)) { _, _, _ in
      calls += 1; return .image(Self.still())
    }
    XCTAssertEqual(calls, 2); XCTAssertEqual(request.references.count, 2)
    calls = 0
    for invalid in ([true, -1, 73, 1.5, "first", NSNull()] as [Any]) {
      XCTAssertThrowsError(try H3StudioRecipe.compileMediaReferences(data: Self.data(task: "ref2va",
        inputs: [input[0], Self.image("bad", frame: invalid)])) { _, _, _ in calls += 1; return .image(Self.still()) })
    }
    XCTAssertEqual(calls, 0)
  }
  func testA2VLaterAnchorsKeepTheOneDriverAndInputOrder() throws {
    let driver: [String: Any] = ["id": "driver", "kind": "audio", "role": "audio_driver", "path": "/media/driver.wav",
      "sha256": Self.digest, "strength": 1, "source_start_seconds": 0, "source_duration_seconds": 2.5]
    let inputs = [Self.image("opening", frame: 0, role: "keyframe"), driver,
      Self.image("later", frame: 50, role: "keyframe")]
    var paths: [String] = []
    let request = try H3StudioRecipe.compileA2V(data: Self.data(task: "a2v", inputs: inputs)) { path, _, _, duration in
      paths.append(path)
      return duration == 0 ? .image(Self.still()) : .audio(.init(samples: [Float](repeating: 0.2, count: 160_000), frames: 80_000))
    }
    XCTAssertEqual(request.references.count, 3); XCTAssertEqual(paths, inputs.map { $0["path"] as! String })
  }
  func testExplicitMovieSidecarIsOneAudiovisualReferenceAndItsHashIsMandatory() throws {
    let movie: [String: Any] = ["id": "movie", "kind": "video", "role": "reference", "path": "/media/movie.mp4",
      "sha256": Self.digest, "strength": 1, "frame_index": 10, "soundtrack_path": "/media/sidecar.wav", "soundtrack_sha256": Self.digest]
    var calls: [String] = []
    let request = try H3StudioRecipe.compileMediaReferences(data: Self.data(task: "ref2va", inputs: [movie])) { path, kind, _ in
      calls.append(path)
      if kind == "audio" { return .audio(.init(samples: [Float](repeating: 0.2, count: 64_000), frames: 32_000)) }
      return .video(.init(rgb8: Data(repeating: 17, count: 22 * 64 * 64 * 3), frameCount: 22, width: 64, height: 64))
    }
    XCTAssertEqual(calls, ["/media/movie.mp4", "/media/sidecar.wav"]); XCTAssertEqual(request.references.count, 1)
    var bad = movie; bad.removeValue(forKey: "soundtrack_sha256"); calls = []
    XCTAssertThrowsError(try H3StudioRecipe.compileMediaReferences(data: Self.data(task: "ref2va", inputs: [bad])) { _, _, _ in
      calls.append("unexpected"); return .image(Self.still())
    }); XCTAssertTrue(calls.isEmpty)
  }
  func testStillOnlyCompilerRejectsNonImageBeforeAnyImageResolver() throws {
    let movie: [String: Any] = ["id": "movie", "kind": "video", "role": "reference", "path": "/movie.mp4", "sha256": Self.digest]
    var calls = 0
    XCTAssertThrowsError(try H3StudioRecipe.compileStillReferences(data: Self.data(task: "ref2va", inputs: [Self.image("still", frame: nil), movie])) { _ in
      calls += 1; return Self.still()
    })
    XCTAssertEqual(calls, 0)
  }
  func testDuplicateA2VAnchorsAndInvalidNoiseFailBeforeMediaResolution() throws {
    let driver: [String: Any] = ["id": "driver", "kind": "audio", "role": "audio_driver", "path": "/driver.wav", "sha256": Self.digest,
      "source_start_seconds": 0, "source_duration_seconds": 2.5]
    var calls = 0
    XCTAssertThrowsError(try H3StudioRecipe.compileA2V(data: Self.data(task: "a2v", inputs: [driver,
      Self.image("one", frame: 20, role: "keyframe"), Self.image("two", frame: 20, role: "keyframe")])) { _, _, _, _ in
      calls += 1; return .image(Self.still())
    })
    XCTAssertThrowsError(try H3StudioRecipe.compileMediaReferences(data: Self.data(task: "ref2va", inputs: [Self.image("one", frame: nil)],
      config: ["audio_condition_strength": true])) { _, _, _ in calls += 1; return .image(Self.still()) })
    XCTAssertEqual(calls, 0)
  }

}
