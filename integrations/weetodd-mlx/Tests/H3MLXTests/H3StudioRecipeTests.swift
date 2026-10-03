import Foundation
import XCTest
@testable import H3MLX

final class H3StudioRecipeTests: XCTestCase {
  func testExternalExtensionRequiresPromptAndAnchorsTheTrueLastFrame() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = try XCTUnwrap(root["components"] as? [String: Any])
    components["task"] = "ref2va"
    components["vision_encoder"] = "/tmp/vision.safetensors"
    root["components"] = components
    var config = try XCTUnwrap(root["config"] as? [String: Any])
    config["width"] = 64; config["height"] = 64
    config["duration_seconds"] = 4.0
    root["config"] = config
    let digest = String(repeating: "a", count: 64)
    root["conditioning"] = ["version": 1, "task": "extension",
      "audio_policy": "generated", "inputs": [["id": "source", "kind": "video",
        "role": "reference", "path": "/tmp/source.mp4", "sha256": digest,
        "strength": 1.0]]]
    var resolved = false
    var anchorWidth = 128
    func compile() throws -> H3Ref2VAStillRequest {
      try H3StudioRecipe.compileExtension(
        data: JSONSerialization.data(withJSONObject: root)) { path, hash in
          resolved = true
          XCTAssertEqual(path, "/tmp/source.mp4")
          XCTAssertEqual(hash, digest)
          let movie = H3VideoReference(rgb8: Data(count: 22 * 64 * 64 * 3),
            frameCount: 22, width: 64, height: 64,
            audio: H3AudioReference(samples: [Float](repeating: 0,
              count: 2 * 32_000), frames: 32_000))
          let last = H3StillReference(rgb8: Data(repeating: 17,
            count: 128 * 64 * 3), width: anchorWidth, height: 64)
          return (movie, last)
        }
    }
    XCTAssertThrowsError(try compile())
    XCTAssertFalse(resolved, "Unsupported prompt must fail before media decoding")
    root["prompt"] = "subject_definitions: robot <Video 1> <Picture 1> " +
      "summary: [video continuation] retention_analysis: detailed_description: " +
      "overall_soundscape: non_diegetic_music:"
    let request = try compile()
    XCTAssertTrue(resolved)
    XCTAssertEqual(request.references.count, 2)
    if case .timedImage(let anchor, let frame) = request.references[1] {
      XCTAssertEqual(frame, 0)
      XCTAssertEqual(anchor.rgb8.first, 17)
      XCTAssertEqual(anchor.width, 128)
      XCTAssertEqual(anchor.height, 64)
    } else { XCTFail("External extension did not preserve the seam anchor") }
    anchorWidth = Int.max
    XCTAssertThrowsError(try compile(), "Validate the canvas before RGB byte-count arithmetic")
    anchorWidth = 160
    XCTAssertThrowsError(try compile(), "An anchor must contain every admitted RGB pixel")
  }

  private func recipe(controls: [String: Any] = [:],
    conditioning: [String: Any] = ["version": 1, "task": "t2v",
      "inputs": [], "audio_policy": "generated"],
    components: [String: Any]? = nil) throws -> Data {
    let paths = components ?? ["transformer": "/tmp/h3.safetensors",
      "text_encoder": "/tmp/qwen-pages", "tokenizer": "/tmp/tokenizer.json",
      "video_vae": "/tmp/video.safetensors",
      "audio_vae": "/tmp/audio.safetensors", "task": "t2va"]
    let config: [String: Any] = ["width": 32, "height": 32,
      "duration_seconds": 2.5, "steps": 5, "seed": 123]
      .merging(controls) { _, new in new }
    return try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2",
      "engine": "h3", "prompt": "A person walks", "components": paths,
      "config": config, "conditioning": conditioning])
  }

  func testTextOnlyRecipeRetainsSeedAndRejectsUnimplementedInputs() throws {
    let request = try H3StudioRecipe.compile(data: recipe())
    XCTAssertEqual(request.seed, 123)
    XCTAssertEqual(request.requestedSteps, 5)
    XCTAssertEqual(request.geometry.frames, 73)
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(
      controls: ["sampling_method": "res_multistep"])))
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(
      conditioning: ["version": 1, "task": "fflf", "inputs": []])))
    var adapter = ["transformer": "/tmp/h3.safetensors",
      "text_encoder": "/tmp/qwen-pages", "tokenizer": "/tmp/tokenizer.json",
      "video_vae": "/tmp/video.safetensors",
      "audio_vae": "/tmp/audio.safetensors", "task": "t2va"] as [String: Any]
    adapter["loras"] = [["/tmp/turbo.safetensors", 1.0]]
    let withTurbo = try H3StudioRecipe.compile(data: recipe(components: adapter))
    XCTAssertEqual(withTurbo.turboLoRA?.path, "/tmp/turbo.safetensors")
    XCTAssertEqual(withTurbo.turboLoRAStrength, 1)
    adapter["loras"] = [["/tmp/turbo.safetensors", 1.0],
      ["/tmp/second.safetensors", 0.4]]
    let stacked = try H3StudioRecipe.compile(data: recipe(components: adapter))
    XCTAssertEqual(stacked.loRAAdapters.map(\.url.path),
      ["/tmp/turbo.safetensors", "/tmp/second.safetensors"])
    XCTAssertEqual(stacked.loRAAdapters.map(\.strength), [1, 0.4])
    adapter["loras"] = [["/tmp/turbo.safetensors", 1.0],
      ["/tmp/turbo.safetensors", 0.4]]
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(components: adapter)))
    adapter["loras"] = [["/tmp/turbo.safetensors", "strong"]]
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(components: adapter)))
  }

  func testUnimplementedOrMistypedExecutionControlsFailBeforeWeights() throws {
    for control: [String: Any] in [
      ["memory_mode": "resident"],
      ["drop_adaln": "true"],
      ["paging_cache_gb": "0"],
      ["attention_head_chunk_size": "2"],
      ["transformer_backend": "nnc"],
      ["inference_optimization": "compiled"],
    ] {
      XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(controls: control)),
        "Unexpectedly accepted \(control)")
    }
    let invalidConditioning: [String: Any] = ["version": 1,
      "task": "t2v", "inputs": [], "audio_policy": false]
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(
      conditioning: invalidConditioning)))
  }

  func testStillReferenceRecipePreservesInputOrderAndRejectsDroppedControls() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = try XCTUnwrap(root["components"] as? [String: Any])
    components["task"] = "ref2va"
    components["vision_encoder"] = "/tmp/qwen-vision.safetensors"
    root["components"] = components
    var config = try XCTUnwrap(root["config"] as? [String: Any])
    config["width"] = 64
    config["height"] = 64
    root["config"] = config
    root["conditioning"] = ["version": 1, "task": "ref2va",
      "audio_policy": "generated", "inputs": [
        ["id": "first", "kind": "image", "role": "reference", "path": "/tmp/a.png",
          "sha256": String(repeating: "a", count: 64)],
        ["id": "second", "kind": "image", "role": "reference", "path": "/tmp/b.png",
          "sha256": String(repeating: "b", count: 64)]]]
    var paths: [String] = []
    func compile(_ object: [String: Any]) throws -> H3Ref2VAStillRequest {
      try H3StudioRecipe.compileStillReferences(
        data: JSONSerialization.data(withJSONObject: object)) { path in
          paths.append(path)
          return H3StillReference(rgb8: Data(count: 64 * 64 * 3),
            width: 64, height: 64)
        }
    }
    let request = try compile(root)
    XCTAssertEqual(paths, ["/tmp/a.png", "/tmp/b.png"])
    XCTAssertEqual(request.references.count, 2)
    XCTAssertEqual(request.qwenVision.path, "/tmp/qwen-vision.safetensors")
    var rejected = root
    var conditioning = try XCTUnwrap(rejected["conditioning"] as? [String: Any])
    var inputs = try XCTUnwrap(conditioning["inputs"] as? [[String: Any]])
    inputs[0]["frame_index"] = 0
    conditioning["inputs"] = inputs
    rejected["conditioning"] = conditioning
    XCTAssertThrowsError(try compile(rejected))
    XCTAssertEqual(paths.count, 2, "No media should load after control rejection")
  }

  func testMixedMediaRecipeRetainsInputOrderAndRequiresVisualForAudio() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = try XCTUnwrap(root["components"] as? [String: Any])
    components["task"] = "ref2va"
    components["vision_encoder"] = "/tmp/qwen-vision.safetensors"
    root["components"] = components
    var config = try XCTUnwrap(root["config"] as? [String: Any])
    config["width"] = 64; config["height"] = 64
    root["config"] = config
    let digest = String(repeating: "a", count: 64)
    root["conditioning"] = ["version": 1, "task": "ref2va",
      "audio_policy": "generated", "inputs": [
        ["id": "still", "kind": "image", "role": "reference",
          "path": "/tmp/face.png", "sha256": digest],
        ["id": "movie", "kind": "video", "role": "reference",
          "path": "/tmp/motion.mp4", "sha256": digest],
        ["id": "voice", "kind": "audio", "role": "reference",
          "path": "/tmp/voice.wav", "sha256": digest]]]
    var seen: [String] = []
    func compile(_ object: [String: Any]) throws -> H3Ref2VAStillRequest {
      try H3StudioRecipe.compileMediaReferences(
        data: JSONSerialization.data(withJSONObject: object)) { path, kind, _ in
          seen.append("\(kind):\(path)")
          if kind == "image" {
            return .image(H3StillReference(rgb8: Data(count: 64 * 64 * 3),
              width: 64, height: 64))
          }
          if kind == "audio" {
            return .audio(H3AudioReference(samples: [Float](repeating: 0,
              count: 3_200), frames: 1_600))
          }
          return .video(H3VideoReference(rgb8: Data(count: 5 * 64 * 64 * 3),
            frameCount: 5, width: 64, height: 64))
        }
    }
    let result = try compile(root)
    XCTAssertEqual(result.references.count, 3)
    XCTAssertEqual(seen, ["image:/tmp/face.png", "video:/tmp/motion.mp4",
      "audio:/tmp/voice.wav"])
    var invalid = root
    var conditioning = invalid["conditioning"] as! [String: Any]
    let inputs = conditioning["inputs"] as! [[String: Any]]
    conditioning["inputs"] = [inputs[2]]
    invalid["conditioning"] = conditioning
    XCTAssertThrowsError(try compile(invalid))
    XCTAssertEqual(seen.count, 3, "Audio-only must fail before media loading")
  }

  func testA2VRecipePlacesDriverAtTargetStartAndRetainsSourceInterval() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = root["components"] as! [String: Any]
    components["task"] = "ref2va"
    components["vision_encoder"] = "/tmp/qwen-vision.safetensors"
    root["components"] = components
    let digest = String(repeating: "a", count: 64)
    let driver: [String: Any] = ["id": "voice", "kind": "audio",
      "role": "audio_driver", "path": "/tmp/voice.wav", "sha256": digest,
      "strength": 1.0, "source_start_seconds": 2.0,
      "source_duration_seconds": 2.5]
    root["conditioning"] = ["version": 1, "task": "a2v",
      "audio_policy": "generated", "inputs": [driver]]
    var loaded: [(String, Double, Double)] = []
    func compile(_ object: [String: Any]) throws -> H3Ref2VAStillRequest {
      try H3StudioRecipe.compileA2V(data: JSONSerialization.data(withJSONObject: object)) {
        path, _, start, duration in
        loaded.append((path, start, duration))
        return .audio(H3AudioReference(samples: [Float](repeating: 0,
          count: 2 * 80_000), frames: 80_000))
      }
    }
    let audioOnly = try compile(root)
    XCTAssertEqual(loaded.map(\.0), ["/tmp/voice.wav"])
    XCTAssertEqual(loaded[0].1, 2)
    XCTAssertEqual(loaded[0].2, 2.5)
    if case .timedAudio(_, let frame) = audioOnly.references[0] {
      XCTAssertEqual(frame, 0)
    } else { XCTFail("Expected timed audio") }
    var withImage = root
    var withImageConditioning = withImage["conditioning"] as! [String: Any]
    var withImageInputs = withImageConditioning["inputs"] as! [[String: Any]]
    withImageInputs.insert(["id": "first", "kind": "image", "role": "keyframe",
      "path": "/tmp/first.png", "sha256": digest, "strength": 1.0,
      "frame_index": 0], at: 0)
    withImageConditioning["inputs"] = withImageInputs
    withImage["conditioning"] = withImageConditioning
    let imageAndSound = try H3StudioRecipe.compileA2V(
      data: JSONSerialization.data(withJSONObject: withImage)) {
        path, _, _, _ in
        if path.hasSuffix(".png") {
          return .image(H3StillReference(rgb8: Data(count: 64 * 64 * 3),
            width: 64, height: 64))
        }
        return .audio(H3AudioReference(samples: [Float](repeating: 0,
          count: 2 * 80_000), frames: 80_000))
      }
    XCTAssertEqual(imageAndSound.references.count, 2)
    if case .timedImage(_, let frame) = imageAndSound.references[0] {
      XCTAssertEqual(frame, 0)
    } else { XCTFail("Expected timed opening image first") }
    var invalid = root
    var conditioning = invalid["conditioning"] as! [String: Any]
    var inputs = conditioning["inputs"] as! [[String: Any]]
    inputs[0]["strength"] = 0.5
    conditioning["inputs"] = inputs
    invalid["conditioning"] = conditioning
    XCTAssertThrowsError(try compile(invalid))
    XCTAssertEqual(loaded.count, 1, "Malformed drivers fail before media loading")
  }

  func testFL2VARecipeKeepsOrderedEndpointRolesAndRejectsUnportedInputs() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = try XCTUnwrap(root["components"] as? [String: Any])
    components["task"] = "fl2va"
    components["vision_encoder"] = "/tmp/vision.safetensors"
    root["components"] = components
    let digest = String(repeating: "a", count: 64)
    let first: [String: Any] = ["id": "first", "kind": "image", "role": "first",
      "path": "/tmp/first.png", "frame_index": 0, "sha256": digest, "strength": 1.0]
    let last: [String: Any] = ["id": "last", "kind": "image", "role": "last",
      "path": "/tmp/last.png", "frame_index": "last", "sha256": digest, "strength": 1.0]
    root["conditioning"] = ["version": 1, "task": "fflf", "audio_policy": "generated",
      "inputs": [first, last]]
    var loaded: [String] = []
    func compile(_ value: [String: Any]) throws -> H3FL2VARequest {
      try H3StudioRecipe.compileFL2VA(data: JSONSerialization.data(withJSONObject: value)) {
        path, _, width, height in
        loaded.append(path)
        return H3StillReference(rgb8: Data(count: width * height * 3),
          width: width, height: height)
      }
    }
    let admitted = try compile(root)
    XCTAssertEqual(admitted.anchors, [.first, .last])
    XCTAssertEqual(loaded, ["/tmp/first.png", "/tmp/last.png"])
    var visibleLast = root
    var visibleConditioning = try XCTUnwrap(visibleLast["conditioning"] as? [String: Any])
    visibleConditioning["inputs"] = [first, last.merging(["frame_index": 59]) { _, new in new }]
    visibleLast["conditioning"] = visibleConditioning
    XCTAssertEqual(try compile(visibleLast).anchors, [.first, .frame(59)])
    var bad = root
    bad["conditioning"] = ["version": 1, "task": "fflf", "audio_policy": "generated",
      "inputs": [last, first]]
    XCTAssertThrowsError(try compile(bad))
    bad["conditioning"] = ["version": 1, "task": "fflf", "audio_policy": "generated",
      "inputs": [first.merging(["strength": 0.5]) { _, new in new }, last]]
    XCTAssertThrowsError(try compile(bad))
    XCTAssertEqual(loaded.count, 4, "Invalid controls must fail before media decoding")
  }

  func testFL2VARecipeAdmitsOrderedTimedKeyframesAndRejectsDuplicatesBeforeMediaLoad() throws {
    var root = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe()) as? [String: Any])
    var components = try XCTUnwrap(root["components"] as? [String: Any])
    components["task"] = "fl2va"
    components["vision_encoder"] = "/tmp/vision.safetensors"
    root["components"] = components
    let digest = String(repeating: "a", count: 64)
    func input(_ id: String, _ frame: Any) -> [String: Any] {
      ["id": id, "kind": "image", "role": "keyframe", "path": "/tmp/\(id).png",
        "frame_index": frame, "sha256": digest, "strength": 1.0]
    }
    var loaded: [String] = []
    func compile(_ entries: [[String: Any]]) throws -> H3FL2VARequest {
      root["conditioning"] = ["version": 1, "task": "fflf", "audio_policy": "generated",
        "inputs": entries]
      return try H3StudioRecipe.compileFL2VA(data: JSONSerialization.data(withJSONObject: root)) {
        path, _, width, height in
        loaded.append(path)
        return H3StillReference(rgb8: Data(count: width * height * 3),
          width: width, height: height)
      }
    }
    var last = input("c", "last")
    last["role"] = "last"
    let admitted = try compile([input("a", 0), input("b", 24), last])
    XCTAssertEqual(admitted.anchors, [.first, .frame(24), .last])
    XCTAssertEqual(loaded, ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"])
    XCTAssertThrowsError(try compile([input("a", 0), input("b", 0)]))
    XCTAssertThrowsError(try compile([input("b", 24), input("a", 0)]))
    XCTAssertThrowsError(try compile([input("a", 0), input("z", 999)]))
    XCTAssertEqual(loaded.count, 3)
  }
}
