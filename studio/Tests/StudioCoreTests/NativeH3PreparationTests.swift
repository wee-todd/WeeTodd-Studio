import Foundation
import XCTest
@testable import StudioCore

final class NativeH3PreparationTests: XCTestCase {
  func testInstalledCompatibleFL2VAProfileAppearsAsFirstLastTask() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_FL2VA_PROFILE"] else {
      throw XCTSkip("Set an installed FL2VA Studio profile for catalog admission.")
    }
    let url = URL(fileURLWithPath: path)
    let catalog = try NativeH3Preparation.catalog(
      directory: url.deletingLastPathComponent().path)
    let entry = try XCTUnwrap(catalog.first {
      ($0["id"] as? String) == url.resolvingSymlinksInPath().path
    })
    XCTAssertEqual(entry["engine"] as? String, "h3")
    XCTAssertEqual(entry["task"] as? String, "fflf")
    let generation = try XCTUnwrap(entry["generation"] as? [String: Any])
    XCTAssertEqual(generation["supportedTasks"] as? [String], ["fflf"])
  }

  private func fixture() throws -> (URL, StudioProject, [String: Any]) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let recipe: [String: Any] = [
      "format": "weetodd-headless-v2", "engine": "h3", "prompt": "stale",
      "components": ["task": "t2va", "transformer": "/model/transformer.safetensors",
        "text_encoder": "/model/qwen", "tokenizer": "/model/tokenizer.json",
        "video_vae": "/model/video.safetensors", "audio_vae": "/model/audio.safetensors",
        "loras": []],
      "config": ["width": 768, "height": 448, "duration_seconds": 5.0,
        "steps": 5, "seed": 1, "drop_adaln": true,
        "sampling_method": "euler", "transformer_backend": "mlx"],
      "conditioning": ["version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated"]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("h3.json"))
    var project = StudioProject()
    var clip = Clip(engine: .h3); clip.profileID = root.appendingPathComponent("h3.json").path
    clip.prompt = "  Beowulf waits beside a fire.  "; clip.duration = 5; clip.seed = 42
    clip.generationWidth = 768; clip.generationHeight = 448
    clip.generationSelection = GenerationSelection(task: "t2v")
    project.clips = [clip]
    return (root, project, ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"])
  }

  private func request(_ project: StudioProject, _ runtime: [String: Any]) throws -> [String: Any] {
    ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": project.clips[0].id.uuidString, "runtime": runtime]
  }

  func testComposesTextOnlyRecipeWithoutDroppingUserInputs() throws {
    let (_, project, runtime) = try fixture()
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = try XCTUnwrap(result["recipe"] as? [String: Any])
    let prompt = try XCTUnwrap(recipe["prompt"] as? String)
    XCTAssertTrue(prompt.hasPrefix("integrated_multimodal_description: [Shot 1] Beowulf waits beside a fire."))
    XCTAssertTrue(prompt.contains("overall_soundscape: Natural location sound. No dialogue."))
    XCTAssertTrue(prompt.contains("non_diegetic_music: N/A"))
    let config = try XCTUnwrap(recipe["config"] as? [String: Any])
    XCTAssertEqual(config["seed"] as? Int, 42)
    XCTAssertEqual(config["duration_seconds"] as? Double, 5)
    XCTAssertEqual((result["report"] as? [String: Any])?["nativePreparation"] as? String, "swift")
  }

  func testRejectsUnsupportedMediaSamplingAndProfileOptions() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    project.clips[0].negativePrompt = "no grain"
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original
    project.clips[0].generationSelection?.cfg = 2
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original
    project.clips[0].attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["loras"] = [["/model/turbo.safetensors", 0.8]]
    recipe["components"] = components
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let prepared = try NativeH3Preparation.compose(request: request(project, runtime))
    let selected = (prepared["recipe"] as! [String: Any])["components"] as! [String: Any]
    XCTAssertEqual((selected["loras"] as! [[Any]])[0][0] as? String,
      "/model/turbo.safetensors")
    components["loras"] = [["/model/turbo.safetensors", 0.8],
      ["/model/second.safetensors", 1.0]]
    recipe["components"] = components
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testSelectedEvaluationsBecomeOneExtraSigmaGridPoint() throws {
    let (_, original, runtime) = try fixture()
    var project = original
    project.clips[0].generationSelection?.steps = 4
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = result["recipe"] as! [String: Any]
    let config = recipe["config"] as! [String: Any]
    let generation = (result["report"] as! [String: Any])["generation"] as! [String: Any]
    let controls = generation["controls"] as! [String: Any]
    XCTAssertEqual(config["steps"] as? Int, 5)
    XCTAssertEqual(controls["evaluations"] as? Int, 4)
    project.clips[0].generationSelection?.steps = Int.max
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testEnabledTurboAttachmentIsIncludedAndDuplicateProfileAdapterFails() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    let path = root.appendingPathComponent("turbo.safetensors")
    try Data([0]).write(to: path)
    var asset = MediaAsset(name: "Turbo", kind: .lora, path: path.path)
    asset.loraModel = .h3
    asset.loraProfile = "turbo"
    asset.loraLayout = "contiguous_qkv"
    project.assets = [asset]
    var attachment = Attachment(assetID: asset.id, role: .lora)
    attachment.strength = 0.8
    project.clips[0].attachments = [attachment]
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let components = (result["recipe"] as! [String: Any])["components"] as! [String: Any]
    let pair = (components["loras"] as! [[Any]])[0]
    XCTAssertEqual(pair[0] as? String, path.path)
    XCTAssertEqual(pair[1] as? Double, 0.8)
    var recipe = try JSONSerialization.jsonObject(with:
      Data(contentsOf: root.appendingPathComponent("h3.json"))) as! [String: Any]
    var configured = recipe["components"] as! [String: Any]
    configured["loras"] = [[path.path, 1.0]]
    recipe["components"] = configured
    try JSONSerialization.data(withJSONObject: recipe)
      .write(to: root.appendingPathComponent("h3.json"))
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testDescriptionAndAtomicPreparation() throws {
    let (root, project, runtime) = try fixture()
    let body = try request(project, runtime)
    let catalog = try NativeH3Preparation.catalog(directory: root.path)
    XCTAssertEqual(catalog.count, 1)
    let description = try NativeH3Preparation.describe(request: body)
    XCTAssertEqual(description["readinessErrors"] as? [String], [])
    let destination = root.appendingPathComponent("prepared")
    let result = try NativeH3Preparation.prepare(request: body, destination: destination)
    XCTAssertTrue(FileManager.default.fileExists(atPath: result["recipePath"] as! String))
    XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent("editor-request.json").path))
    XCTAssertThrowsError(try NativeH3Preparation.prepare(request: body, destination: destination))
  }

  func testStillReferenceProfileMapsOrderedImagesWithoutDroppingThem() throws {
    let (root, original, runtime) = try fixture()
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["task"] = "ref2va"
    components["vision_encoder"] = "/model/qwen-vision.safetensors"
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "ref2va",
      "inputs": [], "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    var project = original
    project.clips[0].generationSelection = GenerationSelection(task: "ref2va")
    let firstPath = root.appendingPathComponent("first.png")
    let secondPath = root.appendingPathComponent("second.png")
    try Data([1]).write(to: firstPath)
    try Data([2]).write(to: secondPath)
    let first = MediaAsset(name: "First", kind: .image, path: firstPath.path)
    let second = MediaAsset(name: "Second", kind: .image, path: secondPath.path)
    project.assets = [first, second]
    project.clips[0].attachments = [Attachment(assetID: first.id, role: .reference),
      Attachment(assetID: second.id, role: .reference)]
    XCTAssertEqual(try NativeH3Preparation.catalog(directory: root.path).first?["task"] as? String,
      "ref2va")
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let composed = result["recipe"] as! [String: Any]
    let inputs = (composed["conditioning"] as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.map { $0["path"] as? String }, [firstPath.path, secondPath.path])
    XCTAssertEqual((result["report"] as! [String: Any])["task"] as? String, "ref2va")
    project.clips[0].attachments[0].strength = 0.7
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testFL2VAProfileComposesTrueFirstAndLastEndpoints() throws {
    let (root, original, runtime) = try fixture()
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["task"] = "fl2va"
    components["vision_encoder"] = "/model/qwen-vision.safetensors"
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "fflf", "inputs": [],
      "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    var project = original
    project.clips[0].generationSelection = GenerationSelection(task: "fflf")
    let firstPath = root.appendingPathComponent("first.png")
    let lastPath = root.appendingPathComponent("last.png")
    try Data([1]).write(to: firstPath)
    try Data([2]).write(to: lastPath)
    let first = MediaAsset(name: "First", kind: .image, path: firstPath.path)
    let last = MediaAsset(name: "Last", kind: .image, path: lastPath.path)
    project.assets = [first, last]
    project.clips[0].attachments = [Attachment(assetID: first.id, role: .first),
      Attachment(assetID: last.id, role: .last)]
    XCTAssertEqual(try NativeH3Preparation.catalog(directory: root.path).first?["task"] as? String,
      "fflf")
    let composed = try NativeH3Preparation.compose(request: request(project, runtime))
    let prepared = composed["recipe"] as! [String: Any]
    let conditioning = prepared["conditioning"] as! [String: Any]
    let inputs = conditioning["inputs"] as! [[String: Any]]
    XCTAssertEqual(conditioning["task"] as? String, "fflf")
    XCTAssertEqual(inputs.map { $0["role"] as? String }, ["first", "last"])
    XCTAssertEqual(inputs[0]["frame_index"] as? Int, 0)
    XCTAssertEqual(inputs[1]["frame_index"] as? String, "last")
    XCTAssertEqual((composed["report"] as! [String: Any])["task"] as? String, "fflf")
    project.clips[0].attachments.reverse()
    let reordered = try NativeH3Preparation.compose(request: request(project, runtime))
    let reorderedInputs = ((reordered["recipe"] as! [String: Any])["conditioning"]
      as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(reorderedInputs.map { $0["role"] as? String }, ["first", "last"])
    project.clips[0].attachments[1].strength = 0.5
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testExistingRootTurboProfileNormalizesWithoutCopyingModelFiles() throws {
    let (root, original, runtime) = try fixture()
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["task"] = "ref2va"
    let tokenizer = root.appendingPathComponent("processor")
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: false)
    try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
    components["tokenizer"] = tokenizer.path
    recipe["components"] = components
    recipe["loras"] = ["adapters": [["path": "/model/turbo.safetensors",
      "strength": 1.0, "profile": "turbo", "qkv_layout": "contiguous_qkv"]]]
    var config = recipe["config"] as! [String: Any]
    config["memory_mode"] = "low_memory_bf16"
    config["attention_head_chunk_size"] = "disabled"
    recipe["config"] = config
    recipe["conditioning"] = ["version": 1, "task": "ref2va",
      "inputs": [], "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    var project = original
    project.clips[0].generationSelection = GenerationSelection(task: "ref2va")
    let image = root.appendingPathComponent("subject.png")
    try Data([1, 2, 3]).write(to: image)
    let asset = MediaAsset(name: "Subject", kind: .image, path: image.path)
    project.assets = [asset]
    project.clips[0].attachments = [Attachment(assetID: asset.id, role: .reference)]
    XCTAssertEqual(try NativeH3Preparation.catalog(directory: root.path).count, 1)
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let composed = result["recipe"] as! [String: Any]
    XCTAssertNil(composed["loras"])
    let mapped = composed["components"] as! [String: Any]
    XCTAssertEqual(mapped["tokenizer"] as? String,
      tokenizer.appendingPathComponent("tokenizer.json").path)
    XCTAssertEqual((mapped["loras"] as! [[Any]])[0][0] as? String,
      "/model/turbo.safetensors")
  }
}
