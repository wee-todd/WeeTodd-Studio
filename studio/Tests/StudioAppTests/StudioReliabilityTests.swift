import Foundation
import Combine
import CryptoKit
import StudioCore
import XCTest
@testable import WeeToddStudio

@MainActor private final class SuspendedBridge {
  var command: String
  var suspendLimit = Int.max
  var inspectRuntimeRoot: String?
  private var suspensionCount = 0
  var calls: [(String, [String: Any])] = []
  var continuation: CheckedContinuation<[String: Any], Error>?
  var entered: (() -> Void)?
  init(_ command: String) { self.command = command }
  func invoke(_ name: String, _ runtime: RuntimeSettings, _ payload: [String: Any], _ output: URL?) async throws -> [String: Any] {
    calls.append((name, payload))
    if name == command && suspensionCount < suspendLimit {
      suspensionCount += 1
      return try await withCheckedThrowingContinuation { continuation in
        self.continuation = continuation
        entered?()
      }
    }
    if name == "inspect" {
      if let inspectRuntimeRoot, runtime.root != inspectRuntimeRoot {
        throw StudioError.invalid("Wrong runtime for completed output")
      }
      return ["duration": payload["path"] as? String == "/tmp/context.mov" ? 8.0 : 12.0]
    }
    if ["prepare", "ltx-native-prepare", "h3-native-prepare"].contains(name) {
      return ["recipePath": "/tmp/prepared/recipe.json", "prompt": "prompt", "report": [:]]
    }
    return [:]
  }
}

final class StudioReliabilityTests: XCTestCase {
  @MainActor func testExperimentalNativeH3PreflightGatesPreparedRecipeWithoutPython() async throws {
    let fake = SuspendedBridge("h3-native-preflight")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: fake.invoke)
    store.runtime.pythonPath = "/unavailable/python"
    store.runtime.nativeH3Enabled = true
    store.addClip(); store.editClip { $0.engine = .h3; $0.prompt = "A quiet lake." }
    let entered = expectation(description: "H3 native preflight")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    XCTAssertNil(store.preparedRecipe)
    fake.continuation?.resume(throwing: StudioError.invalid("Unsupported H3 controls"))
    await task.value
    XCTAssertNil(store.preparedRecipe)
    XCTAssertEqual(store.error, "Unsupported H3 controls")
    XCTAssertTrue(fake.calls.contains { $0.0 == "h3-native-describe" })
    XCTAssertTrue(fake.calls.contains { $0.0 == "h3-native-prepare" })
    XCTAssertFalse(fake.calls.contains { $0.0 == "describe-generation" || $0.0 == "prepare" || $0.0 == "render" })
  }

  @MainActor func testExperimentalNativeH3RenderRoutesWithoutPythonAndKeepsPartialTakeUnpublished() async throws {
    let fake = SuspendedBridge("h3-native-render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: fake.invoke)
    store.runtime.pythonPath = "/unavailable/python"; store.runtime.nativeH3Enabled = true
    store.addClip(); store.editClip { $0.engine = .h3; $0.prompt = "A quiet lake." }
    store.preparedRecipe = "/tmp/h3/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "H3 native render")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    XCTAssertNotNil(store.nativePreviewOwner)
    fake.continuation?.resume(throwing: StudioError.invalid("Worker cancelled"))
    await task.value
    XCTAssertNil(store.selectedClip?.versions.last)
    XCTAssertTrue(fake.calls.contains { $0.0 == "h3-native-render" })
    XCTAssertFalse(fake.calls.contains { $0.0 == "render" || $0.0 == "inspect" })
  }

  @MainActor func testNativeH3RenderCreatesOutputParentBeforeWorkerLaunch() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: { command, _, _, output in
        if command == "h3-native-render" {
          guard let output,
            FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path),
            !FileManager.default.fileExists(atPath: output.path) else {
            throw StudioError.invalid("Missing new native output destination")
          }
          throw StudioError.invalid("Stop before inference")
        }
        return [:]
      })
    store.runtime.pythonPath = "/missing/python"
    store.runtime.nativeH3Enabled = true
    store.addClip(); store.editClip { $0.engine = .h3 }
    store.preparedRecipe = "/tmp/h3/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    await store.renderPrepared()
    XCTAssertEqual(store.error, "Stop before inference")
  }

  @MainActor func testInstalledNativeH3Ref2VAStudioLifecycle() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_LIFECYCLE"] else {
      throw XCTSkip("Opt-in installed H3 Ref2VA generation")
    }
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let source = URL(fileURLWithPath: try XCTUnwrap(config["recipe"]))
    var profile = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
    let originalInputs = try XCTUnwrap((profile["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertEqual(originalInputs.count, 2)
    let directory = URL(fileURLWithPath: try XCTUnwrap(config["output"]))
    let profiles = directory.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    var conditioning = profile["conditioning"] as! [String: Any]
    conditioning["inputs"] = []
    profile["conditioning"] = conditioning
    let profileURL = profiles.appendingPathComponent("matched.json")
    try JSONSerialization.data(withJSONObject: profile, options: [.prettyPrinted, .sortedKeys])
      .write(to: profileURL, options: .atomic)
    let generation = profile["config"] as! [String: Any]
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true
    store.runtime.h3WorkerPath = try XCTUnwrap(config["worker"])
    store.runtime.ffmpegPath = try XCTUnwrap(profile["ffmpeg"] as? String)
    var clip = Clip(engine: .h3)
    clip.profileID = profileURL.path
    clip.prompt = profile["prompt"] as! String
    clip.duration = generation["duration_seconds"] as! Double
    clip.generationWidth = generation["width"] as! Int
    clip.generationHeight = generation["height"] as! Int
    clip.seed = generation["seed"] as! Int
    clip.generationSelection = GenerationSelection(task: "ref2va")
    var assets: [MediaAsset] = []
    for input in originalInputs {
      let asset = MediaAsset(name: "Reference", kind: .image, path: input["path"] as! String)
      assets.append(asset)
      clip.attachments.append(Attachment(assetID: asset.id, role: .reference))
    }
    store.project.clips = [clip]
    store.project.assets = assets
    store.selectedClipID = clip.id
    await store.reloadProfiles()
    XCTAssertEqual(store.profiles.count, 1)
    await store.prepareSelected()
    XCTAssertNil(store.error)
    let prepared = URL(fileURLWithPath: try XCTUnwrap(store.preparedRecipe))
    let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: prepared)) as! [String: Any]
    let inputs = try XCTUnwrap((recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertEqual(inputs.map { $0["path"] as? String }, originalInputs.map { $0["path"] as? String })
    XCTAssertEqual(inputs.map { $0["sha256"] as? String }, originalInputs.map { $0["sha256"] as? String })
    XCTAssertTrue(store.preparedReport.contains("swift-mlx"))
    var previews = Set<Int>()
    let observation = store.bridge.$livePreview.sink { event in
      if let revision = event?.previewRevision { previews.insert(revision) }
    }
    defer { observation.cancel() }
    let started = Date()
    await store.renderPrepared()
    XCTAssertNil(store.error)
    let version = try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertEqual(store.selectedClip?.sourcePath, version.path)
    XCTAssertEqual(store.selectedClip?.renderedSignature, store.signature(for: try XCTUnwrap(store.selectedClip)))
    XCTAssertEqual(store.project.assets.last?.path, version.path)
    XCTAssertGreaterThan(previews.count, 0)
    let savedProject = directory.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: savedProject)
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    reopened.load(savedProject)
    XCTAssertEqual(reopened.selectedClip?.sourcePath, version.path)
    XCTAssertEqual(reopened.selectedClip?.versions.last?.path, version.path)
    let movie = try await StudioStore.inspectNativeMovie(version.path)
    XCTAssertEqual(movie["width"] as? Int, clip.generationWidth)
    XCTAssertEqual(movie["height"] as? Int, clip.generationHeight)
    let evidence: [String: Any] = ["video": version.path, "recipe": prepared.path,
      "renderAndAcceptanceSeconds": Date().timeIntervalSince(started),
      "pythonPath": store.runtime.pythonPath, "decodedPreviewCount": previews.count,
      "duration": movie["duration"] as? Double ?? 0]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: directory.appendingPathComponent("studio-qualification.json"), options: .atomic)
  }

  @MainActor func testInstalledNativeH3FL2VAStudioLifecycle() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_FL2VA_LIFECYCLE"] else {
      throw XCTSkip("Opt-in installed H3 FL2VA generation")
    }
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let source = URL(fileURLWithPath: try XCTUnwrap(config["recipe"]))
    var profile = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
    let originalInputs = try XCTUnwrap((profile["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertEqual(originalInputs.map { $0["role"] as? String }, ["first", "last"])
    let directory = URL(fileURLWithPath: try XCTUnwrap(config["output"]))
    let profiles = directory.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    var conditioning = profile["conditioning"] as! [String: Any]
    conditioning["inputs"] = []
    profile["conditioning"] = conditioning
    let profileURL = profiles.appendingPathComponent("matched-fl2va.json")
    try JSONSerialization.data(withJSONObject: profile, options: [.prettyPrinted, .sortedKeys])
      .write(to: profileURL, options: .atomic)
    let generation = profile["config"] as! [String: Any]
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true
    store.runtime.h3WorkerPath = try XCTUnwrap(config["worker"])
    store.runtime.ffmpegPath = try XCTUnwrap(profile["ffmpeg"] as? String)
    var clip = Clip(engine: .h3)
    clip.profileID = profileURL.path
    clip.prompt = profile["prompt"] as! String
    clip.duration = generation["duration_seconds"] as! Double
    clip.generationWidth = generation["width"] as! Int
    clip.generationHeight = generation["height"] as! Int
    clip.seed = generation["seed"] as! Int
    clip.generationSelection = GenerationSelection(task: "fflf")
    var assets: [MediaAsset] = []
    for (index, input) in originalInputs.enumerated() {
      let asset = MediaAsset(name: index == 0 ? "First" : "Last", kind: .image,
        path: input["path"] as! String)
      assets.append(asset)
      clip.attachments.append(Attachment(assetID: asset.id, role: index == 0 ? .first : .last))
    }
    store.project.clips = [clip]
    store.project.assets = assets
    store.selectedClipID = clip.id
    await store.reloadProfiles()
    XCTAssertEqual(store.profiles.count, 1)
    await store.prepareSelected()
    XCTAssertNil(store.error)
    let prepared = URL(fileURLWithPath: try XCTUnwrap(store.preparedRecipe))
    let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: prepared)) as! [String: Any]
    let inputs = try XCTUnwrap((recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertEqual(inputs.map { $0["path"] as? String }, originalInputs.map { $0["path"] as? String })
    XCTAssertEqual(inputs.map { $0["sha256"] as? String }, originalInputs.map { $0["sha256"] as? String })
    var previews = Set<Int>()
    let observation = store.bridge.$livePreview.sink { event in
      if let revision = event?.previewRevision { previews.insert(revision) }
    }
    defer { observation.cancel() }
    let started = Date()
    await store.renderPrepared()
    XCTAssertNil(store.error)
    let version = try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertGreaterThan(previews.count, 0)
    let savedProject = directory.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: savedProject)
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    reopened.load(savedProject)
    XCTAssertEqual(reopened.selectedClip?.sourcePath, version.path)
    XCTAssertEqual(reopened.selectedClip?.versions.last?.path, version.path)
    let movie = try await StudioStore.inspectNativeMovie(version.path)
    XCTAssertEqual(movie["width"] as? Int, clip.generationWidth)
    XCTAssertEqual(movie["height"] as? Int, clip.generationHeight)
    let evidence: [String: Any] = ["video": version.path, "recipe": prepared.path,
      "renderAndAcceptanceSeconds": Date().timeIntervalSince(started),
      "pythonPath": store.runtime.pythonPath, "decodedPreviewCount": previews.count]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: directory.appendingPathComponent("studio-qualification.json"), options: .atomic)
  }

  @MainActor func testNativeH3RejectsAudioDriverBeforePythonPreparation() async throws {
    let fake = SuspendedBridge("unused")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: fake.invoke)
    store.runtime.pythonPath = "/unavailable/python"; store.runtime.nativeH3Enabled = true
    store.addClip(); store.editClip {
      $0.engine = .h3; $0.prompt = "A quiet lake."
      $0.audioDriverSelection = AudioDriverSelection(mode: .voice)
    }
    await store.prepareSelected()
    XCTAssertFalse(fake.calls.contains { $0.0 == "audio-driver" })
    XCTAssertTrue(fake.calls.contains { $0.0 == "h3-native-describe" })
  }
  @MainActor func testNativeOnlyMovieDoesNotDemandPythonSetup() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.runtime.pythonPath = "/unavailable/python"; store.runtime.nativeLTX25Enabled = true
    store.addClip(); store.editClip { $0.engine = .ltx25 }
    XCTAssertFalse(store.actionItems.contains { $0.id == "runtime" })
    store.runtime.nativeLTX25Enabled = false
    XCTAssertTrue(store.actionItems.contains { $0.id == "runtime" })
    store.runtime.nativeLTX25Enabled = true; store.editClip { $0.engine = .h3 }
    XCTAssertTrue(store.actionItems.contains { $0.id == "runtime" })
  }

  @MainActor func testInstalledNativeLTXLifecycleWithPythonUnavailable() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_LTX_LIFECYCLE"] else {
      throw XCTSkip("Opt-in full installed native LTX generation")
    }
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let recipeURL = URL(fileURLWithPath: try XCTUnwrap(config["recipe"]))
    let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: recipeURL)) as! [String: Any]
    let generation = recipe["config"] as! [String: Any]
    let root = URL(fileURLWithPath: try XCTUnwrap(config["output"]))
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: recipeURL, to: profiles.appendingPathComponent("matched.json"))
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeLTX25Enabled = true; store.runtime.ltx25WorkerPath = config["worker"]
    store.runtime.ffmpegPath = recipe["ffmpeg"] as? String ?? "/opt/homebrew/bin/ffmpeg"
    var clip = Clip(); clip.prompt = recipe["prompt"] as! String
    clip.duration = generation["duration_seconds"] as! Double; clip.seed = generation["seed"] as! Int
    clip.generationWidth = generation["width"] as! Int; clip.generationHeight = generation["height"] as! Int
    var assets: [MediaAsset] = []
    let inputs = (recipe["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]] ?? []
    for input in inputs {
      let asset = MediaAsset(name: "Endpoint", kind: .image, path: input["path"] as! String)
      assets.append(asset)
      let last = input["frame_index"] as? String == "last" || (input["frame_index"] as? Int ?? 0) > 0
      var attachment = Attachment(assetID: asset.id, role: last ? .last : .first)
      attachment.strength = input["strength"] as? Double ?? 1; clip.attachments.append(attachment)
    }
    clip.generationSelection = GenerationSelection(task: inputs.isEmpty ? "t2v" : inputs.count == 1 ? "i2v" : "fflf")
    if let requestPath=config["editorRequest"] {
      let original=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:requestPath))) as! [String:Any]
      let projectData=try JSONSerialization.data(withJSONObject:original["project"]!)
      store.project=try JSONDecoder().decode(StudioProject.self,from:projectData)
      let selectedID=try XCTUnwrap(UUID(uuidString:original["clipID"] as! String))
      clip=try XCTUnwrap(store.project.clips.first { $0.id == selectedID })
      store.globalAssets=try JSONDecoder().decode([MediaAsset].self,from:JSONSerialization.data(withJSONObject:original["globalAssets"] ?? []))
    } else {
      store.project.clips = [clip]; store.project.assets = assets
    }
    store.selectedClipID = clip.id
    let untouchedClips=store.project.clips.filter { $0.id != clip.id }
    await store.reloadProfiles()
    XCTAssertEqual(store.profiles.count, 1)
    await store.describeGeneration()
    XCTAssertNil(store.validationErrors[clip.id])
    await store.prepareSelected()
    XCTAssertNil(store.error)
    let prepared = try XCTUnwrap(store.preparedRecipe)
    XCTAssertTrue(store.preparedReport.contains("swift-mlx"))
    let preparedValue = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: prepared))) as! [String: Any]
    let preparedConfig = preparedValue["config"] as! [String: Any]
    XCTAssertEqual(preparedConfig["seed"] as? Int, clip.seed)
    XCTAssertEqual(try XCTUnwrap(preparedConfig["duration_seconds"] as? Double), clip.duration, accuracy: 0.000001)
    var previewRevisions=Set<Int>()
    let previewObserver=store.bridge.$livePreview.sink { event in
      if let revision=event?.previewRevision, let path=event?.previewPath {
        previewRevisions.insert(revision)
        try? FileManager.default.copyItem(at:URL(fileURLWithPath:path),to:root.appendingPathComponent("preview-\(revision).png"))
      }
    }
    defer { previewObserver.cancel() }
    let started = Date()
    await store.renderPrepared()
    XCTAssertNil(store.error)
    XCTAssertGreaterThan(previewRevisions.count,0,"Installed Swift render must deliver decoded previews")
    try store.bridge.log.write(to:root.appendingPathComponent("worker-progress.log"),atomically:true,encoding:.utf8)
    let version = try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertEqual(store.selectedClip?.sourcePath, version.path)
    XCTAssertEqual(store.selectedClip?.duration, clip.duration)
    XCTAssertEqual(store.project.clips.filter { $0.id != clip.id },untouchedClips)
    let savedProject = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: savedProject)
    let reopened = StudioStore(dataDirectory: root, restoreSession: false)
    reopened.load(savedProject)
    XCTAssertEqual(reopened.selectedClip?.sourcePath, version.path)
    XCTAssertEqual(reopened.selectedClip?.versions.last?.path, version.path)
    let movie = try await StudioStore.inspectNativeMovie(version.path)
    XCTAssertEqual(movie["width"] as? Int, clip.generationWidth)
    XCTAssertEqual(movie["height"] as? Int, clip.generationHeight)
    let hash = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: version.path))).map { String(format: "%02x", $0) }.joined()
    if let expected = config["expectedSHA256"] { XCTAssertEqual(hash, expected) }
    let evidence: [String: Any] = ["video": version.path, "sha256": hash, "recipe": prepared,
      "renderAndAcceptanceSeconds": Date().timeIntervalSince(started), "pythonPath": store.runtime.pythonPath,
      "root": store.runtime.root, "worker": store.runtime.ltx25WorkerPath!, "width": clip.generationWidth,
      "height": clip.generationHeight, "duration": clip.duration, "seed": clip.seed,
      "decodedPreviewCount":previewRevisions.count,"untouchedOtherClips":untouchedClips.count]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("qualification.json"))
  }

  @MainActor func testInstalledNativeLTXA2VStudioLifecycle() async throws {
    guard let manifest=ProcessInfo.processInfo.environment["WEETODD_NATIVE_LTX_A2V_LIFECYCLE"] else {
      throw XCTSkip("Opt-in installed LTX 2.5 A2V Studio lifecycle")
    }
    let options=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:manifest))) as! [String:String]
    let source=URL(fileURLWithPath:try XCTUnwrap(options["recipe"]))
    let model=try JSONSerialization.jsonObject(with:Data(contentsOf:source)) as! [String:Any]
    let generation=model["config"] as! [String:Any]
    let conditioning=model["conditioning"] as! [String:Any]
    let inputs=try XCTUnwrap(conditioning["inputs"] as? [[String:Any]])
    XCTAssertTrue((1...2).contains(inputs.count))
    let input=try XCTUnwrap(inputs.first { $0["role"] as? String == "audio_driver" })
    let opening=inputs.first { $0["role"] as? String == "keyframe" }
    XCTAssertEqual(conditioning["task"] as? String,"a2v")
    let root=URL(fileURLWithPath:try XCTUnwrap(options["output"]))
    let profiles=root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at:profiles,withIntermediateDirectories:true)
    let profile=profiles.appendingPathComponent("a2v.json")
    try FileManager.default.copyItem(at:source,to:profile)
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.runtime=RuntimeSettings(root:"/unavailable",pythonPath:"/unavailable/python",profilesDirectory:profiles.path)
    store.runtime.nativeLTX25Enabled=true
    store.runtime.ltx25WorkerPath=try XCTUnwrap(options["worker"])
    store.runtime.ffmpegPath=try XCTUnwrap(model["ffmpeg"] as? String)
    var clip=Clip(engine:.ltx25)
    clip.profileID=profile.path;clip.prompt=try XCTUnwrap(model["prompt"] as? String)
    clip.duration=try XCTUnwrap(generation["duration_seconds"] as? Double)
    clip.generationWidth=try XCTUnwrap(generation["width"] as? Int)
    clip.generationHeight=try XCTUnwrap(generation["height"] as? Int)
    clip.seed=try XCTUnwrap(generation["seed"] as? Int)
    clip.generationSelection=GenerationSelection(task:"a2v")
    var asset=MediaAsset(name:"Spoken source",kind:.audio,path:try XCTUnwrap(input["path"] as? String))
    let audioStart=try XCTUnwrap(input["source_start_seconds"] as? Double)
    let audioDuration=try XCTUnwrap(input["source_duration_seconds"] as? Double)
    asset.duration=audioStart+audioDuration
    var attachment=Attachment(assetID:asset.id,role:.audioDriver)
    attachment.audioSourceStart=audioStart
    attachment.audioSourceDuration=audioDuration
    clip.attachments=[attachment]
    var assets=[asset]
    if let opening {
      XCTAssertEqual(opening["frame_index"] as? Int,0)
      let image=MediaAsset(name:"Opening frame",kind:.image,path:try XCTUnwrap(opening["path"] as? String))
      assets.append(image)
      clip.attachments.append(Attachment(assetID:image.id,role:.first))
    }
    store.project.clips=[clip];store.project.assets=assets;store.selectedClipID=clip.id
    await store.reloadProfiles();XCTAssertEqual(store.profiles.count,1)
    await store.describeGeneration();XCTAssertNil(store.validationErrors[clip.id])
    await store.prepareSelected();XCTAssertNil(store.error)
    let prepared=URL(fileURLWithPath:try XCTUnwrap(store.preparedRecipe))
    let preparedModel=try JSONSerialization.jsonObject(with:Data(contentsOf:prepared)) as! [String:Any]
    let preparedInputs=try XCTUnwrap((preparedModel["conditioning"] as? [String:Any])?["inputs"] as? [[String:Any]])
    XCTAssertEqual(preparedInputs.count,inputs.count)
    let preparedInput=try XCTUnwrap(preparedInputs.first { $0["role"] as? String == "audio_driver" })
    XCTAssertEqual(preparedInput["path"] as? String,asset.path)
    XCTAssertEqual(preparedInput["source_start_seconds"] as? Double,attachment.audioSourceStart)
    XCTAssertEqual(preparedInput["source_duration_seconds"] as? Double,attachment.audioSourceDuration)
    if let opening {
      let first=try XCTUnwrap(preparedInputs.first { $0["role"] as? String == "keyframe" })
      XCTAssertEqual(first["path"] as? String,opening["path"] as? String)
      XCTAssertEqual(first["frame_index"] as? Int,0)
    }
    var previewRevisions=Set<Int>()
    let observer=store.bridge.$livePreview.sink { if let revision=$0?.previewRevision { previewRevisions.insert(revision) } }
    defer { observer.cancel() }
    await store.renderPrepared();XCTAssertNil(store.error)
    XCTAssertGreaterThan(previewRevisions.count,0)
    let version=try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertEqual(store.selectedClip?.sourcePath,version.path)
    let movie=try await StudioStore.inspectNativeMovie(version.path)
    XCTAssertEqual(movie["width"] as? Int,clip.generationWidth)
    XCTAssertEqual(movie["height"] as? Int,clip.generationHeight)
    let audio=URL(fileURLWithPath:version.path).deletingLastPathComponent().appendingPathComponent("audio.wav")
    let digest=SHA256.hash(data:try Data(contentsOf:audio)).map { String(format:"%02x",$0) }.joined()
    XCTAssertEqual(digest,try XCTUnwrap(options["expectedAudioSHA256"]))
    let saved=root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project,to:saved)
    let reopened=StudioStore(dataDirectory:root,restoreSession:false);reopened.load(saved)
    XCTAssertEqual(reopened.selectedClip?.sourcePath,version.path)
    XCTAssertEqual(reopened.selectedClip?.versions.last?.path,version.path)
    let evidence:[String:Any]=["video":version.path,"audio":audio.path,"audioSHA256":digest,
      "preparedRecipe":prepared.path,"decodedPreviewCount":previewRevisions.count,
      "pythonPath":store.runtime.pythonPath,"width":clip.generationWidth,"height":clip.generationHeight]
    try JSONSerialization.data(withJSONObject:evidence,options:[.prettyPrinted,.sortedKeys])
      .write(to:root.appendingPathComponent("a2v-studio-qualification.json"),options:.atomic)
  }

  @MainActor func testInstalledNativeMovieResultAndLateEditHandling() async throws {
    guard let movie=ProcessInfo.processInfo.environment["WEETODD_NATIVE_LTX_TEST_MOVIE"] else {
      throw XCTSkip("Optional installed native audiovisual output")
    }
    let info=try await StudioStore.inspectNativeMovie(movie)
    XCTAssertEqual(info["width"] as? Int,1344);XCTAssertEqual(info["height"] as? Int,768)
    XCTAssertEqual(info["fps"] as? Double,24)
    for editDuringRender in [false,true] {
      let fake=SuspendedBridge("ltx-native-render")
      let store=StudioStore(dataDirectory:try temporaryDirectory(),restoreSession:false,invocation:fake.invoke)
      store.runtime.nativeLTX25Enabled=true
      store.addClip();store.editClip { $0.engine = .ltx25;$0.duration=88.0/24;$0.prompt="Original" }
      store.preparedRecipe="/tmp/native/prepared/recipe.json"
      store.preparedFingerprint=store.signature(for:store.selectedClip!)
      let entered=expectation(description:"Native render result")
      fake.entered = { entered.fulfill() }
      let task=Task { await store.renderPrepared() }
      await fulfillment(of:[entered],timeout:2)
      if editDuringRender { store.editClip { $0.prompt="Edited" } }
      fake.continuation?.resume(returning:["video":movie,"nativeRuntime":"swift-mlx"])
      await task.value
      XCTAssertNil(store.error)
      XCTAssertEqual(store.selectedClip?.versions.last?.path,movie)
      XCTAssertEqual(store.selectedClip?.sourcePath,editDuringRender ? "":movie)
      XCTAssertEqual(store.selectedClip?.duration,88.0/24)
      XCTAssertFalse(fake.calls.contains { $0.0 == "inspect" || $0.0 == "render" })
    }
  }

  @MainActor func testNativePreviewCannotCrossDocumentSession() async throws {
    let fake=SuspendedBridge("ltx-native-render")
    let store=StudioStore(dataDirectory:try temporaryDirectory(),restoreSession:false,invocation:fake.invoke)
    store.runtime.nativeLTX25Enabled=true
    store.addClip();store.editClip { $0.engine = .ltx25 }
    store.preparedRecipe="/tmp/native/prepared/recipe.json"
    store.preparedFingerprint=store.signature(for:store.selectedClip!)
    let entered=expectation(description:"Native render")
    fake.entered = { entered.fulfill() }
    let task=Task { await store.renderPrepared() }
    await fulfillment(of:[entered],timeout:2)
    var stream=BridgeProgressStream()
    let event=try XCTUnwrap(stream.append(Data("{\"event\":\"progress\",\"message\":\"Frame\",\"previewPath\":\"/tmp/frame.png\",\"previewRevision\":1}\n".utf8)).first)
    store.bridge.livePreview=event
    XCTAssertNotNil(store.nativeRenderPreview)
    store.newProject()
    store.bridge.livePreview=event // Later worker events must also remain hidden.
    XCTAssertNil(store.nativeRenderPreview)
    fake.continuation?.resume(throwing:CancellationError())
    await task.value
  }

  @MainActor func testNativeLTXPreflightMustPassBeforeRecipeCanRender() async throws {
    let fake=SuspendedBridge("ltx-native-preflight")
    let store=StudioStore(dataDirectory:try temporaryDirectory(),restoreSession:false,invocation:fake.invoke)
    store.runtime.nativeLTX25Enabled=true
    store.addClip();store.editClip { $0.engine = .ltx25 }
    let entered=expectation(description:"Native preflight")
    fake.entered = { entered.fulfill() }
    let task=Task { await store.prepareSelected() }
    await fulfillment(of:[entered],timeout:2)
    XCTAssertNil(store.preparedRecipe)
    fake.continuation?.resume(throwing:StudioError.invalid("Unsupported native mode"))
    await task.value
    XCTAssertNil(store.preparedRecipe)
    XCTAssertEqual(store.error,"Unsupported native mode")
    XCTAssertTrue(fake.calls.contains { $0.0 == "ltx-native-describe" })
    XCTAssertTrue(fake.calls.contains { $0.0 == "ltx-native-prepare" })
    XCTAssertFalse(fake.calls.contains { $0.0 == "describe-generation" || $0.0 == "prepare" })
    XCTAssertFalse(fake.calls.contains { $0.0 == "render" || $0.0 == "ltx-native-render" })
  }

  @MainActor func testPreparedReferenceAttachesAtomicallyAndUndoRestoresShot() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: { command, _, _, _ in
        XCTAssertEqual(command, "prepare-reference")
        return ["path": "/tmp/reference-sheet.png", "kind": "image", "width": 1152, "height": 480]
      })
    store.addClip()
    store.editClip { $0.engine = .ltx25 }
    let source = MediaAsset(name: "Story", kind: .video, path: "/tmp/story.mov")
    store.change { $0.assets.append(source) }
    let before = store.project
    let action = try XCTUnwrap(store.selectedClip?.referenceActions(for: source).first)
    await store.useReference(source, action: action)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.inferredTask, "control")
    XCTAssertEqual(store.selectedClip?.attachments.first?.controlType, "ingredients_reference_sheet")
    XCTAssertEqual(store.project.assets.count, before.assets.count + 1)
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testLateReferencePreparationDoesNotAttachAfterModelChange() async throws {
    let fake = SuspendedBridge("prepare-reference")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.engine = .ltx25 }
    let source = MediaAsset(name: "Story", kind: .video, path: "/tmp/story.mov")
    store.change { $0.assets.append(source) }
    let action = try XCTUnwrap(store.selectedClip?.referenceActions(for: source).first)
    let entered = expectation(description: "reference preparation suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.useReference(source, action: action) }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.engine = .h3 }
    fake.continuation?.resume(returning: ["path": "/tmp/reference-sheet.png", "kind": "image"])
    await task.value
    XCTAssertTrue(store.selectedClip?.attachments.isEmpty == true)
    XCTAssertEqual(store.project.assets.last?.scope, .project)
    XCTAssertEqual(store.project.assets.last?.path, "/tmp/reference-sheet.png")
  }

  func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    return directory
  }

  @MainActor func testOpenSeparatesUndoAndPreservesDepartingDirtyDocument() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Unsaved A" }
    let a = store.project
    var b = StudioProject(); b.name = "Saved B"
    let url = directory.appendingPathComponent("B.weetodd")
    try ProjectStorage.write(b, to: url)
    store.load(url)
    XCTAssertFalse(store.canUndo)
    store.undo(); store.save()
    XCTAssertEqual(try ProjectStorage.read(url).name, "Saved B")
    let recovered = (FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? [])
      .filter { $0.pathExtension == "weetodd" }.compactMap { try? ProjectStorage.read($0) }
    XCTAssertTrue(recovered.contains(a), "Departed dirty content must be recoverable even before debounce fires")
  }

  @MainActor func testNewProjectClearsUndoAndSelections() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.selectedAssetID = UUID(); store.selectedAudioID = UUID()
    store.newProject()
    XCTAssertFalse(store.canUndo)
    XCTAssertNil(store.selectedAssetID); XCTAssertNil(store.selectedAudioID)
    store.undo()
    XCTAssertTrue(store.project.clips.isEmpty)
  }

  @MainActor func testSuspendedPreflightNeverPreparesAnotherSelection() async throws {
    let fake = SuspendedBridge("describe-generation")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.addClip()
    let first = store.project.clips[0].id
    let entered = expectation(description: "description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    store.select(first)
    fake.continuation?.resume(returning: ["fingerprint": "resolved"])
    await task.value
    XCTAssertFalse(fake.calls.contains { $0.0 == "prepare" })
    XCTAssertNil(store.preparedRecipe)
    XCTAssertTrue(store.validationErrors.isEmpty)
  }

  @MainActor func testLateNativeRenderDoesNotPromoteOverEdits() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.prompt = "Original"; $0.sourcePath = "/tmp/original.mov" }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.prompt = "Revised"; $0.sourceIn = 2; $0.duration = 3 }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/original.mov")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2)
    XCTAssertEqual(store.selectedClip?.duration, 3)
    XCTAssertEqual(store.selectedClip?.versions.last?.path, "/tmp/completed.mov")
  }

  @MainActor func testLateNativeRenderCannotAttachToReopenedCopy() async throws {
    let directory = try temporaryDirectory()
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let url = directory.appendingPathComponent("copy.weetodd")
    try ProjectStorage.write(store.project, to: url)
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.load(url)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertTrue(store.selectedClip?.versions.isEmpty == true)
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
    XCTAssertTrue(store.error?.contains("/tmp/completed.mov") == true)
  }
  @MainActor func testMovieTransportEndUsesFullMovieDuration() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.duration = 3 }
    store.addClip(); store.editClip { $0.duration = 7 }
    store.select(store.project.clips[0].id)
    store.previewMode = "Movie"
    store.seekToEnd()
    XCTAssertEqual(store.effectivePreviewDuration, 10)
    XCTAssertEqual(store.playhead, 10)
    store.previewMode = "Timeline"
    store.seekToEnd()
    XCTAssertEqual(store.playhead, 10)
  }

  @MainActor func testMoviePreviewAcceptsUnchangedMultiShotProject() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: { _, _, _, _ in [:] })
    for index in 1...6 {
      var clip = Clip(name: "Shot \(index)", engine: .h3)
      clip.selectLocalModel(.ltx25)
      clip.duration = 5
      clip.sourcePath = "/tmp/shot-\(index).mp4"
      clip.continuity = ClipContinuity(mode: index == 1 ? "independent" : "frame")
      store.project.clips.append(clip)
    }
    let snapshot = store.project
    await store.previewMovie()
    XCTAssertEqual(store.project, snapshot)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.previewMode, "Movie")
    XCTAssertEqual(store.effectivePreviewDuration, 30)
  }

  @MainActor func testMoviePreviewRejectsEditsWhileRendering() async throws {
    let fake = SuspendedBridge("preview")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false,
      invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "preview suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.previewMovie() }
    await fulfillment(of: [entered], timeout: 2)
    store.editClip { $0.duration = 7 }
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertNotEqual(store.previewMode, "Movie")
    XCTAssertNotNil(store.error)
  }

  @MainActor func testMoviePreviewRejectsReopenedIdenticalDocument() async throws {
    let directory = try temporaryDirectory()
    let fake = SuspendedBridge("preview")
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let url = directory.appendingPathComponent("same-project.weetodd")
    try ProjectStorage.write(store.project, to: url)
    let entered = expectation(description: "preview suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.previewMovie() }
    await fulfillment(of: [entered], timeout: 2)
    store.load(url)
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertNotEqual(store.previewMode, "Movie")
    XCTAssertNotNil(store.error)
  }

  @MainActor func testFailedRecoveryKeepsCurrentDocumentAndURL() throws {
    let directory = try temporaryDirectory()
    try Data("occupied".utf8).write(to: directory.appendingPathComponent("Recovery"))
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let originalURL = directory.appendingPathComponent("original.weetodd")
    store.projectURL = originalURL
    store.change { $0.name = "Must survive" }
    let before = store.project
    store.newProject()
    XCTAssertEqual(store.project, before)
    XCTAssertEqual(store.projectURL, originalURL)
    XCTAssertTrue(store.dirty)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testDeletedPreflightDestinationReturnsSafelyAndBlocksDuplicateRequest() async throws {
    let fake = SuspendedBridge("describe-generation")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    XCTAssertTrue(store.operationBusy)
    await store.prepareSelected()
    store.deleteClip()
    fake.continuation?.resume(returning: ["fingerprint": "resolved"])
    await task.value
    XCTAssertEqual(fake.calls.map { $0.0 }, ["describe-generation"])
    XCTAssertNil(store.preparedRecipe)
    XCTAssertFalse(store.operationBusy)
  }

  @MainActor func testDeletedRenderDestinationReportsCompletedOutput() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.deleteClip()
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.error?.contains("/tmp/completed.mov") == true)
  }

  @MainActor func testPreflightCanStartWhileBackgroundDescriptionIsPending() async throws {
    let fake = SuspendedBridge("describe-generation"); fake.suspendLimit = 1
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    let entered = expectation(description: "background description suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.describeGeneration() }
    await fulfillment(of: [entered], timeout: 2)
    await store.prepareSelected()
    XCTAssertNotNil(store.preparedRecipe)
    fake.continuation?.resume(returning: [:])
    await task.value
  }

  @MainActor func testSuccessfulAppendRenderPersistsUsableSegment() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.editClip { $0.extensionSource = "/tmp/context.mov"; $0.extensionDirection = "after" }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourceIn, 8)
    XCTAssertEqual(store.selectedClip?.duration, 4)
    XCTAssertEqual(store.selectedClip?.versions.last?.usableSourceIn, 8)
    XCTAssertEqual(store.selectedClip?.versions.last?.usableDuration, 4)
  }

  @MainActor func testNativeDurationRoundingKeepsAcceptedRenderCurrent() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(.h3)
    store.editClip { $0.duration = 5.17; $0.prompt = "A robot lifts a lantern." }
    await store.prepareSelected()
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov",
      "usable_source_in": 0.0, "usable_duration": 124.0 / 24])
    await task.value
    await store.describeGeneration()
    let clip = try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(clip.duration, 124.0 / 24)
    XCTAssertEqual(clip.renderedSignature, store.signature(for: clip))
  }

  @MainActor func testDurationRefreshCannotMarkAReplacedTakeCurrent() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(.h3)
    store.editClip { $0.duration = 5.17; $0.prompt = "A robot lifts a lantern." }
    await store.prepareSelected()
    let rendered = expectation(description: "render suspended")
    fake.entered = { rendered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [rendered], timeout: 2)
    let refreshed = expectation(description: "duration refresh suspended")
    fake.command = "describe-generation"
    fake.entered = { refreshed.fulfill() }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov",
      "usable_source_in": 0.0, "usable_duration": 124.0 / 24])
    await fulfillment(of: [refreshed], timeout: 2)
    store.editClip { $0.sourcePath = "/tmp/older-take.mov"; $0.renderedSignature = "" }
    fake.continuation?.resume(returning: [:])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/older-take.mov")
    XCTAssertEqual(store.selectedClip?.renderedSignature, "")
  }

  @MainActor func testLateNativeRenderUsesSubmittedRuntimeToInspectOutput() async throws {
    let fake = SuspendedBridge("render"); fake.inspectRuntimeRoot = "/runtime/submitted"
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.runtime.root = "/runtime/submitted"
    store.addClip()
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.runtime.root = "/runtime/next"
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(store.selectedClip?.versions.last?.path, "/tmp/completed.mov")
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
  }

  @MainActor func testOpeningProjectImmediatelyReplacesActiveRestoreSnapshot() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Dirty A" }
    let departed = store.project
    try ProjectStorage.write(departed, to: directory.appendingPathComponent("Autosave.weetodd"))
    var opened = StudioProject(); opened.name = "Saved B"
    let url = directory.appendingPathComponent("B.weetodd")
    try ProjectStorage.write(opened, to: url)
    store.load(url)
    let restarted = StudioStore(dataDirectory: directory, restoreSession: false)
    restarted.restoreAutosavedProject()
    XCTAssertEqual(restarted.project, opened)
    let recovery = try FileManager.default.contentsOfDirectory(at: directory.appendingPathComponent("Recovery"), includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "weetodd" }.map { try ProjectStorage.read($0) }
    XCTAssertTrue(recovery.contains(departed))
  }

  @MainActor func testNewMovieImmediatelyReplacesActiveRestoreSnapshot() throws {
    let directory = try temporaryDirectory()
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.change { $0.name = "Dirty A" }
    try ProjectStorage.write(store.project, to: directory.appendingPathComponent("Autosave.weetodd"))
    store.newProject()
    let blank = store.project
    let restarted = StudioStore(dataDirectory: directory, restoreSession: false)
    restarted.restoreAutosavedProject()
    XCTAssertEqual(restarted.project, blank)
  }

  @MainActor func testGenerateUsesAlreadyReviewedPreparedRecipe() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/reviewed/recipe.json"
    store.preparedPrompt = "The reviewed prompt."
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertFalse(fake.calls.contains { ["describe-generation", "prepare"].contains($0.0) })
    XCTAssertEqual(store.selectedClip?.versions.last?.recipePath, "/tmp/reviewed/recipe.json")
    XCTAssertEqual(store.selectedClip?.versions.last?.prompt, "The reviewed prompt.")
  }

  @MainActor func testGenerateRepreparesStaleRecipe() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip()
    store.preparedRecipe = "/tmp/reviewed/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    store.runtime.root = "/runtime/changed"
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 2)
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov"])
    await task.value
    XCTAssertEqual(fake.calls.prefix(3).map { $0.0 }, ["describe-generation", "prepare", "render"])
    XCTAssertEqual(store.selectedClip?.versions.last?.recipePath, "/tmp/prepared/recipe.json")
  }

}

extension StudioReliabilityTests {
  @MainActor func testPredecessorTrimInvalidatesSuspendedContinuityPreparation() async throws {
    let fake = SuspendedBridge("prepare")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip { $0.continuity = ClipContinuity(mode: "frame") }
    let oldKey = store.generationRequestKey(for: store.selectedClip!)
    let entered = expectation(description: "prepare suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.prepareSelected() }
    await fulfillment(of: [entered], timeout: 2)
    store.change { $0.clips[0].sourceIn = 1 }
    XCTAssertNotEqual(oldKey, store.generationRequestKey(for: store.selectedClip!))
    fake.continuation?.resume(returning: ["recipePath": "/tmp/prepared/recipe.json", "prompt": "prompt", "report": [:]])
    await task.value
    XCTAssertNil(store.preparedRecipe)
  }

  @MainActor func testPredecessorTakeChangeRetainsLateContinuityRenderAsInactiveVersion() async throws {
    let fake = SuspendedBridge("render")
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: fake.invoke)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip { $0.continuity = ClipContinuity(mode: "frame") }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    let entered = expectation(description: "render suspended")
    fake.entered = { entered.fulfill() }
    let task = Task { await store.renderPrepared() }
    await fulfillment(of: [entered], timeout: 2)
    store.change { $0.clips[0].sourcePath = "/tmp/new-accepted.mov" }
    fake.continuation?.resume(returning: ["video": "/tmp/completed.mov", "usable_source_in": 2.0,
      "usable_duration": 4.0, "continuation_artifact": ["manifest": "/tmp/context/manifest.json", "manifest_sha256": "a", "payload_sha256": "b"]])
    await task.value
    XCTAssertEqual(store.selectedClip?.sourcePath, "")
    XCTAssertEqual(store.selectedClip?.duration, 5)
    let version = try XCTUnwrap(store.selectedClip?.versions.last)
    XCTAssertEqual(version.path, "/tmp/completed.mov")
    XCTAssertEqual(version.usableSourceIn, 2)
    XCTAssertEqual(version.usableDuration, 4)
    XCTAssertEqual(version.continuationArtifact?.manifest, "/tmp/context/manifest.json")
  }
}

extension StudioReliabilityTests {
  @MainActor func testFrameContinuityDoesNotRequireReplacedStoredFirstAttachment() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.sourcePath = "/tmp/source.mov" }
    store.addClip(); store.editClip {
      $0.continuity = ClipContinuity(mode: "frame")
      $0.attachments = [Attachment(assetID: UUID(), role: .first)]
    }
    XCTAssertFalse(store.issues(for: store.selectedClip!).contains("Relink a missing attachment"))
    XCTAssertEqual(store.selectedClip?.attachments.count, 1)
  }
}

extension StudioReliabilityTests {
  @MainActor func testDrawThingsSignaturePreservesLegacyPartsWithoutContinuity() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    store.addClip(); store.editClip { $0.engine = .drawThings }
    let clip = try XCTUnwrap(store.selectedClip)
    let parts = [clip.generationFingerprint, "generationFPS:\(clip.settings(in: store.project).fps)"]
    let expected = SHA256.hash(data: Data(parts.joined(separator: "\n").utf8))
      .map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(store.signature(for: clip), expected)
  }
}
