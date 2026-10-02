import CryptoKit
import AVFoundation
import Foundation
import XCTest
@testable import StudioCore

final class NativeH3PreparationTests: XCTestCase {
  func testExtensionPromptRetainsStructuredAudioWithoutEmbeddingHeadersInAction() throws {
    var clip=Clip(engine:.h3)
    clip.prompt="integrated_multimodal_description: [Shot 1] The robot lowers its arm.\n\noverall_soundscape: Rain and a metallic whir.\n\nnon_diegetic_music: A quiet cello."
    clip.soundscape="stale room tone";clip.music="stale music"
    let prompt=try NativeH3Preparation.extensionPrompt(clip:clip)
    XCTAssertFalse(prompt.contains("integrated_multimodal_description:"))
    XCTAssertFalse(prompt.contains("stale"))
    XCTAssertTrue(prompt.contains("detailed_description:\n[Shot 1] The robot lowers its arm."))
    XCTAssertTrue(prompt.contains("overall_soundscape:\nRain and a metallic whir."))
    XCTAssertTrue(prompt.contains("non_diegetic_music:\nA quiet cello."))
    XCTAssertEqual(prompt.components(separatedBy:"overall_soundscape:").count,2)
    clip.prompt="subject_definitions: <Video 1>\nsummary: [video continuation] continue"
    XCTAssertThrowsError(try NativeH3Preparation.extensionPrompt(clip:clip))
    clip.prompt="integrated_multimodal_description: walk\nnon_diegetic_music: N/A"
    XCTAssertThrowsError(try NativeH3Preparation.extensionPrompt(clip:clip))
  }
  func testExternalExtensionDescriptionChoosesRef2VAAndRequiresAsyncPreparation() throws {
    let (root,original,runtime)=try fixture(),profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va"
    components["vision_encoder"]="/model/vision.safetensors";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    let movie=root.appendingPathComponent("source.mov");try Data([0]).write(to:movie)
    var project=original;project.clips[0].extensionDirection="after";project.clips[0].extensionSource=movie.path
    let description=try NativeH3Preparation.describe(request:request(project,runtime))
    XCTAssertEqual(description["readinessErrors"] as? [String],[])
    XCTAssertTrue((description["sourcePaths"] as? [String] ?? []).contains(movie.path))
    XCTAssertTrue(((description["generation"] as? [String:Any])?["supportedTasks"] as? [String] ?? []).contains("extension"))
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
    project.clips[0].duration=3
    XCTAssertThrowsError(try NativeH3Preparation.describe(request:request(project,runtime)))
    project.clips[0].duration=5;project.clips[0].extensionDirection="before"
    XCTAssertThrowsError(try NativeH3Preparation.describe(request:request(project,runtime)))
  }
  func testInstalledExtensionPreparationFreezesBoundedTailAndPreservesEditor() async throws {
    guard let source=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_SOURCE"],
      let ffmpeg=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_FFMPEG"] else { throw XCTSkip("Opt-in audiovisual extension preparation") }
    let (root,original,originalRuntime)=try fixture(),profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va"
    components["vision_encoder"]="/model/vision.safetensors";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    var project=original,runtime=originalRuntime;runtime["ffmpegPath"]=ffmpeg
    project.clips[0].extensionDirection="after";project.clips[0].extensionSource=source
    project.clips[0].duration=4;project.clips[0].generationWidth=384;project.clips[0].generationHeight=256
    let body=try request(project,runtime),target=root.appendingPathComponent("prepared")
    let prepared=try await NativeH3Preparation.prepareWithMedia(request:body,destination:target)
    let recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prepared["recipePath"] as! String))) as! [String:Any]
    let contract=recipe["conditioning"] as! [String:Any],inputs=contract["inputs"] as! [[String:Any]]
    XCTAssertEqual(contract["task"] as? String,"extension");XCTAssertEqual(inputs.count,1)
    let file=URL(fileURLWithPath:inputs[0]["path"] as! String),bytes=try Data(contentsOf:file)
    XCTAssertEqual(inputs[0]["sha256"] as? String,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
    let report=prepared["report"] as! [String:Any],dependency=report["continuity"] as! [String:Any]
    let frames=dependency["contextFrames"] as! Int
    XCTAssertLessThanOrEqual(frames,175);XCTAssertEqual((frames-5)%17,0)
    let asset=AVURLAsset(url:file),duration=try await asset.load(.duration).seconds
    XCTAssertEqual(duration,Double(frames)/24,accuracy:0.1)
    let audio=try await asset.loadTracks(withMediaType:.audio);XCTAssertEqual(audio.count,1)
    XCTAssertEqual((try JSONSerialization.jsonObject(with:Data(contentsOf:target.appendingPathComponent("editor-request.json")))) as! NSDictionary,body as NSDictionary)
    XCTAssertTrue((recipe["prompt"] as! String).contains("<Picture 1>: fully_preserved"))
    XCTAssertEqual(project.clips[0].extensionSource,source)
    do { _=try await NativeH3Preparation.prepareWithMedia(request:body,destination:target);XCTFail("Must preserve existing prepared job") }
    catch { XCTAssertEqual(try Data(contentsOf:file),bytes) }
  }
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

  private func writeControlHeader(_ value: [String: Any], to url: URL) throws {
    let header = try JSONSerialization.data(withJSONObject: value)
    var size = UInt64(header.count).littleEndian
    let prefix = withUnsafeBytes(of: &size) { Data($0) }
    try (prefix + header).write(to: url)
  }

  private func controlFixture() throws -> (URL, StudioProject, [String: Any]) {
    let (root, original, runtime) = try fixture()
    let control = root.appendingPathComponent("fun.safetensors")
    let transformer = root.appendingPathComponent("base.safetensors")
    var controlHeader: [String: Any] = ["control_proj_in.weight": ["dtype": "F32", "shape": [5376, 196]]]
    for index in 0..<5 {
      controlHeader["control_blocks.\(index).adaln_proj.linear.weight"] = ["dtype": "BF16", "shape": [96768, 2688]]
    }
    try writeControlHeader(controlHeader, to: control)
    try writeControlHeader(["diffusion_model.time_embedder.proj_out.weight": ["dtype": "BF16", "shape": [2688, 5376]],
      "diffusion_model.blocks.0.adaln_proj.linear.weight": ["dtype": "I8", "shape": [96768, 2688]]], to: transformer)
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["fun_controlnet"] = control.path; components["transformer"] = transformer.path
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "control", "inputs": [], "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let video = root.appendingPathComponent("preprocessed-pose.mp4")
    try Data([1, 2, 3, 4]).write(to: video)
    let guide = MediaAsset(name: "Pose guide", kind: .video, path: video.path)
    var project = original; project.assets = [guide]
    project.clips[0].generationSelection = GenerationSelection(task: "control")
    var attachment = Attachment(assetID: guide.id, role: .control)
    attachment.controlType = "pose_skeleton"; attachment.strength = 0.75
    project.clips[0].attachments = [attachment]
    return (root, project, runtime)
  }

  func testFunControlProfileExposesControlAndPreservesOneHashedGuide() throws {
    let (root, project, runtime) = try controlFixture()
    let catalog = try NativeH3Preparation.catalog(directory: root.path)
    XCTAssertEqual(catalog.count, 1)
    XCTAssertEqual(catalog[0]["task"] as? String, "control")
    XCTAssertEqual((catalog[0]["generation"] as? [String: Any])?["supportedTasks"] as? [String], ["control"])
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = result["recipe"] as! [String: Any]
    XCTAssertEqual((recipe["components"] as! [String: Any])["fun_controlnet"] as? String,
      root.appendingPathComponent("fun.safetensors").path)
    let conditioning = recipe["conditioning"] as! [String: Any]
    XCTAssertEqual(conditioning["task"] as? String, "control")
    let inputs = conditioning["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.count, 1); XCTAssertEqual(inputs[0]["strength"] as? Double, 0.75)
    XCTAssertEqual(inputs[0]["control_type"] as? String, "pose_skeleton")
    XCTAssertEqual((inputs[0]["sha256"] as? String)?.count, 64)
    let target = root.appendingPathComponent("prepared")
    let prepared = try NativeH3Preparation.prepare(request: request(project, runtime), destination: target)
    XCTAssertTrue(FileManager.default.fileExists(atPath: prepared["recipePath"] as! String))
    XCTAssertEqual(project.clips[0].attachments[0].strength, 0.75)
  }

  func testFunControlRejectsChangedInjectionDeclarationsBeforePublishing() throws {
    let (root, project, runtime) = try controlFixture()
    let control = root.appendingPathComponent("fun.safetensors")
    var header = try JSONSerialization.jsonObject(with: Data(contentsOf: control).dropFirst(8)) as! [String: Any]
    for bad in ["[0,5,10,15,20]", "[false,10,20,30,40]"] {
      header["__metadata__"] = ["control_blocks_places": bad]
      try writeControlHeader(header, to: control)
      XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    }
  }

  func testFunControlRejectsPrunedAdaLNAndInvalidMediaContracts() throws {
    let (root, original, runtime) = try controlFixture()
    var project = original
    project.clips[0].attachments[0].controlType = "unprocessed_video"
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original; project.clips[0].attachments[0].strength = 1.1
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original; project.clips[0].attachments += project.clips[0].attachments
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project = original; project.clips[0].continuity = ClipContinuity(mode: "motion")
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    var pruned: [String: Any] = ["control_proj_in.weight": ["dtype": "F32", "shape": [5376, 196]]]
    for index in 0..<5 {
      pruned["control_blocks.\(index).adaln_proj.linear.weight"] = ["dtype": "F32", "shape": [96768, 8]]
    }
    try writeControlHeader(pruned, to: root.appendingPathComponent("fun.safetensors"))
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(original, runtime)))
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

  func testFrameContinuityUsesVisibleVFRFrameWithoutMutatingStoredInputs() async throws {
    let (root,original,runtime)=try fixture()
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="fl2va"
    definition["components"]=components
    definition["conditioning"]=["version":1,"task":"fflf","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    let movie=try await NativeLTXPreparationTests().continuityMovie(root)
    var source=Clip(engine:.movie);source.sourcePath=movie.path;source.sourceIn=0.15;source.duration=0.8
    var target=original.clips[0];target.continuity=ClipContinuity(mode:"frame")
    let stale=MediaAsset(name:"Old first",kind:.image,path:root.appendingPathComponent("missing.png").path)
    target.attachments=[Attachment(assetID:stale.id,role:.first)]
    var project=original;project.clips=[source,target];project.assets=[stale]
    var body=try request(project,runtime);body["clipID"]=target.id.uuidString
    let description=try NativeH3Preparation.describe(request:body)
    XCTAssertEqual(description["readinessErrors"] as? [String],[])
    let result=try await NativeH3Preparation.prepareWithMedia(request:body,destination:root.appendingPathComponent("prepared"))
    let recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as! [String:Any]
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.count,1);XCTAssertEqual(inputs[0]["frame_index"] as? Int,0)
    XCTAssertTrue((inputs[0]["path"] as! String).hasSuffix("previous-frame.png"))
    let report=(result["report"] as! [String:Any])["continuity"] as! [String:Any]
    XCTAssertEqual(report["engine"] as? String,"h3")
    XCTAssertEqual(report["sourceFrameTime"] as! Double,0.8,accuracy:0.001)
    let saved=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("prepared/editor-request.json"))) as! NSDictionary
    XCTAssertEqual(saved,body as NSDictionary)
    XCTAssertEqual(project.clips[1].attachments,target.attachments)
  }

  func testSaveMotionContextBuildsNativeVersionTwoContract() throws {
    let (_,original,runtime)=try fixture();var project=original
    project.clips[0].continuity=ClipContinuity(saveContext:true)
    let result=try NativeH3Preparation.compose(request:request(project,runtime))
    let recipe=result["recipe"] as! [String:Any]
    let context=try XCTUnwrap(recipe["continuation"] as? [String:Any])
    XCTAssertEqual(context["version"] as? Int,2)
    XCTAssertEqual(context["context_frames"] as? Int,22)
    XCTAssertEqual(context["save_context"] as? Bool,true)
    XCTAssertNil(context["source_context"])
    project.clips[0].duration=362.0/24
    let rerender=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    XCTAssertEqual((rerender["config"] as! [String:Any])["duration_seconds"] as? Double,15)
  }

  func testMotionContextKeepsHashesAlignsSavedTailAndRejectsTampering() throws {
    let (root,original,runtime)=try fixture()
    let folder=root.appendingPathComponent("context");try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    let payload=Data(repeating:0,count:7*2*2*96*4+2*37*32*4)
    func hash(_ data:Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    let manifest:[String:Any]=["format":"weetodd-h3-swift-continuation-v2","contextFrames":22,
      "width":64,"height":64,"generatedFrames":90,"publishedFrames":90,"overlapFrames":0,
      "identity":String(repeating:"c",count:64),"payloadBytes":payload.count,"payloadSHA256":hash(payload)]
    let bytes=try JSONSerialization.data(withJSONObject:manifest)
    try bytes.write(to:folder.appendingPathComponent("manifest.json"));try payload.write(to:folder.appendingPathComponent("latents.f32"))
    let movie=root.appendingPathComponent("source.mp4");try Data([0]).write(to:movie)
    var source=Clip(engine:.h3);source.sourcePath=movie.path;source.duration=3.75
    source.versions=[RenderVersion(path:movie.path,seed:1,prompt:"",recipePath:"",usableSourceIn:0,usableDuration:3.75,
      continuationArtifact:ContinuationArtifact(manifest:folder.appendingPathComponent("manifest.json").path,
        manifestSHA256:hash(bytes),payloadSHA256:hash(payload),payloadFilename:"latents.f32"))]
    var target=original.clips[0];target.duration=3;target.generationWidth=64;target.generationHeight=64
    target.continuity=ClipContinuity(mode:"motion",saveContext:true)
    var project=original;project.clips=[source,target]
    func body() throws -> [String:Any] { var value=try request(project,runtime);value["clipID"]=target.id.uuidString;return value }
    let result=try NativeH3Preparation.compose(request:body()),recipe=result["recipe"] as! [String:Any]
    let context=recipe["continuation"] as! [String:Any]
    XCTAssertEqual(context["source_manifest_sha256"] as? String,hash(bytes))
    XCTAssertEqual(context["source_context"] as? String,folder.appendingPathComponent("manifest.json").path)
    XCTAssertEqual((recipe["config"] as! [String:Any])["duration_seconds"] as? Double,85.0/24)
    XCTAssertFalse(((result["report"] as! [String:Any])["warnings"] as! [String]).isEmpty)
    project.clips[1].duration=15;XCTAssertThrowsError(try NativeH3Preparation.compose(request:body()))
    project.clips[1].duration=3
    var changed=payload;changed[0]=1;try changed.write(to:folder.appendingPathComponent("latents.f32"))
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:body()))
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

  func testComposesTimedAudioDriverFromPreparedMix() throws {
    let (root, original, runtime) = try fixture()
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["task"] = "ref2va"
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "ref2va",
      "inputs": [], "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let voicePath = root.appendingPathComponent("voice.wav")
    try Data([1, 2, 3]).write(to: voicePath)
    var project = original
    var voice = MediaAsset(name: "Prepared voice", kind: .audio, path: voicePath.path)
    voice.scope = .clip; voice.owner = project.clips[0].id; voice.duration = 5
    project.assets = [voice]
    project.clips[0].generationSelection = GenerationSelection(task: "a2v")
    project.clips[0].audioDriverSelection = AudioDriverSelection(mode: .voice)
    project.clips[0].audioDriverMixKey = "prepared"
    project.clips[0].attachments = [Attachment(assetID: voice.id, role: .audioDriver)]
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let prepared = result["recipe"] as! [String: Any]
    let contract = prepared["conditioning"] as! [String: Any]
    XCTAssertEqual(contract["task"] as? String, "a2v")
    let inputs = contract["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.count, 1)
    XCTAssertEqual(inputs[0]["role"] as? String, "audio_driver")
    XCTAssertEqual(inputs[0]["source_start_seconds"] as? Double, 0)
    XCTAssertEqual(inputs[0]["source_duration_seconds"] as? Double, 5)
    project.clips[0].audioDriverMixKey = nil
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testA2VOpeningImageUsesKeyframeContract() throws {
    let (root, original, runtime) = try fixture()
    let profile = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["task"] = "ref2va"
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "ref2va",
      "inputs": [], "audio_policy": "generated"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let voiceURL = root.appendingPathComponent("voice.wav")
    let imageURL = root.appendingPathComponent("opening.png")
    try Data([1, 2, 3]).write(to: voiceURL)
    try Data([4, 5, 6]).write(to: imageURL)
    var project = original
    var voice = MediaAsset(name: "Voice", kind: .audio, path: voiceURL.path)
    voice.duration = 5
    let image = MediaAsset(name: "Opening", kind: .image, path: imageURL.path)
    project.assets = [voice, image]
    project.clips[0].generationSelection = GenerationSelection(task: "a2v")
    var driver = Attachment(assetID: voice.id, role: .audioDriver)
    driver.audioSourceStart = 0
    driver.audioSourceDuration = 5
    project.clips[0].attachments = [driver, Attachment(assetID: image.id, role: .first)]
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let prepared = result["recipe"] as! [String: Any]
    let inputs = (prepared["conditioning"] as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.map { $0["role"] as? String }, ["audio_driver", "keyframe"])
    XCTAssertEqual(inputs[1]["frame_index"] as? Int, 0)
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
    let stacked = try NativeH3Preparation.compose(request: request(project, runtime))
    let stackedComponents = (stacked["recipe"] as! [String: Any])["components"] as! [String: Any]
    let entries = stackedComponents["loras"] as! [[Any]]
    XCTAssertEqual(entries.map { $0[0] as! String },
      ["/model/turbo.safetensors", "/model/second.safetensors"])
    XCTAssertEqual(entries.map { ($0[1] as! NSNumber).doubleValue }, [0.8, 1.0])
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
    let secondPath = root.appendingPathComponent("second.safetensors")
    try Data([0]).write(to: secondPath)
    var secondAsset = MediaAsset(name: "Second Turbo", kind: .lora,
      path: secondPath.path)
    secondAsset.loraModel = .h3
    secondAsset.loraProfile = "turbo"
    secondAsset.loraLayout = "contiguous_qkv"
    project.assets.append(secondAsset)
    project.clips[0].attachments.append(Attachment(assetID: secondAsset.id,
      role: .lora))
    let stacked = try NativeH3Preparation.compose(request: request(project, runtime))
    let stackedComponents = (stacked["recipe"] as! [String: Any])["components"] as! [String: Any]
    XCTAssertEqual((stackedComponents["loras"] as! [[Any]]).map { $0[0] as! String },
      [path.path, secondPath.path])
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

  func testMixedReferenceProfilePreservesVideoIdentityAndRejectsFourVideos() throws {
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
    let imagePath = root.appendingPathComponent("subject.png")
    let videoPath = root.appendingPathComponent("motion.mp4")
    let audioPath = root.appendingPathComponent("voice.wav")
    try Data([1, 2, 3]).write(to: imagePath)
    try Data([4, 5, 6]).write(to: videoPath)
    try Data([7, 8, 9]).write(to: audioPath)
    let image = MediaAsset(name: "Subject", kind: .image, path: imagePath.path)
    let video = MediaAsset(name: "Motion", kind: .video, path: videoPath.path)
    let audio = MediaAsset(name: "Voice", kind: .audio, path: audioPath.path)
    project.assets = [image, video, audio]
    project.clips[0].attachments = [
      Attachment(assetID: image.id, role: .reference),
      Attachment(assetID: video.id, role: .reference),
      Attachment(assetID: audio.id, role: .reference)]
    let prepared = try NativeH3Preparation.compose(request: request(project, runtime))
    let inputs = ((prepared["recipe"] as! [String: Any])["conditioning"]
      as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.map { $0["kind"] as? String }, ["image", "video", "audio"])
    XCTAssertEqual(inputs.map { $0["path"] as? String },
      [imagePath.path, videoPath.path, audioPath.path])
    XCTAssertEqual((inputs[1]["sha256"] as? String)?.count, 64)
    project.clips[0].attachments = Array(repeating:
      Attachment(assetID: video.id, role: .reference), count: 4)
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
    XCTAssertEqual(inputs[1]["frame_index"] as? Int, 119,
      "The last reference must land inside the five-second editorial interval")
    XCTAssertEqual((composed["report"] as! [String: Any])["task"] as? String, "fflf")
    project.clips[0].attachments.reverse()
    let reordered = try NativeH3Preparation.compose(request: request(project, runtime))
    let reorderedInputs = ((reordered["recipe"] as! [String: Any])["conditioning"]
      as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(reorderedInputs.map { $0["role"] as? String }, ["first", "last"])
    project.clips[0].attachments[1].strength = 0.5
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
  }

  func testFL2VAProfileComposesTimedKeyframesInFrameOrder() throws {
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
    let paths = (0..<3).map { root.appendingPathComponent("key-\($0).png") }
    for (index, path) in paths.enumerated() { try Data([UInt8(index + 1)]).write(to: path) }
    let assets = paths.enumerated().map { MediaAsset(name: "Key \($0.offset)",
      kind: .image, path: $0.element.path) }
    project.assets = assets
    project.clips[0].attachments = [
      Attachment(assetID: assets[2].id, role: .last),
      Attachment(assetID: assets[1].id, role: .keyframe, time: 1),
      Attachment(assetID: assets[0].id, role: .first)]
    let prepared = try NativeH3Preparation.compose(request: request(project, runtime))
    let inputs = ((prepared["recipe"] as! [String: Any])["conditioning"]
      as! [String: Any])["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs.map { $0["role"] as? String }, ["first", "keyframe", "last"])
    XCTAssertEqual(inputs[1]["frame_index"] as? Int, 24)
    project.clips[0].attachments[1].time = 0
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
    project.clips[0].attachments[1].time = 1e300
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
