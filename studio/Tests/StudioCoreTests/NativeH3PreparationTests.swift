import CryptoKit
import CoreGraphics
import AVFoundation
import Foundation
import ImageIO
import XCTest
@testable import StudioCore

final class NativeH3PreparationTests: XCTestCase {
  func testAudioOnlyRefRequiresEveryInputTimedAndKeepsAuthoredOrder() throws {
    let(root,original,runtime)=try fixture();var project=original
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    project.assets=[];project.clips[0].attachments=[];project.clips[0].generationSelection = .init(task:"ref2va")
    for index in 0..<2 {
      let url=root.appendingPathComponent("timed-audio-\(index).wav");try Data([UInt8(index+1)]).write(to:url)
      let asset=MediaAsset(name:"Timed audio",kind:.audio,path:url.path);project.assets.append(asset)
      var attachment=Attachment(assetID:asset.id,role:.reference)
      attachment.h3ReferencePlacement = .init(frame:index==0 ? .last : .index(0))
      project.clips[0].attachments.append(attachment)
    }
    let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.compactMap {$0["kind"] as? String},["audio","audio"])
    XCTAssertEqual(inputs.compactMap {$0["frame_index"] as? Int},[119,0])
    project.clips[0].attachments[1].h3ReferencePlacement=nil
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
  }

  func testA2VHistorySaveNeedsExactEffectiveAudioCoverageAndNeverPadsPreparedMix() throws {
    let(root,original,runtime)=try fixture();var project=original
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    var target=project.clips[0];target.generationSelection = .init(task:"a2v");target.duration=3
    target.generationWidth=64;target.generationHeight=64;target.continuity = .init(mode:"motion",saveContext:true)
    let directory=root.appendingPathComponent("a2v-context");try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    let payload=Data(count:7*2*2*96*4+2*37*32*4)
    func hash(_ data:Data)->String {SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()}
    try payload.write(to:directory.appendingPathComponent("latents.f32"))
    let manifest:[String:Any]=["format":"weetodd-h3-swift-continuation-v2","task":"ref2va","contextFrames":22,"width":64,"height":64,
      "generatedFrames":90,"publishedFrames":90,"overlapFrames":0,"identity":String(repeating:"c",count:64),"payloadBytes":payload.count,"payloadSHA256":hash(payload)]
    let bytes=try JSONSerialization.data(withJSONObject:manifest),manifestURL=directory.appendingPathComponent("manifest.json")
    try bytes.write(to:manifestURL)
    let sourceMovie=root.appendingPathComponent("a2v-source.mp4");try Data([3]).write(to:sourceMovie)
    var source=Clip(engine:.h3);source.sourcePath=sourceMovie.path;source.duration=3.75
    source.versions=[RenderVersion(path:sourceMovie.path,seed:1,prompt:"",recipePath:"",usableSourceIn:0,usableDuration:3.75,
      continuationArtifact:.init(manifest:manifestURL.path,manifestSHA256:hash(bytes),payloadSHA256:hash(payload),payloadFilename:"latents.f32"))]
    let driverURL=root.appendingPathComponent("a2v-history-driver.wav");try Data([4]).write(to:driverURL)
    var audio=MediaAsset(name:"Driver",kind:.audio,path:driverURL.path,scope:.clip,owner:target.id);audio.duration=3.6
    var attachment=Attachment(assetID:audio.id,role:.audioDriver);attachment.audioSourceStart=0;attachment.audioSourceDuration=3
    target.attachments=[attachment];project.assets=[audio]
    func body() throws->[String:Any] {
      project.clips=[source,target];var value=try request(project,runtime);value["clipID"]=target.id.uuidString;return value
    }
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:body())) {error in
      XCTAssertTrue(error.localizedDescription.contains("effective published"))
    }
    target.attachments[0].audioSourceDuration=3.6
    let recipe=try NativeH3Preparation.compose(request:body())["recipe"] as! [String:Any]
    XCTAssertEqual((recipe["config"] as! [String:Any])["duration_seconds"] as? Double,85.0/24)
    let driver=((recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[0]
    XCTAssertEqual(driver["source_duration_seconds"] as? Double,3.6)
    XCTAssertEqual(driver["source_start_seconds"] as? Double,0)
    XCTAssertNil(driver["frame_index"])
    target.audioDriverSelection=AudioDriverSelection(mode:.voice);target.audioDriverMixKey="prepared"
    target.attachments[0].audioSourceStart=nil;target.attachments[0].audioSourceDuration=nil
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:body())) {error in
      XCTAssertTrue(error.localizedDescription.contains("longer effective H3 published interval"))
      XCTAssertTrue(error.localizedDescription.contains("not padded"))
    }
  }

// Insert inside NativeH3PreparationTests; use its existing private fixture/request.
// These tests compose only. Tiny payloads are intentionally not decoded.
func testA2VFirstLastBudgetAnchorsFreezeAsOrderedKeyframes() throws {
  let(root,original,runtime)=try fixture();var project=original
  let profile=root.appendingPathComponent("h3.json")
  var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
  var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
  definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
  try JSONSerialization.data(withJSONObject:definition).write(to:profile)
  let sound=root.appendingPathComponent("budget-driver.wav");try Data([1]).write(to:sound)
  var audio=MediaAsset(name:"Driver",kind:.audio,path:sound.path);audio.duration=5
  project.assets=[audio];var driver=Attachment(assetID:audio.id,role:.audioDriver)
  driver.audioSourceStart=0;driver.audioSourceDuration=5;project.clips[0].attachments=[driver]
  for role in [MediaRole.last,.first] {
    let url=root.appendingPathComponent("budget-\(role.rawValue).png");try Data([2]).write(to:url)
    let image=MediaAsset(name:"Anchor",kind:.image,path:url.path);project.assets.append(image)
    var anchor=Attachment(assetID:image.id,role:role)
    anchor.h3ReferencePlacement = .init(imagePixelBudgetPercent:200)
    project.clips[0].attachments.append(anchor)
  }
  project.clips[0].generationSelection = .init(task:"a2v")
  let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
  let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
  XCTAssertEqual(inputs.dropFirst().compactMap {$0["role"] as? String},["keyframe","keyframe"])
  XCTAssertEqual(inputs.dropFirst().compactMap {$0["frame_index"] as? Int},[119,0])
  XCTAssertEqual(inputs.dropFirst().compactMap {$0["image_pixel_budget_percent"] as? Int},[200,200])
  XCTAssertEqual(inputs[0]["role"] as? String,"audio_driver")
  XCTAssertNil(inputs[0]["image_pixel_budget_percent"])
}

func testReferenceDensitySidecarAndAudioCapMatchWorkerAdmission() throws {
  let(root,original,runtime)=try fixture();var project=original
  let profile=root.appendingPathComponent("h3.json")
  var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
  var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
  definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
  try JSONSerialization.data(withJSONObject:definition).write(to:profile)
  let movie=root.appendingPathComponent("density.mov"),sidecar=root.appendingPathComponent("density-sidecar.wav")
  try Data([1]).write(to:movie);try Data([2]).write(to:sidecar)
  let visual=MediaAsset(name:"Movie",kind:.video,path:movie.path);project.assets=[visual]
  var attachment=Attachment(assetID:visual.id,role:.reference)
  attachment.h3ReferencePlacement = .init(soundtrackPath:sidecar.path,videoTemporalDensity:.quarter)
  project.clips[0].attachments=[attachment];project.clips[0].generationSelection = .init(task:"ref2va")
  for index in 0..<2 {
    let url=root.appendingPathComponent("standalone-\(index).wav");try Data([UInt8(index+3)]).write(to:url)
    let asset=MediaAsset(name:"Audio",kind:.audio,path:url.path);project.assets.append(asset)
    project.clips[0].attachments.append(Attachment(assetID:asset.id,role:.reference))
  }
  let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
  let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
  XCTAssertEqual(inputs[0]["size_policy"] as? String,"match_output")
  XCTAssertEqual(inputs[0]["temporal_density"] as? String,"quarter")
  XCTAssertEqual(inputs[0]["soundtrack_path"] as? String,sidecar.path)
  let fourth=root.appendingPathComponent("fourth-audio.wav");try Data([6]).write(to:fourth)
  let extra=MediaAsset(name:"Fourth audio source",kind:.audio,path:fourth.path);project.assets.append(extra)
  project.clips[0].attachments.append(Attachment(assetID:extra.id,role:.reference))
  XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)),
    "The movie sidecar is one of the worker's maximum three audio sources.")
}

  func testH3RejectsCraftedLTXKeyframeSelectionBeforePreparation() throws {
    let(_,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:2,experimentalEnabled:true)
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
    project.clips[0].generationSelection?.ltx25Keyframes=nil
    XCTAssertNoThrow(try NativeH3Preparation.compose(request:request(project,runtime)))
  }
  func testResMultistepProfileAndSelectionPreserveStepsWithoutPython() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    project.clips[0].generationSelection?.h3SamplingMethod = .resMultistep
    project.clips[0].generationSelection?.steps = 4
    let composed = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = composed["recipe"] as! [String: Any]
    let config = recipe["config"] as! [String: Any]
    XCTAssertEqual(config["sampling_method"] as? String, "res_multistep")
    XCTAssertEqual(config["steps"] as? Int, 5)
    let url = root.appendingPathComponent("h3.json")
    try JSONSerialization.data(withJSONObject: recipe).write(to: url)
    XCTAssertEqual(try NativeH3Preparation.catalog(directory: root.path).count, 1)
    project.clips[0].generationSelection?.h3SamplingMethod = nil
    let inherited = try NativeH3Preparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((inherited["config"] as! [String: Any])["sampling_method"] as? String, "res_multistep")
    project.clips[0].generationSelection?.h3SamplingMethod = .euler
    let overridden = try NativeH3Preparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((overridden["config"] as! [String: Any])["sampling_method"] as? String, "euler")
  }

  func testResMultistepRejectsExplicitAndHeaderTurboBeforeWeights() throws {
    for metadata in [[:], ["adapter_profile": "turbo", "inference_steps": "4"]] {
      let (root, original, runtime) = try fixture()
      let adapter = try loraFile(root, metadata: metadata)
      var project = original
      attachLoRA(adapter, profile: metadata.isEmpty ? "turbo" : nil, to: &project)
      project.clips[0].generationSelection?.h3SamplingMethod = .resMultistep
      XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime))) { error in
        XCTAssertTrue(error.localizedDescription.contains("Turbo"))
        XCTAssertTrue(error.localizedDescription.contains("Euler"))
      }
      project.clips[0].generationSelection?.h3SamplingMethod = .euler
      XCTAssertEqual(try preparedSteps(project, runtime), 5)
    }
    let (root, original, runtime) = try fixture()
    let adapter = try loraFile(root)
    try setProfile(root, rootTurbo: adapter.path)
    var project = original
    project.clips[0].generationSelection?.h3SamplingMethod = .resMultistep
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime))) { error in
      XCTAssertTrue(error.localizedDescription.contains("Turbo requires Euler"))
    }
  }

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

  func testFunControlPreservesReleasedOneMegapixelCanvasAndRejectsLargerArea() throws {
    let (_, original, runtime) = try controlFixture()
    var project = original
    project.clips[0].generationWidth = 1376
    project.clips[0].generationHeight = 768
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = result["recipe"] as! [String: Any]
    let config = recipe["config"] as! [String: Any]
    XCTAssertEqual(config["width"] as? Int, 1376)
    XCTAssertEqual(config["height"] as? Int, 768)
    project.clips[0].generationWidth = 1408
    XCTAssertThrowsError(try NativeH3Preparation.compose(request: request(project, runtime)))
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

  // Tiny, valid paired SafeTensors files exercise admission without reading model weights.
  private func loraFile(_ root: URL, name: String = "adapter", metadata: [String: String] = [:],
    target: String = "diffusion_model.blocks.0.attn.qkv_proj") throws -> URL {
    let header: [String: Any] = ["__metadata__": metadata,
      target + ".lora_A.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
      target + ".lora_B.weight": ["dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]]]
    let bytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }
    data.append(bytes); data.append(Data(repeating: 0, count: 8))
    let url = root.appendingPathComponent(name + ".safetensors")
    try data.write(to: url); return url
  }

  private func setProfile(_ root: URL, steps: Int = 20, pairs: [[Any]] = [],
    rootTurbo: String? = nil) throws {
    let url = root.appendingPathComponent("h3.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var config = recipe["config"] as! [String: Any]; config["steps"] = steps
    var components = recipe["components"] as! [String: Any]; components["loras"] = pairs
    recipe["config"] = config; recipe["components"] = components
    if let rootTurbo { recipe["loras"] = ["adapters": [["path": rootTurbo,
      "strength": 1.0, "profile": "turbo", "qkv_layout": "contiguous_qkv"]]] }
    try JSONSerialization.data(withJSONObject: recipe).write(to: url)
  }

  private func attachLoRA(_ url: URL, profile: String?, to project: inout StudioProject) {
    var asset = MediaAsset(name: "Adapter", kind: .lora, path: url.path)
    asset.loraModel = .h3; asset.loraProfile = profile; asset.loraLayout = "contiguous_qkv"
    project.assets.append(asset)
    project.clips[0].attachments.append(Attachment(assetID: asset.id, role: .lora))
  }

  private func preparedSteps(_ project: StudioProject, _ runtime: [String: Any]) throws -> Int {
    let result = try NativeH3Preparation.compose(request: request(project, runtime))
    let recipe = result["recipe"] as! [String: Any]
    let steps = (recipe["config"] as! [String: Any])["steps"] as! Int
    let controls = ((result["report"] as! [String: Any])["generation"] as! [String: Any])["controls"] as! [String: Any]
    XCTAssertEqual(controls["evaluations"] as? Int, steps - 1)
    return steps
  }

  func testStandardLoRAAttachmentPreservesUserEvaluationsAndEditor() throws {
    let (root, original, runtime) = try fixture()
    let url = try loraFile(root, metadata: ["adapter_profile": "standard", "inference_steps": "19"])
    var project = original; attachLoRA(url, profile: "standard", to: &project)
    project.clips[0].generationSelection?.steps = 19
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let before = try encoder.encode(project)
    XCTAssertEqual(try preparedSteps(project, runtime), 20)
    XCTAssertEqual(try encoder.encode(project), before)
  }

  func testTurboAttachmentResolvesDefaultButRejectsExplicitWrongEvaluations() throws {
    let (root, original, runtime) = try fixture(); try setProfile(root)
    let url = try loraFile(root)
    var project = original; attachLoRA(url, profile: "turbo", to: &project)
    XCTAssertEqual(try preparedSteps(project, runtime), 5)
    project.clips[0].generationSelection?.steps = 4
    XCTAssertEqual(try preparedSteps(project, runtime), 5)
    for evaluations in [3, 5, 8, 19] {
      project.clips[0].generationSelection?.steps = evaluations
      XCTAssertThrowsError(try preparedSteps(project, runtime)) { error in
        XCTAssertTrue(error.localizedDescription.contains("4 evaluations"), error.localizedDescription)
        XCTAssertTrue(error.localizedDescription.contains("5"), error.localizedDescription)
      }
    }
    project.clips[0].attachments[0].enabled = false
    XCTAssertEqual(try preparedSteps(project, runtime), 20)
  }

  func testDeclaredTurboAndUnknownComponentPairsUseDistinctSamplingContracts() throws {
    let (root, original, runtime) = try fixture()
    let unknown = try loraFile(root, name: "unknown")
    try setProfile(root, pairs: [[unknown.path, 0.8]])
    XCTAssertEqual(try preparedSteps(original, runtime), 20)
    let turbo = try loraFile(root, name: "declared", metadata: ["schedule_points": "5"])
    try setProfile(root, pairs: [[unknown.path, 0.8], [turbo.path, 1.0]])
    XCTAssertEqual(try preparedSteps(original, runtime), 5)
    var project = original; project.clips[0].generationSelection?.steps = 19
    XCTAssertThrowsError(try preparedSteps(project, runtime))
    try setProfile(root, rootTurbo: unknown.path)
    XCTAssertEqual(try preparedSteps(original, runtime), 5)
    XCTAssertThrowsError(try preparedSteps(project, runtime))
  }

  func testTurboHeaderSamplingCountsAndLayoutAreAdmittedBeforePublication() throws {
    let (root, original, runtime) = try fixture()
    let badCount = try loraFile(root, name: "count", metadata: ["inference_steps": "12"])
    var project = original; attachLoRA(badCount, profile: "turbo", to: &project)
    let destination = root.appendingPathComponent("not-published")
    XCTAssertThrowsError(try NativeH3Preparation.prepare(request: request(project, runtime), destination: destination))
    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    let conflict = try loraFile(root, name: "conflict", metadata: ["adapter_profile": "turbo",
      "inference_steps": "4", "schedule_points": "6"])
    try setProfile(root, pairs: [[conflict.path, 1.0]])
    XCTAssertThrowsError(try preparedSteps(original, runtime))
    for (name, metadata, target) in [
      ("layout", ["adapter_profile": "standard", "qkv_layout": "native_interleaved"], "diffusion_model.blocks.0.attn.qkv_proj"),
      ("adaln", ["adapter_profile": "standard"], "diffusion_model.blocks.0.adaln_proj.linear")] {
      let url = try loraFile(root, name: name, metadata: metadata, target: target)
      try setProfile(root, pairs: [[url.path, 1.0]])
      if name == "layout" { XCTAssertEqual(try preparedSteps(original,runtime),20) }
      else { XCTAssertThrowsError(try preparedSteps(original,runtime)) }
    }
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

  private func flContextFixture() throws -> (URL,StudioProject,[String:Any]) {
    let (root,original,runtime)=try fixture()
    let profile=root.appendingPathComponent("h3.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=recipe["components"] as! [String:Any]
    components["task"]="fl2va";components["vision_encoder"]="/model/vision"
    recipe["components"]=components
    recipe["conditioning"]=["version":1,"task":"fflf","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    let image=root.appendingPathComponent("anchor.png")
    let context=try XCTUnwrap(CGContext(data:nil,width:64,height:64,bitsPerComponent:8,bytesPerRow:256,
      space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(red:0.2,green:0.3,blue:0.4,alpha:1));context.fill(CGRect(x:0,y:0,width:64,height:64))
    let destination=try XCTUnwrap(CGImageDestinationCreateWithURL(image as CFURL,"public.png" as CFString,1,nil))
    CGImageDestinationAddImage(destination,try XCTUnwrap(context.makeImage()),nil)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    let asset=MediaAsset(name:"Anchor",kind:.image,path:image.path)
    var project=original;project.assets=[asset]
    project.clips[0].generationWidth=64;project.clips[0].generationHeight=64
    project.clips[0].duration=3;project.clips[0].generationSelection=GenerationSelection(task:"fflf")
    project.clips[0].attachments=[Attachment(assetID:asset.id,role:.last),Attachment(assetID:asset.id,role:.first)]
    return(root,project,runtime)
  }

  func testFL2VASaveContextUsesVersionThreeAndEffectiveVisibleLastFrameWithoutPython() throws {
    let (root,original,runtime)=try flContextFixture();var project=original
    project.clips[0].continuity=ClipContinuity(saveContext:true)
    let composed=try NativeH3Preparation.compose(request:request(project,runtime))
    let recipe=composed["recipe"] as! [String:Any],context=recipe["continuation"] as! [String:Any]
    XCTAssertEqual(context["version"] as? Int,3);XCTAssertNil(context["source_context"])
    XCTAssertEqual((recipe["components"] as! [String:Any])["task"] as? String,"fl2va")
    XCTAssertEqual((recipe["config"] as! [String:Any])["duration_seconds"] as? Double,3)
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.compactMap { $0["frame_index"] as? Int },[0,72])
    XCTAssertEqual(project.clips[0].attachments,original.clips[0].attachments)
    let prepared=try NativeH3Preparation.prepare(request:request(project,runtime),destination:root.appendingPathComponent("prepared"))
    XCTAssertEqual((prepared["report"] as! [String:Any])["nativePreparation"] as? String,"swift")
    XCTAssertEqual(try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("prepared/editor-request.json"))) as! NSDictionary,
      try request(project,runtime) as NSDictionary)
    project.clips[0].duration=362.0/24
    let rerender=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let repeatedInputs=(rerender["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(repeatedInputs.compactMap { $0["frame_index"] as? Int },[0,361])
    XCTAssertEqual((rerender["config"] as! [String:Any])["duration_seconds"] as? Double,15)
  }

  func testFL2VAMotionKeepsVisibleKeyframesAndRequiresTaskBoundNativeContext() throws {
    let (root,original,runtime)=try flContextFixture();var project=original
    let directory=root.appendingPathComponent("fl-context");try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
    let payload=Data(repeating:0,count:7*2*2*96*4+2*37*32*4)
    func hash(_ bytes:Data)->String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
    var manifest:[String:Any]=["format":"weetodd-h3-swift-continuation-v2","task":"fl2va","contextFrames":22,
      "width":64,"height":64,"generatedFrames":90,"publishedFrames":90,"overlapFrames":0,
      "identity":String(repeating:"c",count:64),"payloadBytes":payload.count,"payloadSHA256":hash(payload)]
    let manifestURL=directory.appendingPathComponent("manifest.json")
    let movie=root.appendingPathComponent("source.mp4");try Data([0]).write(to:movie)
    try payload.write(to:directory.appendingPathComponent("latents.f32"))
    var source=original.clips[0];source.sourcePath=movie.path;source.duration=3.75
    var target=original.clips[0];target.id=UUID();target.duration=3;target.continuity=ClipContinuity(mode:"motion",saveContext:true)
    var timed=Attachment(assetID:project.assets[0].id,role:.keyframe);timed.time=71.0/24
    target.attachments.append(timed)
    func body() throws->[String:Any] {
      let bytes=try JSONSerialization.data(withJSONObject:manifest)
      try bytes.write(to:manifestURL)
      source.versions=[RenderVersion(path:movie.path,seed:1,prompt:"",recipePath:"",usableSourceIn:0,usableDuration:3.75,
        continuationArtifact:ContinuationArtifact(manifest:manifestURL.path,manifestSHA256:hash(bytes),payloadSHA256:hash(payload),payloadFilename:"latents.f32"))]
      project.clips=[source,target]
      var value=try request(project,runtime);value["clipID"]=target.id.uuidString;return value
    }
    let composed=try NativeH3Preparation.compose(request:body()),recipe=composed["recipe"] as! [String:Any]
    XCTAssertEqual((recipe["continuation"] as! [String:Any])["version"] as? Int,3)
    XCTAssertEqual((recipe["config"] as! [String:Any])["duration_seconds"] as? Double,85.0/24)
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.compactMap { $0["frame_index"] as? Int },[0,71,84])
    for task in [nil,"t2va","ref2va"] as [String?] {
      manifest["task"]=task
      XCTAssertThrowsError(try NativeH3Preparation.compose(request:body())) { error in
        XCTAssertTrue(error.localizedDescription.contains("context"))
      }
    }
    manifest["task"]="fl2va";manifest["format"]="weetodd-h3-continuation-v1"
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:body()))
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

  private var referenceDialoguePrompt: String {
    """
    subject_definitions:
    <Subject 1> is Beowulf from <Picture 1>, the sole speaker (S1). <Audio 1> guides his warm male voice.

    summary:
    [reference generation + audio reference] Beowulf addresses the camera in a quiet room.

    retention_analysis:
    <Subject 1>: fully_preserved - retain his identity. <Audio 1>: reference - generate new speech without copying the source waveform.

    detailed_description:
    [Shot 1] <Subject 1> (S1) faces a locked camera and says, <d>[English] I stand guard.</d>

    overall_soundscape:
    Quiet room tone beneath the single speaking voice.

    non_diegetic_music:
    A soft instrumental cello, without voices.
    """
  }

  func testCompleteReferenceDialoguePromptSurvivesNativePreparationVerbatimWithoutDefaultNoDialogue() throws {
    for task in ["ref2va", "a2v"] {
      let (root, original, runtime) = try fixture()
      let profileURL = root.appendingPathComponent("h3.json")
      var profile = try JSONSerialization.jsonObject(with: Data(contentsOf: profileURL)) as! [String: Any]
      var components = profile["components"] as! [String: Any]
      components["task"] = "ref2va"; components["vision_encoder"] = "/unloaded/vision.safetensors"
      profile["components"] = components
      profile["conditioning"] = ["version": 1, "task": "ref2va", "inputs": [], "audio_policy": "generated"]
      try JSONSerialization.data(withJSONObject: profile).write(to: profileURL)
      let audioURL = root.appendingPathComponent("voice.wav")
      try Data([1, 2, 3]).write(to: audioURL) // Composition hashes sources; media decoding is a later stage.
      var audio = MediaAsset(name: "Voice reference", kind: .audio, path: audioURL.path); audio.duration = 6
      var project = original; project.assets = [audio]
      var attachment = Attachment(assetID: audio.id, role: task == "a2v" ? .audioDriver : .reference)
      if task == "a2v" { attachment.audioSourceStart = 0.3; attachment.audioSourceDuration = 5 }
      project.clips[0].attachments = [attachment]
      if task == "ref2va" {
        let imageURL = root.appendingPathComponent("beowulf.png")
        let context = try XCTUnwrap(CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8,
          bytesPerRow: 64 * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.3, blue: 0.4, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(imageURL as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let image = MediaAsset(name: "Beowulf identity", kind: .image, path: imageURL.path)
        project.assets.append(image)
        project.clips[0].attachments.insert(Attachment(assetID: image.id, role: .reference), at: 0)
      }
      project.clips[0].generationSelection = GenerationSelection(task: task)
      project.clips[0].prompt = " \n\t" + referenceDialoguePrompt + "\n "
      XCTAssertEqual(project.clips[0].soundscape, "Natural location sound. No dialogue.")
      let prepared = try NativeH3Preparation.prepare(request: request(project, runtime), destination: root.appendingPathComponent("prepared"))
      let recipeURL = URL(fileURLWithPath: try XCTUnwrap(prepared["recipePath"] as? String))
      let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: recipeURL)) as! [String: Any]
      XCTAssertEqual(recipe["prompt"] as? String, referenceDialoguePrompt)
      XCTAssertFalse((recipe["prompt"] as! String).contains("No dialogue."))
      XCTAssertEqual((recipe["conditioning"] as! [String: Any])["task"] as? String, task)
      XCTAssertEqual((recipe["conditioning"] as! [String: Any])["audio_policy"] as? String, "generated")
      XCTAssertEqual((recipe["config"] as! [String: Any])["seed"] as? Int, 42)
      XCTAssertEqual(project.clips[0].prompt, " \n\t" + referenceDialoguePrompt + "\n ")
    }
  }

  func testPartialQuotedAndMisorderedReferenceHeadingsStillUseSimplePromptComposer() throws {
    let (_, original, runtime) = try fixture()
    let malformed = [
      referenceDialoguePrompt.replacingOccurrences(of: "summary:\n", with: ""),
      "A sign quotes \"" + referenceDialoguePrompt.replacingOccurrences(of: "\n", with: " ") + "\".",
      "Show the following text on a sign:\n" + referenceDialoguePrompt,
      referenceDialoguePrompt.replacingOccurrences(of: "summary:", with: "retention_analysis:")
        .replacingOccurrences(of: "retention_analysis:\n<Subject", with: "summary:\n<Subject"),
      referenceDialoguePrompt.replacingOccurrences(of: "non_diegetic_music:\nA soft instrumental cello, without voices.", with: "non_diegetic_music:"),
      referenceDialoguePrompt.uppercased()
    ]
    for text in malformed {
      var project = original; project.clips[0].prompt = text
      let recipe = try NativeH3Preparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
      let prompt = try XCTUnwrap(recipe["prompt"] as? String)
      XCTAssertTrue(prompt.hasPrefix("integrated_multimodal_description: [Shot 1] "), text)
      XCTAssertTrue(prompt.hasSuffix("overall_soundscape: Natural location sound. No dialogue.\n\nnon_diegetic_music: N/A"), text)
    }
  }

  func testExistingIntegratedAndContinuationPromptPassThroughRemainUnchanged() throws {
    let (_, original, runtime) = try fixture()
    for text in ["integrated_multimodal_description: [Shot 1] Beowulf speaks.\n\noverall_soundscape: His voice.\n\nnon_diegetic_music: N/A",
      "[video continuation] Continue the existing action and sound."] {
      var project = original; project.clips[0].prompt = "\n " + text + " \n"
      let recipe = try NativeH3Preparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
      XCTAssertEqual(recipe["prompt"] as? String, text)
    }
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
    let first = try loraFile(root, name: "profile-first")
    let second = try loraFile(root, name: "profile-second")
    components["loras"] = [[first.path, 0.8]]
    recipe["components"] = components
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let prepared = try NativeH3Preparation.compose(request: request(project, runtime))
    let selected = (prepared["recipe"] as! [String: Any])["components"] as! [String: Any]
    XCTAssertEqual((selected["loras"] as! [[Any]])[0][0] as? String,
      first.path)
    components["loras"] = [[first.path, 0.8], [second.path, 1.0]]
    recipe["components"] = components
    try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
    let stacked = try NativeH3Preparation.compose(request: request(project, runtime))
    let stackedComponents = (stacked["recipe"] as! [String: Any])["components"] as! [String: Any]
    let entries = stackedComponents["loras"] as! [[Any]]
    XCTAssertEqual(entries.map { $0[0] as! String },
      [first.path, second.path])
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
    let path = try loraFile(root, name: "turbo")
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
    let secondPath = try loraFile(root, name: "second")
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
    let turbo = try loraFile(root, name: "root-turbo")
    recipe["loras"] = ["adapters": [["path": turbo.path,
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
      turbo.path)
  }
  func testMovieSourceIntervalsCannotHideOnOrdinaryOrDisabledAttachments() async throws {
    let (root,original,runtime)=try fixture()
    var baseline=original
    let unused=MediaAsset(name:"Disabled stale adapter",kind:.lora,path:"/missing/unused.safetensors")
    baseline.assets.append(unused)
    var attachment=Attachment(assetID:unused.id,role:.lora);attachment.enabled=false
    baseline.clips[0].attachments.append(attachment)
    XCTAssertNoThrow(try NativeH3Preparation.compose(request:request(baseline,runtime)))
    for durationField in [false,true] {
      var project=baseline
      if durationField { project.clips[0].attachments[0].sourceDurationSeconds=1 }
      else { project.clips[0].attachments[0].sourceStartSeconds=0 }
      let frozen=try request(project,runtime)
      XCTAssertThrowsError(try NativeH3Preparation.compose(request:frozen)) {
        XCTAssertTrue($0.localizedDescription.contains("Source movie interval fields"))
      }
      XCTAssertThrowsError(try NativeH3Preparation.describe(request:frozen)) {
        XCTAssertTrue($0.localizedDescription.contains("Source movie interval fields"))
      }
      let destination=root.appendingPathComponent("rejected-movie-field-"+String(durationField))
      do {
        _ = try await NativeH3Preparation.prepareWithMedia(request:frozen,destination:destination)
        XCTFail("Ordinary preparation silently ignored a movie interval")
      } catch { XCTAssertTrue(error.localizedDescription.contains("Source movie interval fields")) }
      XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
    }
  }
// Insert in existing NativeH3PreparationTests to reuse its bounded real headers.
  func testEightSignedAdaptersWithExplicitDeferredQKVContract() throws {
    let(root,original,runtime)=try fixture();var project=original
    for index in 0..<8 {
      attachLoRA(try loraFile(root,name:"signed-\(index)"),profile:"standard",to:&project)
      project.clips[0].attachments[index].strength=index==0 ? -2 : 0.5
      project.clips[0].attachments[index].h3LoRA = .init(profile:.standard,qkvLayout:.nativeInterleaved,startAfterEvaluations:index==0 ? 2 : 0)
    }
    let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let stack=try XCTUnwrap(recipe["loras"] as? [String:Any]);let adapters=try XCTUnwrap(stack["adapters"] as? [[String:Any]])
    XCTAssertEqual(stack["version"] as? Int,1);XCTAssertEqual(adapters.count,8)
    XCTAssertEqual(adapters[0]["strength"] as? Double,-2);XCTAssertEqual(adapters[0]["start_after_evaluations"] as? Int,2)
    XCTAssertEqual(adapters[0]["qkv_layout"] as? String,"native_interleaved")
    XCTAssertNil((recipe["components"] as? [String:Any])?["loras"])
    project.clips[0].attachments[0].h3LoRA?.startAfterEvaluations=4
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
    project.clips[0].attachments[0].h3LoRA?.startAfterEvaluations=0
    attachLoRA(try loraFile(root,name:"ninth"),profile:"standard",to:&project)
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
  }
  func testReferenceTimingSoundtrackAndGlobalNoiseAreFrozenSeparately() throws {
    let(root,original,runtime)=try fixture();var project=original
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    let movie=root.appendingPathComponent("reference.mov"),audio=root.appendingPathComponent("sound.wav")
    try Data([1,2,3]).write(to:movie);try Data([4,5,6]).write(to:audio)
    let asset=MediaAsset(name:"AV reference",kind:.video,path:movie.path);project.assets=[asset]
    var input=Attachment(assetID:asset.id,role:.reference)
    input.h3ReferencePlacement = .init(frame:.last,soundtrackPath:audio.path)
    project.clips[0].attachments=[input];project.clips[0].generationSelection = .init(task:"ref2va")
    project.clips[0].generationSelection?.h3Reference = .init(visualConditionStrength:0.3,audioConditionStrength:0.8)
    let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let config=recipe["config"] as! [String:Any],inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(config["visual_condition_strength"] as? Double,0.3);XCTAssertEqual(config["audio_condition_strength"] as? Double,0.8)
    XCTAssertEqual(inputs[0]["frame_index"] as? Int,119);XCTAssertEqual(inputs[0]["strength"] as? Double,1)
    XCTAssertEqual(inputs[0]["soundtrack_path"] as? String,audio.path)
    XCTAssertEqual(inputs[0]["soundtrack_sha256"] as? String,SHA256.hash(data:Data([4,5,6])).map { String(format:"%02x",$0) }.joined())
    let description=try NativeH3Preparation.describe(request:request(project,runtime));XCTAssertTrue((description["sourcePaths"] as! [String]).contains(audio.path))
    project.clips[0].attachments[0].h3ReferencePlacement?.frame = .index(120)
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
  }
  func testA2VAdmitsDistinctTimedImagesWithoutSortingReferenceOrder() throws {
    let(root,original,runtime)=try fixture();var project=original
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    let sound=root.appendingPathComponent("driver.wav");try Data([1]).write(to:sound)
    var audio=MediaAsset(name:"Driver",kind:.audio,path:sound.path);audio.duration=5
    project.assets=[audio];var driver=Attachment(assetID:audio.id,role:.audioDriver)
    driver.audioSourceStart=0;driver.audioSourceDuration=5;project.clips[0].attachments=[driver]
    for index in [2,0,1] {
      let url=root.appendingPathComponent("image-\(index).png");try Data([UInt8(index+2)]).write(to:url)
      let image=MediaAsset(name:"Anchor",kind:.image,path:url.path);project.assets.append(image)
      project.clips[0].attachments.append(Attachment(assetID:image.id,role:.keyframe,time:Double(index)))
    }
    project.clips[0].generationSelection = .init(task:"a2v")
    let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.dropFirst().compactMap { $0["frame_index"] as? Int },[48,0,24])
    project.clips[0].attachments[3].time=0
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
  }

  func testFunAcceptsSignedStandardBaseStreamLoRAWithoutChangingGuide() throws {
    let(root,original,runtime)=try controlFixture();var project=original
    let adapter=try loraFile(root);attachLoRA(adapter,profile:"standard",to:&project)
    project.clips[0].attachments[1].strength = -0.5
    project.clips[0].attachments[1].h3LoRA = .init(profile:.standard,qkvLayout:.contiguousQKV)
    let result=try NativeH3Preparation.compose(request:request(project,runtime)),recipe=result["recipe"] as! [String:Any]
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.count,1);XCTAssertEqual(inputs[0]["control_type"] as? String,"pose_skeleton")
    XCTAssertEqual(inputs[0]["strength"] as? Double,0.75)
    let adapters=(recipe["loras"] as! [String:Any])["adapters"] as! [[String:Any]]
    XCTAssertEqual(adapters[0]["strength"] as? Double,-0.5)
  }
  func testAdvancedControlsRejectAutoProjectionBeforeMediaPublication() throws {
    let(_,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.h3Joint = .init(saveFullLatents:true)
    project.clips[0].generationSelection?.projectionBackend="auto"
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime))) { error in
      XCTAssertTrue(error.localizedDescription.contains("explicit MLX"))
    }
  }

// Insert inside NativeH3PreparationTests, reusing its existing tiny native headers.
  func testRefAndA2VSaveNativeTaskBoundContextVersion4WithoutPreoffsettingMedia() throws {
    for task in ["ref2va","a2v"] {
      let(root,original,runtime)=try fixture();var project=original
      let profile=root.appendingPathComponent("h3.json")
      var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
      var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
      definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
      try JSONSerialization.data(withJSONObject:definition).write(to:profile)
      project.clips[0].generationSelection = .init(task:task);project.clips[0].duration=3
      project.clips[0].continuity = .init(saveContext:true)
      let imageURL=root.appendingPathComponent("identity.png");try Data([1]).write(to:imageURL)
      let image=MediaAsset(name:"Identity",kind:.image,path:imageURL.path);project.assets=[image]
      if task == "ref2va" {
        var ref=Attachment(assetID:image.id,role:.reference);ref.h3ReferencePlacement = .init(frame:.last)
        project.clips[0].attachments=[ref]
      } else {
        let audioURL=root.appendingPathComponent("driver.wav");try Data([2]).write(to:audioURL)
        var sound=MediaAsset(name:"Driver",kind:.audio,path:audioURL.path);sound.duration=3.6;project.assets.append(sound)
        var driver=Attachment(assetID:sound.id,role:.audioDriver);driver.audioSourceStart=0;driver.audioSourceDuration=3.6
        project.clips[0].attachments=[driver,Attachment(assetID:image.id,role:.keyframe,time:71.0/24)]
      }
      let recipe=try NativeH3Preparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
      let continuation=recipe["continuation"] as! [String:Any]
      XCTAssertEqual(continuation["version"] as? Int,4);XCTAssertEqual(continuation["save_context"] as? Bool,true)
      XCTAssertNil(continuation["source_context"])
      let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
      if task == "ref2va" { XCTAssertEqual(inputs[0]["frame_index"] as? Int,72) }
      else { XCTAssertEqual(inputs[1]["frame_index"] as? Int,71);XCTAssertNil(inputs[0]["frame_index"]) }
    }
  }
  func testRefNativeContextLoadRejectsOtherTaskAndPreservesVisiblePlacement() throws {
    let(root,original,runtime)=try fixture();var project=original
    let profile=root.appendingPathComponent("h3.json")
    var definition=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    var components=definition["components"] as! [String:Any];components["task"]="ref2va";definition["components"]=components
    definition["conditioning"]=["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"]
    try JSONSerialization.data(withJSONObject:definition).write(to:profile)
    let imageURL=root.appendingPathComponent("identity.png");try Data([1]).write(to:imageURL)
    let image=MediaAsset(name:"Identity",kind:.image,path:imageURL.path);project.assets=[image]
    var target=project.clips[0];target.generationSelection = .init(task:"ref2va");target.duration=3
    target.generationWidth=64;target.generationHeight=64;target.continuity = .init(mode:"motion",saveContext:true)
    var ref=Attachment(assetID:image.id,role:.reference);ref.h3ReferencePlacement = .init(frame:.index(71));target.attachments=[ref]
    let directory=root.appendingPathComponent("ref-context");try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    let payload=Data(count:7*2*2*96*4+2*37*32*4)
    func hash(_ bytes:Data)->String { SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined() }
    try payload.write(to:directory.appendingPathComponent("latents.f32"))
    var manifest:[String:Any]=["format":"weetodd-h3-swift-continuation-v2","task":"ref2va","contextFrames":22,"width":64,"height":64,
      "generatedFrames":90,"publishedFrames":90,"overlapFrames":0,"identity":String(repeating:"c",count:64),"payloadBytes":payload.count,"payloadSHA256":hash(payload)]
    let sourceMovie=root.appendingPathComponent("source.mp4");try Data([3]).write(to:sourceMovie)
    var source=Clip(engine:.h3);source.sourcePath=sourceMovie.path;source.duration=3.75
    func body() throws->[String:Any] {
      let bytes=try JSONSerialization.data(withJSONObject:manifest),url=directory.appendingPathComponent("manifest.json");try bytes.write(to:url)
      source.versions=[RenderVersion(path:sourceMovie.path,seed:1,prompt:"",recipePath:"",usableSourceIn:0,usableDuration:3.75,
        continuationArtifact:.init(manifest:url.path,manifestSHA256:hash(bytes),payloadSHA256:hash(payload),payloadFilename:"latents.f32"))]
      project.clips=[source,target];var body=try request(project,runtime);body["clipID"]=target.id.uuidString;return body
    }
    let recipe=try NativeH3Preparation.compose(request:body())["recipe"] as! [String:Any]
    XCTAssertEqual((recipe["continuation"] as! [String:Any])["version"] as? Int,4)
    XCTAssertEqual((recipe["config"] as! [String:Any])["duration_seconds"] as? Double,85.0/24)
    XCTAssertEqual(((recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[0]["frame_index"] as? Int,71)
    for wrong in [nil,"t2va","fl2va"] as [String?] { manifest["task"]=wrong;XCTAssertThrowsError(try NativeH3Preparation.compose(request:body())) }
  }

}
