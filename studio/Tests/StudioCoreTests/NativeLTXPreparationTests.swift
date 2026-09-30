import XCTest
import AVFoundation
import AppKit
import CryptoKit
@testable import StudioCore

final class NativeLTXPreparationTests: XCTestCase {
  func testA2VCompositionRetainsNonzeroSourceIntervalAndRejectsMissingControls() throws {
    let (root,original,runtime)=try fixture()
    var project=original
    let file=root.appendingPathComponent("voice.wav");try Data([1]).write(to:file)
    var asset=MediaAsset(name:"Voice",kind:.audio,path:file.path);asset.duration=4
    project.assets=[asset]
    project.clips[0].duration=2
    project.clips[0].generationSelection?.task="a2v"
    var driver=Attachment(assetID:asset.id,role:.audioDriver)
    driver.audioSourceStart=1.25;driver.audioSourceDuration=49.0/24.0
    project.clips[0].attachments=[driver]
    let composed=try NativeLTXPreparation.compose(request:request(project,runtime))
    let recipe=composed["recipe"] as! [String:Any]
    let conditioning=recipe["conditioning"] as! [String:Any]
    let inputs=conditioning["inputs"] as! [[String:Any]]
    XCTAssertEqual(conditioning["task"] as? String,"a2v")
    XCTAssertEqual(inputs.count,1)
    XCTAssertEqual(inputs[0]["role"] as? String,"audio_driver")
    XCTAssertEqual(inputs[0]["source_start_seconds"] as? Double,1.25)
    XCTAssertEqual(inputs[0]["source_duration_seconds"] as? Double,49.0/24.0)
    XCTAssertTrue((try NativeLTXPreparation.catalog(directory:root.path)[0]["generation"] as! [String:Any])["supportedTasks"] as! [String] == ["t2v","i2v","fflf","a2v","extension"])
    project.clips[0].attachments[0].audioSourceStart=nil
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[0].attachments[0]=driver
    project.clips[0].attachments[0].strength=0.5
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    let imageFile=root.appendingPathComponent("opening.png");try Data([1]).write(to:imageFile)
    let image=MediaAsset(name:"Opening",kind:.image,path:imageFile.path)
    project.assets.append(image)
    project.clips[0].attachments[0]=driver
    project.clips[0].attachments.append(Attachment(assetID:image.id,role:.first))
    let combined=try NativeLTXPreparation.compose(request:request(project,runtime))
    let combinedInputs=((combined["recipe"] as! [String:Any])["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(combinedInputs.map { $0["role"] as! String },["audio_driver","keyframe"])
  }
  func testPreparedTimelineAudioDriverFeedsNativeA2VWithoutDiscardingSelection() throws {
    let (root,original,runtime)=try fixture()
    let mixed=root.appendingPathComponent("timeline-driver.wav")
    try Data([1]).write(to:mixed)
    var project=original
    project.clips[0].duration=2
    project.clips[0].generationSelection?.task="a2v"
    project.clips[0].audioDriverSelection=AudioDriverSelection(mode:.voice)
    project.clips[0].audioDriverMixKey="frozen-mix-key"
    var asset=MediaAsset(name:"Timeline mix",kind:.audio,path:mixed.path,scope:.clip,owner:project.clips[0].id)
    asset.duration=2
    project.assets=[asset]
    project.clips[0].attachments=[Attachment(assetID:asset.id,role:.audioDriver)]
    let composed=try NativeLTXPreparation.compose(request:request(project,runtime))
    let inputs=((composed["recipe"] as! [String:Any])["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs[0]["source_start_seconds"] as? Double,0)
    XCTAssertEqual(inputs[0]["source_duration_seconds"] as? Double,2)
    project.clips[0].audioDriverMixKey=nil
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
  func fixture() throws -> (URL, StudioProject, [String: Any]) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "prompt": "old prompt", "components": ["transformer_path": "/models/transformer", "loras": [["/models/base.safetensors", 0.5]]],
      "config": ["pipeline_mode": "distilled", "stage1_steps": 8, "stage2_steps": 3, "frame_rate": 24,
        "width": 768, "height": 448, "seed": 1, "duration_seconds": 5],
      "conditioning": ["version": 1, "task": "fflf", "inputs": [["path": "/old.png"]]]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("model.json"))
    var project = StudioProject()
    var clip = Clip(); clip.prompt = "  Beowulf raises his cup.  "; clip.duration = 3.7; clip.seed = 43
    clip.generationWidth = 1344; clip.generationHeight = 768
    clip.generationSelection = GenerationSelection(task: "t2v")
    project.clips = [clip]
    return (root, project, ["profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"])
  }
  func request(_ project: StudioProject, _ runtime: [String: Any]) throws -> [String: Any] {
    ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": project.clips[0].id.uuidString, "runtime": runtime]
  }
  func testNativeCompositionUsesEditorialCoverageAndReplacesHiddenMedia() throws {
    let (_, project, runtime) = try fixture()
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    let recipe = try XCTUnwrap(result["recipe"] as? [String: Any])
    let config = try XCTUnwrap(recipe["config"] as? [String: Any])
    XCTAssertEqual(config["duration_seconds"] as? Double, 4)
    XCTAssertEqual(config["seed"] as? Int, 43)
    XCTAssertEqual(config["width"] as? Int, 1344)
    XCTAssertEqual(recipe["prompt"] as? String, "Beowulf raises his cup.")
    XCTAssertEqual(((recipe["conditioning"] as? [String: Any])?["inputs"] as? [Any])?.count, 0)
    XCTAssertEqual((result["report"] as? [String: Any])?["preserveEditorialDuration"] as? Bool, true)
  }
  func testCatalogSkipsInvalidFilesAndDescriptionRetainsControlsForMissingInput() throws {
    let (root, original, runtime) = try fixture()
    try Data("broken".utf8).write(to: root.appendingPathComponent("broken.json"))
    XCTAssertEqual(try NativeLTXPreparation.catalog(directory: root.path).count, 1)
    var project = original; project.clips[0].generationSelection?.task = "fflf"
    let description = try NativeLTXPreparation.describe(request: request(project, runtime))
    XCTAssertFalse((description["readinessErrors"] as? [String] ?? []).isEmpty)
    let generation = try XCTUnwrap(description["generation"] as? [String: Any])
    XCTAssertEqual((generation["controls"] as? [String: Any])?["evaluations"] as? Int, 8)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testCatalogDoesNotAdvertiseProfileRejectedBySwiftWorker() throws {
    let (root, _, _) = try fixture()
    let url = root.appendingPathComponent("model.json")
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let unsupported: [(String, String, Any)] = [
      ("components", "msr_lora_path", "/models/msr.safetensors"),
      ("components", "distilled_lora_path", "/models/distilled.safetensors"),
      ("components", "duration_head_path", "/models/duration.safetensors"),
      ("config", "stage1_steps", 7),
      ("config", "duration_mode", "automatic"),
      ("config", "stage1_sampler", "euler"),
      ("config", "dfr_enabled", true),
    ]
    for (section, key, value) in unsupported {
      var recipe = original
      var fields = recipe[section] as! [String: Any]
      fields[key] = value
      recipe[section] = fields
      try JSONSerialization.data(withJSONObject: recipe).write(to: url)
      XCTAssertTrue(try NativeLTXPreparation.catalog(directory: root.path).isEmpty,
        "The Swift worker rejects \(section).\(key); Studio must not offer this profile.")
    }
  }
  func testLegacyProfileNegativePromptIsReportedAndOmittedFromSwiftRecipe() throws {
    let (root, project, runtime) = try fixture()
    let url = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var config = recipe["config"] as! [String: Any]
    config["negative_prompt"] = "blurry"
    recipe["config"] = config
    try JSONSerialization.data(withJSONObject: recipe).write(to: url)

    XCTAssertEqual(try NativeLTXPreparation.catalog(directory: root.path).count, 1)
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    let effective = (result["recipe"] as! [String: Any])["config"] as! [String: Any]
    XCTAssertEqual(effective["negative_prompt"] as? String, "")
    let warnings = (result["report"] as! [String: Any])["warnings"] as! [String]
    XCTAssertTrue(warnings.contains { $0.contains("negative prompt") })
  }
  func testRejectsOverridesAndContinuousSceneLeader() throws {
    let (_, original, runtime) = try fixture()
    var project = original; project.clips[0].generationSelection?.steps = 7
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project = original
    var follower = Clip(); follower.continuity = ClipContinuity(mode: "scene")
    project.clips.append(follower)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testSwiftSceneCompositionUsesWholeGroupAndRejectsUnsupportedInputs() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    project.clips[0].duration = 2
    var follower = project.clips[0]
    follower.id = UUID(); follower.prompt = "The same warrior turns toward the fire."
    follower.seed = 44
    follower.soundscape = "Stale individual sound"
    follower.continuity = ClipContinuity(mode: "scene", sourceClipID: project.clips[0].id)
    project.clips.append(follower)
    let body = try request(project, runtime)
    let composed = try NativeLTXPreparation.compose(request: body)
    let recipe = try XCTUnwrap(composed["recipe"] as? [String: Any])
    let scene = try XCTUnwrap(recipe["scene"] as? [String: Any])
    let segments = try XCTUnwrap(scene["segments"] as? [[String: Any]])
    XCTAssertEqual(segments.map { $0["seed"] as? Int }, [43, 44])
    XCTAssertFalse((segments[1]["prompt"] as! String).contains("Stale individual sound"))
    XCTAssertTrue((segments[1]["prompt"] as! String).contains(project.clips[0].soundscape))
    XCTAssertEqual((recipe["config"] as? [String: Any])?["duration_seconds"] as? Double, 4)
    let report = try XCTUnwrap((composed["report"] as? [String: Any])?["scene"] as? [String: Any])
    let ranges = try XCTUnwrap(report["members"] as? [[String: Any]])
    XCTAssertEqual(ranges.map { $0["source_in"] as? Double }, [0, 2])
    XCTAssertEqual(ranges.map { $0["duration"] as? Double }, [2, 2])
    let prepared = try NativeLTXPreparation.prepare(request: body,
      destination: root.appendingPathComponent("scene-job"))
    XCTAssertNotNil((prepared["report"] as? [String: Any])?["scene"])
    project.clips[1].attachments.append(Attachment(assetID: UUID(), role: .first))
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[1].attachments = []
    project.clips[1].generationWidth = 768
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[1].generationWidth = project.clips[0].generationWidth
    project.clips[0].duration = 0.5
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testSwiftSceneCarriesOpeningAndLaterShotImagesInTheirOwnWindows() throws {
    let (root,original,runtime)=try fixture()
    let opening=root.appendingPathComponent("opening.png")
    try Data([1]).write(to:opening)
    let image=MediaAsset(name:"Opening",kind:.image,path:opening.path)
    let laterImage=root.appendingPathComponent("later.png")
    try Data([2]).write(to:laterImage)
    let laterAsset=MediaAsset(name:"Later",kind:.image,path:laterImage.path)
    var project=original
    project.assets=[image,laterAsset]
    project.clips[0].duration=2
    project.clips[0].generationSelection?.task="i2v"
    project.clips[0].attachments=[Attachment(assetID:image.id,role:.first)]
    var follower=project.clips[0]
    follower.id=UUID();follower.prompt="The warrior turns."
    follower.generationSelection?.task="t2v"
    follower.attachments=[]
    follower.continuity=ClipContinuity(mode:"scene",sourceClipID:project.clips[0].id)
    project.clips.append(follower)
    let composed=try NativeLTXPreparation.compose(request:request(project,runtime))
    let recipe=composed["recipe"] as! [String:Any]
    let conditioning=recipe["conditioning"] as! [String:Any]
    XCTAssertEqual(conditioning["task"] as? String,"fflf")
    XCTAssertEqual((conditioning["inputs"] as? [[String:Any]])?.count,1)
    XCTAssertEqual(((composed["report"] as! [String:Any])["conditioning"] as! [String:Any])["inputs"] as? Int,1)
    project.clips[1].attachments=[Attachment(assetID:laterAsset.id,role:.first)]
    project.clips[1].generationSelection?.task="i2v"
    let laterSong=root.appendingPathComponent("later-song.wav")
    try Data([3]).write(to:laterSong)
    let laterHash=SHA256.hash(data:Data([3])).map { String(format:"%02x",$0) }.joined()
    project.clips[1].musicSource=MusicShotSource(path:laterSong.path,sha256:laterHash,
      start:0,duration:2,task:"t2v")
    let later=try NativeLTXPreparation.compose(request:request(project,runtime))
    let laterRecipe=later["recipe"] as! [String:Any]
    let segments=((laterRecipe["scene"] as! [String:Any])["segments"] as! [[String:Any]])
    XCTAssertNil(segments[0]["image_input"])
    XCTAssertEqual((segments[1]["image_input"] as? [String:Any])?["path"] as? String,laterImage.path)
    XCTAssertEqual((segments[1]["image_input"] as? [String:Any])?["frame_index"] as? Int,0)
    let described=try NativeLTXPreparation.describe(request:request(project,runtime))
    XCTAssertTrue((described["sourcePaths"] as? [String] ?? []).contains(laterImage.path))
    XCTAssertTrue((described["sourcePaths"] as? [String] ?? []).contains(laterSong.path))
    XCTAssertTrue((described["readinessErrors"] as? [String] ?? []).isEmpty)
    let prepared=try NativeLTXPreparation.prepare(request:request(project,runtime),
      destination:root.appendingPathComponent("two-image-scene"))
    XCTAssertNotNil(prepared["recipePath"])
  }
  func testSwiftSceneUsesConsecutiveIntervalsOfOneSourceAudioFile() throws {
    let (root,original,runtime)=try fixture()
    let music=root.appendingPathComponent("song.wav");try Data([1]).write(to:music)
    var asset=MediaAsset(name:"Song",kind:.audio,path:music.path);asset.duration=8
    var project=original;project.assets=[asset]
    project.clips[0].duration=2
    project.clips[0].generationSelection?.task="a2v"
    var first=Attachment(assetID:asset.id,role:.audioDriver)
    first.audioSourceStart=1;first.audioSourceDuration=2
    project.clips[0].attachments=[first]
    var follower=project.clips[0]
    follower.id=UUID();follower.duration=3;follower.seed=44
    follower.continuity=ClipContinuity(mode:"scene",sourceClipID:project.clips[0].id)
    var second=Attachment(assetID:asset.id,role:.audioDriver)
    second.audioSourceStart=3;second.audioSourceDuration=3
    follower.attachments=[second];project.clips.append(follower)
    let composed=try NativeLTXPreparation.compose(request:request(project,runtime))
    let recipe=composed["recipe"] as! [String:Any]
    let conditioning=recipe["conditioning"] as! [String:Any]
    XCTAssertEqual(conditioning["task"] as? String,"a2v")
    let input=(conditioning["inputs"] as! [[String:Any]])[0]
    XCTAssertEqual(input["source_start_seconds"] as? Double,1)
    XCTAssertEqual(input["source_duration_seconds"] as? Double,5)
    let hash=SHA256.hash(data:Data([1])).map { String(format:"%02x",$0) }.joined()
    project.clips[0].musicSource=MusicShotSource(path:music.path,sha256:hash,start:1,duration:2,task:"a2v")
    project.clips[1].musicSource=MusicShotSource(path:music.path,sha256:hash,start:3,duration:3,task:"a2v")
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[1].musicSource?.start=3.25
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[1].musicSource?.start=3
    project.clips[1].attachments[0].audioSourceStart=3.25
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
  func testSwiftSceneFrozenRecipeUsesSavedBoundedDecodeChoice() throws {
    let (_, original, runtime) = try fixture()
    var project = original
    project.clips[0].duration = 2
    project.clips[0].continuity = try JSONDecoder().decode(ClipContinuity.self,
      from: Data(#"{"mode":"independent","sceneDecodeMode":"windowed"}"#.utf8))
    var follower = project.clips[0]
    follower.id = UUID()
    follower.prompt = "The same warrior turns toward the fire."
    follower.continuity = ClipContinuity(mode: "scene", sourceClipID: project.clips[0].id)
    project.clips.append(follower)
    let composed = try NativeLTXPreparation.compose(request: request(project, runtime))
    let recipe = try XCTUnwrap(composed["recipe"] as? [String: Any])
    let scene = try XCTUnwrap(recipe["scene"] as? [String: Any])
    XCTAssertEqual(scene["decode_mode"] as? String, "windowed")
    XCTAssertEqual(scene["decode_window_frames"] as? Int, 361)
    let report = try XCTUnwrap((composed["report"] as? [String: Any])?["scene"] as? [String: Any])
    XCTAssertEqual(report["publication_mode"] as? String, "windowed_decode_native_latent_chain")
  }
  func testSwiftSceneRejectsUnknownSavedDecodeChoiceBeforeRender() throws {
    let (_, original, runtime) = try fixture()
    var project = original
    project.clips[0].duration = 2
    project.clips[0].continuity = try JSONDecoder().decode(ClipContinuity.self,
      from: Data(#"{"mode":"independent","sceneDecodeMode":"unbounded-fast"}"#.utf8))
    var follower = project.clips[0]
    follower.id = UUID()
    follower.prompt = "The same warrior turns toward the fire."
    follower.continuity = ClipContinuity(mode: "scene", sourceClipID: project.clips[0].id)
    project.clips.append(follower)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testRejectsNegativePromptThatDistilledSwiftCannotEvaluate() throws {
    let (_, original, runtime) = try fixture()
    var project = original
    project.clips[0].negativePrompt = "no ghosting"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testPreparationPublishesSnapshotAndNeverOverwritesExistingJob() throws {
    let (root, project, runtime) = try fixture()
    let output = root.appendingPathComponent("job")
    let result = try NativeLTXPreparation.prepare(request: request(project, runtime), destination: output)
    XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(result["recipePath"] as? String)))
    XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("editor-request.json").path))
    XCTAssertThrowsError(try NativeLTXPreparation.prepare(request: request(project, runtime), destination: output))
  }
  func testLoRA23CompatibilityOrderAndConflictingHeaderRejection() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    let file = root.appendingPathComponent("style.safetensors")
    func writeHeader(_ model: String) throws {
      let header = try JSONSerialization.data(withJSONObject: ["__metadata__": ["model_version": model]])
      var length = UInt64(header.count).littleEndian
      var bytes = withUnsafeBytes(of: &length) { Data($0) }; bytes.append(header)
      try bytes.write(to: file)
    }
    try writeHeader("ltx-2.3")
    var asset = MediaAsset(name: "Style", kind: .lora, path: file.path)
    asset.loraModel = .ltx23; project.assets = [asset]
    var adapter = Attachment(assetID: asset.id, role: .lora); adapter.strength = 0.8
    project.clips[0].attachments = [adapter]
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    let recipe = result["recipe"] as! [String: Any]
    let pairs = (recipe["components"] as! [String: Any])["loras"] as! [[Any]]
    XCTAssertEqual(pairs.count, 2); XCTAssertEqual(pairs[1][0] as? String, file.path)
    XCTAssertEqual(pairs[1][1] as? Double, 0.8)
    try writeHeader("minimax-h3")
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[0].attachments[0].enabled = false
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }

  func testEndpointRolesAndSingleImageDoNotBecomeGenericReferences() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    let file = root.appendingPathComponent("image.png"); try Data([1]).write(to: file)
    let first = MediaAsset(name: "First", kind: .image, path: file.path)
    let last = MediaAsset(name: "Last", kind: .image, path: file.path)
    project.assets = [first, last]
    project.clips[0].generationSelection?.task = "fflf"
    project.clips[0].attachments = [Attachment(assetID: last.id, role: .last), Attachment(assetID: first.id, role: .first)]
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    let contract = (result["recipe"] as! [String: Any])["conditioning"] as! [String: Any]
    let inputs = contract["inputs"] as! [[String: Any]]
    XCTAssertEqual(inputs[0]["frame_index"] as? String, "last")
    XCTAssertEqual(inputs[1]["frame_index"] as? Int, 0)
    XCTAssertEqual(inputs[0]["role"] as? String, "keyframe")
    project.clips[0].attachments.removeFirst(); project.clips[0].generationSelection?.task = "i2v"
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[0].attachments.append(Attachment(assetID: last.id, role: .keyframe, time: 1))
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }

  func testMalformedProfileLoRAsAndH3OnlyAssetControlsFailClosed() throws {
    let (root, original, runtime) = try fixture()
    var project = original
    let file = root.appendingPathComponent("style.safetensors")
    let header = Data("{}".utf8); var size = UInt64(header.count).littleEndian
    var bytes = withUnsafeBytes(of: &size) { Data($0) }; bytes.append(header); try bytes.write(to: file)
    var asset = MediaAsset(name: "Style", kind: .lora, path: file.path); asset.loraModel = .ltx25
    project.assets = [asset]; project.clips[0].attachments = [Attachment(assetID: asset.id, role: .lora)]
    for field in ["profile", "layout", "grid"] {
      project.assets = [asset]
      if field == "profile" { project.assets[0].loraProfile = "standard" }
      if field == "layout" { project.assets[0].loraLayout = "auto" }
      if field == "grid" { project.assets[0].loraAdalnInputGrid = "/unused" }
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)), field)
    }
    project.assets = [asset]
    let url = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]; components["loras"] = ["invalid", [file.path, 1]] as [Any]
    recipe["components"] = components; try JSONSerialization.data(withJSONObject: recipe).write(to: url)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
  }
  func testDescriptionIncludesModelProvenanceFiles() throws {
    let (root, project, runtime) = try fixture()
    let model = root.appendingPathComponent("model")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    let names = ["paged_manifest.json", "model_identity.json", "conversion_provenance.json"]
    for name in names { try Data("{}".utf8).write(to: model.appendingPathComponent(name)) }
    let url = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]; components["transformer_path"] = model.path
    recipe["components"] = components; try JSONSerialization.data(withJSONObject: recipe).write(to: url)
    let description = try NativeLTXPreparation.describe(request: request(project, runtime))
    let paths = description["sourcePaths"] as? [String] ?? []
    for name in names { XCTAssertTrue(paths.contains(model.appendingPathComponent(name).path)) }
  }

  func continuityMovie(_ root: URL) async throws -> URL {
    let url = root.appendingPathComponent("variable.mov")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: 64, AVVideoHeightKey: 64])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB])
    writer.add(input); XCTAssertTrue(writer.startWriting()); writer.startSession(atSourceTime: .zero)
    for (frame, time) in [0.0, 0.2, 0.8, 1.1].enumerated() {
      while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
      var buffer: CVPixelBuffer?
      XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32ARGB, nil, &buffer), kCVReturnSuccess)
      let pixel = try XCTUnwrap(buffer); CVPixelBufferLockBaseAddress(pixel, [])
      let bytes = CVPixelBufferGetBaseAddress(pixel)!.assumingMemoryBound(to: UInt8.self)
      for y in 0..<64 { for x in 0..<64 {
        let offset = y * CVPixelBufferGetBytesPerRow(pixel) + x * 4
        bytes[offset] = 255
        for c in 0..<3 { bytes[offset + c + 1] = c == frame % 3 ? 255 : 0 }
      } }
      CVPixelBufferUnlockBaseAddress(pixel, [])
      XCTAssertTrue(adaptor.append(pixel, withPresentationTime: CMTime(seconds: time, preferredTimescale: 600)))
    }
    input.markAsFinished(); writer.endSession(atSourceTime: CMTime(seconds: 1.5, preferredTimescale: 600))
    await writer.finishWriting(); XCTAssertEqual(writer.status, .completed); return url
  }

  func testMatchPreviousFrameUsesLastVisibleVFRTimestampAndPreservesOriginalRequest() async throws {
    let (root, original, runtime) = try fixture()
    let movie = try await continuityMovie(root)
    for (end, expectedChannel) in [(0.8, 1), (0.95, 2)] {
      var project = original
      var source = Clip(engine: .movie); source.sourcePath = movie.path; source.sourceIn = 0.15; source.duration = end - 0.15
      var target = project.clips[0]; target.continuity = ClipContinuity(mode: "frame")
      let stored = MediaAsset(name: "Original first", kind: .image, path: root.appendingPathComponent("missing-original.png").path)
      project.assets.append(stored); target.attachments = [Attachment(assetID: stored.id, role: .first)]
      project.clips = [source, target]
      var body = try request(project, runtime); body["clipID"] = target.id.uuidString
      let described = try NativeLTXPreparation.describe(request: body)
      XCTAssertEqual(described["readinessErrors"] as? [String], [])
      XCTAssertTrue((described["sourcePaths"] as? [String] ?? []).contains(movie.path))
      let output = root.appendingPathComponent("prepared-\(end)")
      let result = try await NativeLTXPreparation.prepareWithMedia(request: body, destination: output)
      let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: result["recipePath"] as! String))) as! [String: Any]
      let inputs = (recipe["conditioning"] as! [String: Any])["inputs"] as! [[String: Any]]
      XCTAssertEqual(inputs.count, 1); XCTAssertEqual(inputs[0]["frame_index"] as? Int, 0)
      let imagePath = inputs[0]["path"] as! String
      XCTAssertTrue(imagePath.hasPrefix(output.path + "/"))
      let image = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: URL(fileURLWithPath: imagePath))))
      let color = try XCTUnwrap(image.colorAt(x: 32, y: 32)?.usingColorSpace(.deviceRGB))
      XCTAssertGreaterThan([color.redComponent, color.greenComponent, color.blueComponent][expectedChannel], 0.8)
      let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("editor-request.json"))) as! NSDictionary
      XCTAssertEqual(saved, body as NSDictionary)
      XCTAssertEqual(project.clips[1].attachments, target.attachments)
      let continuity = (result["report"] as! [String: Any])["continuity"] as! [String: Any]
      XCTAssertEqual(continuity["mode"] as? String, "frame")
      XCTAssertEqual(continuity["sourceFrameTime"] as! Double, expectedChannel == 1 ? 0.2 : 0.8, accuracy: 0.001)
    }
  }

  func testMatchPreviousFrameRejectsInvalidSourceAndInvalidatesChangedTrim() async throws {
    let (root, original, runtime) = try fixture(); let movie = try await continuityMovie(root)
    var project = original
    var source = Clip(engine: .movie); source.sourcePath = movie.path; source.duration = 0.8
    project.clips[0].continuity = ClipContinuity(mode: "frame")
    let targetID = project.clips[0].id
    project.clips.insert(source, at: 0)
    func body() throws -> [String: Any] { var result = try request(project, runtime); result["clipID"] = targetID.uuidString; return result }
    let before = try NativeLTXPreparation.describe(request: body())["fingerprint"] as? String
    project.clips[0].duration = 0.95
    XCTAssertNotEqual(try NativeLTXPreparation.describe(request: body())["fingerprint"] as? String, before)
    project.clips[1].continuity?.sourceClipID = targetID
    XCTAssertThrowsError(try NativeLTXPreparation.describe(request: body()))
    project.clips[1].continuity?.sourceClipID = nil; project.clips[0].duration = 10
    let output = root.appendingPathComponent("bad-job")
    do { _ = try await NativeLTXPreparation.prepareWithMedia(request: body(), destination: output); XCTFail("Out-of-range trim accepted") }
    catch { XCTAssertFalse(FileManager.default.fileExists(atPath: output.path)) }
  }

  func testExtensionExtractsOnlyVisibleTailAndPublishesFrozenRecipe() async throws {
    let ffmpeg = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else { throw XCTSkip("FFmpeg unavailable") }
    let (root, original, oldRuntime) = try fixture()
    let movie = root.appendingPathComponent("source.mp4")
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-f", "lavfi", "-i",
      "testsrc2=size=128x64:rate=24:duration=4", "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=48000:duration=4", "-map", "0:v:0",
      "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
      "-c:a", "aac", "-shortest", movie.path]
    try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
    var source = Clip(engine: .movie)
    source.sourcePath = movie.path; source.sourceIn = 1; source.duration = 2
    var target = original.clips[0]
    target.duration = 1; target.generationWidth = 128; target.generationHeight = 64
    target.extensionDirection = "after"; target.extensionSource = movie.path
    target.extensionClipID = source.id; target.generationSelection = GenerationSelection(task: "extension")
    var project = original; project.clips = [source, target]
    var runtime = oldRuntime; runtime["ffmpegPath"] = ffmpeg.path
    var body = try request(project, runtime); body["clipID"] = target.id.uuidString
    let description = try NativeLTXPreparation.describe(request: body)
    XCTAssertEqual(description["readinessErrors"] as? [String], [])
    let output = root.appendingPathComponent("extension-job")
    let prepared = try await NativeLTXPreparation.prepareWithMedia(request: body, destination: output)
    let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: prepared["recipePath"] as! String))) as! [String: Any]
    let condition = recipe["conditioning"] as! [String: Any]
    XCTAssertEqual(condition["task"] as? String, "extension")
    XCTAssertEqual((condition["extension"] as? [String: Any])?["context_frames"] as? Int, 25)
    XCTAssertEqual((condition["extension"] as? [String: Any])?["additional_frames"] as? Int, 24)
    let input = (condition["inputs"] as! [[String: Any]])[0]
    let tail = URL(fileURLWithPath: input["path"] as! String)
    XCTAssertTrue(tail.path.hasPrefix(output.path + "/"))
    XCTAssertEqual((input["sha256"] as? String)?.count, 64)
    let asset = AVURLAsset(url: tail)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    XCTAssertEqual(videoTracks.count, 1)
    XCTAssertEqual(audioTracks.count, 1)
    let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("editor-request.json"))) as! NSDictionary
    XCTAssertEqual(saved, body as NSDictionary)
    XCTAssertEqual(project.clips[1].extensionSource, movie.path)
    target.attachments = [Attachment(assetID: UUID(), role: .first)]
    project.clips[1] = target
    body = try request(project, runtime); body["clipID"] = target.id.uuidString
    XCTAssertFalse((try NativeLTXPreparation.describe(request: body)["readinessErrors"] as? [String] ?? []).isEmpty)
    target.attachments = []
    target.extensionDirection = ""; target.extensionSource = ""; target.extensionClipID = nil
    target.continuity = ClipContinuity(mode: "motion", sourceClipID: source.id)
    target.generationSelection = GenerationSelection(task: "t2v")
    source.sourceIn = 0.5; source.duration = 3
    project.clips = [source, target]
    body = try request(project, runtime); body["clipID"] = target.id.uuidString
    let motion = try await NativeLTXPreparation.prepareWithMedia(request: body,
      destination: root.appendingPathComponent("motion-job"))
    let motionRecipe = try JSONSerialization.jsonObject(with:
      Data(contentsOf: URL(fileURLWithPath: motion["recipePath"] as! String))) as! [String: Any]
    let motionCondition = motionRecipe["conditioning"] as! [String: Any]
    XCTAssertEqual((motionCondition["extension"] as? [String: Any])?["context_frames"] as? Int, 49)
    XCTAssertEqual((motion["report"] as? [String: Any])?["task"] as? String, "extension")
  }

}
