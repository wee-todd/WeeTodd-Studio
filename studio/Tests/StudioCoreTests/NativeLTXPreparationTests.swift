import XCTest
import AVFoundation
import AppKit
import CryptoKit
@testable import StudioCore

final class NativeLTXPreparationTests: XCTestCase {
  func automaticFixture() throws -> (URL,StudioProject,[String:Any]) {
    let (root,original,runtime)=try fixture(), head=try NativeLTXAutomaticDurationTests.headFixture(at:root)
    let file=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    var components=recipe["components"] as! [String:Any]
    components["duration_head_path"]=head.path
    components["duration_head_header_sha256"]=try NativeLTXAutomaticDuration.validateHead(at:head)
    recipe["components"]=components
    try JSONSerialization.data(withJSONObject:recipe).write(to:file)
    var project=original
    project.clips[0].generationSelection?.ltx25AutomaticDuration=LTX25AutomaticDurationSettings(experimentalEnabled:true,minimumSeconds:0.25,maximumSeconds:30)
    return(root,project,runtime)
  }
  func testAutomaticDurationFreezesHeaderAndMaximumAdmissionWithoutChangingEditorOrManualDefaults() throws {
    let (_,original,runtime)=try automaticFixture()
    let snapshot=original,encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
    let before=try encoder.encode(original)
    let output=try NativeLTXPreparation.compose(request:request(original,runtime))
    let recipe=output["recipe"] as! [String:Any], config=recipe["config"] as! [String:Any], report=output["report"] as! [String:Any]
    XCTAssertEqual(config["duration_mode"] as? String,"automatic")
    XCTAssertEqual(config["auto_duration_min_seconds"] as? Double,0.25)
    XCTAssertEqual(config["auto_duration_max_seconds"] as? Double,30)
    XCTAssertEqual((report["conditioning"] as? [String:Any])?["frames"] as? Int,713)
    XCTAssertEqual(report["preserveEditorialDuration"] as? Bool,false)
    XCTAssertEqual(((recipe["components"] as! [String:Any])["duration_head_header_sha256"] as? String)?.count,64)
    XCTAssertEqual(original,snapshot)
    XCTAssertEqual(try encoder.encode(original),before)
    var manual=original;manual.clips[0].generationSelection?.ltx25AutomaticDuration=nil
    let manualOutput=try NativeLTXPreparation.compose(request:request(manual,runtime))
    let manualRecipe=manualOutput["recipe"] as! [String:Any], manualConfig=manualRecipe["config"] as! [String:Any]
    XCTAssertNil(manualConfig["duration_mode"])
    XCTAssertEqual((manualOutput["report"] as? [String:Any])?["preserveEditorialDuration"] as? Bool,true)
  }
  func testAutomaticRequiresOptInHeadPinAndSupportedOneShotBeforeConditioning() throws {
    let (root,original,runtime)=try automaticFixture()
    var project=original;project.clips[0].generationSelection?.ltx25AutomaticDuration?.experimentalEnabled=false
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project=original;project.clips[0].generationSelection?.ltx25AutomaticDuration?.minimumSeconds=2.4
    project.clips[0].generationSelection?.ltx25AutomaticDuration?.maximumSeconds=2.5
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project=original;project.clips[0].generationSelection?.task="a2v"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime))) {
      XCTAssertTrue($0.localizedDescription.contains("require manual timing"))
    }
    project=original
    var follower=project.clips[0];follower.id=UUID()
    follower.continuity=ClipContinuity(mode:"scene",sourceClipID:project.clips[0].id)
    project.clips.append(follower)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime))) {
      XCTAssertTrue($0.localizedDescription.contains("continuous-scene shot intervals"))
    }
    let file=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    var components=recipe["components"] as! [String:Any];components["duration_head_header_sha256"]=String(repeating:"b",count:64)
    recipe["components"]=components;try JSONSerialization.data(withJSONObject:recipe).write(to:file)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(original,runtime))) {
      XCTAssertTrue($0.localizedDescription.contains("duration-head header changed"))
    }
  }
  func guidedFixture(_ mode: LTX25GuidanceMode) throws -> (URL, StudioProject, [String: Any]) {
    let (root, original, runtime) = try fixture()
    let url = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var config = recipe["config"] as! [String: Any]
    config["pipeline_mode"] = mode.rawValue
    config["stage1_steps"] = mode == .guided ? 30 : 15
    config["stage1_sampler"] = mode == .guided ? "euler_guided" : "res_2s_guided"
    config["video_cfg_scale"] = 3.0; config["audio_cfg_scale"] = 7.0
    recipe["config"] = config
    var components = recipe["components"] as! [String: Any]
    components["distilled_lora_path"] = "/models/official-distilled-helper.safetensors"
    recipe["components"] = components
    recipe["conditioning"] = ["version": 1, "task": "t2v", "inputs": []]
    try JSONSerialization.data(withJSONObject: recipe).write(to: url)
    var project = original
    project.clips[0].generationSelection?.ltx25Guidance = LTX25GuidanceSettings(mode: mode, experimentalEnabled: true)
    return (root, project, runtime)
  }

  func testGuidedModesPreserveNegativeAndAdvancedSettingsWithoutPython() throws {
    for mode in LTX25GuidanceMode.allCases {
      let (_, original, runtime) = try guidedFixture(mode)
      var project = original
      project.clips[0].negativePrompt = "  blur, extra fingers  "
      project.clips[0].generationSelection?.steps = 12
      project.clips[0].generationSelection?.cfg = 4
      project.clips[0].generationSelection?.ltx25Guidance?.audioCFG = 8
      project.clips[0].generationSelection?.ltx25Guidance?.stgScale = 0.5
      project.clips[0].generationSelection?.ltx25Guidance?.stgBlocks = [0, 28, 47]
      let snapshot=project,encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
      let before = try encoder.encode(project)
      let result = try NativeLTXPreparation.compose(request: request(project, runtime))
      let recipe = result["recipe"] as! [String: Any], config = recipe["config"] as! [String: Any]
      XCTAssertEqual(config["pipeline_mode"] as? String, mode.rawValue)
      XCTAssertEqual(config["stage1_steps"] as? Int, 12)
      XCTAssertEqual(config["stage2_steps"] as? Int, 3)
      XCTAssertEqual(config["negative_prompt"] as? String, "blur, extra fingers")
      XCTAssertEqual(config["video_cfg_scale"] as? Double, 4)
      XCTAssertEqual(config["audio_cfg_scale"] as? Double, 8)
      XCTAssertEqual(config["stg_scale"] as? Double, 0.5)
      XCTAssertEqual(config["stg_blocks"] as? [Int], [0, 28, 47])
      XCTAssertEqual(project,snapshot)
      XCTAssertEqual(try encoder.encode(project), before)
      let controls = (result["report"] as! [String: Any])["generation"] as! [String: Any]
      XCTAssertTrue(((controls["controls"] as! [String: Any])["stepsEditable"] as? Bool) == true)
      XCTAssertTrue((result["report"] as! [String: Any])["productionQualified"] as? Bool == false)
    }
  }

  func testGuidedAdmissionRequiresExplicitOptInAndCompatibleMode() throws {
    let (_, original, runtime) = try guidedFixture(.guided)
    var project = original
    project.clips[0].generationSelection?.ltx25Guidance?.experimentalEnabled = false
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[0].generationSelection?.ltx25Guidance = nil
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project = original
    project.clips[0].generationSelection?.ltx25Guidance?.audioCFG = .nan
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project = original
    project.clips[0].generationSelection?.ltx25Guidance?.stgBlocks = [28, 28]
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project = original
    project.clips[0].generationSelection?.refinementSteps = 4
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    let (_, distilled, distilledRuntime) = try fixture()
    var incompatible = distilled
    incompatible.clips[0].generationSelection?.ltx25Guidance = LTX25GuidanceSettings(mode: .guided, experimentalEnabled: true)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(incompatible, distilledRuntime)))
  }

  func testGuidedCustomSigmasRetainPartialNoiseAndRejectInvalidSchedules() throws {
    let (_, original, runtime) = try guidedFixture(.guided)
    var project = original
    project.clips[0].generationSelection?.steps = 2
    project.clips[0].generationSelection?.ltx25Guidance?.sigmas = [0.8, 0.3, 0]
    let recipe = try NativeLTXPreparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((recipe["config"] as! [String: Any])["stage1_sigmas"] as? [Double], [0.8, 0.3, 0])
    for sigmas in [[1.0, 0], [1, 0.5, 0.1], [1, 1, 0], [0, 0.5, 0], [1.1, 0.5, 0]] {
      project.clips[0].generationSelection?.ltx25Guidance?.sigmas = sigmas
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    }
  }

  func testGuidedProfileRejectsMalformedInheritedGuidanceBeforePublication() throws {
    for (key, value) in [("stage1_steps", true as Any), ("stage1_steps", 30.5 as Any),
      ("audio_cfg_scale", true as Any), ("stg_blocks", [true] as Any),
      ("stage1_sigmas", "adaptive" as Any), ("video_rescale_scale", 2 as Any)] {
      let (root, project, runtime) = try guidedFixture(.guided)
      let url = root.appendingPathComponent("model.json")
      var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
      var config = recipe["config"] as! [String: Any]; config[key] = value; recipe["config"] = config
      try JSONSerialization.data(withJSONObject: recipe).write(to: url)
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)), key)
    }
  }

  func testGuidedAdaptiveOverrideClearsProfileSigmasExplicitly() throws {
    let (root, original, runtime) = try guidedFixture(.guided)
    let url = root.appendingPathComponent("model.json")
    var definition = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var config = definition["config"] as! [String: Any]
    config["stage1_steps"] = 2; config["stage1_sigmas"] = [0.8, 0.3, 0]
    definition["config"] = config
    try JSONSerialization.data(withJSONObject: definition).write(to: url)
    let inherited = try NativeLTXPreparation.compose(request: request(original, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((inherited["config"] as! [String: Any])["stage1_sigmas"] as? [Double], [0.8, 0.3, 0])
    var project = original; project.clips[0].generationSelection?.ltx25Guidance?.sigmas = []
    let adaptive = try NativeLTXPreparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
    XCTAssertTrue((adaptive["config"] as! [String: Any])["stage1_sigmas"] is NSNull)
  }

  func specializedControlProfile(_ root: URL, family: String) throws {
    let file = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    let adapters: [[Any]] = family == "crossview_ingredients"
      ? [["/models/crossview.safetensors",1.0],["/models/ingredients.safetensors",1.0]]
      : [["/models/control.safetensors",1.0]]
    recipe["components"] = ["transformer_path":"/models/transformer","loras":[],"ic_loras":adapters]
    recipe["conditioning"] = ["version":1,"task":"control","inputs":[],"control_family":family]
    try JSONSerialization.data(withJSONObject:recipe).write(to:file)
  }
  func testMotionTrackCompositionKeepsTwoStageScheduleAndRejectsWrongGuide() throws {
    let (root,original,runtime) = try fixture(); try specializedControlProfile(root,family:"motion_track")
    var project = original; project.clips[0].generationSelection = GenerationSelection(task:"control")
    project.clips[0].generationWidth = 512; project.clips[0].generationHeight = 256
    let file = root.appendingPathComponent("track.mp4"); try Data([1,2]).write(to:file)
    let asset = MediaAsset(name:"Trajectories",kind:.video,path:file.path); project.assets = [asset]
    var guide = Attachment(assetID:asset.id,role:.control); guide.controlType = "motion_track"; guide.strength = 0.7
    project.clips[0].attachments = [guide]
    let frozen = project.clips[0].attachments
    let result = try NativeLTXPreparation.compose(request:request(project,runtime))
    let recipe = result["recipe"] as! [String:Any], config = recipe["config"] as! [String:Any]
    let condition = recipe["conditioning"] as! [String:Any], inputs = condition["inputs"] as! [[String:Any]]
    XCTAssertEqual(config["stage1_steps"] as? Int,8); XCTAssertEqual(config["stage2_steps"] as? Int,3)
    XCTAssertEqual(condition["control_family"] as? String,"motion_track")
    XCTAssertEqual(condition["audio_policy"] as? String,"generated")
    XCTAssertEqual(inputs[0]["path"] as? String,file.path); XCTAssertEqual(inputs[0]["strength"] as? Double,0.7)
    XCTAssertEqual(project.clips[0].attachments,frozen)
    project.clips[0].attachments[0].controlType = "pose_skeleton"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[0].attachments = [guide,guide]
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[0].attachments = [guide]; project.clips[0].generationSelection?.refinementSteps = 4
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
  func testCrossViewCompositionOrdersWarpSourceAndSheetAndRejectsMissingLabels() throws {
    let (root,original,runtime) = try fixture()
    for family in ["crossview_warp","crossview_ingredients"] {
      try specializedControlProfile(root,family:family)
      var project = original; project.clips[0].generationSelection = GenerationSelection(task:"control")
      project.clips[0].duration = 5; project.clips[0].generationWidth = 512; project.clips[0].generationHeight = 256
      var assets: [MediaAsset] = [], attachments: [Attachment] = []
      for role in ["source","warp"] {
        let file = root.appendingPathComponent(role+".mp4"); try Data(role.utf8).write(to:file)
        let asset = MediaAsset(name:role,kind:.video,path:file.path); assets.append(asset)
        var guide = Attachment(assetID:asset.id,role:.control); guide.controlType = "crossview_warp"; guide.referenceRole = role
        attachments.append(guide)
      }
      if family == "crossview_ingredients" {
        let file = root.appendingPathComponent("sheet.png"); try Data([1]).write(to:file)
        let asset = MediaAsset(name:"Sheet",kind:.image,path:file.path); assets.append(asset)
        var sheet = Attachment(assetID:asset.id,role:.control); sheet.controlType = "ingredients_reference_sheet"; sheet.description = "A hero in three views"
        attachments.insert(sheet,at:0)
      }
      project.assets = assets; project.clips[0].attachments = attachments
      let result = try NativeLTXPreparation.compose(request:request(project,runtime))
      let recipe = result["recipe"] as! [String:Any], condition = recipe["conditioning"] as! [String:Any]
      let inputs = condition["inputs"] as! [[String:Any]]
      XCTAssertEqual(inputs.prefix(2).compactMap { $0["reference_role"] as? String },["warp","source"])
      XCTAssertEqual(inputs.prefix(2).compactMap { $0["path"] as? String },[root.appendingPathComponent("warp.mp4").path,root.appendingPathComponent("source.mp4").path])
      XCTAssertEqual(condition["audio_policy"] as? String,"source")
      XCTAssertEqual((recipe["config"] as! [String:Any])["stage2_steps"] as? Int,3)
      XCTAssertEqual(project.clips[0].attachments,attachments)
      if family == "crossview_ingredients" { XCTAssertEqual(inputs.last?["control_type"] as? String,"ingredients_reference_sheet") }
      project.clips[0].attachments.removeLast()
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
      project.clips[0].attachments = attachments
      let sourceIndex = try XCTUnwrap(project.clips[0].attachments.firstIndex { $0.referenceRole == "source" })
      project.clips[0].attachments[sourceIndex].referenceRole = "warp"
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
      project.clips[0].attachments[sourceIndex].referenceRole = nil
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    }
  }
  func testMotionTrackAndCrossViewGuidePreparationUseTheirRespectiveCanvases() async throws {
    let (root,_,_) = try fixture(), movie = try await continuityMovie(root)
    for (downscale,width,height) in [(2,128,64),(1,256,128)] {
      let output = root.appendingPathComponent("guide-\(downscale).rgb24")
      let digest = try await NativeLTXControlGuide.prepare(source:movie,destination:output,width:512,height:256,
        frames:33,fps:24,editorialDuration:1,referenceDownscale:downscale)
      let bytes = try Data(contentsOf:output), frameBytes = width*height*3
      XCTAssertEqual(bytes.count,33*frameBytes)
      XCTAssertEqual(digest,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
      for (frame,channel) in [(0,0),(5,1),(20,2),(32,0)] {
        let center = frame*frameBytes+((height/2)*width+width/2)*3
        XCTAssertGreaterThan(bytes[center+channel],200)
        XCTAssertLessThan(bytes[center+(channel+1)%3],35)
      }
    }
  }
  func testIngredientsSheetFreezesRepeatedRGBFramesAndDoesNotReplaceExistingPublication() throws {
    let (root,_,_) = try fixture(), image = root.appendingPathComponent("sheet.png")
    let bitmap = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:32,pixelsHigh:32,
      bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0))
    let pixels = try XCTUnwrap(bitmap.bitmapData)
    for y in 0..<32 { for x in 0..<32 {
      let offset = y*bitmap.bytesPerRow+x*4
      pixels[offset] = 255; pixels[offset+1] = 0; pixels[offset+2] = 0; pixels[offset+3] = 255
    } }
    try XCTUnwrap(bitmap.representation(using:.png,properties:[:])).write(to:image)
    let output = root.appendingPathComponent("sheet.rgb24")
    let digest = try NativeLTXControlGuide.prepareSheet(source:image,destination:output,width:64,height:32,frames:121)
    let bytes = try Data(contentsOf:output), frameBytes = 64*32*3
    XCTAssertEqual(bytes.count,121*frameBytes)
    XCTAssertEqual(bytes.prefix(frameBytes),bytes.suffix(frameBytes))
    let center=((32/2)*64+64/2)*3
    XCTAssertGreaterThan(bytes[center],240); XCTAssertLessThan(bytes[center+1],10)
    XCTAssertEqual(Array(bytes.prefix(3)),[0,0,0])
    XCTAssertEqual(digest,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
    XCTAssertThrowsError(try NativeLTXControlGuide.prepareSheet(source:image,destination:output,width:64,height:32,frames:121))
    XCTAssertEqual(try Data(contentsOf:output),bytes)
    XCTAssertThrowsError(try NativeLTXControlGuide.prepareSheet(source:image,destination:root.appendingPathComponent("too-short.rgb24"),width:64,height:32,frames:113))
  }
  func testIngredientsPortraitSheetRetainsHeadAndFooterWithCenteredBlackPadding() throws {
    let (root,_,_)=try fixture(),image=root.appendingPathComponent("portrait-sheet.png")
    let bitmap=try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:128,pixelsHigh:384,
      bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0))
    let pixels=try XCTUnwrap(bitmap.bitmapData)
    for y in 0..<384 { for x in 0..<128 {
      let offset=y*bitmap.bytesPerRow+x*4,channel=y<64 ? 0 : y>=320 ? 2 : 1
      for c in 0..<3 { pixels[offset+c]=c == channel ? 255 : 0 };pixels[offset+3]=255
    } }
    try XCTUnwrap(bitmap.representation(using:.png,properties:[:])).write(to:image)
    let destination=root.appendingPathComponent("portrait-guide.rgb24")
    _=try NativeLTXControlGuide.prepareSheet(source:image,destination:destination,width:256,height:128,frames:121)
    let bytes=try Data(contentsOf:destination),frameBytes=256*128*3
    XCTAssertEqual(bytes.count,121*frameBytes);XCTAssertEqual(bytes.prefix(frameBytes),bytes.suffix(frameBytes))
    // The head, middle and footer bands must all survive the fit.
    for (y,channel) in [(8,0),(64,1),(120,2)] {
      let offset=(y*256+128)*3
      XCTAssertGreaterThan(bytes[offset+channel],240)
      XCTAssertLessThan(bytes[offset+(channel+1)%3],10)
    }
    for x in [0,64,192,255] {
      let offset=(64*256+x)*3;XCTAssertEqual(Array(bytes[offset..<offset+3]),[0,0,0])
    }
  }
  func testCrossViewPreparationFreezesOrderedMoviesAndOriginalSourceAudioWithoutMutatingEditor() async throws {
    let ffmpeg = URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("FFmpeg unavailable") }
    let (root,original,runtime) = try fixture(); try specializedControlProfile(root,family:"crossview_warp")
    let movie = root.appendingPathComponent("original-source.mp4"), process = Process()
    process.executableURL = ffmpeg; process.arguments = ["-v","error","-f","lavfi","-i","color=c=red:size=64x64:rate=24:duration=1.5",
      "-f","lavfi","-i","sine=frequency=440:sample_rate=48000:duration=1.5","-map","0:v:0","-map","1:a:0",
      "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",movie.path]
    try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus,0)
    let warp = try await continuityMovie(root)
    var project = original; project.clips[0].generationSelection = GenerationSelection(task:"control")
    project.clips[0].duration = 1; project.clips[0].generationWidth = 512; project.clips[0].generationHeight = 256
    var attachments: [Attachment] = []
    for (role,file) in [("source",movie),("warp",warp)] {
      let asset = MediaAsset(name:role,kind:.video,path:file.path); project.assets.append(asset)
      var attachment = Attachment(assetID:asset.id,role:.control); attachment.controlType = "crossview_warp"; attachment.referenceRole = role
      attachments.append(attachment)
    }
    project.clips[0].attachments = attachments
    let body = try request(project,runtime), destination = root.appendingPathComponent("crossview-job")
    let result = try await NativeLTXPreparation.prepareWithMedia(request:body,destination:destination)
    let recipe = try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as! [String:Any]
    let conditioning = recipe["conditioning"] as! [String:Any], inputs = conditioning["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs.compactMap { $0["reference_role"] as? String },["warp","source"])
    for input in inputs {
      let file = URL(fileURLWithPath:input["path"] as! String), bytes = try Data(contentsOf:file)
      XCTAssertEqual(bytes.count,25*256*128*3)
      XCTAssertEqual(input["sha256"] as? String,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
      XCTAssertEqual(input["format"] as? String,"rgb24")
    }
    let audio = try XCTUnwrap(conditioning["publication_audio"] as? [String:Any])
    let audioFile = URL(fileURLWithPath:audio["path"] as! String), bytes = try Data(contentsOf:audioFile)
    XCTAssertFalse(bytes.isEmpty); XCTAssertEqual(audio["sha256"] as? String,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
    let duration = try await AVURLAsset(url:audioFile).load(.duration).seconds
    XCTAssertEqual(duration,1,accuracy:0.025)
    let saved = try JSONSerialization.jsonObject(with:Data(contentsOf:destination.appendingPathComponent("editor-request.json"))) as! NSDictionary
    XCTAssertEqual(saved,body as NSDictionary); XCTAssertEqual(project.clips[0].attachments,attachments)
  }
  func testCrossViewAudioFreezesExactFloatPCMAndPreservesDelayedMovieTiming() async throws {
    let ffmpeg = URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("FFmpeg unavailable") }
    let (root,_,_) = try fixture(), source = root.appendingPathComponent("float-source.wav")
    let rate = 48000, channels = 2, frames = rate*3/2, frameBytes = channels*4
    var pcm = Data()
    for frame in 0..<frames {
      let value = Float((frame % 201)-100)/128
      // AVFoundation canonicalizes negative zero; use only positive zero so
      // bit equality measures waveform preservation across its PCM decoder.
      for sample in [value,value == 0 ? 0 : -value] {
        var bits = sample.bitPattern.littleEndian
        withUnsafeBytes(of:&bits) { pcm.append(contentsOf:$0) }
      }
    }
    var wav = Data()
    func text(_ value:String) { wav.append(contentsOf:value.utf8) }
    func word<T:FixedWidthInteger>(_ value:T) { var little=value.littleEndian;withUnsafeBytes(of:&little) { wav.append(contentsOf:$0) } }
    text("RIFF");word(UInt32(36+pcm.count));text("WAVEfmt ");word(UInt32(16));word(UInt16(3))
    word(UInt16(channels));word(UInt32(rate));word(UInt32(rate*frameBytes));word(UInt16(frameBytes));word(UInt16(32))
    text("data");word(UInt32(pcm.count));wav.append(pcm);try wav.write(to:source)
    let movie = root.appendingPathComponent("delayed-float-source.mov"), process = Process()
    process.executableURL = ffmpeg;process.arguments = ["-v","error","-f","lavfi","-i","color=c=red:size=64x64:rate=24:duration=2",
      "-itsoffset","0.125","-i",source.path,"-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p",
      "-c:a","pcm_f32le",movie.path]
    try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
    func decodedPCM(_ url:URL) async throws -> Data {
      let asset = AVURLAsset(url:url), tracks = try await asset.loadTracks(withMediaType:.audio)
      let track = try XCTUnwrap(tracks.first)
      let reader = try AVAssetReader(asset:asset), output = AVAssetReaderTrackOutput(track:track,outputSettings:[
        AVFormatIDKey:kAudioFormatLinearPCM,AVLinearPCMBitDepthKey:32,AVLinearPCMIsFloatKey:true,
        AVLinearPCMIsBigEndianKey:false,AVLinearPCMIsNonInterleaved:false])
      reader.add(output);XCTAssertTrue(reader.startReading());var result = Data()
      while let sample = output.copyNextSampleBuffer() {
        let format = try XCTUnwrap(CMSampleBufferGetFormatDescription(sample))
        let info = try XCTUnwrap(CMAudioFormatDescriptionGetStreamBasicDescription(format))
        XCTAssertEqual(info.pointee.mSampleRate,Double(rate));XCTAssertEqual(info.pointee.mChannelsPerFrame,UInt32(channels))
        let block = try XCTUnwrap(CMSampleBufferGetDataBuffer(sample)), count = CMBlockBufferGetDataLength(block)
        var bytes = Data(count:count)
        let status = bytes.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block,atOffset:0,dataLength:count,destination:$0.baseAddress!) }
        XCTAssertEqual(status,kCMBlockBufferNoErr);result.append(bytes)
      }
      XCTAssertEqual(reader.status,.completed);return result
    }
    for (name,input,delay) in [("wav",source,0),("movie",movie,6000)] {
      let destination = root.appendingPathComponent(name+"-frozen.wav")
      let hash = try await NativeLTXControlGuide.prepareAudio(source:input,destination:destination,duration:1)
      let bytes = try Data(contentsOf:destination)
      XCTAssertEqual(hash,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
      var expected = Data(repeating:0,count:delay*frameBytes);expected.append(pcm.prefix((rate-delay)*frameBytes))
      let actual = try await decodedPCM(destination), duration = try await AVURLAsset(url:destination).load(.duration).seconds
      let difference = zip(actual,expected).enumerated().first { $0.element.0 != $0.element.1 }
      XCTAssertEqual(actual,expected,"\(name) first difference: \(String(describing:difference))")
      XCTAssertEqual(duration,1,accuracy:1.0/Double(rate))
    }
    let original = try await decodedPCM(source)
    let difference = zip(original,pcm).enumerated().first { $0.element.0 != $0.element.1 }
    XCTAssertEqual(original,pcm,"source first difference: \(String(describing:difference))")
  }
  func testControlPublicationRejectsSheetAndLaterMovieChangedAfterComposition() async throws {
    let ffmpeg=URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("FFmpeg unavailable") }
    let (root,original,runtime)=try fixture();try specializedControlProfile(root,family:"crossview_ingredients")
    let template=root.appendingPathComponent("template.mp4"),process=Process()
    process.executableURL=ffmpeg;process.arguments=["-v","error","-f","lavfi","-i","color=c=red:size=64x64:rate=24:duration=5.5",
      "-f","lavfi","-i","sine=frequency=440:sample_rate=48000:duration=5.5","-map","0:v:0","-map","1:a:0",
      "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",template.path]
    try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
    let warp=root.appendingPathComponent("warp.mp4");try FileManager.default.copyItem(at:template,to:warp)
    func png(_ channel:Int) throws -> Data {
      let bitmap=try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:32,pixelsHigh:32,
        bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0))
      let pixels=try XCTUnwrap(bitmap.bitmapData)
      for y in 0..<32 { for x in 0..<32 { let offset=y*bitmap.bytesPerRow+x*4
        for c in 0..<3 { pixels[offset+c]=c == channel ? 255 : 0 };pixels[offset+3]=255
      } }
      return try XCTUnwrap(bitmap.representation(using:.png,properties:[:]))
    }
    let red=try png(0),green=try png(1),movieBytes=try Data(contentsOf:template)
    for changedKind in ["sheet","movie"] {
      let source=root.appendingPathComponent(changedKind+"-source.mp4"),sheet=root.appendingPathComponent(changedKind+"-sheet.png")
      try movieBytes.write(to:source);try red.write(to:sheet)
      var project=original;project.clips[0].generationSelection=GenerationSelection(task:"control")
      project.clips[0].duration=5;project.clips[0].generationWidth=512;project.clips[0].generationHeight=256
      var attachments:[Attachment]=[]
      for (role,file) in [("warp",warp),("source",source),("ingredients",sheet)] {
        let asset=MediaAsset(name:role,kind:role == "ingredients" ? .image : .video,path:file.path);project.assets.append(asset)
        var attachment=Attachment(assetID:asset.id,role:.control)
        attachment.controlType=role == "ingredients" ? "ingredients_reference_sheet" : "crossview_warp"
        if role == "ingredients" { attachment.description="A red reference sheet" }
        else { attachment.referenceRole=role };attachments.append(attachment)
      }
      project.clips[0].attachments=attachments
      let body=try request(project,runtime),destination=root.appendingPathComponent(changedKind+"-job")
      let mutation=Task.detached { () throws -> Bool in
        for _ in 0..<2000 {
          try Task.checkCancellation()
          let staged=(try? FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)) ?? []
          if staged.contains(where:{ $0.lastPathComponent.hasPrefix(".prepare-") &&
            FileManager.default.fileExists(atPath:$0.appendingPathComponent("control-guide-0.rgb24").path) }) {
            if changedKind == "sheet" { try green.write(to:sheet,options:.atomic) }
            else { var changed=movieBytes;changed.append(0);try changed.write(to:source,options:.atomic) }
            return true
          }
          try await Task.sleep(nanoseconds:1_000_000)
        }
        return false
      }
      do {
        _=try await NativeLTXPreparation.prepareWithMedia(request:body,destination:destination)
        XCTFail("Publishing must reject source bytes changed since composition")
      } catch { XCTAssertTrue(error.localizedDescription.contains("changed after composition"),error.localizedDescription) }
      let mutated=try await mutation.value;XCTAssertTrue(mutated,"Mutation must occur after composition while the first guide is staged")
      XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
      XCTAssertFalse(try FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)
        .contains { $0.lastPathComponent.hasPrefix(".prepare-") })
      XCTAssertEqual(project.clips[0].attachments,attachments)
    }
  }

  func testUnionCompositionRetainsOneVideoAndRejectsWrongFamily() throws {
    let (root,original,runtime)=try fixture();let profile=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    recipe["components"]=["transformer_path":"/models/transformer","loras":[],"ic_loras":[["/models/union.safetensors",1.0]]]
    recipe["conditioning"]=["version":1,"task":"control","inputs":[]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    var project=original;project.clips[0].generationSelection?.task="control"
    project.clips[0].generationWidth=512;project.clips[0].generationHeight=256
    let file=root.appendingPathComponent("guide.mp4");try Data([1]).write(to:file)
    let asset=MediaAsset(name:"Pose guide",kind:.video,path:file.path);project.assets=[asset]
    var attachment=Attachment(assetID:asset.id,role:.control);attachment.controlType="pose_skeleton";attachment.strength=0.7
    project.clips[0].attachments=[attachment]
    XCTAssertEqual(try NativeLTXPreparation.catalog(directory:root.path).count,1)
    let result=try NativeLTXPreparation.compose(request:request(project,runtime))
    let input=(((result["recipe"] as! [String:Any])["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[0]
    XCTAssertEqual(input["kind"] as? String,"video");XCTAssertEqual(input["control_type"] as? String,"pose_skeleton")
    var ingredients=recipe,ingredientsConfig=recipe["config"] as! [String:Any]
    ingredientsConfig["ic_lora_single_stage"]=true;ingredients["config"]=ingredientsConfig
    try JSONSerialization.data(withJSONObject:ingredients).write(to:root.appendingPathComponent("aaa-ingredients.json"))
    let selected=try NativeLTXPreparation.compose(request:request(project,runtime))
    XCTAssertEqual((selected["report"] as? [String:Any])?["profile"] as? String,"model")
    project.clips[0].generationWidth=1344
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[0].generationWidth=512
    project.clips[0].attachments[0].controlType="motion_track"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    project.clips[0].attachments=[attachment,attachment]
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
  func testMSRCompositionFreezesReferencesAndRetainsEditorParameters() throws {
    let (root,original,runtime)=try fixture()
    let profile=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    recipe["components"]=["transformer_path":"/models/transformer","loras":[],
      "msr_lora_path":"/models/msr.safetensors","msr_lora_strength":1.0,
      "ic_loras":[["/models/msr.safetensors",1.0]]]
    var config=recipe["config"] as! [String:Any];config["ic_lora_single_stage"]=true
    recipe["config"]=config;recipe["conditioning"]=["version":1,"task":"ref2va","inputs":[]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    var project=original;project.clips[0].generationSelection?.task="ref2va"
    let file=root.appendingPathComponent("hero.png");try Data([1,2,3]).write(to:file)
    let asset=MediaAsset(name:"Hero",kind:.image,path:file.path)
    project.assets=[asset]
    var attachment=Attachment(assetID:asset.id,role:.reference)
    attachment.description="Warrior with braided hair";attachment.referenceFrames="25"
    attachment.referenceSizePolicy="balanced";attachment.attentionStrength=0.7
    project.clips[0].attachments=[attachment]
    let result=try NativeLTXPreparation.compose(request:request(project,runtime))
    let content=result["recipe"] as! [String:Any]
    let inputs=(content["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs[0]["sha256"] as? String,SHA256.hash(data:Data([1,2,3])).map { String(format:"%02x",$0) }.joined())
    XCTAssertEqual(inputs[0]["reference_frames"] as? String,"25")
    XCTAssertEqual(inputs[0]["attention_strength"] as? Double,0.7)
    XCTAssertEqual(inputs[0]["description"] as? String,attachment.description)
    project.clips[0].attachments[0].attentionStrength=2
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
  func testIngredientsCompositionRequiresOneDescribedSheetAndFiveSeconds() throws {
    let (root,original,runtime)=try fixture()
    let profile=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    recipe["components"]=["transformer_path":"/models/transformer","loras":[],
      "ic_loras":[["/models/ingredients.safetensors",1.0]]]
    var config=recipe["config"] as! [String:Any];config["ic_lora_single_stage"]=true
    recipe["config"]=config;recipe["conditioning"]=["version":1,"task":"control","inputs":[]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    var project=original;project.clips[0].generationSelection?.task="control";project.clips[0].duration=5
    let file=root.appendingPathComponent("sheet.png");try Data([1]).write(to:file)
    let asset=MediaAsset(name:"Character sheet",kind:.image,path:file.path);project.assets=[asset]
    var attachment=Attachment(assetID:asset.id,role:.control)
    attachment.controlType="ingredients_reference_sheet";attachment.description="A warrior in four views"
    project.clips[0].attachments=[attachment]
    let result=try NativeLTXPreparation.compose(request:request(project,runtime))
    XCTAssertEqual(((result["recipe"] as! [String:Any])["conditioning"] as! [String:Any])["task"] as? String,"control")
    config["single_stage_sampler"]="euler_ancestral_cfg_pp";recipe["config"]=config
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    let authored=try NativeLTXPreparation.compose(request:request(project,runtime))
    let generation=try XCTUnwrap((authored["report"] as? [String:Any])?["generation"] as? [String:Any])
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,16)
    XCTAssertEqual(((authored["recipe"] as! [String:Any])["config"] as! [String:Any])["single_stage_sampler"] as? String,"euler_ancestral_cfg_pp")
    project.clips[0].duration=4
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
  }
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
  func testDFRProfileOffersOnlyEndpointTasksAndPreservesControls() throws {
    let (root, original, runtime)=try fixture()
    let url=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    var config=recipe["config"] as! [String:Any]
    config["dfr_enabled"]=true
    config["dfr_detailing_lora_path"]="/models/detail.safetensors"
    config["dfr_detailing_lora_strength"]=0.5
    config["dfr_temporal_rounds"]=1
    config["dfr_temporal_upsampler_path"]="/models/temporal.safetensors"
    recipe["config"]=config
    var components=recipe["components"] as! [String:Any]
    components["loras"]=[]
    recipe["components"]=components
    try JSONSerialization.data(withJSONObject:recipe).write(to:url)
    let profiles=try NativeLTXPreparation.catalog(directory:root.path)
    XCTAssertEqual(profiles.count,1)
    XCTAssertEqual((profiles[0]["generation"] as? [String:Any])?["supportedTasks"] as? [String],["t2v","i2v","fflf"])
    var project=original
    project.clips[0].profileID=url.path
    let content=try NativeLTXPreparation.compose(request:request(project,runtime))["recipe"] as! [String:Any]
    let effective=content["config"] as! [String:Any]
    XCTAssertEqual(effective["dfr_temporal_rounds"] as? Int,1)
    XCTAssertEqual(effective["dfr_detailing_lora_strength"] as? Double,0.5)
    project.clips[0].generationSelection?.task="a2v"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request(project,runtime)))
    config["dfr_temporal_upsampler_path"]=""
    recipe["config"]=config
    try JSONSerialization.data(withJSONObject:recipe).write(to:url)
    XCTAssertTrue(try NativeLTXPreparation.catalog(directory:root.path).isEmpty)
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

  func testUnionGuideResamplesVFRInOrderAndFreezesQuarterCanvas() async throws {
    let (root,original,runtime)=try fixture(),movie=try await continuityMovie(root)
    let target=root.appendingPathComponent("test.rgb24")
    let digest=try await NativeLTXControlGuide.prepare(source:movie,destination:target,width:512,height:256,
      frames:33,fps:24,editorialDuration:1)
    let bytes=try Data(contentsOf:target),frameBytes=128*64*3
    XCTAssertEqual(bytes.count,33*frameBytes)
    XCTAssertEqual(digest,SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
    for (frame,channel) in [(0,0),(5,1),(20,2),(32,0)] {
      let offset=frame*frameBytes+(32*128+64)*3
      XCTAssertGreaterThan(bytes[offset+channel],200)
      XCTAssertLessThan(bytes[offset+(channel+1)%3],35)
    }
    do {
      _=try await NativeLTXControlGuide.prepare(source:movie,destination:root.appendingPathComponent("too-long.rgb24"),
        width:512,height:256,frames:49,fps:24,editorialDuration:2)
      XCTFail("Short guide should fail before publication")
    } catch { XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("too-long.rgb24").path)) }
    let profile=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as! [String:Any]
    recipe["components"]=["transformer_path":"/models/transformer","loras":[],"ic_loras":[["/models/union.safetensors",1.0]]]
    recipe["conditioning"]=["version":1,"task":"control","inputs":[]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:profile)
    var project=original;project.clips[0].generationSelection?.task="control"
    project.clips[0].duration=4.0/3;project.clips[0].generationWidth=512;project.clips[0].generationHeight=256
    let asset=MediaAsset(name:"Guide",kind:.video,path:movie.path);project.assets=[asset]
    project.clips[0].attachments=[Attachment(assetID:asset.id,role:.control)]
    let output=root.appendingPathComponent("job"),body=try request(project,runtime)
    let prepared=try await NativeLTXPreparation.prepareWithMedia(request:body,destination:output)
    let frozen=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prepared["recipePath"] as! String))) as! [String:Any]
    let input=((frozen["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[0]
    XCTAssertEqual(input["format"] as? String,"rgb24");XCTAssertEqual(input["sha256"] as? String,digest)
    XCTAssertEqual(input["path"] as? String,output.appendingPathComponent("union-guide.rgb24").path)
    XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("union-guide.rgb24")),bytes)
    XCTAssertTrue(FileManager.default.fileExists(atPath:output.appendingPathComponent("editor-request.json").path))
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

  func testMovieSourceIntervalsCannotHideOnOrdinaryOrDisabledAttachments() async throws {
    let (root,original,runtime)=try fixture()
    var baseline=original
    let unused=MediaAsset(name:"Disabled stale adapter",kind:.lora,path:"/missing/unused.safetensors")
    baseline.assets.append(unused)
    var attachment=Attachment(assetID:unused.id,role:.lora);attachment.enabled=false
    baseline.clips[0].attachments.append(attachment)
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request:request(baseline,runtime)))
    for durationField in [false,true] {
      var project=baseline
      if durationField { project.clips[0].attachments[0].sourceDurationSeconds=1 }
      else { project.clips[0].attachments[0].sourceStartSeconds=0 }
      let frozen=try request(project,runtime)
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request:frozen)) {
        XCTAssertTrue($0.localizedDescription.contains("Source movie interval fields"))
      }
      XCTAssertThrowsError(try NativeLTXPreparation.describe(request:frozen)) {
        XCTAssertTrue($0.localizedDescription.contains("Source movie interval fields"))
      }
      let destination=root.appendingPathComponent("rejected-movie-field-"+String(durationField))
      do {
        _ = try await NativeLTXPreparation.prepareWithMedia(request:frozen,destination:destination)
        XCTFail("Ordinary preparation silently ignored a movie interval")
      } catch { XCTAssertTrue(error.localizedDescription.contains("Source movie interval fields")) }
      XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
    }
  }
}
