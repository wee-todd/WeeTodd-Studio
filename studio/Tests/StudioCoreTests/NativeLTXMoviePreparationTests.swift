import AVFoundation
import XCTest
@testable import StudioCore
final class NativeLTXMoviePreparationTests:XCTestCase {
  func testCQOnlyEnhancementKeepsSizeClockAndSeparatesDenoiseFromAmount() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let backend=try VideoProcessingFixture(root:root),source=try backend.source(fps:25)
    var settings=LTX25MovieUpscaleSettings();settings.experimentalEnabled=true
    settings.mode = try XCTUnwrap(LTX25MovieUpscaleSettings.Mode(rawValue:"enhance"))
    settings.refinementStrength=0.5;settings.sizePolicy = .strict
    let components=["video_checkpoint":"/models/video.safetensors","gemma_root":"/models/gemma","connector_checkpoint":"/models/fixed.safetensors","transformer_root":"/models/pages","audio_checkpoint":"/models/audio.safetensors"]
    let result=try await NativeLTXMoviePreparation.prepare(source:URL(fileURLWithPath:source.path),settings:settings,
      components:components,prompt:"Preserve the source.",seed:1,ffmpeg:try ffmpeg(),
      destination:root.appendingPathComponent("prepared"),outputDirectory:root.appendingPathComponent("render"),
      refinementLoRAs:[.init(path:"/models/cq.safetensors",strength:1)])
    let recipe=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as? [String:Any])
    let body=try XCTUnwrap(recipe["movie_upscale"] as? [String:Any]),frozen=try XCTUnwrap(body["source"] as? [String:Any])
    let report=try XCTUnwrap(result["report"] as? [String:Any])
    XCTAssertEqual(body["mode"] as? String,"enhance")
    XCTAssertEqual(body["refinement_strength"] as? Double,0.5)
    XCTAssertEqual((body["refinement_loras"] as? [[String:Any]])?.first?["strength"] as? Double,1)
    XCTAssertNil((body["components"] as? [String:String])?["spatial_upscaler_checkpoint"])
    XCTAssertNil((body["components"] as? [String:String])?["pixel_spatial_adapter"])
    XCTAssertEqual(report["width"] as? Int,frozen["width"] as? Int)
    XCTAssertEqual(report["height"] as? Int,frozen["height"] as? Int)
    XCTAssertEqual(frozen["fps"] as? Double,25)
    XCTAssertEqual(body["audio_policy"] as? String,"source")
  }
  func testMovieUpscaleRejectsH3PrecisionAndWeightCacheBeforeMedia() throws {
    var clip = Clip(engine: .ltx25)
    var settings = LTX25MovieUpscaleSettings(); settings.experimentalEnabled = true
    clip.generationSelection = GenerationSelection(task: "video_upscale")
    XCTAssertNoThrow(try NativeLTXMoviePreparation.validateEditor(clip: clip, settings: settings))
    for precision in [true, false] {
      var invalid = clip
      if precision { invalid.generationSelection?.h3VideoDecodePrecision = .float16 }
      else { invalid.generationSelection?.h3TransformerWeightCacheGB = 8 }
      XCTAssertThrowsError(try NativeLTXMoviePreparation.validateEditor(clip: invalid, settings: settings),
        "Movie upscaling must reject H3-only settings before decoding media")
    }
  }

  private func ffmpeg() throws -> URL {
    for path in [ProcessInfo.processInfo.environment["WEETODD_FFMPEG"],"/opt/homebrew/bin/ffmpeg"].compactMap({ $0 }) {
      if FileManager.default.isExecutableFile(atPath:path) { return URL(fileURLWithPath:path) }
    }
    throw XCTSkip("Existing FFmpeg required for model-free source movie fixture")
  }
  func testNativeV2ExportFreezesExactVisibleMovieAndOriginalPCMWithoutSamplerOrPython() async throws {
    let fm=FileManager.default,root=fm.temporaryDirectory.appendingPathComponent(UUID().uuidString),ffmpeg=try ffmpeg()
    try fm.createDirectory(at:root,withIntermediateDirectories:false);defer { try? fm.removeItem(at:root) }
    let movie=root.appendingPathComponent("source.mov"),audio=root.appendingPathComponent("original44100.wav")
    try NativeMovieFrozenMedia.run(ffmpeg,["-v","error","-nostdin","-n","-f","lavfi","-i","color=c=red:s=64x32:r=24","-frames:v","49","-c:v","qtrle","-pix_fmt","rgb24","-an",movie.path],log:root.appendingPathComponent("source.log"))
    try NativeMovieFrozenMedia.run(ffmpeg,["-v","error","-nostdin","-n","-f","lavfi","-i","sine=frequency=220:sample_rate=44100","-af","atrim=end_sample=90038","-ac","1","-c:a","pcm_f32le",audio.path],log:root.appendingPathComponent("audio.log"))
    let originalMovie=try NativeMovieFrozenMedia.digest(movie),originalAudio=try NativeMovieFrozenMedia.digest(audio)
    let inspected=try await NativeLTXMoviePreparation.inspectSource(movie)
    XCTAssertEqual(inspected["frames"] as? Int,49);XCTAssertEqual(inspected["fps"] as? Double,24)
    XCTAssertEqual(inspected["width"] as? Int,64);XCTAssertEqual(inspected["height"] as? Int,32)
    XCTAssertEqual(inspected["duration"] as? Double,49.0/24);XCTAssertEqual(inspected["startSeconds"] as? Double,0)
    var settings=LTX25MovieUpscaleSettings();settings.experimentalEnabled=true;settings.audioPolicy = .sidecar;settings.sizePolicy = .strict
    let components=["video_checkpoint":"/models/video.safetensors","spatial_upscaler_checkpoint":"/models/spatial.safetensors","gemma_root":"/models/gemma","connector_checkpoint":"/models/fixed.safetensors","transformer_root":"/models/pages","audio_checkpoint":"/models/audio.safetensors"]
    let alias=root.appendingPathComponent("same-movie-as-audio.mov")
    try fm.createSymbolicLink(at:alias,withDestinationURL:movie)
    let rejected=root.appendingPathComponent("duplicate-roles")
    do {
      _ = try await NativeLTXMoviePreparation.prepare(source:movie,settings:settings,components:components,prompt:"A robot lowers its arm.",seed:42,
        sidecar:.init(path:alias.path),ffmpeg:ffmpeg,destination:rejected,outputDirectory:root.appendingPathComponent("rejected-take"))
      XCTFail("Two source roles aliased the same canonical movie")
    } catch { XCTAssertTrue(error.localizedDescription.contains("distinct canonical media paths")) }
    XCTAssertFalse(fm.fileExists(atPath:rejected.path))
    let prepared=root.appendingPathComponent("prepared")
    let result=try await NativeLTXMoviePreparation.prepare(source:movie,settings:settings,components:components,prompt:"The same red robot lowers its arm.",seed:42,
      sidecar:.init(path:audio.path),ffmpeg:ffmpeg,destination:prepared,outputDirectory:root.appendingPathComponent("take"),editorRequest:["frozenEditorIntent":"movie-test"])
    let editor=try JSONSerialization.jsonObject(with:Data(contentsOf:prepared.appendingPathComponent("editor-request.json"))) as! [String:Any]
    XCTAssertEqual(editor["frozenEditorIntent"] as? String,"movie-test")
    let recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as! [String:Any]
    XCTAssertEqual(recipe["format"] as? String,"weetodd-headless-v2")
    let body=recipe["movie_upscale"] as! [String:Any],source=body["source"] as! [String:Any]
    XCTAssertEqual(source["frames"] as? Int,49);XCTAssertEqual(source["fps"] as? Double,24)
    XCTAssertEqual(source["duration_seconds"] as? Double,49.0/24)
    XCTAssertEqual(source["sha256"] as? String,originalMovie)
    let sidecar=body["audio_source"] as! [String:Any]
    XCTAssertEqual(sidecar["path"] as? String,audio.path);XCTAssertEqual(sidecar["sha256"] as? String,originalAudio)
    XCTAssertTrue(sidecar["duration_seconds"] is NSNull)
    let file=try AVAudioFile(forReading:audio)
    XCTAssertEqual(file.fileFormat.sampleRate,44100);XCTAssertEqual(file.length,90038)
    XCTAssertEqual(try NativeMovieFrozenMedia.digest(audio),originalAudio)
    XCTAssertEqual(try NativeMovieFrozenMedia.digest(movie),originalMovie)
    let inputs=(recipe["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(Set(inputs.compactMap { $0["path"] as? String }),[movie.path,audio.path,prepared.appendingPathComponent("source.rgb24").path])
    let report=result["report"] as! [String:Any]
    XCTAssertEqual(report["usable_duration"] as? Double,49.0/24)
    XCTAssertEqual(report["width"] as? Int,128);XCTAssertEqual(report["height"] as? Int,64)
    let descriptor=try JSONDecoder().decode(GenerationDescriptor.self,from:JSONSerialization.data(withJSONObject:report["generation"]!))
    XCTAssertEqual(descriptor.controls.evaluations,3);XCTAssertFalse(descriptor.controls.stepsEditable)
    XCTAssertFalse(fm.fileExists(atPath:prepared.appendingPathComponent("audio.wav").path))
  }
  func testPixelCQWirePreserves25FPSAndOriginalAudioPolicy() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let backend=try VideoProcessingFixture(root:root),source=try backend.source(fps:25)
    var settings=LTX25MovieUpscaleSettings();settings.experimentalEnabled=true;settings.mode = .pixelSpatial
    settings.refinementStrength=0.75;settings.sizePolicy = .strict
    let components=["video_checkpoint":"/models/video.safetensors","spatial_upscaler_checkpoint":"/models/spatial.safetensors","gemma_root":"/models/gemma","connector_checkpoint":"/models/fixed.safetensors","transformer_root":"/models/pages","audio_checkpoint":"/models/audio.safetensors","pixel_spatial_adapter":"/models/pixel.safetensors"]
    let extra=NativeLTXMoviePreparation.RefinementLoRA(path:"/models/cq.safetensors",strength:0.5)
    let result=try await NativeLTXMoviePreparation.prepare(source:URL(fileURLWithPath:source.path),settings:settings,
      components:components,prompt:"Preserve the source.",seed:1,ffmpeg:try ffmpeg(),
      destination:root.appendingPathComponent("prepared"),outputDirectory:root.appendingPathComponent("render"),refinementLoRAs:[extra])
    let recipe=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as? [String:Any])
    let body=try XCTUnwrap(recipe["movie_upscale"] as? [String:Any])
    XCTAssertEqual((body["source"] as? [String:Any])?["fps"] as? Double,25)
    XCTAssertEqual((body["source"] as? [String:Any])?["frames"] as? Int,25)
    XCTAssertEqual(body["audio_policy"] as? String,"source")
    XCTAssertEqual(body["pixel_strength"] as? Double,1)
    let adapters=try XCTUnwrap(body["refinement_loras"] as? [[String:Any]])
    XCTAssertEqual(adapters.count,1);XCTAssertEqual(adapters[0]["strength"] as? Double,0.5)
    XCTAssertEqual(adapters[0]["path"] as? String,"/models/cq.safetensors")
    XCTAssertEqual(adapters[0]["enabled"] as? Bool,true)
  }
  func testGridMatchesFrozenNativePreparationAndExperimentalRejectsBeforeSourceIO() async throws {
    let grid=try NativeLTXMoviePreparation.grid(width:1280,height:720,policy:.fitNearest)
    XCTAssertEqual(grid.width,1312);XCTAssertEqual(grid.height,736);XCTAssertTrue(grid.resize)
    XCTAssertThrowsError(try NativeLTXMoviePreparation.grid(width:1280,height:720,policy:.strict))
    do {
      _ = try await NativeLTXMoviePreparation.prepare(source:URL(fileURLWithPath:"/missing.mov"),settings:LTX25MovieUpscaleSettings(),components:[:],prompt:"",seed:42,
        ffmpeg:URL(fileURLWithPath:"/missing-ffmpeg"),destination:URL(fileURLWithPath:"/not-created"),outputDirectory:URL(fileURLWithPath:"/take"))
      XCTFail("Default settings launched preparation")
    } catch { XCTAssertTrue(String(describing:error).contains("experimental")) }
  }
}
