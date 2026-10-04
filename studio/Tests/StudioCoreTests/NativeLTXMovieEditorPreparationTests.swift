import XCTest
@testable import StudioCore

final class NativeLTXMovieEditorPreparationTests:XCTestCase {
  private func fixture(ffmpeg:String="/usr/bin/true") throws -> (URL,StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let pages=root.appendingPathComponent("pages")
    try FileManager.default.createDirectory(at:pages,withIntermediateDirectories:false)
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":"old profile prompt",
      "components":["transformer_path":pages.path,"video_vae_path":"/models/video.safetensors",
        "spatial_upscaler_path":"/models/spatial.safetensors","text_encoder_path":"/models/gemma","audio_vae_path":"/models/audio.safetensors"],
      "config":["pipeline_mode":"distilled","stage1_steps":8,"stage2_steps":3,"frame_rate":24,"width":768,"height":448,"seed":1,"duration_seconds":5],
      "conditioning":["version":1,"task":"t2v","inputs":[]]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("model.json"))
    var settings=LTX25MovieUpscaleSettings();settings.experimentalEnabled=true
    var clip=Clip();clip.prompt="The same source character lowers its arm.";clip.duration=9.5
    clip.generationWidth=1344;clip.generationHeight=768;clip.seed=77
    clip.generationSelection=GenerationSelection(task:"video_upscale");clip.generationSelection?.ltx25MovieUpscale=settings
    let source=MediaAsset(name:"Original movie",kind:.video,path:root.appendingPathComponent("source.mov").path)
    var attachment=Attachment(assetID:source.id,role:.reference)
    attachment.sourceStartSeconds=8.0/24;attachment.sourceDurationSeconds=13.0/24
    clip.attachments=[attachment]
    var project=StudioProject();project.assets=[source];project.clips=[clip]
    return (root,project,["profilesDirectory":root.path,"ffmpegPath":ffmpeg])
  }
  private func request(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),"clipID":project.clips[0].id.uuidString,"runtime":runtime]
  }
  func testActualMovieEditorIgnoresNoOverridesBeforeComponentOnlyResolution() throws {
    let (_,original,runtime)=try fixture()
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    let snapshot=try encoder.encode(original)
    let described=try NativeLTXMovieEditorPreparation.describe(request:request(original,runtime))
    XCTAssertEqual((described["generation"] as? [String:Any])?["supportedTasks"] as? [String],["video_upscale"])
    XCTAssertEqual(try encoder.encode(original),snapshot)
    for name in ["negative","projection","memory","continuity","slot","lora"] {
      var project=original
      switch name {
      case "negative":project.clips[0].negativePrompt="blur"
      case "projection":project.clips[0].generationSelection?.projectionBackend="resident"
      case "memory":project.clips[0].generationSelection?.memoryPolicy="retain"
      case "continuity":project.clips[0].continuity=ClipContinuity(mode:"frame")
      case "slot":project.clips[0].generationSelection?.ltx25Keyframes=LTX25KeyframeSettings(generatedCount:1,experimentalEnabled:true)
      default:project.clips[0].attachments.append(Attachment(assetID:project.assets[0].id,role:.lora))
      }
      XCTAssertThrowsError(try NativeLTXMovieEditorPreparation.describe(request:request(project,runtime)),name)
    }
    var noSettings=original;noSettings.clips[0].generationSelection?.ltx25MovieUpscale=nil
    XCTAssertTrue(try NativeLTXMovieEditorPreparation.matches(request:request(noSettings,runtime)))
    XCTAssertThrowsError(try NativeLTXMovieEditorPreparation.describe(request:request(noSettings,runtime)))
  }
  func testEmbeddedProfileAdaptersRejectRatherThanDisappear() throws {
    let (root,project,runtime)=try fixture(),file=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    var components=recipe["components"] as! [String:Any]
    components["loras"]=[["/models/user.safetensors",0.5]];recipe["components"]=components
    try JSONSerialization.data(withJSONObject:recipe).write(to:file)
    XCTAssertThrowsError(try NativeLTXMovieEditorPreparation.describe(request:request(project,runtime))) {
      XCTAssertTrue($0.localizedDescription.contains("embedded"))
    }
  }
  func testSelectedMovieIntervalPreserves13VisibleFramesAndActualSourceFPS() async throws {
    guard FileManager.default.isExecutableFile(atPath:"/opt/homebrew/bin/ffmpeg") else { throw XCTSkip("Existing FFmpeg required") }
    let (root,project,runtime)=try fixture(ffmpeg:"/opt/homebrew/bin/ffmpeg")
    let source=URL(fileURLWithPath:project.assets[0].path)
    try NativeMovieFrozenMedia.run(URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg"),["-v","error","-nostdin","-n","-f","lavfi","-i","color=c=red:s=64x32:r=24","-frames:v","49","-c:v","qtrle","-pix_fmt","rgb24","-an",source.path],log:root.appendingPathComponent("source.log"))
    let sourceSHA=try NativeMovieFrozenMedia.digest(source),body=try request(project,runtime)
    let destination=root.appendingPathComponent("prepared")
    let result=try await NativeLTXMovieEditorPreparation.prepare(request:body,destination:destination)
    let recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result["recipePath"] as! String))) as! [String:Any]
    let movie=recipe["movie_upscale"] as! [String:Any],frozen=movie["source"] as! [String:Any]
    XCTAssertEqual(frozen["frames"] as? Int,13);XCTAssertEqual(frozen["fps"] as? Double,24)
    XCTAssertEqual(frozen["start_seconds"] as? Double,8.0/24)
    XCTAssertEqual(frozen["duration_seconds"] as? Double,13.0/24)
    XCTAssertEqual(movie["seed"] as? Int,77);XCTAssertEqual(movie["prompt"] as? String,project.clips[0].prompt)
    XCTAssertEqual((result["report"] as? [String:Any])?["width"] as? Int,128)
    XCTAssertEqual((result["report"] as? [String:Any])?["height"] as? Int,64)
    XCTAssertEqual((try Data(contentsOf:destination.appendingPathComponent("source.rgb24"))).count,13*64*32*3)
    let editor=try JSONSerialization.jsonObject(with:Data(contentsOf:destination.appendingPathComponent("editor-request.json"))) as! [String:Any]
    let restored=try JSONDecoder().decode(StudioProject.self,from:JSONSerialization.data(withJSONObject:editor["project"]!))
    XCTAssertEqual(restored,project)
    XCTAssertEqual(try NativeMovieFrozenMedia.digest(source),sourceSHA)
  }
}
