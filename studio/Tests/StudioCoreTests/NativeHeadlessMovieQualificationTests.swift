import Foundation
import XCTest
@testable import StudioCore

final class NativeHeadlessMovieQualificationTests:XCTestCase {
  /// Opt-in media preparation/export only. This method never invokes a model worker.
  func testInstalledMovieFixtureExportsRealPendingJob() async throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_NATIVE_HEADLESS_MOVIE_EXPORT"] else {
      throw XCTSkip("Installed immutable movie fixture export; CPU/media preparation only")
    }
    let options=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:path))) as! [String:String]
    let root=URL(fileURLWithPath:try XCTUnwrap(options["output"])),fm=FileManager.default
    XCTAssertFalse(fm.fileExists(atPath:root.path));guard !fm.fileExists(atPath:root.path) else { throw StudioError.invalid("Never overwrite a qualification export") }
    let bytes=try Data(contentsOf:URL(fileURLWithPath:try XCTUnwrap(options["recipe"])))
    let original=try JSONSerialization.jsonObject(with:bytes) as! [String:Any]
    let source=original["source"] as! [String:Any],components=original["components"] as! [String:String]
    let audio=original["audio_source"] as! [String:Any]
    var settings=LTX25MovieUpscaleSettings();settings.experimentalEnabled=true
    settings.mode=try XCTUnwrap(.init(rawValue:original["mode"] as! String))
    settings.sizePolicy=try XCTUnwrap(.init(rawValue:original["size_policy"] as! String))
    settings.anchors=try XCTUnwrap(.init(rawValue:original["anchors"] as! String))
    settings.audioPolicy=try XCTUnwrap(.init(rawValue:original["audio_policy"] as! String))
    settings.refinementStrength=original["refinement_strength"] as! Double
    settings.anchorStrength=original["anchor_strength"] as! Double;settings.pixelStrength=original["pixel_strength"] as! Double
    settings.maximumAudioDriftSeconds=original["maximum_audio_drift_seconds"] as! Double
    settings.chunkFrameMegapixelBudget=original["chunk_frame_megapixel_budget"] as! Double
    settings.chunking=original["chunking"] as! Bool;settings.resume=original["resume"] as! Bool;settings.keepChunks=original["keep_chunks"] as! Bool
    XCTAssertEqual(settings.audioPolicy,.sidecar);XCTAssertEqual(settings.mode,.refine)
    let id=try XCTUnwrap(UUID(uuidString:try XCTUnwrap(options["clipID"])))
    let fps=source["fps"] as! Double,count=source["frames"] as! Int
    var clip=Clip(engine:.ltx25);clip.id=id;clip.sourcePath=source["path"] as! String
    clip.sourceIn=source["start_seconds"] as! Double;clip.duration=Double(count)/fps
    clip.prompt=original["prompt"] as! String;clip.seed=(original["seed"] as! NSNumber).intValue
    clip.generationWidth=source["width"] as! Int;clip.generationHeight=source["height"] as! Int
    var selection=GenerationSelection(task:"video_upscale");selection.ltx25MovieUpscale=settings;clip.generationSelection=selection
    var project=StudioProject();project.name="Frozen learned2x movie headless qualification"
    project.settings.width=2*(source["width"] as! Int);project.settings.height=2*(source["height"] as! Int);project.settings.fps=fps
    project.clips=[clip]
    XCTAssertTrue(project.clips[0].versions.isEmpty,"Export must start from pending input, never fabricated acceptance")
    try NativeLTXMoviePreparation.validateEditor(clip:clip,settings:settings)
    let target=root.appendingPathComponent("cli-render/take-"+id.uuidString)
    let endpoints=(original["reference_images"] as! [[String:Any]]).map {
      NativeLTXMoviePreparation.Endpoint(path:$0["path"] as! String,role:$0["role"] as! String,
        strength:$0["strength"] as! Double,crf:$0["crf"] as! Int)
    }
    let prepared=try await NativeLTXMoviePreparation.prepare(source:URL(fileURLWithPath:clip.sourcePath),startSeconds:clip.sourceIn,
      visibleFrames:count,settings:settings,components:components,prompt:clip.prompt,seed:UInt64(clip.seed),endpoints:endpoints,
      sidecar:.init(path:audio["path"] as! String,startSeconds:audio["start_seconds"] as! Double,durationSeconds:audio["duration_seconds"] as? Double),
      ffmpeg:URL(fileURLWithPath:try XCTUnwrap(options["ffmpeg"])),destination:root.appendingPathComponent("prepared-inputs"),outputDirectory:target)
    let frozenBytes=try Data(contentsOf:URL(fileURLWithPath:prepared["recipePath"] as! String))
    let wrapper=try JSONSerialization.jsonObject(with:frozenBytes) as! [String:Any]
    let actual=wrapper["movie_upscale"] as! [String:Any]
    var expected=original;expected["output_directory"]=actual["output_directory"]
    var expectedSource=source;expectedSource["rgb_path"]=(actual["source"] as! [String:Any])["rgb_path"];expected["source"]=expectedSource
    func canonical(_ value:Any) throws -> Data { try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes]) }
    XCTAssertEqual(try canonical(actual),try canonical(expected),"Only output and byte-identical prepared RGB path may relocate")
    XCTAssertEqual((actual["source"] as! [String:Any])["rgb_sha256"] as? String,source["rgb_sha256"] as? String)
    let worker=try XCTUnwrap(options["worker"]),ffmpeg=try XCTUnwrap(options["ffmpeg"])
    let job=try NativeHeadlessJob(project:project,recipes:[id.uuidString:.init(engine:"ltx25",bytes:frozenBytes,
      signature:"movie-frozen:"+NativeHeadlessJob.hash(frozenBytes),report:try canonical(prepared["report"]!))],workers:["ltx25":worker],ffmpeg:ffmpeg)
    let exported=root.appendingPathComponent("movie.weetodd-job.json");try job.write(to:exported)
    let reopened=try NativeHeadlessJob.read(from:exported)
    XCTAssertEqual(reopened.project,project);XCTAssertEqual(reopened.recipes[id.uuidString]?.bytes,frozenBytes)
    XCTAssertTrue(reopened.project.clips[0].versions.isEmpty);try reopened.verifySources()
    try ProjectStorage.write(project,to:root.appendingPathComponent("pending.weetodd"))
    try canonical(["status":"prepared","inferenceExecuted":false,"workerInvocations":0,"pythonModelInference":false,
      "exportedJob":exported.path,"recipe":prepared["recipePath"]!,"recipeSHA256":NativeHeadlessJob.hash(frozenBytes),
      "clipID":id.uuidString,"pendingVersions":0,"sourceRGBSHA256":source["rgb_sha256"]!,"creativeContractEqual":true,
      "sourceFrames":count,"fps":fps,"outputWidth":project.settings.width,"outputHeight":project.settings.height])
      .write(to:root.appendingPathComponent("export-qualification.json"),options:.withoutOverwriting)
  }
}
