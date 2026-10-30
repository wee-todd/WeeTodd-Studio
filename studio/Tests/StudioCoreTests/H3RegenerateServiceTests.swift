import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

final class H3RegenerateServiceTests:XCTestCase {
  func testSourceClockFailsBeforeMissingModelRead() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
    let probe=root.appendingPathComponent("probe.sh"),source=root.appendingPathComponent("source.mov")
    try Data([1]).write(to:source)
    let script="""
    #!/bin/sh
    case "$*" in
      *-show_streams*) printf '%s\\n' '{"streams":[{"codec_type":"video","width":32,"height":32,"avg_frame_rate":"30/1","pix_fmt":"yuv420p"}]}' ;;
      *) printf '0\\n0.033333\\n' ;;
    esac
    """
    try Data(script.utf8).write(to:probe);try FileManager.default.setAttributes([.posixPermissions:0o755],ofItemAtPath:probe.path)
    var settings=H3RegenerateSettings();settings.baseProfilePath="/missing/profile";settings.modelLibraryRoot="/missing/models"
    let request=H3RegenerateRequest(sourcePath:source.path,settings:settings,runtime:.init(ffmpegPath:"/usr/bin/true",ffprobePath:probe.path,workerPath:"/usr/bin/true"))
    do {_ = try await H3RegenerateService().preflight(request:request);XCTFail("Source rate must reject")}
    catch {XCTAssertTrue(error.localizedDescription.contains("24 FPS"),error.localizedDescription)}
  }
  func testLosslessMediaAndOriginalAudioWithSimulatedWorker() throws {
    try mediaCase(native:false,alterDecoder:false)
  }
  func testExplicitNativeRowsReachWorkerWithoutPixelEncoder() throws {
    try mediaCase(native:true,alterDecoder:false)
  }
  func testChangedWorkerDecoderDoesNotPublishVideo() throws {
    try mediaCase(native:false,alterDecoder:true)
  }
  func testIntermediateUsesOnlyVerifiedSilentLosslessMovie() throws {
    try mediaCase(native:false,alterDecoder:false,intermediateOnly:true)
  }
  private func mediaCase(native:Bool,alterDecoder:Bool,intermediateOnly:Bool=false) throws {
    let ffmpeg="/opt/homebrew/bin/ffmpeg",ffprobe="/opt/homebrew/bin/ffprobe"
    guard FileManager.default.isExecutableFile(atPath:ffmpeg),FileManager.default.isExecutableFile(atPath:ffprobe) else{throw XCTSkip("Existing media utilities unavailable")}
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
    let runtime=H3RegenerateRuntime(ffmpegPath:ffmpeg,ffprobePath:ffprobe,workerPath:"/usr/bin/true"),cancel=NativeHeadlessCancellation()
    let movie=root.appendingPathComponent("source.mkv")
    var inputs=["-f","lavfi","-i","testsrc2=size=32x32:rate=24:duration=1"]
    if !intermediateOnly {inputs += ["-f","lavfi","-i","sine=frequency=440:sample_rate=32000:duration=1"]}
    inputs += ["-map","0:v:0","-c:v","ffv1"]
    inputs += intermediateOnly ? ["-an"]:["-map","1:a:0","-c:a","pcm_f32le"]
    try H3RegenerateService.ffmpeg(inputs+[movie.path],runtime:runtime,directory:root,name:"fixture",cancellation:cancel)
    let source=try FlashVSRFile.capture(movie)
    let media=try FlashVSRMediaInfo.inspect(movie,runtime:runtime.mediaRuntime,directory:root,cancellation:cancel)
    let placeholder=root.appendingPathComponent("placeholder");try Data([1]).write(to:placeholder)
    let file=try FlashVSRFile.capture(placeholder)
    func c(_ role:NativeComponentRole)->H3RegenerateJob.Component{.init(path:file.path,identity:.init(family:.h3,role:role,sourceSHA256:file.sha256,layout:"fixture"))}
    let resources=H3RegenerateResources(components:.init(transformer:c(.transformer),refiner:c(.refiner),sharpCurve:c(.sharpCurve),sharpInput:c(.sharpInput),latentUpscaler:c(.latentUpscaler),videoVAE:c(.videoVAE),audioVAE:c(.audioVAE)),qwen:.init(checkpointRoot:root.path,tokenizer:.init(path:file.path,sha256:file.sha256),files:[.init(path:file.path,sha256:file.sha256)]),modelFiles:[file],engineFiles:[],profile:file,runtime:runtime)
    var settings=H3RegenerateSettings();settings.targetWidth=64;settings.targetHeight=64;settings.encoderProfile = .nativePosteriorMeanF16;settings.posteriorSeed=nil
    var request=H3RegenerateRequest(sourcePath:movie.path,settings:settings,runtime:runtime);request.firstFrame=1;request.frameCount=22
    if native {
      let rows=root.appendingPathComponent("native.f32");try Data(count:7*96*4).write(to:rows)
      request.nativeLatents = .init(file:try .capture(rows),sourceSHA256:source.sha256,firstFrame:1,visibleFrames:22,paddedFrames:22,width:32,height:32)
    }
    let grid=try H3RegeneratePreparation.geometry(media:media,settings:settings,firstFrame:1,frameCount:22,latent:request.nativeLatents)
    let plan=H3RegenerateResolvedPlan(request:request,source:source,media:media,resources:resources,frames:22,paddedFrames:22,sourceWidth:32,sourceHeight:32,temporaryBytes:grid.temporaryBytes,preparationPolicy:"source-ceil32-center-pad;target-ceil32-center-crop;repeat-last-to-5plus17-v1")
    let output=root.appendingPathComponent("result")
    let work: (H3RegenerateJob,URL)throws->[String:Any] = {job,directory in
      XCTAssertEqual(job.source.kind,native ? .normalizedVideoRows:.rgb24)
      XCTAssertEqual(job.source.encoderProfile,native ? .nativeNormalizedLatents:.nativePosteriorMeanF16)
      XCTAssertNil(job.posteriorSeed);XCTAssertNil(job.components.publisherEncoder)
      try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
      let rgb=directory.appendingPathComponent("video.rgb24"),rows=directory.appendingPathComponent("refined-video.f32")
      try Data(repeating:127,count:22*64*64*3).write(to:rgb);try Data(count:7*4*96*4).write(to:rows)
      let rgbFile=try FlashVSRFile.capture(rgb),rowFile=try FlashVSRFile.capture(rows)
      let metadata:[String:Any] = ["nativeRuntime":"swift-mlx","engine":"h3","task":"h3_regenerate","inferenceExecuted":true,"sourceSHA256":job.source.originalSHA256,"sourcePayloadSHA256":job.source.payload.sha256,"sourceAudioSHA256":job.source.audioSHA256,"decoder":alterDecoder ? "fastH3":job.decoder.rawValue,"encoderProfile":job.source.encoderProfile.rawValue,"fps":24,"visibleFrames":22,"paddedFrames":22,"width":64,"height":64,"audioDecoded":false,"sourceAudioPreservationRequired":true,"outputRGBPath":rgb.path,"outputRGBSHA256":rgbFile.sha256,"outputRGBBytes":rgbFile.bytes,"refinedLatentsPath":rows.path,"refinedLatentsSHA256":rowFile.sha256,"refinedLatentsShape":[1,28,96],"refinedLatentsDType":"F32LE"]
      try JSONSerialization.data(withJSONObject:metadata).write(to:directory.appendingPathComponent("result.json"))
      try JSONEncoder().encode(job).write(to:directory.appendingPathComponent("worker-job.json"));return metadata
    }
    if alterDecoder {
      XCTAssertThrowsError(try H3RegenerateService.execute(plan:plan,directory:output,cancellation:cancel,emit:{_ in},resourceResolver:{_,_,_ in resources},worker:work))
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.appendingPathComponent("output.mov").path));return
    }
    let receipt=try H3RegenerateService.execute(plan:plan,directory:output,cancellation:cancel,intermediateOnly:intermediateOnly,emit:{_ in},resourceResolver:{_,_,_ in resources},worker:work)
    XCTAssertEqual(receipt.frames,22);XCTAssertEqual(receipt.fps.value,24);XCTAssertEqual(receipt.source.sha256,source.sha256)
    if intermediateOnly {
      XCTAssertNil(receipt.audio);XCTAssertEqual(receipt.video,receipt.restored)
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.appendingPathComponent("output.mov").path))
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.appendingPathComponent("deliver.log").path))
    } else {XCTAssertNotNil(receipt.audio)}
    XCTAssertEqual(receipt.refinedLatentsShape,[1,28,96])
    XCTAssertEqual(receipt.restored.pathExtension,"mkv")
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.appendingPathComponent("source.rgb24").path) == native)
  }
}
private extension FlashVSRFile { var pathExtension:String {URL(fileURLWithPath:path).pathExtension} }
