import XCTest
import CryptoKit
import LTX25Engine
@testable import LTX25MLX

final class MLXMovieSourceVideoTests:XCTestCase {
  private func ffmpeg() throws -> URL {
    for path in [ProcessInfo.processInfo.environment["WEETODD_FFMPEG"],"/opt/homebrew/bin/ffmpeg","/usr/local/bin/ffmpeg"].compactMap({ $0 }) {
      if FileManager.default.isExecutableFile(atPath:path) { return URL(fileURLWithPath:path) }
    }
    throw XCTSkip("An existing FFmpeg is required for model-free movie media fixtures.")
  }
  private func source(_ root:URL,ffmpeg:URL) throws -> (movie:URL,rgb:URL) {
    let raw=root.appendingPathComponent("source.rgb24"),movie=root.appendingPathComponent("source.mov")
    let red=Data([UInt8](repeating:0,count:64*32*3).enumerated().map { item -> UInt8 in item.offset%3==0 ? 255 : 0 })
    let blue=Data([UInt8](repeating:0,count:64*32*3).enumerated().map { item -> UInt8 in item.offset%3==2 ? 255 : 0 })
    let out=try FileHandle(forWritingTo: { _ = FileManager.default.createFile(atPath:raw.path,contents:nil);return raw }())
    for frame in 0..<98 { try out.write(contentsOf:frame<49 ? red : blue) };try out.close()
    try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-f","rawvideo","-pixel_format","rgb24","-video_size","64x32","-framerate","24","-i",raw.path,"-frames:v","98","-c:v","qtrle","-pix_fmt","rgb24","-an",movie.path],log:root.appendingPathComponent("source.log"))
    return (movie,raw)
  }
  func testActualCFRSourceSceneCutAndCausalTailPreserveAllVisibleFrames() async throws {
    let ffmpeg=try ffmpeg(),root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer { try? FileManager.default.removeItem(at:root) }
    let input=try source(root,ffmpeg:ffmpeg)
    let plan=try MLXMovieUpscalePlan(mode:.refine,width:64,height:32,frames:98,fps:24,sizePolicy:.strict)
    let source=try MLXMovieSourceVideo(source:input.movie,sha256:MLXMovieFiles.digest(input.movie),plan:plan)
    let prepared=try await source.prepare(ffmpeg:ffmpeg,directory:root.appendingPathComponent("prepared"))
    XCTAssertEqual(prepared.sourceFrameRange,0..<98)
    XCTAssertEqual(try Data(contentsOf:prepared.rgb24),try Data(contentsOf:input.rgb))
    XCTAssertEqual(try MLXMovieSourceVideo.sceneCuts(rgb24:prepared.rgb24,plan:plan),[49])
    let padded=root.appendingPathComponent("padded.rgb24")
    try MLXMovieFiles.copyRGBRange(source:prepared.rgb24,to:padded,frameBytes:64*32*3,visibleRange:0..<98,paddedFrames:105)
    let data=try Data(contentsOf:padded),original=try Data(contentsOf:input.rgb)
    XCTAssertEqual(data.prefix(original.count),original)
    for frame in 98..<105 { XCTAssertEqual(data[frame*6144..<(frame+1)*6144],original.suffix(6144)) }
    XCTAssertThrowsError(try MLXMovieFiles.copyRGBRange(source:prepared.rgb24,to:root.appendingPathComponent("bad"),frameBytes:6144,visibleRange:98..<99,paddedFrames:1))
  }
  private final class Counter:@unchecked Sendable {
    private let lock=NSLock();private var count=0
    func add() { lock.lock();count+=1;lock.unlock() }
    var value:Int { lock.lock();defer { lock.unlock() };return count }
  }
  func testMalformedFrozenEndpointRejectsBeforeCheckpointProvidersOrArrays() async throws {
    let ffmpeg=try ffmpeg(),root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer { try? FileManager.default.removeItem(at:root) }
    let input=try source(root,ffmpeg:ffmpeg),endpoint=root.appendingPathComponent("invalid-endpoint.png")
    try Data("not an image".utf8).write(to:endpoint)
    let sha=try MLXMovieFiles.digest(endpoint),output=root.appendingPathComponent("render")
    let body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale",
      "source":["path":input.movie.path,"sha256":try MLXMovieFiles.digest(input.movie),"rgb_path":input.rgb.path,"rgb_sha256":try MLXMovieFiles.digest(input.rgb),"width":64,"height":32,"frames":98,"fps":24,"start_seconds":0,"duration_seconds":98.0/24],
      "components":["video_checkpoint":"/missing/video.safetensors","spatial_upscaler_checkpoint":"/missing/upscaler.safetensors","gemma_root":"/missing/gemma","connector_checkpoint":"/missing/fixed.safetensors","transformer_root":"/missing/pages","audio_checkpoint":"/missing/audio.safetensors"],
      "mode":"refine","size_policy":"strict_32","output_directory":output.path,"prompt":"Same robot lowers its arm.","seed":42,"refinement_strength":0.35,"anchors":"first_last","anchor_strength":0.7,"pixel_strength":1,
      "reference_images":[["path":endpoint.path,"role":"last","strength":0.7,"crf":33]],"reference_image_sha256":[endpoint.path:sha],
      "maximum_audio_drift_seconds":0.05,"chunking":false,"chunk_frame_megapixel_budget":260,"resume":false,"keep_chunks":false]
    let request=try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:body),outputDirectory:output)
    do {
      _ = try await MLXMovieUpscalePipeline(request:request).preflight()
      XCTFail("Malformed endpoint reached checkpoint admission")
    } catch LTXError.invalid(let reason) {
      XCTAssertTrue(reason.contains("single image"),reason)
    } catch { XCTFail("Expected malformed-image admission before checkpoints: \(error)") }
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
  }
  func testActualChunkCancellationThenResumeReusesOnlyVerifiedMediaAndRejectsChangedSource() async throws {
    let ffmpeg=try ffmpeg(),root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer { try? FileManager.default.removeItem(at:root) }
    let input=try source(root,ffmpeg:ffmpeg),counter=Counter(),directory=root.appendingPathComponent("chunks")
    func request(resume:Bool) throws -> MLXMovieUpscaleRequest {
      let body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale","source":["path":input.movie.path,"sha256":try MLXMovieFiles.digest(input.movie),"rgb_path":input.rgb.path,"rgb_sha256":try MLXMovieFiles.digest(input.rgb),"width":64,"height":32,"frames":98,"fps":24,"start_seconds":0,"duration_seconds":98.0/24],"components":["video_checkpoint":"/models/video.safetensors","spatial_upscaler_checkpoint":"/models/upscaler.safetensors"],"mode":"latent_only","size_policy":"strict_32","output_directory":root.appendingPathComponent("render").path,"prompt":"","seed":42,"refinement_strength":0.35,"anchors":"none","anchor_strength":0.7,"pixel_strength":1,"reference_images":[],"maximum_audio_drift_seconds":0.05,"chunking":true,"chunk_frame_megapixel_budget":0.40140801,"resume":resume,"keep_chunks":true]
      return try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:body),outputDirectory:root.appendingPathComponent("render"))
    }
    let first=try request(resume:false),chunks=try first.chunkPlans(cutFrames:[49])
    let binding=try MLXMovieChunkCoordinator.Binding(request:first,audioSHA256:String(repeating:"c",count:64),componentIdentitySHA256:String(repeating:"d",count:64),workerSHA256:String(repeating:"e",count:64))
    do {
      _ = try await MLXMovieChunkCoordinator.execute(request:first,chunks:chunks,binding:binding,directory:directory) { chunk,index,temp in
        counter.add()
        if index==1 { try Data([1,2,3]).write(to:temp.appendingPathComponent("video.mp4"));throw CancellationError() }
        try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-f","lavfi","-i","color=c=red:s=128x64:r=24","-frames:v",String(chunk.frames),"-c:v","libx264","-pix_fmt","yuv420p","-an",temp.appendingPathComponent("video.mp4").path],log:temp.appendingPathComponent("render.log"))
      }
      XCTFail("Partial second chunk was published")
    } catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertEqual(counter.value,2)
    XCTAssertTrue(FileManager.default.fileExists(atPath:directory.appendingPathComponent("chunk-000000/chunk.json").path))
    XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("chunk-000001").path))
    XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:directory.path).contains { $0.hasPrefix(".chunk-") || $0==".execution.lock" })
    let retry=try request(resume:true)
    XCTAssertEqual(retry.contractSHA256,first.contractSHA256)
    let completed=try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:binding,directory:directory) { chunk,_,temp in
      counter.add();try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-f","lavfi","-i","color=c=blue:s=128x64:r=24","-frames:v",String(chunk.frames),"-c:v","libx264","-pix_fmt","yuv420p","-an",temp.appendingPathComponent("video.mp4").path],log:temp.appendingPathComponent("render.log"))
    }
    XCTAssertEqual(counter.value,3);XCTAssertEqual(completed.map(\.reused),[true,false])
    let zero=try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:binding,directory:directory) { _,_,_ in XCTFail("Completed media invoked the executor") }
    XCTAssertEqual(zero.map(\.reused),[true,true]);XCTAssertEqual(counter.value,3)
    let changedWorker=try MLXMovieChunkCoordinator.Binding(request:retry,audioSHA256:String(repeating:"c",count:64),componentIdentitySHA256:String(repeating:"d",count:64),workerSHA256:String(repeating:"f",count:64))
    do {
      _ = try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:changedWorker,directory:directory) { _,_,_ in XCTFail("Mismatched worker invoked executor") }
      XCTFail("A different worker silently reused a completed chunk")
    } catch { XCTAssertTrue(String(describing:error).contains("different source")) }
    XCTAssertEqual(counter.value,3)
    let oversized=root.appendingPathComponent("oversized.json")
    try Data(repeating:32,count:65537).write(to:oversized)
    XCTAssertThrowsError(try MLXMovieFiles.metadata(oversized,maximumBytes:65536))
    let damaged=try FileHandle(forWritingTo:input.rgb);try damaged.write(contentsOf:Data([0]));try damaged.close()
    do {
      _ = try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:binding,directory:directory) { _,_,_ in XCTFail("Changed source invoked executor") }
      XCTFail("Changed frozen source was admitted")
    } catch { XCTAssertTrue(String(describing:error).contains("changed")) }
  }
}
