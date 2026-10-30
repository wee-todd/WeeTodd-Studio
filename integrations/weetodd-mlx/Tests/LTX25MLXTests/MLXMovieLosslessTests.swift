import XCTest
import CryptoKit
import InferenceMedia
import LTX25Engine
@testable import LTX25MLX

final class MLXMovieLosslessTests: XCTestCase {
  func testFractionalClockAndRGBOrderSurviveManyOneFrameChunks() throws {
    let ffmpeg = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else { throw XCTSkip("Installed FFmpeg unavailable") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    var chunks: [URL] = [], hash = SHA256()
    for frame in 0..<40 {
      let video = root.appendingPathComponent("chunk-\(frame).mkv")
      let bytes = Data((0..<96).map { UInt8(($0*37 + frame*53) % 256) }); hash.update(data: bytes)
      let writer = try RawVideoWriter(ffmpeg: ffmpeg, output: video, width: 8, height: 4, frames: 1, fps: 30000.0/1001, encoding: .ffv1)
      try writer.append(bytes, frame: 0); try writer.finish(); chunks.append(video)
    }
    let proof = try MLXMovieLosslessPublication.concatenate(ffmpeg: ffmpeg, chunks: chunks,
      output: root.appendingPathComponent("restored.mkv"), width: 8, height: 4, frames: 40, fps: 30000.0/1001)
    XCTAssertEqual(proof.rgbSHA256, hash.finalize().map { String(format: "%02x", $0) }.joined())
    XCTAssertEqual(proof.frames, 40); XCTAssertEqual(proof.fpsNumerator, 30000); XCTAssertEqual(proof.fpsDenominator, 1001)
    XCTAssertThrowsError(try MLXMovieLosslessPublication.concatenate(ffmpeg: ffmpeg, chunks: chunks,
      output: root.appendingPathComponent("wrong.mkv"), width: 8, height: 4, frames: 39, fps: 30000.0/1001))
  }
  func testLosslessChunksResumeOnlyTheirVerifiedDecoderRGBProof() async throws {
    let ffmpeg=URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("Installed FFmpeg unavailable") }
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let movie=root.appendingPathComponent("source.mp4"),rgb=root.appendingPathComponent("source.rgb24"),output=root.appendingPathComponent("output")
    let bytes=Data(repeating:73,count:64*32*3)
    try (bytes+bytes).write(to:rgb)
    let sourceWriter=try RawVideoWriter(ffmpeg:ffmpeg,output:movie,width:64,height:32,frames:2,fps:24)
    try sourceWriter.append(bytes,frame:0);try sourceWriter.append(bytes,frame:1);try sourceWriter.finish()
    func request(_ resume:Bool) throws -> MLXMovieUpscaleRequest {
      let body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale",
        "source":["path":movie.path,"sha256":try MLXMovieFiles.digest(movie),"rgb_path":rgb.path,"rgb_sha256":try MLXMovieFiles.digest(rgb),"width":64,"height":32,"frames":2,"fps":24,"start_seconds":0,"duration_seconds":2.0/24],
        "components":["video_checkpoint":"/models/video","spatial_upscaler_checkpoint":"/models/upscale"],
        "mode":"latent_only","size_policy":"strict_32","output_directory":output.path,"prompt":"","seed":42,
        "refinement_strength":0.35,"anchors":"none","anchor_strength":0.7,"pixel_strength":1,"reference_images":[],
        "maximum_audio_drift_seconds":0.05,"chunking":true,"chunk_frame_megapixel_budget":1,"resume":resume,"keep_chunks":true,"intermediate_only":true]
      return try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:body),outputDirectory:output)
    }
    let first=try request(false),retry=try request(true),chunks:[MLXMovieUpscalePlan.Chunk]=[
      .init(startFrame:0,endFrame:1,paddedFrames:1,reason:"fixture"),.init(startFrame:1,endFrame:2,paddedFrames:1,reason:"fixture")]
    let binding=try MLXMovieChunkCoordinator.Binding(request:first,audioSHA256:String(repeating:"a",count:64),componentIdentitySHA256:String(repeating:"b",count:64),workerSHA256:String(repeating:"c",count:64))
    let directory=root.appendingPathComponent("chunks")
    func render(_ chunk:MLXMovieUpscalePlan.Chunk,_ index:Int,_ temp:URL) throws {
      let video=temp.appendingPathComponent("restored.mkv"),frame=Data(repeating:UInt8(17+index),count:128*64*3)
      let writer=try RawVideoWriter(ffmpeg:ffmpeg,output:video,width:128,height:64,frames:chunk.frames,fps:24,encoding:.ffv1)
      try writer.append(frame,frame:0);try writer.finish()
      let proof=try LosslessVideoVerifier.verify(ffmpeg:ffmpeg,video:video,width:128,height:64,frames:1,fps:24,expectedRGBSHA256:writer.rgbSHA256)
      try JSONEncoder().encode(proof).write(to:temp.appendingPathComponent("lossless.json"))
    }
    do {
      _ = try await MLXMovieChunkCoordinator.execute(request:first,chunks:chunks,binding:binding,directory:directory,ffmpeg:ffmpeg) { chunk,index,temp in
        if index==1 { throw CancellationError() };try render(chunk,index,temp)
      };XCTFail("Cancellation was ignored")
    } catch { XCTAssertTrue(error is CancellationError) }
    let completed=try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:binding,directory:directory,ffmpeg:ffmpeg,render:render)
    XCTAssertEqual(completed.map(\.reused),[true,false]);XCTAssertEqual(completed.map(\.receipt.version),[2,2])
    let receipt=directory.appendingPathComponent("chunk-000000/chunk.json")
    var changed=try JSONSerialization.jsonObject(with:Data(contentsOf:receipt)) as! [String:Any]
    changed["rgbSHA256"]=String(repeating:"0",count:64)
    try JSONSerialization.data(withJSONObject:changed).write(to:receipt,options:.atomic)
    do {
      _ = try await MLXMovieChunkCoordinator.execute(request:retry,chunks:chunks,binding:binding,directory:directory,ffmpeg:ffmpeg) { _,_,_ in XCTFail("Invalid proof invoked weighted executor") }
      XCTFail("Changed RGB proof was reused")
    } catch { XCTAssertTrue(String(describing:error).contains("RGB"),"\(error)") }
  }
}
