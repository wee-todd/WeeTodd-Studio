import Foundation
import XCTest
@testable import StudioCore

final class NativeHeadlessDependencyResumeTests: XCTestCase {
  private let ffmpeg = "/opt/homebrew/bin/ffmpeg"
  private func directory() throws -> URL {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent("HeadlessDependencies-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) };return root
  }
  private func project(_ engine:Engine) -> StudioProject {
    var p=StudioProject();p.settings.width=64;p.settings.height=64
    var c=Clip(engine:engine);c.duration=73.0/24;c.generationWidth=64;c.generationHeight=64;c.prompt="Frozen dependency fixture"
    p.clips=[c];return p
  }
  private func recipe(_ engine:Engine,movie:Bool=false) throws -> Data {
    var body:[String:Any]=["format":"weetodd-headless-v2","engine":engine.rawValue,"prompt":"Frozen dependency fixture",
      "config":[:],"components":[:],"conditioning":["task":movie ? "video_upscale":"t2v","inputs":[],"audio_policy":"generated"]]
    if movie { body["movie_upscale"]=["version":1,"engine":"ltx25","task":"video_upscale"] }
    return try JSONSerialization.data(withJSONObject:body,options:[.sortedKeys])
  }
  private func captureWorker(_ root:URL) throws -> (URL,URL) {
    let script=root.appendingPathComponent("worker"),capture=root.appendingPathComponent("captured.json")
    let content = #"""
    #!/bin/sh
    set -eu
    /bin/cp "$3" '\#(capture.path)'
    id=$(/usr/bin/sed -n 's/.*"jobID"[ ]*:[ ]*"\([^"]*\)".*/\1/p' "$3")
    printf '{"status":"success","result":{"jobID":"%s","nativeRuntime":"swift-mlx","task":"fixture"}}\n' "$id"
    """#
    try content.write(to:script,atomically:true,encoding:.utf8)
    try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:script.path)
    return (script,capture)
  }
  func testExportedMoviePreflightPassesFrozenFFmpegThroughActualProcessEnvelope() async throws {
    guard FileManager.default.isExecutableFile(atPath:ffmpeg) else { throw XCTSkip("Tiny media fixture requires FFmpeg") }
    let root=try directory(),(worker,capture)=try captureWorker(root),p=project(.ltx25),bytes=try recipe(.ltx25,movie:true)
    let job=try NativeHeadlessJob(project:p,recipes:[p.clips[0].id.uuidString:.init(engine:"ltx25",bytes:bytes,signature:"movie")],
      workers:["ltx25":worker.path],ffmpeg:ffmpeg)
    let result=try await NativeHeadlessExecutor.run(job:job,output:root.appendingPathComponent("export"),preflightOnly:true)
    let envelope=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:capture)) as? [String:Any])
    XCTAssertEqual(envelope["ffmpegPath"] as? String,job.ffmpeg)
    XCTAssertEqual(envelope["recipeSHA256"] as? String,NativeHeadlessJob.hash(bytes))
    XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:try XCTUnwrap(envelope["recipePath"] as? String))),bytes)
    XCTAssertEqual(result["newlyGenerated"] as? Int,0)
    XCTAssertEqual(result["python_inference"] as? Bool,false)
  }
  func testDirectWorkerDefaultStillOmitsOptionalFFmpeg() throws {
    let root=try directory(),(worker,capture)=try captureWorker(root),path=root.appendingPathComponent("recipe.json")
    try recipe(.ltx25).write(to:path)
    _ = try NativeHeadlessExecutor.worker(worker.path,recipe:path,output:root.appendingPathComponent("out"),mode:"preflight",
      cancellation:NativeHeadlessCancellation(),emit:{ _ in })
    let envelope=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:capture)) as? [String:Any])
    XCTAssertNil(envelope["ffmpegPath"])
  }
  private func movie(_ root:URL) throws -> URL {
    guard FileManager.default.isExecutableFile(atPath:ffmpeg) else { throw XCTSkip("Tiny audiovisual fixture requires FFmpeg") }
    let url=root.appendingPathComponent("fixture.mp4"),child=Process();child.executableURL=URL(fileURLWithPath:ffmpeg)
    child.arguments=["-v","error","-f","lavfi","-i","testsrc2=s=64x64:r=24:d=3.041666666667",
      "-f","lavfi","-i","sine=frequency=440:sample_rate=48000:duration=3.041666666667",
      "-frames:v","73","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-ac","2","-y",url.path]
    child.standardOutput=FileHandle.nullDevice;child.standardError=FileHandle.nullDevice
    try child.run();child.waitUntilExit();XCTAssertEqual(child.terminationStatus,0);return url
  }
  func testCompletedH3JointArtifactMutationOrDeletionRejectsResumeBeforeWorker() async throws {
    let root=try directory(),media=try movie(root),(executable,_)=try captureWorker(root),p=project(.h3),id=p.clips[0].id.uuidString
    let job=try NativeHeadlessJob(project:p,recipes:[id:.init(engine:"h3",bytes:try recipe(.h3),signature:"joint")],
      workers:["h3":executable.path],ffmpeg:ffmpeg),output=root.appendingPathComponent("run")
    var calls=0
    _ = try await NativeHeadlessExecutor.run(job:job,output:output,worker:{ _,_,target,mode in
      calls+=1
      if mode == "preflight" { return ["nativeRuntime":"swift-mlx"] }
      try FileManager.default.createDirectory(at:target,withIntermediateDirectories:true)
      let movie=target.appendingPathComponent("render.mp4");try FileManager.default.copyItem(at:media,to:movie)
      let videoFloats=22*2*2*96,audioFloats=2*122*32
      let payload=Data(repeating:0,count:(videoFloats+audioFloats)*4),payloadSHA=NativeHeadlessJob.hash(payload)
      try payload.write(to:target.appendingPathComponent("joint-latents.f32"))
      let manifest=try JSONSerialization.data(withJSONObject:["format":"weetodd-h3-swift-joint-latents-v1","task":"t2va",
        "width":64,"height":64,"generatedFrames":73,"componentIdentity":String(repeating:"a",count:64),
        "videoFloats":videoFloats,"audioFloats":audioFloats,"payloadBytes":payload.count,"payloadSHA256":payloadSHA],options:[.sortedKeys])
      let manifestURL=target.appendingPathComponent("joint-latents.json");try manifest.write(to:manifestURL)
      return ["nativeRuntime":"swift-mlx","video":movie.path,"metadata":["jointLatentManifest":manifestURL.path,
        "jointLatentManifestSHA256":NativeHeadlessJob.hash(manifest),"jointLatentPayloadSHA256":payloadSHA]]
    })
    XCTAssertEqual(calls,2)
    let accepted=try ProjectStorage.read(output.appendingPathComponent("result.weetodd"))
    let artifact=try XCTUnwrap(accepted.clips[0].versions.last?.jointLatentArtifact)
    let mediaSHA=try NativeHeadlessJob.fileHash(URL(fileURLWithPath:accepted.clips[0].sourcePath))
    let payloadURL=URL(fileURLWithPath:artifact.payloadPath),manifestURL=URL(fileURLWithPath:artifact.manifest)
    let originalPayload=try Data(contentsOf:payloadURL),originalManifest=try Data(contentsOf:manifestURL)
    var resumedCalls=0
    let noWorker:NativeHeadlessExecutor.Worker={ _,_,_,_ in resumedCalls+=1;XCTFail("Completed artifacts must reject before workers");return [:] }
    let resumed=try await NativeHeadlessExecutor.run(job:job,output:output,resume:true,worker:noWorker)
    XCTAssertEqual(resumed["newlyGenerated"] as? Int,0);XCTAssertEqual(resumedCalls,0)
    for kind in ["payload-mutation","payload-deletion","manifest-mutation","manifest-deletion"] {
      try originalPayload.write(to:payloadURL);try originalManifest.write(to:manifestURL)
      switch kind {
      case "payload-mutation":var changed=originalPayload;changed[0]=1;try changed.write(to:payloadURL)
      case "payload-deletion":try FileManager.default.removeItem(at:payloadURL)
      case "manifest-mutation":try Data("{}".utf8).write(to:manifestURL)
      default:try FileManager.default.removeItem(at:manifestURL)
      }
      do { _ = try await NativeHeadlessExecutor.run(job:job,output:output,resume:true,worker:noWorker);XCTFail("Accepted \(kind)") }
      catch { XCTAssertTrue(error.localizedDescription.contains("joint") || error.localizedDescription.contains("H3"),error.localizedDescription) }
      XCTAssertEqual(resumedCalls,0)
      XCTAssertEqual(try NativeHeadlessJob.fileHash(URL(fileURLWithPath:accepted.clips[0].sourcePath)),mediaSHA)
    }
  }
}
