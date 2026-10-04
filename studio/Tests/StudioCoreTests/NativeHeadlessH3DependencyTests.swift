import Foundation
import Darwin
import XCTest
@testable import StudioCore

final class NativeHeadlessH3DependencyTests:XCTestCase {
  private func directory() throws -> URL {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent("H3Dependencies-\(UUID())")
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) };return root
  }
  private func file(_ root:URL,_ name:String,_ bytes:Data=Data("12345678".utf8)) throws -> URL {
    let url=root.appendingPathComponent(name);try bytes.write(to:url)
    // Whole-second fixtures make Foundation's Date restoration exact; SHA still
    // has to catch the same-size mutation with an unchanged modification date.
    try FileManager.default.setAttributes([.modificationDate:Date(timeIntervalSince1970:1_700_000_000)],ofItemAtPath:url.path)
    return url
  }
  private func makeFixture(_ root:URL) throws -> (job:NativeHeadlessJob,inputs:[URL]) {
    let sidecar=try file(root,"sidecar.wav"),movie=try file(root,"motion.mp4")
    let payload=try file(root,"joint-latents.f32"),context=try file(root,"latents.f32")
    func manifest(_ name:String,_ payload:URL) throws -> URL {
      try file(root,name,JSONSerialization.data(withJSONObject:["payloadSHA256":try NativeHeadlessJob.fileHash(payload)],options:.sortedKeys))
    }
    let joint=try manifest("joint.json",payload),continuation=try manifest("continuation.json",context)
    let alias=root.appendingPathComponent("movie-alias.mp4")
    try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:movie)
    var project=StudioProject();project.settings.width=64;project.settings.height=64
    var clip=Clip(engine:.h3);clip.duration=1;clip.prompt="Frozen creative source dependencies.";clip.generationWidth=64;clip.generationHeight=64
    project.clips=[clip]
    let raw:[String:Any]=["format":"weetodd-headless-v2","engine":"h3","prompt":clip.prompt,
      "config":["seed":42],"components":["transformer":"/missing-model/ignored.safetensors"],
      "loras":["adapters":[["path":"/missing-lora/ignored.safetensors"]]],
      "conditioning":["task":"ref2va","inputs":[["path":alias.path,"soundtrack_path":sidecar.path,
        "soundtrack_sha256":try NativeHeadlessJob.fileHash(sidecar)]]],
      "motion_fidelity":["source_video":movie.path,"source_sha256":try NativeHeadlessJob.fileHash(movie)],
      "refinement":["source_manifest":joint.path,"source_manifest_sha256":try NativeHeadlessJob.fileHash(joint)],
      "continuation":["source_context":continuation.path,"source_manifest_sha256":try NativeHeadlessJob.fileHash(continuation)]]
    // Transport-only fixture deliberately combines fields to exercise collection,
    // not to claim a compatible model recipe; native compiler owns those restrictions.
    let bytes=try JSONSerialization.data(withJSONObject:raw,options:.sortedKeys)
    let job=try NativeHeadlessJob(project:project,recipes:[clip.id.uuidString:
      .init(engine:"h3",bytes:bytes,signature:"dependencies")],workers:["h3":"/usr/bin/true"],ffmpeg:"/usr/bin/true")
    return (job,[sidecar,movie,joint,payload,continuation,context])
  }
  func testNewH3CreativeInputsAndSiblingPayloadsAreHashedWithAliasesDeduplicatedAndModelsExcluded() throws {
    let root=try directory(),fixture=try makeFixture(root)
    XCTAssertEqual(fixture.job.sources.count,6)
    XCTAssertEqual(Set(fixture.job.sources.map(\.path)),Set(try fixture.inputs.map { url -> String in
      let pointer=try XCTUnwrap(Darwin.realpath(url.path,nil));defer { free(pointer) };return String(cString:pointer)
    }))
    XCTAssertTrue(fixture.job.sources.allSatisfy { $0.sha256 != nil })
    XCTAssertNoThrow(try fixture.job.verifySources())
    let saved=root.appendingPathComponent("job.json");try fixture.job.write(to:saved)
    let reopened=try NativeHeadlessJob.read(from:saved)
    XCTAssertEqual(reopened.sources,fixture.job.sources)
    XCTAssertEqual(reopened.recipes.values.first?.bytes,fixture.job.recipes.values.first?.bytes)
  }
  func testEveryNewDependencySameSizeRestoredTimestampMutationRejectsBeforeAnyWorkerOrFinishing() async throws {
    let root=try directory(),fixture=try makeFixture(root)
    for (index,path) in fixture.inputs.enumerated() {
      let original=try Data(contentsOf:path)
      let modified=try FileManager.default.attributesOfItem(atPath:path.path)[.modificationDate] as! Date
      var changed=original;changed[changed.startIndex] ^= 1
      try changed.write(to:path);try FileManager.default.setAttributes([.modificationDate:modified],ofItemAtPath:path.path)
      XCTAssertEqual(try FileManager.default.attributesOfItem(atPath:path.path)[.modificationDate] as? Date,modified)
      var calls=0
      do {
        _ = try await NativeHeadlessExecutor.run(job:fixture.job,output:root.appendingPathComponent("reject-\(index)"),
          preflightOnly:true,worker:{ _,_,_,_ in calls += 1;return [:] })
        XCTFail("Changed input must reject before FFmpeg/worker admission")
      } catch { XCTAssertTrue(error.localizedDescription.contains("frozen native job source changed"),error.localizedDescription) }
      XCTAssertEqual(calls,0)
      XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("reject-\(index)").path))
      try original.write(to:path);try FileManager.default.setAttributes([.modificationDate:modified],ofItemAtPath:path.path)
      XCTAssertNoThrow(try fixture.job.verifySources())
    }
  }
  func testMissingSiblingPayloadOrStaleManifestDigestRejectsDuringExport() throws {
    let root=try directory(),fixture=try makeFixture(root)
    try FileManager.default.removeItem(at:fixture.inputs[3])
    XCTAssertThrowsError(try NativeHeadlessJob(project:fixture.job.project,recipes:fixture.job.recipes,workers:fixture.job.workers,ffmpeg:fixture.job.ffmpeg))
    try Data("12345678".utf8).write(to:fixture.inputs[3])
    try Data("bad manifest".utf8).write(to:fixture.inputs[2])
    XCTAssertThrowsError(try NativeHeadlessJob(project:fixture.job.project,recipes:fixture.job.recipes,workers:fixture.job.workers,ffmpeg:fixture.job.ffmpeg))
  }
  func testCompletedTakeResumeRefusesChangedH3SidecarBeforeAnyWorker() async throws {
    let ffmpeg="/opt/homebrew/bin/ffmpeg"
    guard FileManager.default.isExecutableFile(atPath:ffmpeg) else { throw XCTSkip("Tiny accepted-media resume fixture needs FFmpeg") }
    let root=try directory(),fixture=try makeFixture(root)
    var job=fixture.job;job.ffmpeg=ffmpeg
    let media=root.appendingPathComponent("accepted-fixture.mp4")
    let process=Process();process.executableURL=URL(fileURLWithPath:ffmpeg)
    process.arguments=["-v","error","-nostdin","-n","-f","lavfi","-i","color=c=red:s=64x64:r=24:d=1",
      "-f","lavfi","-i","anullsrc=r=48000:cl=stereo","-t","1","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",media.path]
    process.standardOutput=FileHandle.nullDevice;process.standardError=FileHandle.nullDevice
    try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
    let output=root.appendingPathComponent("accepted")
    let first=try await NativeHeadlessExecutor.run(job:job,output:output,worker:{ _,_,target,action in
      if action == "preflight" { return [:] }
      try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
      let take=target.appendingPathComponent("render.mp4");try FileManager.default.copyItem(at:media,to:take)
      return ["video":take.path,"nativeRuntime":"swift-mlx"]
    })
    XCTAssertEqual(first["newlyGenerated"] as? Int,1)
    let state=output.appendingPathComponent("job-state.json")
    let before=try Data(contentsOf:state)
    let sidecar=fixture.inputs[0]
    let modified=try FileManager.default.attributesOfItem(atPath:sidecar.path)[.modificationDate] as! Date
    try Data("87654321".utf8).write(to:sidecar)
    try FileManager.default.setAttributes([.modificationDate:modified],ofItemAtPath:sidecar.path)
    var calls=0
    do {
      _ = try await NativeHeadlessExecutor.run(job:job,output:output,resume:true,worker:{ _,_,_,_ in calls += 1;return [:] })
      XCTFail("Resume cannot accept changed source inputs")
    } catch { XCTAssertTrue(error.localizedDescription.contains("frozen native job source changed"),error.localizedDescription) }
    XCTAssertEqual(calls,0);XCTAssertEqual(try Data(contentsOf:state),before)
  }
}
