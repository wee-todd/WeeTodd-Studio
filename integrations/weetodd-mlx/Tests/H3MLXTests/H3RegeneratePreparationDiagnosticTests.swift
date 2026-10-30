import Foundation
import InferenceContracts
import MLX
import XCTest
@testable import H3MLX

final class H3RegeneratePreparationDiagnosticTests:XCTestCase {
  func testInstalledSourceRoundtripAndPreparedTargetDecode() throws {
    let env=ProcessInfo.processInfo.environment
    guard let jobPath=env["WEETODD_H3_PREPARATION_JOB"],let artifacts=env["WEETODD_H3_PREPARATION_ARTIFACTS"],
      let destination=env["WEETODD_H3_PREPARATION_OUTPUT"],let worker=env["WEETODD_H3_PREPARATION_WORKER"] else {
      throw XCTSkip("Explicit source preparation diagnostics require a frozen job, artifacts and fresh output directory.")
    }
    GPU.metallib=try NativeMLXMetalLibrary.resolve(executable:URL(fileURLWithPath:worker))
    let job=try H3RegenerateJob.decode(Data(contentsOf:URL(fileURLWithPath:jobPath))),root=URL(fileURLWithPath:destination)
    guard !FileManager.default.fileExists(atPath:root.path) else {throw H3CheckpointError.invalid("Diagnostic output already exists.")}
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    let source=try H3Geometry(regenerateWidth:job.source.width,height:job.source.height,paddedFrames:job.source.paddedFrames)
    let target=try H3Geometry(regenerateWidth:job.target.width,height:job.target.height,paddedFrames:job.source.paddedFrames)
    let vae=try H3VideoVAELayout(url:URL(fileURLWithPath:job.components.videoVAE.path))
    func floats(_ url:URL,shape:[Int]) throws -> MLXArray {
      let data=try Data(contentsOf:url),bytes=shape.reduce(4,*)
      guard data.count==bytes else {throw H3CheckpointError.invalid("Diagnostic tensor bytes differ from the frozen grid.")}
      return data.withUnsafeBytes{MLXArray($0,shape,type:Float.self)}
    }
    func decode(_ rows:MLXArray,geometry:H3Geometry,name:String) throws {
      let raw=try H3LatentCodec.videoDecoderInput(rows:rows,latentFrames:geometry.videoLatentFrames,
        latentHeight:geometry.height/16,latentWidth:geometry.width/16,mean:vae.latentsMean,standardDeviation:vae.latentsStandardDeviation)
      let url=root.appendingPathComponent(name+".rgb24")
      guard FileManager.default.createFile(atPath:url.path,contents:nil) else {throw H3CheckpointError.invalid("Cannot create diagnostic RGB.")}
      let handle=try FileHandle(forWritingTo:url);defer{try?handle.close()}
      var written=0
      try H3VideoVAEDecoder.decodeRegenerateChunks(checkpointURL:URL(fileURLWithPath:job.components.videoVAE.path),latent:raw,geometry:geometry){pixels in
        let count=min(pixels.shape[1],job.source.visibleFrames-written)
        if count>0 {
          let rgb=try H3FastVideoDecoder.rgb8(pixels[0..<1,0..<count,.ellipsis])
          try handle.write(contentsOf:H3RegenerateArtifact.copyData(rgb));written += count
        }
      }
      XCTAssertEqual(written,job.source.visibleFrames)
      try handle.synchronize();Stream.gpu.synchronize();Memory.clearCache()
    }
    let sourceRows=try autoreleasepool {
      var rgb=[UInt8](try Data(contentsOf:URL(fileURLWithPath:job.source.payload.path)))
      let last=Array(rgb.suffix(job.source.width*job.source.height*3))
      for _ in job.source.visibleFrames..<job.source.paddedFrames{rgb.append(contentsOf:last)}
      let noise=try floats(URL(fileURLWithPath:artifacts).appendingPathComponent("posterior-noise.f32"),
        shape:[1,source.videoLatentFrames,source.height/16,source.width/16,24])
      let rows=try H3VideoVAEEncoder.encodePublisherVideoRows(checkpointURL:URL(fileURLWithPath:job.components.publisherEncoder!.path),
        rgb8:rgb,frameCount:source.frames,width:source.width,height:source.height,posteriorNoise:noise)
      eval(rows);return rows
    }
    try H3RegenerateArtifact.copyData(sourceRows).write(to:root.appendingPathComponent("source-video.f32"),options:.withoutOverwriting)
    Stream.gpu.synchronize();Memory.clearCache()
    try decode(sourceRows,geometry:source,name:"source-roundtrip")
    let upscaled=try floats(URL(fileURLWithPath:artifacts).appendingPathComponent("prepared-target-video.f32"),shape:[1,target.videoRows,96])
    try decode(upscaled,geometry:target,name:"prepared-target")
  }
}
