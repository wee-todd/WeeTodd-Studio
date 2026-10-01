import Foundation
import InferenceContracts
import LTX25Engine

/// Freeze and encode one MSR image at a time. The source is never sent to a
/// model at its original resolution; only the admitted 25/33-frame grid enters
/// the VAE, with subject/object/clothing images fully visible on white.
enum MLXMSRGuide {
  static func prepare(_ reference:MLXMSRReference,plan:MLXMSRReferencePlan,
    ffmpeg:URL,directory:URL,index:Int) throws -> URL {
    let source=try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256)
    try source.verify()
    let url=URL(fileURLWithPath:reference.path)
    let pixels=try plan.role == "background"
      ? MLXReferenceImage.prepare(url,width:plan.geometry.width,height:plan.geometry.height,
        crf:0,ffmpeg:ffmpeg,temporaryParent:directory)
      : MLXReferenceImage.prepareFitWhite(url,width:plan.geometry.width,height:plan.geometry.height)
    let output=directory.appendingPathComponent("msr-reference-\(index).rgb")
    try writeRepeatedRGB(pixels:pixels,geometry:plan.geometry,to:output)
    try source.verify()
    return output
  }

  static func writeRepeatedRGB(pixels:[Float],geometry:AVGeometry,to output:URL) throws {
    guard [25,33].contains(geometry.frames),pixels.count == geometry.width*geometry.height*3,
      pixels.allSatisfy(\.isFinite),!FileManager.default.fileExists(atPath:output.path) else {
      throw LTXError.invalid("MSR needs one finite still and 25 or 33 causal guide frames.")
    }
    let frame=Data(pixels.map { UInt8((min(1,max(-1,$0))+1)*127.5+0.5) })
    guard FileManager.default.createFile(atPath:output.path,contents:nil) else {
      throw LTXError.invalid("Cannot create MSR RGB24 guide.")
    }
    let handle=try FileHandle(forWritingTo:output)
    defer { try? handle.close() }
    for _ in 0..<geometry.frames {
      try Task.checkCancellation()
      try handle.write(contentsOf:frame)
    }
  }
}
