import Foundation
import MLX
import InferenceContracts
import LTX25Engine

/// Builds one bounded RGB24 guide from a single frozen Ingredients sheet.
/// Static stills are encoded once, then repeated in latent space. Moving
/// guides continue through the separate causal video encoder.
enum MLXIngredientsGuide {
  static func prepare(_ sheet:MLXIngredientsSheet,geometry:AVGeometry,
    ffmpeg:URL,directory:URL,repeatedFrames:Bool=false) throws -> URL {
    try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
    let pixels=try MLXReferenceImage.prepare(URL(fileURLWithPath:sheet.path),
      width:geometry.width,height:geometry.height,crf:0,ffmpeg:ffmpeg,
      temporaryParent:directory)
    let output=directory.appendingPathComponent("ingredients-guide.rgb")
    if repeatedFrames { try writeRepeatedRGB(pixels:pixels,geometry:geometry,to:output) }
    else { try writeStaticRGB(pixels: pixels, geometry: geometry, to: output) }
    try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
    return output
  }

  static func writeRepeatedRGB(pixels:[Float],geometry:AVGeometry,to output:URL) throws {
    try writeRGB(pixels:pixels,geometry:geometry,frames:geometry.frames,to:output)
  }

  static func writeStaticRGB(pixels:[Float],geometry:AVGeometry,to output:URL) throws {
    try writeRGB(pixels:pixels,geometry:geometry,frames:1,to:output)
  }

  private static func writeRGB(pixels:[Float],geometry:AVGeometry,frames:Int,to output:URL) throws {
    guard geometry.frames >= 121, pixels.count == geometry.width*geometry.height*3,
      pixels.allSatisfy(\.isFinite), !FileManager.default.fileExists(atPath:output.path) else {
      throw LTXError.invalid("Ingredients needs one finite full-canvas image and at least 121 frames.")
    }
    let frame=Data(pixels.map { UInt8((min(1,max(-1,$0))+1)*127.5+0.5) })
    guard FileManager.default.createFile(atPath:output.path,contents:nil) else {
      throw LTXError.invalid("Cannot create Ingredients RGB24 guide.")
    }
    let handle=try FileHandle(forWritingTo:output)
    defer { try? handle.close() }
    for _ in 0..<frames {
      try Task.checkCancellation()
      try handle.write(contentsOf:frame)
    }
  }

  static func repeatStaticLatent(_ frame:MLXArray,geometry:AVGeometry) throws -> MLXArray {
    try Task.checkCancellation()
    guard frame.dtype == .float32,
      frame.shape == [1,geometry.latentHeight,geometry.latentWidth,128],
      MLX.isFinite(frame).all().item(Bool.self) else {
      throw LTXError.invalid("Static Ingredients encoding must contain exactly one finite latent frame.")
    }
    let result=broadcast(frame,to:[geometry.latentFrames,geometry.latentHeight,geometry.latentWidth,128])
      .reshaped([geometry.videoTokens,128])
    eval(result)
    try Task.checkCancellation()
    return result
  }

  static func encodeStatic(guide:URL,checkpoint:URL,geometry:AVGeometry,
    maximumOwnedBufferBytes:Int,progress:(Int,Int) throws -> Void) throws -> MLXArray {
    let plan=try MLXVideoEncodeTilePlan(frames:1,width:geometry.width,height:geometry.height,
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    let frame=try MLXTiledVideoEncoder.encode(guide:guide,checkpoint:checkpoint,plan:plan,progress:progress)
    return try repeatStaticLatent(frame,geometry:geometry)
  }
}
