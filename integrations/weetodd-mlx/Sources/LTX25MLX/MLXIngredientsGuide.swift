import Foundation
import InferenceContracts
import LTX25Engine

/// Builds one bounded RGB24 guide from a single frozen Ingredients sheet.
/// Repetition happens on disk before loading the video VAE.
enum MLXIngredientsGuide {
  static func prepare(_ sheet:MLXIngredientsSheet,geometry:AVGeometry,
    ffmpeg:URL,directory:URL) throws -> URL {
    try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
    let pixels=try MLXReferenceImage.prepare(URL(fileURLWithPath:sheet.path),
      width:geometry.width,height:geometry.height,crf:0,ffmpeg:ffmpeg,
      temporaryParent:directory)
    let output=directory.appendingPathComponent("ingredients-guide.rgb")
    try writeRepeatedRGB(pixels: pixels, geometry: geometry, to: output)
    try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
    return output
  }

  static func writeRepeatedRGB(pixels:[Float],geometry:AVGeometry,to output:URL) throws {
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
    for _ in 0..<geometry.frames {
      try Task.checkCancellation()
      try handle.write(contentsOf:frame)
    }
  }
}
