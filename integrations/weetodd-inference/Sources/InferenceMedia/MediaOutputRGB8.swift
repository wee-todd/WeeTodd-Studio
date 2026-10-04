import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

extension MediaOutput {
  /// Preserve the decoder's already-quantized RGB24 bytes for frame exports.
  /// No conversion back to floating point and no second quantization boundary.
  public static func writeRGB8PNG(_ bytes:Data,width:Int,height:Int,to url:URL) throws {
    guard (1...16384).contains(width),(1...16384).contains(height),bytes.count == width*height*3,
      !FileManager.default.fileExists(atPath:url.path) else { throw MediaOutputError.invalid("Invalid or existing RGB24 PNG output.") }
    let png=NSMutableData()
    guard let provider=CGDataProvider(data:bytes as CFData),let color=CGColorSpace(name:CGColorSpace.sRGB),
      let image=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:24,bytesPerRow:width*3,
        space:color,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,
        decode:nil,shouldInterpolate:false,intent:.defaultIntent),
      let destination=CGImageDestinationCreateWithData(png,UTType.png.identifier as CFString,1,nil) else {
      throw MediaOutputError.invalid("Cannot construct a decoder RGB24 PNG.")
    }
    CGImageDestinationAddImage(destination,image,nil)
    guard CGImageDestinationFinalize(destination) else { throw MediaOutputError.invalid("Cannot finalize a decoder RGB24 PNG.") }
    try (png as Data).write(to:url,options:.withoutOverwriting)
  }
}
