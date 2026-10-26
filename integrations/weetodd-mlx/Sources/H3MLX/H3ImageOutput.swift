import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public enum H3ImageOutput {
  public static func png(rgb: Data, width: Int, height: Int) throws -> Data {
    guard (1...4096).contains(width), (1...4096).contains(height),
      rgb.count == width * height * 3,
      let provider = CGDataProvider(data: rgb as CFData),
      let image = CGImage(width: width, height: height, bitsPerComponent: 8,
        bitsPerPixel: 24, bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent) else {
      throw H3CheckpointError.invalid("H3 image output needs complete RGB8 pixels at the requested canvas.")
    }
    let bytes = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(bytes,
      UTType.png.identifier as CFString, 1, nil) else {
      throw H3CheckpointError.invalid("Cannot prepare H3 PNG output.")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw H3CheckpointError.invalid("Cannot finish H3 PNG output.")
    }
    return bytes as Data
  }

  public static func decode(videoRows: [Float], geometry: H3Geometry, decoder: URL,
    mode: H3StillImageMode, selection: H3ImageFrameSelection,
    memoryMode: H3VideoDecodeMemoryMode?, precision: H3VideoDecodePrecision) throws -> Data {
    guard geometry.stillImageMode == mode, videoRows.count == geometry.videoRows * 96,
      videoRows.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 still latent rows do not match the admitted image grid.")
    }
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    let layout = try H3VideoVAELayout(url: decoder)
    try mode.validateDecoder(imageOnly: layout.imageOnly)
    let latent = try H3LatentCodec.videoDecoderInput(
      rows: MLXArray(videoRows, [1,geometry.videoRows,96]),
      latentFrames: geometry.videoLatentFrames, latentHeight: geometry.height / 16,
      latentWidth: geometry.width / 16, mean: layout.latentsMean,
      standardDeviation: layout.latentsStandardDeviation)
    let image = try H3VideoVAEDecoder.decodeStill(checkpointURL: decoder,
      latent: latent, mode: mode, selection: selection, memoryMode: memoryMode, precision: precision)
    let bytes = try H3LatentCodec.videoPixelsRGB8(image).asArray(UInt8.self)
    return try png(rgb: Data(bytes), width: geometry.width, height: geometry.height)
  }
}
