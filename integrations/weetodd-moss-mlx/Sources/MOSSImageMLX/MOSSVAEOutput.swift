import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public enum MOSSVAEOutput {
  public static func rgb8(_ image: MLXArray) -> [UInt8] {
    (clip(image, min: -1, max: 1) * 127.5 + 127.5).round().asType(.uint8).asArray(UInt8.self)
  }
  public static func writePNG(_ image: MLXArray, to url: URL) throws {
    guard image.ndim == 4, image.dim(0) == 1, image.dim(3) == 3 else { throw MOSSError("Expected one NHWC RGB image.") }
    guard isFinite(image).all().item(Bool.self) else { throw MOSSError("MOSS decoded nonfinite pixels; the PNG was not saved.") }
    let width = image.dim(2), height = image.dim(1), bytes = Data(rgb8(image))
    guard let provider = CGDataProvider(data: bytes as CFData),
      let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw MOSSError("Could not encode RGB output.") }
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let partial = url.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).png")
    defer { try? FileManager.default.removeItem(at: partial) }
    guard let destination = CGImageDestinationCreateWithURL(partial as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw MOSSError("Could not create PNG output.") }
    CGImageDestinationAddImage(destination, cg, nil)
    guard CGImageDestinationFinalize(destination) else { throw MOSSError("PNG encoding failed.") }
    guard !FileManager.default.fileExists(atPath: url.path) else { throw MOSSError("Output already exists; choose a new take path.") }
    try FileManager.default.moveItem(at: partial, to: url)
  }
}
