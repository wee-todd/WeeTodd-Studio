import CryptoKit
import Foundation
import ImageIO
import CoreGraphics

struct MOSSLayoutImage {
  let image: CGImage
  let width, height: Int
  let sourceSHA256: String
  static func sha256(_ url: URL) throws -> String {
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, let size = values.fileSize, size > 0, size <= 256 * 1024 * 1024 else {
      throw MOSSError("Choose a regular image file up to 256 MiB.")
    }
    let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }; var digest = SHA256()
    while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty {
      try Task.checkCancellation(); digest.update(data: bytes)
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
  static func prepare(source: URL, expectedSHA256: String, output: URL) throws -> MOSSLayoutImage {
    guard try sha256(source) == expectedSHA256 else { throw MOSSError("Source image changed after selection. Select it again.") }
    guard let decoder = CGImageSourceCreateWithURL(source as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
      let info = CGImageSourceCopyPropertiesAtIndex(decoder, 0, nil) as? [CFString: Any],
      let rawWidth = info[kCGImagePropertyPixelWidth] as? Int, let rawHeight = info[kCGImagePropertyPixelHeight] as? Int,
      rawWidth > 0, rawHeight > 0, rawWidth <= 131_072, rawHeight <= 131_072,
      Double(rawWidth) * Double(rawHeight) <= 128_000_000 else { throw MOSSError("The source is not a supported bounded still image.") }
    let orientation = info[kCGImagePropertyOrientation] as? Int ?? 1
    let width = (5...8).contains(orientation) ? rawHeight : rawWidth
    let height = (5...8).contains(orientation) ? rawWidth : rawHeight
    guard let oriented = CGImageSourceCreateThumbnailAtIndex(decoder, 0, [
      kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: min(2048, max(width, height)),
      kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
      let space = CGColorSpace(name: CGColorSpace.sRGB),
      let context = CGContext(data: nil, width: oriented.width, height: oriented.height, bitsPerComponent: 8,
        bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw MOSSError("Unable to orient and decode the source image.") }
    context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: oriented.width, height: oriented.height))
    context.draw(oriented, in: CGRect(x: 0, y: 0, width: oriented.width, height: oriented.height))
    guard let image = context.makeImage() else { throw MOSSError("Unable to prepare the source image.") }
    guard try sha256(source) == expectedSHA256 else { throw MOSSError("Source image changed while preparing it.") }
    guard !FileManager.default.fileExists(atPath: output.path) else { throw MOSSError("Analysis source snapshot already exists.") }
    let temporary = output.deletingLastPathComponent().appendingPathComponent("." + UUID().uuidString + ".png")
    defer { try? FileManager.default.removeItem(at: temporary) }
    guard let writer = CGImageDestinationCreateWithURL(temporary as CFURL, "public.png" as CFString, 1, nil) else { throw MOSSError("Unable to create the analysis snapshot.") }
    CGImageDestinationAddImage(writer, image, nil)
    guard CGImageDestinationFinalize(writer) else { throw MOSSError("Unable to write the analysis snapshot.") }
    try FileManager.default.moveItem(at: temporary, to: output)
    return .init(image: image, width: width, height: height, sourceSHA256: expectedSHA256)
  }
}
