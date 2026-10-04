import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Decode one FL2VA endpoint directly to its generation canvas. The first
/// endpoint establishes the canvas by stretching; the last uses a cover crop.
public enum H3FL2VAMedia {
  public struct Loaded {
    public let image: H3StillReference
    public let sourceSHA256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
  }

  public static func load(path: String, width: Int, height: Int,
    first: Bool, canvasAdmission: H3CanvasAdmission = .ordinary) throws -> Loaded {
    guard path.hasPrefix("/"), !path.utf8.contains(0),
      width > 0, height > 0, width * height <= canvasAdmission.maximumPixels else {
      throw H3CheckpointError.invalid("Invalid H3 FL2VA source or canvas.")
    }
    if canvasAdmission == .spatialRefinement { try canvasAdmission.validate(width: width,height: height) }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw H3CheckpointError.invalid("Cannot open an H3 endpoint image.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      (1...128 * 1024 * 1024).contains(status.st_size) else {
      throw H3CheckpointError.invalid("H3 endpoint must be a regular image under 128 MiB.")
    }
    let bytes = try file.read(upToCount: 128 * 1024 * 1024 + 1) ?? Data()
    guard bytes.count == Int(status.st_size),
      let source = CGImageSourceCreateWithData(bytes as CFData,
        [kCGImageSourceShouldCache: false] as CFDictionary),
      CGImageSourceGetCount(source) == 1,
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let sourceWidth = properties[kCGImagePropertyPixelWidth] as? Int,
      let sourceHeight = properties[kCGImagePropertyPixelHeight] as? Int,
      (1...20_000).contains(sourceWidth), (1...20_000).contains(sourceHeight),
      sourceWidth * sourceHeight <= 100_000_000 else {
      throw H3CheckpointError.invalid("Invalid H3 endpoint image dimensions.")
    }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: max(width, height) * 2,
      kCGImageSourceShouldCache: false]
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0,
      options as CFDictionary), image.width > 0, image.height > 0 else {
      throw H3CheckpointError.invalid("Cannot decode the H3 endpoint image.")
    }
    let rowBytes = width * 4
    var rgba = [UInt8](repeating: 0, count: height * rowBytes)
    let drawn = rgba.withUnsafeMutableBytes { memory -> Bool in
      guard let context = CGContext(data: memory.baseAddress,
        width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: rowBytes, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
      // Bitmap context bytes are read as top-down RGB rows below. Flipping
      // the CGContext here inverted both Qwen pixels and VAE keyframe latents.
      context.interpolationQuality = .high
      let scaleX = CGFloat(width) / CGFloat(image.width)
      let scaleY = CGFloat(height) / CGFloat(image.height)
      let scale = first ? nil : max(scaleX, scaleY)
      let drawnWidth = first ? CGFloat(width) : CGFloat(image.width) * scale!
      let drawnHeight = first ? CGFloat(height) : CGFloat(image.height) * scale!
      context.draw(image, in: CGRect(x: (CGFloat(width) - drawnWidth) / 2,
        y: (CGFloat(height) - drawnHeight) / 2,
        width: drawnWidth, height: drawnHeight))
      return true
    }
    guard drawn else { throw H3CheckpointError.invalid("Cannot prepare H3 endpoint pixels.") }
    var rgb = Data(count: width * height * 3)
    rgb.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
      for index in 0..<(width * height) {
        target[index * 3] = rgba[index * 4]
        target[index * 3 + 1] = rgba[index * 4 + 1]
        target[index * 3 + 2] = rgba[index * 4 + 2]
      }
    }
    return Loaded(image: H3StillReference(rgb8: rgb, width: width,
      height: height), sourceSHA256: SHA256.hash(data: bytes)
        .map { String(format: "%02x", $0) }.joined(),
      sourceWidth: sourceWidth, sourceHeight: sourceHeight)
  }
}
