import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Reads one compressed source once, records its exact digest, and asks ImageIO
/// for a bounded thumbnail. A 4K source never becomes a full-resolution RGB
/// tensor merely to enter Qwen or the reference video VAE.
public enum H3StillReferenceMedia {
  public struct Loaded {
    public let reference: H3StillReference
    public let sourceSHA256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
  }

  public static func load(path: String) throws -> Loaded {
    guard path.hasPrefix("/"), !path.utf8.contains(0) else {
      throw H3CheckpointError.invalid("H3 reference image needs an absolute local path.")
    }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else {
      throw H3CheckpointError.invalid("Cannot open the H3 reference image.")
    }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      (1...128 * 1024 * 1024).contains(status.st_size) else {
      throw H3CheckpointError.invalid("H3 reference must be a regular image file under 128 MiB.")
    }
    let bytes = try handle.read(upToCount: 128 * 1024 * 1024 + 1) ?? Data()
    guard bytes.count == Int(status.st_size),
      let source = CGImageSourceCreateWithData(bytes as CFData,
        [kCGImageSourceShouldCache: false] as CFDictionary),
      CGImageSourceGetCount(source) == 1,
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let sourceWidth = properties[kCGImagePropertyPixelWidth] as? Int,
      let sourceHeight = properties[kCGImagePropertyPixelHeight] as? Int,
      (1...20_000).contains(sourceWidth), (1...20_000).contains(sourceHeight),
      sourceWidth * sourceHeight <= 100_000_000 else {
      throw H3CheckpointError.invalid("H3 reference image has invalid or excessive source dimensions.")
    }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: 256,
      kCGImageSourceShouldCache: false,
    ]
    guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0,
      options as CFDictionary), thumbnail.width > 0, thumbnail.height > 0 else {
      throw H3CheckpointError.invalid("Cannot decode a bounded H3 reference thumbnail.")
    }
    let side = 256
    let rowBytes = side * 4
    var rgba = [UInt8](repeating: 0, count: side * rowBytes)
    let result = rgba.withUnsafeMutableBytes { memory -> Bool in
      guard let context = CGContext(data: memory.baseAddress, width: side,
        height: side, bitsPerComponent: 8, bytesPerRow: rowBytes,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
      context.setFillColor(CGColor(gray: 0.5, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: side, height: side))
      context.interpolationQuality = .high
      let fit = min(CGFloat(side) / CGFloat(thumbnail.width),
        CGFloat(side) / CGFloat(thumbnail.height))
      let width = CGFloat(thumbnail.width) * fit
      let height = CGFloat(thumbnail.height) * fit
      context.draw(thumbnail, in: CGRect(x: (CGFloat(side) - width) / 2,
        y: (CGFloat(side) - height) / 2, width: width, height: height))
      return true
    }
    guard result else {
      throw H3CheckpointError.invalid("Cannot prepare H3 reference RGB pixels.")
    }
    var rgb = Data(count: side * side * 3)
    rgb.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
      for pixel in 0..<(side * side) {
        target[pixel * 3] = rgba[pixel * 4]
        target[pixel * 3 + 1] = rgba[pixel * 4 + 1]
        target[pixel * 3 + 2] = rgba[pixel * 4 + 2]
      }
    }
    return Loaded(reference: H3StillReference(rgb8: rgb, width: side,
      height: side), sourceSHA256: SHA256.hash(data: bytes)
        .map { String(format: "%02x", $0) }.joined(),
      sourceWidth: sourceWidth, sourceHeight: sourceHeight)
  }
}
