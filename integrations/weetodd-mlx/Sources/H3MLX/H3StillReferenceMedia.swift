import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Reads one compressed source once, records its exact digest, and asks ImageIO
/// for a bounded thumbnail. A 4K source never becomes a full-resolution RGB
/// tensor merely to enter Qwen or the reference video VAE.
public enum H3StillReferenceMedia {
  public static let preparationPolicy = "aspect-grid-65536-v1"
  public struct Canvas: Sendable, Equatable {
    public let width: Int
    public let height: Int
  }

  /// Keep the former 256-square compute budget while preserving source aspect.
  /// Search at most 64 × 64 grid pairs; aspect error takes precedence over area.
  /// Small sources are never enlarged merely to satisfy the model grid.
  public static func resolveCanvas(sourceWidth: Int, sourceHeight: Int,
    orientation: Int = 1) throws -> Canvas {
    guard (32...20_000).contains(sourceWidth), (32...20_000).contains(sourceHeight),
      sourceWidth * sourceHeight <= 100_000_000, (1...8).contains(orientation) else {
      throw H3CheckpointError.invalid("H3 still reference needs valid source dimensions of at least 32 pixels per axis and EXIF orientation 1–8.")
    }
    let swapsAxes = (5...8).contains(orientation)
    let width = swapsAxes ? sourceHeight : sourceWidth
    let height = swapsAxes ? sourceWidth : sourceHeight
    let aspect = Double(width) / Double(height)
    guard (0.25...4).contains(aspect) else {
      throw H3CheckpointError.invalid("H3 still reference aspect must be between 1:4 and 4:1.")
    }
    var best: Canvas?
    var bestError = Double.infinity
    var bestArea = 0
    for preparedHeight in stride(from: 32, through: min(height, 2048), by: 32) {
      for preparedWidth in stride(from: 32, through: min(width, 2048), by: 32) {
        let area = preparedWidth * preparedHeight
        guard (4096...65_536).contains(area) else { continue }
        let error = abs(log((Double(preparedWidth) / Double(preparedHeight)) / aspect))
        if error < bestError - 1e-12 || (abs(error - bestError) <= 1e-12 && area > bestArea) {
          best = Canvas(width: preparedWidth, height: preparedHeight)
          bestError = error
          bestArea = area
        }
      }
    }
    guard let best else {
      throw H3CheckpointError.invalid("H3 still reference is too small for four visual pads on the 32-pixel grid without enlargement.")
    }
    return best
  }

  public static func validateCanvas(width: Int, height: Int, pixelBudgetPercent: Int? = nil) throws {
    if let pixelBudgetPercent {
      guard (50...400).contains(pixelBudgetPercent), (32...2048).contains(width),
        (32...2048).contains(height), width.isMultiple(of: 32), height.isMultiple(of: 32),
        (4096...(4 * H3Geometry.maximumCanvasPixels)).contains(width * height) else {
        throw H3CheckpointError.invalid("Explicit H3 image reference exceeds its bounded 32-pixel canvas.")
      }
      return
    }
    guard (32...2048).contains(width), (32...2048).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      (4096...65_536).contains(width * height) else {
      throw H3CheckpointError.invalid("H3 still reference needs a 32-pixel grid with 4–64 visual pads, at most 65,536 pixels and 2048 pixels per axis.")
    }
  }

  public struct Loaded {
    public let reference: H3StillReference
    public let sourceSHA256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let sourceOrientation: Int
    public let thumbnailWidth: Int
    public let thumbnailHeight: Int
    public let preparedSHA256: String
    public var preparationPolicy: String {
      reference.pixelBudgetPercent == nil ? H3StillReferenceMedia.preparationPolicy : "owned-output-area-down-only32-v1"
    }
  }

  public static func load(path: String, outputGeometry: H3Geometry? = nil,
    pixelBudgetPercent: Int? = nil) throws -> Loaded {
    guard (outputGeometry == nil) == (pixelBudgetPercent == nil) else {
      throw H3CheckpointError.invalid("Explicit image pixel budget needs its output canvas.")
    }
    if let pixelBudgetPercent, !(50...400).contains(pixelBudgetPercent) {
      throw H3CheckpointError.invalid("H3 image reference pixel budget must be50–400percent.")
    }
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
    let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
    guard (1...8).contains(orientation) else {
      throw H3CheckpointError.invalid("H3 image reference has invalid EXIF orientation.")
    }
    let canvas: Canvas
    if let geometry = outputGeometry, let percent = pixelBudgetPercent {
      let swapsAxes = (5...8).contains(orientation)
      let prepared = try H3ReferenceCanvasPolicy.image(sourceWidth: swapsAxes ? sourceHeight : sourceWidth,
        sourceHeight: swapsAxes ? sourceWidth : sourceHeight, outputWidth: geometry.width,
        outputHeight: geometry.height, percent: percent)
      try validateCanvas(width: prepared.width, height: prepared.height, pixelBudgetPercent: percent)
      canvas = Canvas(width: prepared.width, height: prepared.height)
    } else {
      canvas = try resolveCanvas(sourceWidth: sourceWidth, sourceHeight: sourceHeight, orientation: orientation)
    }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: max(canvas.width, canvas.height),
      kCGImageSourceShouldCache: false,
    ]
    guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0,
      options as CFDictionary), thumbnail.width > 0, thumbnail.height > 0 else {
      throw H3CheckpointError.invalid("Cannot decode a bounded H3 reference thumbnail.")
    }
    let rowBytes = canvas.width * 4
    var rgba = [UInt8](repeating: 0, count: canvas.height * rowBytes)
    let result = rgba.withUnsafeMutableBytes { memory -> Bool in
      guard let context = CGContext(data: memory.baseAddress, width: canvas.width,
        height: canvas.height, bitsPerComponent: 8, bytesPerRow: rowBytes,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
      context.setFillColor(CGColor(gray: 0.5, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: canvas.width, height: canvas.height))
      context.interpolationQuality = .high
      context.draw(thumbnail, in: CGRect(x: 0, y: 0,
        width: canvas.width, height: canvas.height))
      return true
    }
    guard result else {
      throw H3CheckpointError.invalid("Cannot prepare H3 reference RGB pixels.")
    }
    var rgb = Data(count: canvas.width * canvas.height * 3)
    rgb.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
      for pixel in 0..<(canvas.width * canvas.height) {
        target[pixel * 3] = rgba[pixel * 4]
        target[pixel * 3 + 1] = rgba[pixel * 4 + 1]
        target[pixel * 3 + 2] = rgba[pixel * 4 + 2]
      }
    }
    return Loaded(reference: H3StillReference(rgb8: rgb, width: canvas.width,
      height: canvas.height, pixelBudgetPercent: pixelBudgetPercent), sourceSHA256: SHA256.hash(data: bytes)
        .map { String(format: "%02x", $0) }.joined(),
      sourceWidth: sourceWidth, sourceHeight: sourceHeight,
      sourceOrientation: orientation, thumbnailWidth: thumbnail.width,
      thumbnailHeight: thumbnail.height,
      preparedSHA256: SHA256.hash(data: rgb).map { String(format: "%02x", $0) }.joined())
  }
}
