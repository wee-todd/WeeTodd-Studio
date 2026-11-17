import Foundation
import MLX
import CoreGraphics
import ImageIO

struct MOSSPreparedImage {
  let pixels: MLXArray
  let sourceWidth, sourceHeight: Int
  let rgbSHA256: String
}

public struct MOSSReferenceImage: Codable, Equatable, Sendable {
  public var path, sha256: String
  public var noiseLevel: Float
  public init(path: String, sha256: String, noiseLevel: Float = 0.08) {
    self.path = path; self.sha256 = sha256; self.noiseLevel = noiseLevel
  }
  public func validate() throws {
    guard path.hasPrefix("/"), !path.utf8.contains(0), sha256.count == 64,
      sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }),
      noiseLevel.isFinite, noiseLevel > 0, noiseLevel < 1 else {
      throw MOSSError("Source reconstruction needs a local image, its SHA-256, and a finite noise level strictly between zero and one.")
    }
  }
  func prepare(width: Int, height: Int) throws -> MOSSPreparedImage {
    try validate()
    let url = URL(fileURLWithPath: path)
    guard try MOSSLayoutImage.sha256(url) == sha256 else { throw MOSSError("Reconstruction source changed or is missing. Analyze the image again.") }
    guard let decoder = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
      let info = CGImageSourceCopyPropertiesAtIndex(decoder, 0, nil) as? [CFString: Any],
      let rawWidth = info[kCGImagePropertyPixelWidth] as? Int, let rawHeight = info[kCGImagePropertyPixelHeight] as? Int,
      rawWidth > 0, rawHeight > 0, rawWidth <= 131_072, rawHeight <= 131_072,
      Double(rawWidth) * Double(rawHeight) <= 128_000_000 else { throw MOSSError("Reconstruction source is not a supported bounded image.") }
    let orientation = info[kCGImagePropertyOrientation] as? Int ?? 1
    let sourceWidth = (5...8).contains(orientation) ? rawHeight : rawWidth
    let sourceHeight = (5...8).contains(orientation) ? rawWidth : rawHeight
    // ImageIO applies EXIF orientation before a whole-image resample; no crop or overlays.
    guard let image = CGImageSourceCreateThumbnailAtIndex(decoder, 0, [
      kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: max(width, height), kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
      let space = CGColorSpace(name: CGColorSpace.sRGB),
      let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
      throw MOSSError("Unable to prepare the reconstruction source.")
    }
    context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let pointer = context.data?.assumingMemoryBound(to: UInt8.self) else { throw MOSSError("Unable to read reconstruction pixels.") }
    var rgb = [UInt8](); rgb.reserveCapacity(width * height * 3)
    for i in 0..<(width * height) { rgb.append(contentsOf: [pointer[i * 4], pointer[i * 4 + 1], pointer[i * 4 + 2]]) }
    guard try MOSSLayoutImage.sha256(url) == sha256 else { throw MOSSError("Reconstruction source changed during preparation.") }
    return .init(pixels: (MLXArray(rgb).asType(.float32) / Float(127.5) - 1).reshaped(1, height, width, 3),
      sourceWidth: sourceWidth, sourceHeight: sourceHeight, rgbSHA256: MOSSImagePreflight.digest(Data(rgb)))
  }
  static func initialize(source: MLXArray, noise: MLXArray, sigma: Float) throws -> MLXArray {
    guard sigma.isFinite, sigma > 0, sigma < 1, source.shape == noise.shape,
      isFinite(source).all().item(Bool.self), isFinite(noise).all().item(Bool.self) else { throw MOSSError("Invalid source latent initialization.") }
    return source.asType(.float32) * (1 - sigma) + noise.asType(.float32) * sigma
  }
  static func sigmas(_ schedule: [Float], noiseLevel: Float) throws -> [Float] {
    guard noiseLevel.isFinite, noiseLevel > 0, noiseLevel < 1, schedule.count >= 2,
      schedule.first == 1, schedule.last == 0, schedule.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      zip(schedule, schedule.dropFirst()).allSatisfy({ $0 > $1 }) else { throw MOSSError("Invalid source reconstruction noise schedule.") }
    let scaled = schedule.map { $0 * noiseLevel }
    guard zip(scaled, scaled.dropFirst()).allSatisfy({ $0 > $1 }) else { throw MOSSError("Source noise schedule collapsed; use a larger change strength.") }
    return scaled
  }
}
