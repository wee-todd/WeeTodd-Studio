import Foundation
import ImageIO
import StudioCore

enum NativeAssetInspection {
  private static let stillExtensions: Set<String> =
    ["png", "jpg", "jpeg", "webp", "tif", "tiff", "heic"]

  /// Read only image metadata. Importing a large still must not decode its pixels
  /// or start the Python/FFprobe bridge just to create a linked asset.
  static func inspectStill(_ url: URL) throws -> [String: Any]? {
    guard stillExtensions.contains(url.pathExtension.lowercased()) else { return nil }
    let source = url.resolvingSymlinksInPath().standardizedFileURL
    let values = try source.resourceValues(forKeys: [.isRegularFileKey])
    guard values.isRegularFile == true,
      let image = CGImageSourceCreateWithURL(source as CFURL,
        [kCGImageSourceShouldCache: false] as CFDictionary),
      CGImageSourceGetCount(image) > 0,
      let properties = CGImageSourceCopyPropertiesAtIndex(image, 0,
        [kCGImageSourceShouldCache: false] as CFDictionary) as? [CFString: Any],
      let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
      let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
      width > 0, height > 0, width <= 100_000, height <= 100_000,
      Int64(width) * Int64(height) <= 100_000_000 else {
      throw StudioError.invalid("The selected image has no supported dimensions or exceeds 100 megapixels.")
    }
    return ["kind": "image", "path": source.path, "width": width,
      "height": height, "duration": 0.0, "fps": 0.0, "hasAudio": false]
  }
}
