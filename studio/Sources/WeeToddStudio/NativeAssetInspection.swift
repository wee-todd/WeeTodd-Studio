import AVFoundation
import Foundation
import ImageIO
import StudioCore

enum NativeAssetInspection {
  private static let stillExtensions: Set<String> =
    ["png", "jpg", "jpeg", "webp", "tif", "tiff", "heic"]
  private static let avExtensions: Set<String> =
    ["mov", "mp4", "m4v", "mp3", "m4a", "wav", "aif", "aiff", "caf"]

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

  /// AVFoundation loads container/track metadata without decoding movie frames or
  /// routing a standard audio/video import through the Python bridge.
  static func inspectAVMedia(_ url: URL) async throws -> [String: Any]? {
    guard avExtensions.contains(url.pathExtension.lowercased()) else { return nil }
    let source = url.resolvingSymlinksInPath().standardizedFileURL
    let values = try source.resourceValues(forKeys: [.isRegularFileKey])
    guard values.isRegularFile == true else {
      throw StudioError.invalid("Select a regular local audio or movie file.")
    }
    let asset = AVURLAsset(url: source)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    guard !videoTracks.isEmpty || !audioTracks.isEmpty else {
      throw StudioError.invalid("The selected file contains no supported audio or movie stream.")
    }
    let video = videoTracks.first
    let duration = CMTimeGetSeconds(try await asset.load(.duration))
    guard duration.isFinite, duration > 0 else {
      throw StudioError.invalid("The selected audio or movie has no finite duration.")
    }
    var width = 0, height = 0, fps = 0.0
    if let video {
      let size = try await video.load(.naturalSize)
      let rate = Double(try await video.load(.nominalFrameRate))
      guard size.width.isFinite, size.height.isFinite,
        size.width > 0, size.height > 0,
        size.width <= 100_000, size.height <= 100_000,
        size.width * size.height <= 100_000_000,
        rate.isFinite, rate >= 0, rate <= 240 else {
        throw StudioError.invalid("The selected movie has unsupported dimensions or frame rate.")
      }
      width = Int(size.width.rounded())
      height = Int(size.height.rounded())
      fps = rate
    }
    return ["kind": video == nil ? "audio" : "video", "path": source.path,
      "width": width, "height": height, "duration": duration,
      "fps": fps, "hasAudio": !audioTracks.isEmpty]
  }
}
