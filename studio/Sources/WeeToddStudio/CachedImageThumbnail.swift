import ImageIO
import SwiftUI

/// ImageIO downsamples before decoding. Actor isolation keeps file reads and decoding
/// off the main actor; the LRU budget counts decoded pixel storage, not compressed bytes.
actor ImageThumbnailCache {
  static let shared = ImageThumbnailCache(byteLimit: 64 * 1024 * 1024)
  private struct Key: Hashable { let revision: MediaFileRevision; let size: Int }
  private var images: [Key: CGImage] = [:]
  private var order: [Key] = []
  private let byteLimit: Int
  private(set) var residentBytes = 0
  init(byteLimit: Int) { self.byteLimit = max(0, byteLimit) }
  func image(path: String, maximumPixelSize: Int) -> CGImage? {
    let key = Key(revision: MediaFileRevision(path), size: max(1, maximumPixelSize))
    if let image = images[key] {
      order.removeAll { $0 == key }; order.append(key); return image
    }
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL,
      [kCGImageSourceShouldCache: false] as CFDictionary),
      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: key.size,
        kCGImageSourceShouldCacheImmediately: true
      ] as CFDictionary) else { return nil }
    let cost = image.bytesPerRow * image.height
    guard cost <= byteLimit else { return image }
    while residentBytes + cost > byteLimit, !order.isEmpty {
      if let removed = images.removeValue(forKey: order.removeFirst()) {
        residentBytes -= removed.bytesPerRow * removed.height
      }
    }
    images[key] = image; order.append(key); residentBytes += cost
    return image
  }
}

struct CachedImageThumbnail: View {
  let path: String
  let maximumPixelSize: Int
  var contentMode: ContentMode = .fit
  @State private var thumbnail: CGImage?
  var body: some View {
    Group {
      if let thumbnail {
        Image(decorative: thumbnail, scale: 1).resizable().aspectRatio(contentMode: contentMode)
      } else { Image(systemName: "photo").foregroundStyle(.secondary) }
    }
    .task(id: "\(MediaFileRevision(path))|\(maximumPixelSize)") {
      thumbnail = nil
      let result = await ImageThumbnailCache.shared.image(path: path, maximumPixelSize: maximumPixelSize)
      guard !Task.isCancelled else { return }
      thumbnail = result
    }
  }
}
