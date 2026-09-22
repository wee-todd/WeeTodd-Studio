import CryptoKit
import Foundation

/// Storage budget covers retained prepared pixels and encoded feature tensors together.
/// Model parameters are deliberately excluded; this is not a total-process memory limit.
struct LocalVisionCache<Value> {
  private struct Entry { let value: Value; let bytes: Int; var touched: UInt64 }
  private var entries: [String: Entry] = [:]
  private var clock: UInt64 = 0
  let entryLimit: Int
  let byteLimit: Int
  private(set) var retainedBytes = 0
  var count: Int { entries.count }

  init(entryLimit: Int = 4, byteLimit: Int = 128 * 1024 * 1024) {
    precondition(entryLimit > 0 && byteLimit > 0)
    self.entryLimit = entryLimit; self.byteLimit = byteLimit
  }
  mutating func value(for key: String) -> Value? {
    guard var entry = entries[key] else { return nil }
    clock += 1; entry.touched = clock; entries[key] = entry
    return entry.value
  }
  mutating func insert(_ value: Value, key: String, bytes: Int) {
    guard bytes >= 0, bytes <= byteLimit else { return }
    if let previous = entries.removeValue(forKey: key) { retainedBytes -= previous.bytes }
    while entries.count >= entryLimit || retainedBytes + bytes > byteLimit {
      guard let oldest = entries.min(by: { $0.value.touched < $1.value.touched }) else { break }
      retainedBytes -= oldest.value.bytes; entries.removeValue(forKey: oldest.key)
    }
    clock += 1; entries[key] = Entry(value: value, bytes: bytes, touched: clock)
    retainedBytes += bytes
  }
  mutating func removeAll() { entries.removeAll(); retainedBytes = 0; clock = 0 }
}

enum LocalVisionIdentity {
  /// Hash the bounded source file, not an mtime/path-only proxy. Image bytes include
  /// EXIF orientation and encoded dimensions; the preparation version covers resizing,
  /// color normalization, grids and precision. Labels and order are intentional keys.
  static func key(images: [LocalPromptImage], model: String, cancelled: () -> Bool) throws -> String {
    var records: [[String: String]] = []
    for image in images {
      guard !cancelled() else { throw LocalTextError("text_cancelled") }
      let url = URL(fileURLWithPath: image.path).resolvingSymlinksInPath()
      guard let attributes = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        attributes.isRegularFile == true, let size = attributes.fileSize, size > 0,
        size <= 64 * 1024 * 1024, let file = try? FileHandle(forReadingFrom: url) else {
        throw LocalTextError("vision_image_unavailable")
      }
      defer { try? file.close() }
      var hash = SHA256(); var count = 0
      while let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty {
        guard !cancelled() else { throw LocalTextError("text_cancelled") }
        count += data.count
        guard count <= 64 * 1024 * 1024 else { throw LocalTextError("vision_image_unavailable") }
        hash.update(data: data)
      }
      guard count == size else { throw LocalTextError("vision_source_changed") }
      records.append(["path": url.path, "label": image.label,
        "sha256": hash.finalize().map { String(format: "%02x", $0) }.joined()])
    }
    let value: [String: Any] = ["model": model, "images": records,
      "preparation": "qwen35-thumbnail512-exif-srgb-white-round32-normalize05-f32-patches-f16-features-v1"]
    return SHA256.hash(data: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
      .map { String(format: "%02x", $0) }.joined()
  }
}
