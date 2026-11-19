import Foundation
import CryptoKit
import Darwin

/// Metadata identity for discovery. Workers still validate tensors and pinned content before execution.
public enum GenerationLibraryFileIdentity {
  private final class ReleaseCache: @unchecked Sendable { let lock = NSLock(); var verified = Set<String>() }
  private static let releaseCache = ReleaseCache()
  public static func verifyRelease(_ file: URL, bytes: Int64, sha256: String) throws {
    guard try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]).isRegularFile == true,
      (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) == bytes else { throw StudioError.invalid("The adapter differs from the supported author release.") }
    let before = signature([file.path]), key = sha256 + "|" + before
    releaseCache.lock.lock(); let cached = releaseCache.verified.contains(key); releaseCache.lock.unlock()
    if cached { return }
    let captured = try FlashVSRFile.capture(file)
    guard captured.bytes == bytes, captured.sha256 == sha256, signature([file.path]) == before else { throw StudioError.invalid("The adapter hash or file identity differs from the supported author release.") }
    releaseCache.lock.lock(); defer { releaseCache.lock.unlock() }
    if releaseCache.verified.count >= 128 { releaseCache.verified.removeAll() }
    releaseCache.verified.insert(key)
  }
  /// Bounded component inventory, never a recursive search of a model library.
  public static func componentFiles(_ directory: URL) -> [String] {
    let supported: Set<String> = ["json", "safetensors", "gguf", "ckpt", "pth"]
    return [directory, directory.appendingPathComponent("pages"), directory.appendingPathComponent("weights")].flatMap { folder in
      ((try? FileManager.default.contentsOfDirectory(at: folder,
      includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? [])
      .filter { supported.contains($0.pathExtension) && (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
      .prefix(1024).map(\.path).sorted()
    }
  }
  public static func signature(_ paths: [String]) -> String {
    let values = Set(paths.map { value -> String in
      let url = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath()
      var status = stat()
      guard lstat(url.path, &status) == 0 else { return url.path + "|missing" }
      return url.path + "|" + [Int64(status.st_dev), Int64(bitPattern: UInt64(status.st_ino)), status.st_size,
        Int64(status.st_mtimespec.tv_sec), Int64(status.st_mtimespec.tv_nsec),
        Int64(status.st_ctimespec.tv_sec), Int64(status.st_ctimespec.tv_nsec)].map(String.init).joined(separator: "|")
    }).sorted().joined(separator: "\n")
    return SHA256.hash(data: Data(values.utf8)).map { String(format: "%02x", $0) }.joined()
  }
  public static func id(_ alias: String) -> String {
    SHA256.hash(data: Data(alias.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
  }
}
