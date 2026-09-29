import CryptoKit
import Darwin
import Foundation
import LTX25Engine

/// The author adapter has no task metadata. A matching name or tensor shape
/// cannot distinguish it from an ordinary LoRA, so verify the pinned release.
public enum MLXRippleAdapterIdentity {
  public static let sha256 = "bde543610144e5bba78782fb0ac18daa663fc087c2212244b8116deb1925ee2a"
  public static let bytes: Int64 = 654_443_392

  private struct Signature: Hashable {
    let path: String
    let device: UInt64
    let inode: UInt64
    let size: Int64
    let modifiedSeconds: Int64
    let modifiedNanoseconds: Int64
    let changedSeconds: Int64
    let changedNanoseconds: Int64

    init(path: String, status: stat) {
      self.path = path
      device = UInt64(status.st_dev)
      inode = status.st_ino
      size = status.st_size
      modifiedSeconds = Int64(status.st_mtimespec.tv_sec)
      modifiedNanoseconds = Int64(status.st_mtimespec.tv_nsec)
      changedSeconds = Int64(status.st_ctimespec.tv_sec)
      changedNanoseconds = Int64(status.st_ctimespec.tv_nsec)
    }
  }

  private final class Cache: @unchecked Sendable {
    let lock = NSLock()
    var verified: Set<Signature> = []
  }
  private static let cache = Cache()

  public static func verify(_ location: URL) throws {
    guard location.isFileURL else { throw LTXError.invalid("Ripple adapter must be a local file.") }
    let path = location.resolvingSymlinksInPath().standardizedFileURL.path
    let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard descriptor >= 0 else { throw LTXError.invalid("Cannot open Ripple adapter.") }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? handle.close() }
    var before = stat()
    guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
      before.st_size == bytes else {
      throw LTXError.invalid("Ripple adapter differs from the pinned author release.")
    }
    let signature = Signature(path: path, status: before)
    cache.lock.lock()
    let cached = cache.verified.contains(signature)
    cache.lock.unlock()
    if cached { return }
    var digest = SHA256()
    while let block = try handle.read(upToCount: 8 * 1024 * 1024), !block.isEmpty {
      try Task.checkCancellation()
      digest.update(data: block)
    }
    var after = stat()
    guard fstat(descriptor, &after) == 0,
      Signature(path: path, status: after) == signature,
      digest.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
      throw LTXError.invalid("Ripple adapter hash or file identity does not match the pinned author release.")
    }
    cache.lock.lock()
    cache.verified.insert(signature)
    cache.lock.unlock()
  }
}
