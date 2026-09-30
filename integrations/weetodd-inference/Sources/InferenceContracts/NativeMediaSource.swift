import CryptoKit
import Darwin
import Foundation

/// Frozen local media identity shared by native video tasks. Reads in bounded
/// chunks and refuses symlinks or a source changed while it is being checked.
public struct NativeMediaSource: Sendable {
  public let path: String
  public let sha256: String

  public init(path: String, sha256: String) throws {
    guard path.hasPrefix("/"), path.utf8.count <= 4096, !path.utf8.contains(0),
      sha256.utf8.count == 64,
      sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw ContractError.invalid("Media source needs an absolute local path and a lowercase SHA-256 digest.")
    }
    self.path = path
    self.sha256 = sha256
  }

  public func verify() throws {
    let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard descriptor >= 0 else {
      throw ContractError.invalid("Cannot open the frozen media source as a regular file.")
    }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? handle.close() }
    var before = stat()
    guard fstat(descriptor, &before) == 0,
      before.st_mode & S_IFMT == S_IFREG, before.st_size > 0 else {
      throw ContractError.invalid("Frozen media source must be a nonempty regular file.")
    }
    var digest = SHA256()
    while let bytes = try handle.read(upToCount: 8 * 1024 * 1024), !bytes.isEmpty {
      try Task.checkCancellation()
      digest.update(data: bytes)
    }
    var after = stat()
    let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
    guard fstat(descriptor, &after) == 0,
      before.st_dev == after.st_dev, before.st_ino == after.st_ino,
      before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
      actual == sha256 else {
      throw ContractError.invalid("Media source changed since the take was prepared. Prepare it again.")
    }
  }
}
