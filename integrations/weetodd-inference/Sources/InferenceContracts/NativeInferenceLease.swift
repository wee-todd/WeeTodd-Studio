import Darwin
import Foundation

/// Coordinates weighted Swift workers with the Python/MLX host's existing
/// Runtime/native-inference.lock. Acquire before loading any model weights.
public final class NativeInferenceLease {
  private var descriptor: Int32

  private init(descriptor: Int32) { self.descriptor = descriptor }

  public static var defaultURL: URL {
    let root = ProcessInfo.processInfo.environment["WEETODD_STUDIO_DATA"]
      .flatMap { $0.isEmpty ? nil : $0 }
      ?? FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/WeeTodd Studio").path
    return URL(fileURLWithPath: root, isDirectory: true)
      .appendingPathComponent("Runtime/native-inference.lock")
  }

  public static func tryAcquire(at path: URL = defaultURL) throws -> NativeInferenceLease? {
    guard path.isFileURL else {
      throw NSError(domain: "NativeInferenceLease", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Inference lock must be a local file."])
    }
    try FileManager.default.createDirectory(
      at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    let fd = Darwin.open(path.path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
      mode_t(0o600))
    guard fd >= 0 else {
      throw NSError(domain: "NativeInferenceLease", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "Cannot open the native inference lock."])
    }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
      Darwin.close(fd)
      throw NSError(domain: "NativeInferenceLease", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Inference lock is not a regular file."])
    }
    if flock(fd, LOCK_EX | LOCK_NB) == 0 {
      return NativeInferenceLease(descriptor: fd)
    }
    let issue = errno
    Darwin.close(fd)
    if issue == EWOULDBLOCK { return nil }
    throw NSError(domain: "NativeInferenceLease", code: 4,
      userInfo: [NSLocalizedDescriptionKey: "Cannot acquire the native inference lock."])
  }

  public static func acquire(at path: URL = defaultURL,
    onWait: () -> Void = {}) throws -> NativeInferenceLease {
    var reported = false
    while true {
      try Task.checkCancellation()
      if let lease = try tryAcquire(at: path) { return lease }
      if !reported { onWait(); reported = true }
      usleep(100_000)
    }
  }

  public func release() {
    if descriptor >= 0 {
      flock(descriptor, LOCK_UN)
      Darwin.close(descriptor)
      descriptor = -1
    }
  }

  deinit { release() }
}
