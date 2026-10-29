import Darwin
import Foundation

public enum Drama3ContractVersion { public static let current = 1 }

/// The stable lock file is retained; kernel ownership ends when its process closes the
/// descriptor or exits, so interrupted setup never leaves a permanent busy marker.
public final class Drama3SceneSetupLock {
  private let descriptor: Int32
  public init(model: URL) throws {
    let file = model.appendingPathComponent(".drama3-scene-controls.lock")
    let descriptor = Darwin.open(file.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0o600)
    guard descriptor >= 0 else { throw Drama3Error.invalid("Cannot lock scene-control setup in this model folder.") }
    var status = stat()
    guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
      Darwin.close(descriptor)
      throw Drama3Error.invalid("Scene-control setup requires a regular lock file.")
    }
    guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
      Darwin.close(descriptor)
      throw Drama3Error.invalid("Another scene-control setup owns this model folder.")
    }
    self.descriptor = descriptor
  }
  deinit { _ = flock(descriptor, LOCK_UN); Darwin.close(descriptor) }
}
