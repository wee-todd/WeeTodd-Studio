import Foundation

/// Process-local reuse of authorized credentials. No cache is persisted to disk.
/// The lock covers the backing operation as well as the cache, so concurrent reads
/// share a lookup and an in-flight read cannot restore a removed/changed credential.
/// Call potentially interactive operations off the UI executor (BackgroundCredential).
final class SessionCredentialStore {
  private let lock = NSLock()
  private var values: [String: String] = [:]
  private let lookup: (String) throws -> String?
  private let write: (String, String) throws -> Void
  private let delete: (String) throws -> Void

  init(read: @escaping (String) throws -> String?,
       save: @escaping (String, String) throws -> Void,
       remove: @escaping (String) throws -> Void) {
    lookup = read; write = save; delete = remove
  }

  func read(_ reference: String) throws -> String? {
    lock.lock(); defer { lock.unlock() }
    if let value = values[reference] { return value }
    // A missing entry or denied authorization must be retryable.
    if let value = try lookup(reference) {
      values[reference] = value
      return value
    }
    return nil
  }

  func save(_ value: String, reference: String) throws {
    lock.lock(); defer { lock.unlock() }
    values.removeValue(forKey: reference)
    try write(value, reference)
    // Require an authorized read after editing, rather than bypassing the ACL
    // merely because a caller knows (and saved) a value.
  }

  func remove(_ reference: String) throws {
    lock.lock(); defer { lock.unlock() }
    values.removeValue(forKey: reference)
    try delete(reference)
  }

  func clear() {
    lock.lock(); defer { lock.unlock() }
    values.removeAll()
  }
}
