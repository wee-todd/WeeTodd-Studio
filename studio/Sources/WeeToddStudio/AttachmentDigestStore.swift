import CryptoKit
import Darwin
import Foundation
import Combine

/// Cheap identity for a local file. Nanosecond change time also detects atomic replacement
/// and same-size rewrites whose modification date has been restored by another application.
struct MediaFileRevision: Hashable, Sendable {
  let path: String
  let identity: String
  init(_ path: String) {
    self.path = path
    var info = stat()
    if stat(path, &info) == 0 {
      identity = "\(info.st_dev):\(info.st_ino):\(info.st_size):\(info.st_mtimespec.tv_sec):\(info.st_mtimespec.tv_nsec):\(info.st_ctimespec.tv_sec):\(info.st_ctimespec.tv_nsec)"
    } else { identity = "missing" }
  }
}

private actor DigestReader {
  let read: @Sendable (String) throws -> String
  init(_ read: @escaping @Sendable (String) throws -> String) { self.read = read }
  func digest(_ revision: MediaFileRevision) throws -> String {
    let hash = try read(revision.path)
    guard MediaFileRevision(revision.path) == revision else {
      throw CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey:
        "\(URL(fileURLWithPath: revision.path).lastPathComponent) changed while being read. Prepare again."])
    }
    return hash
  }
}

/// UI status only reads cached digests. Preparation awaits the shared, streamed reader;
/// submission can force a fresh read without retaining the attachment bytes in memory.
@MainActor final class AttachmentDigestStore: ObservableObject {
  private struct Entry { let revision: MediaFileRevision; let result: Result<String, Error> }
  private struct Pending { let revision: MediaFileRevision; let id: UUID; let task: Task<String, Error> }
  private var entries: [String: Entry] = [:]
  private var pending: [String: Pending] = [:]
  private var order: [String] = []
  private let reader: DigestReader
  init(reader: @escaping @Sendable (String) throws -> String = { try AttachmentDigestStore.hashFile($0) }) {
    self.reader = DigestReader(reader)
  }
  nonisolated static func hashFile(_ path: String) throws -> String {
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? handle.close() }
    var hash = SHA256()
    while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty { hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  func fingerprint(_ path: String) -> String {
    let revision = MediaFileRevision(path)
    if let entry = entries[path], entry.revision == revision {
      switch entry.result {
      case .success(let hash): return "\(path)|\(hash)"
      case .failure: return "\(path)|\(revision.identity)|unreadable"
      }
    }
    if pending[path]?.revision != revision {
      // Defer publication until outside SwiftUI's current body evaluation.
      Task { try? await resolve([path]) }
    }
    return "\(path)|\(revision.identity)|pending"
  }
  func resolve(_ paths: [String], force: Bool = false) async throws {
    for path in Set(paths).sorted() {
      let revision = MediaFileRevision(path)
      if !force, let entry = entries[path], entry.revision == revision,
        case .success = entry.result {
        continue
      }
      let work: Pending
      if let existing = pending[path], existing.revision == revision { work = existing }
      else {
        let worker = reader
        work = Pending(revision: revision, id: UUID(), task: Task { try await worker.digest(revision) })
        pending[path] = work
      }
      let result = await work.task.result
      if pending[path]?.id == work.id {
        pending[path] = nil
        if MediaFileRevision(path) == revision {
          entries[path] = Entry(revision: revision, result: result)
          order.removeAll { $0 == path }; order.append(path)
          while order.count > 512 { entries[order.removeFirst()] = nil }
          objectWillChange.send()
        }
      }
      _ = try result.get()
      guard MediaFileRevision(path) == revision else {
        throw CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey:
          "\(URL(fileURLWithPath: path).lastPathComponent) changed while being read. Prepare again."])
      }
    }
  }
}
