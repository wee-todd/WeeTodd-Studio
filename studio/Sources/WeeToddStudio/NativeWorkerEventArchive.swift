import Foundation

/// Confined to the worker pipe's drain queue. A write failure stops archiving,
/// never draining, so diagnostics cannot deadlock the inference subprocess.
final class NativeWorkerEventArchive: @unchecked Sendable {
  let url: URL
  private let handle: FileHandle
  private let maximumBytes: Int
  private let writer: (FileHandle, Data) throws -> Void
  private(set) var bytesWritten = 0
  private(set) var truncated = false
  private(set) var errorMessage: String?
  private var finished = false

  init(output: URL, jobID: String, maximumBytes: Int = 64 * 1024 * 1024,
    writer: @escaping (FileHandle, Data) throws -> Void = { try $0.write(contentsOf: $1) }) throws {
    let directory = output.deletingLastPathComponent().appendingPathComponent("NativeWorkerLogs")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let identifier = UUID(uuidString: jobID)?.uuidString ?? UUID().uuidString
    url = directory.appendingPathComponent(identifier + "-" + UUID().uuidString + ".events.jsonl")
    try Data().write(to: url, options: .withoutOverwriting)
    handle = try FileHandle(forWritingTo: url)
    self.maximumBytes = max(0, maximumBytes); self.writer = writer
  }
  func append(_ data: Data) {
    guard !finished, !truncated else { return }
    let available = maximumBytes - bytesWritten
    let retained = Data(data.prefix(available))
    do { try writer(handle, retained); bytesWritten += retained.count }
    catch { truncated = true; errorMessage = "Native worker event archive write failed: " + error.localizedDescription }
    if !truncated, data.count > available {
      truncated = true; errorMessage = "Native worker event archive reached its \(maximumBytes)-byte limit."
    }
  }
  func finish() {
    guard !finished else { return }; finished = true
    do { try handle.synchronize(); try handle.close() }
    catch { truncated = true; errorMessage = "Native worker event archive close failed: " + error.localizedDescription }
    struct Metadata: Encodable { var bytesWritten: Int; var maximumBytes: Int; var truncated: Bool; var error: String? }
    do {
      let metadata = Metadata(bytesWritten: bytesWritten, maximumBytes: maximumBytes, truncated: truncated, error: errorMessage)
      try JSONEncoder().encode(metadata).write(to: url.appendingPathExtension("metadata.json"), options: .atomic)
    } catch { truncated = true; errorMessage = "Native worker event archive metadata failed: " + error.localizedDescription }
  }
  deinit { try? handle.close() }
}
