import Foundation

/// Resume checkpoints retain analysis and human review evidence beyond definition imports.
public enum WorkflowCheckpoint {
  public static let maximumBytes = 16 * 1024 * 1024

  public static func read<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
    guard try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
      throw StudioError.invalid("Workflow checkpoint cannot be a symlink.")
    }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: maximumBytes + 1) ?? Data()
    guard data.count <= maximumBytes else {
      throw StudioError.invalid("Checkpoint exceeds the 16 MiB inspection limit.")
    }
    return try JSONDecoder().decode(type, from: data)
  }
}
