import CryptoKit
import Foundation

/// A media-only snapshot of the last visible frame, independent of model setup.
public enum NativeContinuityFrame {
  public static func freeze(project: StudioProject, clip: Clip, destination: URL) async throws -> [String: Any] {
    guard [.h3, .ltx25].contains(clip.engine), clip.continuityMode == "frame" else {
      throw StudioError.invalid("Select a native H3 or LTX 2.5 clip using Match previous frame.")
    }
    return try await freeze(source: NativeLTXFrameSource(project: project, clip: clip),
      engine: clip.engine, destination: destination)
  }

  static func freeze(source: NativeLTXFrameSource, engine: Engine, destination: URL) async throws -> [String: Any] {
    try Task.checkCancellation(); try source.verify()
    let parent = destination.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw StudioError.invalid("The frozen-frame destination already exists.")
    }
    let staging = parent.appendingPathComponent(".frame-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    var published = false, complete = false
    defer {
      try? FileManager.default.removeItem(at: staging)
      if published && !complete { try? FileManager.default.removeItem(at: destination) }
    }
    let sourceSHA = try digest(source.url)
    try source.verify()
    let image = staging.appendingPathComponent("first-frame.png")
    let time = try await source.extract(to: image)
    let frozenSHA = try digest(image)
    try source.verify(); try Task.checkCancellation()
    var report = source.report
    report["engine"] = engine.rawValue
    report["sourceFrameTime"] = time
    report["sourceSHA256"] = sourceSHA
    report["sourceFrozenSHA256"] = frozenSHA
    report["path"] = destination.appendingPathComponent(image.lastPathComponent).path
    report["pythonExecuted"] = false
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: staging.appendingPathComponent("continuity.json"), options: .withoutOverwriting)
    try source.verify(); try Task.checkCancellation()
    try FileManager.default.moveItem(at: staging, to: destination)
    published = true
    try source.verify(); try Task.checkCancellation()
    complete = true
    return report
  }

  private static func digest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let bytes = try file.read(upToCount: 64 * 1024), !bytes.isEmpty {
      try Task.checkCancellation(); hash.update(data: bytes)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
