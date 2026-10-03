import Foundation
import XCTest
@testable import StudioCore

final class NativeContinuityFrameTests: XCTestCase {
  func testChangedSourceSnapshotFailsBeforeDecodeOrPublication() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let movie = root.appendingPathComponent("source.mp4")
    let executable = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].first {
      FileManager.default.isExecutableFile(atPath: $0)
    }
    guard let executable else { throw XCTSkip("Direct FFmpeg required for the bounded media-only fixture") }
    let process = Process(); process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = ["-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
      "color=c=red:s=64x64:r=1:d=3", "-frames:v", "3", "-c:v", "libx264", "-crf", "12", movie.path]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
    var source = Clip(engine: .movie); source.sourcePath = movie.path; source.duration = 1
    var target = Clip(engine: .ltx25); target.continuity = ClipContinuity(mode: "frame")
    var project = StudioProject(); project.clips = [source, target]
    let frozenSource = try NativeLTXFrameSource(project: project, clip: target)
    let file = try FileHandle(forWritingTo: movie)
    try file.seekToEnd(); try file.write(contentsOf: Data("source-mutation".utf8)); try file.close()
    let output = root.appendingPathComponent("must-not-publish")
    do {
      _ = try await NativeContinuityFrame.freeze(source: frozenSource, engine: .ltx25, destination: output)
      XCTFail("Source mutation must fail before media work")
    } catch { XCTAssertTrue(error.localizedDescription.contains("continuity source changed")) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
  }
}
