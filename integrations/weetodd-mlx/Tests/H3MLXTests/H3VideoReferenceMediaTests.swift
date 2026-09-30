import Foundation
import XCTest
@testable import H3MLX

final class H3VideoReferenceMediaTests: XCTestCase {
  private func ffmpeg() throws -> URL {
    let path = ProcessInfo.processInfo.environment["WEETODD_FFMPEG"]
      ?? "/opt/homebrew/bin/ffmpeg"
    guard FileManager.default.isExecutableFile(atPath: path) else {
      throw XCTSkip("Set WEETODD_FFMPEG for local video reference decode tests.")
    }
    return URL(fileURLWithPath: path)
  }

  func testSilentMovieDecodesBoundedAlignedReference() throws {
    let executable = try ffmpeg()
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("silent.mp4")
    let process = Process()
    process.executableURL = executable
    process.arguments = ["-v", "error", "-f", "lavfi", "-i",
      "testsrc2=size=96x64:rate=30", "-t", "1", "-an",
      "-c:v", "mpeg4", movie.path]
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    let loaded = try H3VideoReferenceMedia.load(path: movie.path,
      ffmpeg: executable)
    XCTAssertEqual(loaded.decodedFrames, 24)
    XCTAssertEqual(loaded.reference.frameCount, 22)
    XCTAssertEqual(loaded.reference.width, 256)
    XCTAssertEqual(loaded.reference.height, 256)
    XCTAssertEqual(loaded.reference.rgb8.count, 22 * 256 * 256 * 3)
    XCTAssertNil(loaded.reference.audio)
  }

  func testMovieSoundtrackIsBoundedToPreparedVideoInterval() throws {
    let executable = try ffmpeg()
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("soundtrack.mp4")
    let process = Process()
    process.executableURL = executable
    process.arguments = ["-v", "error", "-f", "lavfi", "-i",
      "testsrc2=size=96x64:rate=24", "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=32000", "-t", "1",
      "-c:v", "mpeg4", "-c:a", "aac", movie.path]
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    let loaded = try H3VideoReferenceMedia.load(path: movie.path,
      ffmpeg: executable)
    XCTAssertEqual(loaded.reference.frameCount, 22)
    let soundtrack = try XCTUnwrap(loaded.reference.audio)
    XCTAssertGreaterThan(soundtrack.frames, 29_000)
    XCTAssertLessThanOrEqual(soundtrack.frames, 29_334)
    XCTAssertEqual(soundtrack.samples.count, 2 * soundtrack.frames)
  }
}
