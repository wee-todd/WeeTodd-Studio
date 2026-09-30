import Foundation
import XCTest
@testable import H3MLX

final class H3AudioReferenceMediaTests: XCTestCase {
  func testAudioDecodesBoundedStereoSamples() throws {
    let path = ProcessInfo.processInfo.environment["WEETODD_FFMPEG"]
      ?? "/opt/homebrew/bin/ffmpeg"
    guard FileManager.default.isExecutableFile(atPath: path) else {
      throw XCTSkip("Set WEETODD_FFMPEG for local audio decode tests.")
    }
    let ffmpeg = URL(fileURLWithPath: path)
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = directory.appendingPathComponent("tone.wav")
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=44100", "-t", "1",
      "-ac", "1", source.path]
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    let loaded = try H3AudioReferenceMedia.load(path: source.path,
      ffmpeg: ffmpeg)
    XCTAssertEqual(loaded.frames, 32_000)
    XCTAssertEqual(loaded.samples.count, 64_000)
    XCTAssertTrue(loaded.samples.allSatisfy(\.isFinite))
    XCTAssertEqual(loaded.samples[8_000], loaded.samples[40_000],
      accuracy: 1e-6, "Mono source duplicates into stereo")
    let interval = try H3AudioReferenceMedia.load(path: source.path,
      ffmpeg: ffmpeg, startSeconds: 0.25, durationSeconds: 0.5)
    XCTAssertEqual(interval.frames, 16_000)
    XCTAssertEqual(interval.samples.count, 32_000)
    XCTAssertThrowsError(try H3AudioReferenceMedia.load(path: source.path,
      ffmpeg: ffmpeg, startSeconds: 0.75, durationSeconds: 0.5),
      "A driver interval extending past the source must fail")
  }
}
