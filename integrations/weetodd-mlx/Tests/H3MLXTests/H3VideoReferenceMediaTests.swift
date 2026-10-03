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

  func testExtensionSeamUsesUnpaddedTrueFinalSourceFrame() throws {
    let executable = try ffmpeg()
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let raw = directory.appendingPathComponent("source.rgb")
    let frameBytes = 256 * 160 * 3
    var bytes = Data(count: 21 * frameBytes)
    bytes.withUnsafeMutableBytes { (pixels: UnsafeMutableRawBufferPointer) in
      for frame in 0..<21 {
        for pixel in 0..<(256 * 160) {
          let start = frame * frameBytes + pixel * 3
          pixels[start] = frame == 20 ? 0 : 255
          pixels[start + 2] = frame == 20 ? 255 : 0
        }
      }
    }
    try bytes.write(to: raw)
    let movie = directory.appendingPathComponent("source.mp4")
    let process = Process(); process.executableURL = executable
    process.arguments = ["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
      "-s", "256x160", "-r", "24", "-i", raw.path, "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=32000", "-frames:v", "21", "-t", "0.875",
      "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-c:a", "aac", movie.path]
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    let loaded = try H3VideoReferenceMedia.load(path: movie.path, ffmpeg: executable,
      retainCompleteAudio: true, preserveTail: true)
    XCTAssertEqual(loaded.decodedFrames, 21)
    XCTAssertEqual(loaded.reference.frameCount, 22)
    XCTAssertEqual(loaded.reference.width, 256)
    XCTAssertEqual(loaded.reference.height, 256)
    XCTAssertEqual(loaded.reference.rgb8.count, 22 * 256 * 256 * 3)
    XCTAssertEqual(loaded.reference.rgb8.prefix(256 * 256 * 3).prefix(3), Data([128, 128, 128]))
    XCTAssertEqual(loaded.lastFrame.width, 256)
    XCTAssertEqual(loaded.lastFrame.height, 160)
    XCTAssertEqual(loaded.lastFrame.rgb8.count, frameBytes)
    // The true last decoded frame is blue. The temporal padding clone must not
    // change its timestamp, and video-analysis gray padding must not be pixels.
    let last = Array(loaded.lastFrame.rgb8)
    for pixel in [0, 255, 256 * 159, 256 * 160 - 1] {
      XCTAssertLessThan(Int(last[pixel * 3]), 5)
      XCTAssertLessThan(Int(last[pixel * 3 + 1]), 5)
      XCTAssertGreaterThan(Int(last[pixel * 3 + 2]), 250)
    }
    XCTAssertNotNil(loaded.reference.audio)
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
    let extensionSource = try H3VideoReferenceMedia.load(path: movie.path,
      ffmpeg: executable, retainCompleteAudio: true, preserveTail: true)
    XCTAssertEqual(extensionSource.decodedFrames, 24)
    XCTAssertEqual(extensionSource.reference.frameCount, 39)
    let lastDecodedSquare = extensionSource.reference.rgb8.subdata(in:
      23 * 256 * 256 * 3..<24 * 256 * 256 * 3)
    XCTAssertEqual(extensionSource.reference.rgb8.suffix(256 * 256 * 3), lastDecodedSquare)
    XCTAssertEqual(extensionSource.lastFrame.width, 96)
    XCTAssertEqual(extensionSource.lastFrame.height, 64)
    XCTAssertGreaterThan(extensionSource.reference.audio?.frames ?? 0, 31_000)
  }
}
