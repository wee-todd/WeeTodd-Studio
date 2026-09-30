import CryptoKit
import Foundation
import XCTest
@testable import LTX25MLX
import LTX25Engine

final class MLXSourceMovieIntervalTests: XCTestCase {
  private func run(_ executable: URL, _ args: [String]) throws {
    let process = Process()
    process.executableURL = executable
    process.arguments = args
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0, "FFmpeg fixture failed")
  }

  func testExtractsExactCausalTailAndAudioWithoutLoadingModel() async throws {
    let ffmpeg = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw XCTSkip("FFmpeg is needed for the media fixture")
    }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("source.mp4")
    try run(ffmpeg, ["-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
      "testsrc2=size=768x448:rate=24", "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=48000", "-t", "5.041666667", "-frames:v", "121",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source.path])
    let digest = try SHA256.hash(data: Data(contentsOf: source))
      .map { String(format: "%02x", $0) }.joined()
    let window = try LTX25ExtensionWindow(contextFrames: 25, additionalFrames: 96,
      width: 768, height: 448, fps: 24)
    let interval = try MLXSourceMovieInterval(source: source, sha256: digest, window: window)
    let prepared = try await interval.prepare(ffmpeg: ffmpeg,
      directory: folder.appendingPathComponent("prepared"))
    XCTAssertEqual(prepared.sourceFrames, 121)
    XCTAssertEqual(prepared.sourceRange, 96..<121)
    XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: prepared.rgb24.path)[.size] as? Int,
      25 * 768 * 448 * 3)
    XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: prepared.lowRGB24.path)[.size] as? Int,
      25 * 384 * 224 * 3)
    XCTAssertTrue(FileManager.default.fileExists(atPath: prepared.audio16k.path))
    try Data("replacement".utf8).write(to: source)
    XCTAssertThrowsError(try interval.validateSource())
  }
}
