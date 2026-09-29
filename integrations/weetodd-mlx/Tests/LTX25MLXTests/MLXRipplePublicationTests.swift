import AVFoundation
import Foundation
import InferenceMedia
import XCTest
@testable import LTX25MLX

final class MLXRipplePublicationTests: XCTestCase {
  private func run(_ binary: URL, _ arguments: [String]) throws {
    let process = Process()
    process.executableURL = binary
    process.arguments = arguments
    let stderr = Pipe()
    process.standardError = stderr
    process.standardOutput = stderr
    try process.run()
    process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0,
      String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
  }

  func testSourceAudioMuxRetainsAnAudioStreamAndEditorialDuration() async throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_FFMPEG"] else {
      throw XCTSkip("Installed FFmpeg source-audio qualification is opt-in.")
    }
    let ffmpeg = URL(fileURLWithPath: path)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mp4")
    try run(ffmpeg, ["-v", "error", "-nostdin", "-f", "lavfi", "-i",
      "color=c=red:s=64x64:r=24:d=1", "-f", "lavfi", "-i",
      "sine=frequency=440:sample_rate=48000:duration=1", "-shortest",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source.path])
    let video = root.appendingPathComponent("video.mp4")
    let writer = try RawVideoWriter(ffmpeg: ffmpeg, output: video,
      width: 64, height: 64, frames: 10, fps: 24)
    defer { writer.cancel() }
    for frame in 0..<10 {
      try writer.append(Data(repeating: UInt8(frame * 20), count: 64 * 64 * 3),
        frame: frame)
    }
    try writer.finish()
    let output = root.appendingPathComponent("ripple.mp4")
    try MLXRipplePipeline.muxSourceAudio(ffmpeg: ffmpeg, video: video,
      source: source, target: output, start: 0.125, duration: 10.0 / 24,
      log: root.appendingPathComponent("mux.log"))
    let movie = AVURLAsset(url: output)
    let duration = try await movie.load(.duration).seconds
    let videoTracks = try await movie.loadTracks(withMediaType: .video)
    let audioTracks = try await movie.loadTracks(withMediaType: .audio)
    XCTAssertEqual(duration, 10.0 / 24, accuracy: 1.0 / 24 + 0.002)
    XCTAssertEqual(videoTracks.count, 1)
    XCTAssertEqual(audioTracks.count, 1)
  }
}
