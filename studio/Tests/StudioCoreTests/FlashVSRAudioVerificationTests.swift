import AVFoundation
import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRAudioVerificationTests: XCTestCase {
  func testDelayedAudioKeepsItsOffsetAcrossTrimAndSilentInterval() throws {
    let candidates = [ProcessInfo.processInfo.environment["WEETODD_TEST_FFMPEG"] ?? "", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    guard let binary = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else { throw XCTSkip("FFmpeg fixture tool unavailable") }
    let ffmpeg = URL(fileURLWithPath: binary)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("delayed.mkv")
    let result = try ExternalEngineProcess.run(executable: ffmpeg, arguments: ["-v", "error", "-nostdin", "-n",
      "-f", "lavfi", "-i", "color=size=128x128:rate=24:duration=1", "-itsoffset", "0.5", "-f", "lavfi", "-i",
      "sine=frequency=1000:sample_rate=32000:duration=0.5", "-c:v", "ffv1", "-c:a", "pcm_f32le", source.path],
      directory: root, log: root.appendingPathComponent("fixture.log"))
    XCTAssertEqual(result.exitCode, 0)
    let fps = try FlashVSRFrameRate(numerator: 24, denominator: 1)
    func samples(_ start: Double, _ count: Int, _ name: String) throws -> [Float] {
      let output = root.appendingPathComponent(name + ".wav")
      try FlashVSRAudioPreparation.prepare(source: source, output: output, ffmpeg: ffmpeg, directory: root,
        startTimestamp: start, frames: count, fps: fps, sampleRate: 32000, cancellation: NativeHeadlessCancellation())
      let audio = try AVAudioFile(forReading: output)
      XCTAssertEqual(audio.length, Int64(Double(count) / 24 * 32000))
      guard audio.length > 0 else { return [] }
      let raw = root.appendingPathComponent(name + ".f32le")
      let decoded = try ExternalEngineProcess.run(executable: ffmpeg,
        arguments: ["-v", "error", "-nostdin", "-n", "-i", output.path, "-f", "f32le", raw.path],
        directory: root, log: root.appendingPathComponent(name + "-decode.log"))
      XCTAssertEqual(decoded.exitCode, 0)
      let bytes = try Data(contentsOf: raw)
      return bytes.withUnsafeBytes { raw in
        stride(from: 0, to: bytes.count, by: 4).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self))) }
      }
    }
    let trimmed = try samples(0.25, 12, "trimmed")
    XCTAssertEqual(trimmed.count, 16000)
    if trimmed.count == 16000 {
      XCTAssertLessThan(trimmed.prefix(7900).map { abs($0) }.max()!, 0.00001)
      XCTAssertGreaterThan(trimmed.suffix(7900).map { abs($0) }.max()!, 0.05)
    }
    let silent = try samples(0, 6, "silent")
    XCTAssertEqual(silent.count, 8000)
    XCTAssertLessThan(silent.map { abs($0) }.max() ?? 0, 0.00001)
  }
  func testEncoderPaddingIsAllowedButTruncationAndRateChangesFail() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func wave(_ name: String, _ frames: Int, rate: Double = 32000) throws -> URL {
      let url = root.appendingPathComponent(name + ".wav")
      let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2)!
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
      buffer.frameLength = AVAudioFrameCount(frames)
      try AVAudioFile(forWriting: url, settings: format.settings).write(from: buffer)
      return url
    }
    let source = try wave("source", 8192)
    XCTAssertNoThrow(try FlashVSRAudioVerification.verify(source: source, decoded: wave("padded", 9216)))
    XCTAssertThrowsError(try FlashVSRAudioVerification.verify(source: source, decoded: wave("short", 4096)))
    XCTAssertThrowsError(try FlashVSRAudioVerification.verify(source: source, decoded: wave("rate", 8192, rate: 44100)))
  }
}
