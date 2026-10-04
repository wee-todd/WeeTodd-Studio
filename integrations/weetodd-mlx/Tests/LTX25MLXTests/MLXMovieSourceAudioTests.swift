import AVFoundation
import CryptoKit
import XCTest
@testable import LTX25MLX

final class MLXMovieSourceAudioTests: XCTestCase {
  private func ffmpeg() throws -> URL {
    for path in [ProcessInfo.processInfo.environment["WEETODD_FFMPEG"], "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].compactMap({ $0 }) {
      if FileManager.default.isExecutableFile(atPath: path) { return URL(fileURLWithPath: path) }
    }
    throw XCTSkip("Native FFmpeg executable unavailable; no dependency installation.")
  }
  private func fixture(_ root: URL) throws -> (URL, String, [Float]) {
    let path = root.appendingPathComponent("mono-44100.wav")
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let values = (0..<44100).map { Float(($0 % 127) - 63) / 128 }
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44100)!
    buffer.frameLength = 44100
    for index in values.indices { buffer.floatChannelData![0][index] = values[index] }
    let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM,
      AVSampleRateKey: 44100, AVNumberOfChannelsKey: 1,
      AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsBigEndianKey: false]
    try autoreleasepool { let file = try AVAudioFile(forWriting: path, settings: settings); try file.write(from: buffer) }
    return (path, SHA256.hash(data: try Data(contentsOf: path)).map { String(format: "%02x", $0) }.joined(), values)
  }
  func testActualMonoPCMRetainsRateAndExactWaveformWhileConditioningIs16k() async throws {
    let binary = try ffmpeg(), root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let (source, sha, expected) = try fixture(root)
    let input = try MLXMovieSourceAudio(source: source, sha256: sha, sourceDurationSeconds: 1, frames: 24, fps: 24)
    let p = try await input.prepare(ffmpeg: binary, directory: root.appendingPathComponent("prepared"))
    XCTAssertEqual(p.contract.publicationSampleRate, 44100); XCTAssertEqual(p.contract.publicationSamples, 44100)
    XCTAssertTrue(p.contract.duplicateMono)
    let file = try AVAudioFile(forReading: p.publication, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 44100)!
    try file.read(into: buffer)
    XCTAssertEqual(buffer.frameLength, 44100)
    for channel in 0..<2 { XCTAssertEqual(Array(UnsafeBufferPointer(start: buffer.floatChannelData![channel], count: 44100)), expected) }
    let context = try AVAudioFile(forReading: p.conditioning)
    XCTAssertEqual(context.fileFormat.sampleRate, 16000); XCTAssertEqual(context.length, 16000)
    XCTAssertEqual(SHA256.hash(data: try Data(contentsOf: source)).map { String(format: "%02x", $0) }.joined(), sha)
  }
  func testActualCancellationAfterPublicationDoesNotPublishPartialAudioAndRetrySucceeds() async throws {
    let binary = try ffmpeg(), root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let (source, sha, _) = try fixture(root), destination = root.appendingPathComponent("prepared")
    let input = try MLXMovieSourceAudio(source: source, sha256: sha, sourceDurationSeconds: 1, frames: 24, fps: 24)
    do {
      _ = try await input.prepare(ffmpeg: binary, directory: destination) { phase in
        if phase == "movie_audio_publication_ready" { throw CancellationError() }
      }
      XCTFail("Expected cancellation")
    } catch is CancellationError { }
    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".movie-audio-") })
    let p = try await input.prepare(ffmpeg: binary, directory: destination)
    XCTAssertEqual(p.contract.publicationSamples, 44100)
  }
  func testActualSilentPreparationAndMutationRejectionBeforePublication() async throws {
    let binary = try ffmpeg(), root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let silence = try MLXMovieSourceAudio(source: nil, frames: 24, fps: 24)
    let p = try await silence.prepare(ffmpeg: binary, directory: root.appendingPathComponent("silence"))
    XCTAssertFalse(p.contract.sourceSupplied); XCTAssertEqual(p.contract.publicationSamples, 48000)
    let (source, sha, _) = try fixture(root)
    let input = try MLXMovieSourceAudio(source: source, sha256: sha, sourceDurationSeconds: 1, frames: 24, fps: 24)
    var modified = try Data(contentsOf: source); modified[modified.count - 1] ^= 1; try modified.write(to: source)
    let destination = root.appendingPathComponent("mutated")
    do { _ = try await input.prepare(ffmpeg: binary, directory: destination); XCTFail("Expected frozen source rejection") }
    catch { XCTAssertTrue(String(describing: error).contains("changed")) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
  }
}
