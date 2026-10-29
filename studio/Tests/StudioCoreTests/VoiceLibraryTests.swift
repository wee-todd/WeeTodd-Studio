import AVFoundation
import XCTest

@testable import StudioCore

final class VoiceLibraryTests: XCTestCase {
  func testSavedCloneOwnsSelectedSampleAfterOriginalIsDeleted() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("original.wav")
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44100)!
    buffer.frameLength = 44100
    for i in 0..<44100 { buffer.floatChannelData![0][i] = sin(Float(i) * 0.03) * 0.1 }
    do {
      let file = try AVAudioFile(forWriting: source, settings: format.settings)
      try file.write(from: buffer)
    }
    let library = VoiceLibrary(root: root.appendingPathComponent("voices"))
    let ref = VoiceReference(
      path: source.path, start: 0.25, duration: 0.5, transcript: "Hello there.")
    let saved = try await library.save(name: "Alex", samples: [ref], description: "Warm voice")
    try FileManager.default.removeItem(at: source)
    let restored = try await library.load(saved.reference)
    XCTAssertEqual(restored.samples.count, 1)
    XCTAssertEqual(restored.samples[0].transcript, "Hello there.")
    XCTAssertEqual(restored.samples[0].start, 0)
    let archived = try AVAudioFile(forReading: URL(fileURLWithPath: restored.samples[0].path))
    XCTAssertEqual(archived.length, 22050)
    XCTAssertEqual(restored.engine, .fishDrama3)
    let profiles = try await library.list()
    XCTAssertEqual(profiles.count, 1)
  }
  func testProfileVersionsPinContentAndDeduplicateIdenticalSamples() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("sample.wav")
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4410)!
    buffer.frameLength = 4410
    buffer.floatChannelData![0].initialize(repeating: 0.1, count: 4410)
    try AVAudioFile(forWriting: source, settings: format.settings).write(from: buffer)
    let library = VoiceLibrary(root: root.appendingPathComponent("voices"))
    let ref = VoiceReference(path: source.path, duration: 0.1, transcript: "Hi")
    let a = try await library.save(name: "A", samples: [ref], description: "")
    let b = try await library.save(name: "B", samples: [ref], description: "")
    XCTAssertEqual(a.samples[0].path, b.samples[0].path)
    let prior = try await library.load(a.reference)
    XCTAssertEqual(prior.name, "A")
    let decoded = try JSONDecoder().decode(VoiceProfileSnapshot.self, from: JSONEncoder().encode(a))
    XCTAssertEqual(a, decoded)
    try Data("corrupted owned sample".utf8).write(to: URL(fileURLWithPath: a.samples[0].path))
    do {
      _ = try await library.save(name: "Corrupt", samples: [ref], description: "")
      XCTFail("A content-addressed filename must not hide corrupted retained audio.")
    } catch {}
  }
}
