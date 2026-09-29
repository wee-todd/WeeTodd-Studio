import Foundation
import XCTest
@testable import LTX25MLX

final class MLXSourceAudioTests: XCTestCase {
  private func wave(_ url: URL) throws {
    let rate = 48_000, count = rate * 2
    var bytes = Data()
    func word(_ value: UInt32) { var little = value.littleEndian; bytes.append(Data(bytes: &little, count: 4)) }
    func half(_ value: UInt16) { var little = value.littleEndian; bytes.append(Data(bytes: &little, count: 2)) }
    bytes.append(Data("RIFF".utf8)); word(UInt32(36 + count * 2)); bytes.append(Data("WAVEfmt ".utf8))
    word(16); half(1); half(1); word(UInt32(rate)); word(UInt32(rate * 2)); half(2); half(16)
    bytes.append(Data("data".utf8)); word(UInt32(count * 2))
    for index in 0..<count { half(UInt16(bitPattern: index < rate ? 0 : 8192)) }
    try bytes.write(to: url)
  }
  private func samples(_ url: URL) throws -> [Float] {
    let bytes = try Data(contentsOf: url)
    guard let marker = bytes.range(of: Data("data".utf8)) else { return [] }
    let start = marker.upperBound + 4
    return bytes[start...].withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self))
    }
  }
  func testNonzeroInPointAndExactTrimmedOutputs() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.wav"); try wave(source)
    let interval = try MLXSourceAudioInterval(source: source, sourceStartSeconds: 1,
      durationSeconds: 0.5)
    let output = try await interval.extract(ffmpeg: URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
      directory: root.appendingPathComponent("prepared"))
    let publication = try samples(output.publication)
    let conditioning = try samples(output.conditioning)
    XCTAssertEqual(publication.count, 48_000)
    XCTAssertEqual(conditioning.count, 16_000)
    XCTAssertEqual(publication[100], 0.25, accuracy: 0.005)
    XCTAssertEqual(conditioning[100], 0.25, accuracy: 0.005)
    let mel = try MLXAudioMel.encode(wav: output.conditioning)
    XCTAssertEqual(mel.shape, [1, 2, 51, 64])
  }
  func testRejectsBadIntervalsAndOutOfRangeInPoint() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.wav"); try wave(source)
    XCTAssertThrowsError(try MLXSourceAudioInterval(source: source, sourceStartSeconds: -1, durationSeconds: 1))
    XCTAssertThrowsError(try MLXSourceAudioInterval(source: source, sourceStartSeconds: .nan, durationSeconds: 1))
    XCTAssertThrowsError(try MLXSourceAudioInterval(source: source, sourceStartSeconds: 0, durationSeconds: 0))
    let interval = try MLXSourceAudioInterval(source: source, sourceStartSeconds: 2.1, durationSeconds: 1)
    do {
      _ = try await interval.extract(ffmpeg: URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
        directory: root.appendingPathComponent("prepared"))
      XCTFail("Out-of-range source start accepted")
    } catch { XCTAssertTrue(String(describing: error).contains("outside"), String(describing: error)) }
  }
  func testExplicitSourceDurationPadsPublicationAfterNonzeroInPoint() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let source=root.appendingPathComponent("source.wav");try wave(source)
    let interval=try MLXSourceAudioInterval(source:source,sourceStartSeconds:1,
      sourceDurationSeconds:0.25,durationSeconds:0.5)
    let prepared=try await interval.extract(ffmpeg:URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg"),
      directory:root.appendingPathComponent("prepared"))
    let publication=try samples(prepared.publication)
    XCTAssertEqual(publication.count,48_000)
    XCTAssertEqual(publication[100],0.25,accuracy:0.005)
    XCTAssertEqual(publication[30_000],0,accuracy:0.0001)
    let copied=root.appendingPathComponent("published.wav")
    try MLXMediaPipeline.publishSourceAudio(prepared,to:copied)
    XCTAssertEqual(try Data(contentsOf:copied),try Data(contentsOf:prepared.publication))
    XCTAssertThrowsError(try MLXMediaPipeline.publishSourceAudio(prepared,to:copied))
    let pastEnd=try MLXSourceAudioInterval(source:source,sourceStartSeconds:1.8,
      sourceDurationSeconds:0.5,durationSeconds:0.5)
    do {
      _ = try await pastEnd.extract(ffmpeg:URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg"),
        directory:root.appendingPathComponent("past-end"))
      XCTFail("An overlong source interval was accepted")
    } catch { XCTAssertTrue(String(describing:error).contains("outside")) }
  }
}
