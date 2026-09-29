import XCTest
import MLX
@testable import LTX25MLX

final class MLXAudioMelTests: XCTestCase {
  func testStereoMelMatchesIndependentLTXProcessorFixture() throws {
    let rate = 16_000, count = 8_000
    let left = (0..<count).map { sample in Float(0.2 * sin(2 * Double.pi * 440 * Double(sample) / Double(rate))) }
    let waveform = left + [Float](repeating: 0, count: count)
    let mel = try MLXAudioMel.encode(planar: waveform, sampleRate: rate)
    XCTAssertEqual(mel.shape, [1, 2, 51, 64])
    let values = mel.asArray(Float.self)
    func at(_ channel: Int, _ frame: Int, _ band: Int) -> Float {
      values[(channel * 51 + frame) * 64 + band]
    }
    XCTAssertEqual(at(0, 10, 8), 0.16946925, accuracy: 0.002)
    XCTAssertEqual(at(0, 10, 9), 0.06357598, accuracy: 0.002)
    XCTAssertEqual(at(0, 10, 10), -4.417991, accuracy: 0.02)
    XCTAssertEqual(at(1, 10, 8), -11.512925, accuracy: 0.0001)
  }
  func testRejectsInvalidAudioBeforeFFT() throws {
    XCTAssertThrowsError(try MLXAudioMel.encode(planar: [], sampleRate: 16_000))
    XCTAssertThrowsError(try MLXAudioMel.encode(planar: [Float.nan, 0], sampleRate: 16_000))
    XCTAssertThrowsError(try MLXAudioMel.encode(planar: [0, 0], sampleRate: 48_000))
  }
}
