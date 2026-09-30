import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3AudioVAEEncoderTests: XCTestCase {
  func testInstalledStereoWaveformMatchesReferenceEncoder() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let checkpoint = environment["WEETODD_H3_AUDIO_VAE"],
      let fixture = environment["WEETODD_H3_AUDIO_ENCODER_ORACLE"] else {
      throw XCTSkip("Set the installed audio VAE and encoder oracle directory.")
    }
    let root = URL(fileURLWithPath: fixture)
    let waveform = try Data(contentsOf: root.appendingPathComponent("h3-audio-input.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 1600, 1], type: Float.self) }
    let expected = try Data(contentsOf: root.appendingPathComponent("h3-audio-oracle.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 2, 32], type: Float.self) }
    let actual = try H3AudioVAEEncoder.encode(
      checkpointURL: URL(fileURLWithPath: checkpoint), waveform: waveform,
      observe: { name, value in
        let channels = name == "encoder" ? 2048 : 32
        let reference = try Data(contentsOf: root.appendingPathComponent(
          "h3-audio-\(name).f32")).withUnsafeBytes {
            MLXArray($0, [2, 2, channels], type: Float.self)
          }
        XCTAssertLessThan(max(abs(value - reference)).item(Float.self), 0.003,
          name)
      })
    XCTAssertEqual(actual.shape, expected.shape)
    XCTAssertLessThan(max(abs(actual - expected)).item(Float.self), 0.003)
  }
}
