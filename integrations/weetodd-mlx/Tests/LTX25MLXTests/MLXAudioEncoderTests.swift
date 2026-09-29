import XCTest
import MLX
import InferenceTestSupport
@testable import LTX25MLX

final class MLXAudioEncoderTests: XCTestCase {
  func testInstalledCheckpointEncodesFiniteSourceTokens() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_LTX_AUDIO_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_LTX_AUDIO_CHECKPOINT for the real-checkpoint probe.")
    }
    let rate = 16_000, count = 8_000
    let left = (0..<count).map { sample in Float(0.2 * sin(2 * Double.pi * 440 * Double(sample) / Double(rate))) }
    let mel = try MLXAudioMel.encode(planar: left + [Float](repeating: 0, count: count), sampleRate: rate)
    let encoder = try MLXAudioEncoder(checkpoint: URL(fileURLWithPath: path), maximumMelFrames: 51)
    let tokens = try encoder.encode(mel: mel)
    XCTAssertEqual(tokens.shape, [13, 128])
    XCTAssertTrue(MLX.isFinite(tokens).all().item(Bool.self))
  }
  func testReleasedEncoderHeaderAndCausalTokenClock() throws {
    let shapes = MLXAudioEncoder.expectedShapes()
    XCTAssertEqual(shapes.count, 46)
    XCTAssertEqual(shapes["audio_vae.encoder.conv_in.conv.weight"], [128, 2, 3, 3])
    XCTAssertEqual(shapes["audio_vae.encoder.down.1.block.0.nin_shortcut.conv.weight"], [256, 128, 1, 1])
    XCTAssertEqual(shapes["audio_vae.encoder.conv_out.conv.weight"], [16, 512, 3, 3])
    XCTAssertEqual(try MLXAudioEncoder.latentFrames(melFrames: 101), 26)
    XCTAssertEqual(try MLXAudioEncoder.latentFrames(melFrames: 2001), 501)
    XCTAssertEqual(try MLXAudioEncoder.latentFrames(melFrames: 2011), 503)
    XCTAssertThrowsError(try MLXAudioEncoder.latentFrames(melFrames: 2012))
    let tensors = shapes.map { ($0.key, $0.value, "BF16") }
    try withTensorFile(metadata: ["model_version": "2.5.0"], tensors: tensors) { url in
      XCTAssertNoThrow(try MLXAudioEncoder(checkpoint: url, maximumMelFrames: 101))
    }
    try withTensorFile(metadata: ["model_version": "2.3.0"], tensors: tensors) { url in
      XCTAssertThrowsError(try MLXAudioEncoder(checkpoint: url))
    }
  }
}
