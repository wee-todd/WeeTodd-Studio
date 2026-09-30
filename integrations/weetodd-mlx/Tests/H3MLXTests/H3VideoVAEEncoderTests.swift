import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEEncoderTests: XCTestCase {
  func testInstalledMovingVideoMatchesReferenceEncoder() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let checkpoint = environment["H3_VIDEO_VAE_CHECKPOINT"],
      let fixture = environment["H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set the installed VAE and moving-video oracle directory.")
    }
    for count in [5, 22] {
      let source = try Data(contentsOf: URL(fileURLWithPath: fixture)
        .appendingPathComponent("h3-video-input-\(count).rgb"))
      let reference = try Data(contentsOf: URL(fileURLWithPath: fixture)
        .appendingPathComponent("h3-video-oracle-\(count).f32"))
      let actual = try H3VideoVAEEncoder.encodeVideo(
        checkpointURL: URL(fileURLWithPath: checkpoint), rgb8: Array(source),
        frameCount: count, width: 64, height: 64)
      let frames = (count - 5) / 17 * 5 + 2
      let expected = reference.withUnsafeBytes {
        MLXArray($0, [1, frames, 4, 4, 24], type: Float.self)
      }
      XCTAssertEqual(actual.shape, expected.shape)
      XCTAssertLessThan(max(abs(actual - expected)).item(Float.self), 0.035,
        "moving VAE reference with \(count) frames")
    }
  }

  func testVideoReferenceRejectsUnalignedOrOversizedClipBeforeCheckpointLoad() {
    let unavailable = URL(fileURLWithPath: "/nonexistent/video-vae.safetensors")
    XCTAssertThrowsError(try H3VideoVAEEncoder.encodeVideo(
      checkpointURL: unavailable, rgb8: [UInt8](repeating: 0,
        count: 6 * 64 * 64 * 3), frameCount: 6, width: 64, height: 64))
    XCTAssertThrowsError(try H3VideoVAEEncoder.encodeVideo(
      checkpointURL: unavailable, rgb8: [UInt8](repeating: 0,
        count: 5 * 288 * 64 * 3), frameCount: 5, width: 288, height: 64))
  }

  func testInstalledVideoReferenceEncodesAlignedClip() throws {
    guard let path = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set H3_VIDEO_VAE_CHECKPOINT for installed video encoder qualification.")
    }
    let frames = [UInt8](repeating: 127, count: 5 * 64 * 64 * 3)
    let latents = try H3VideoVAEEncoder.encodeVideo(
      checkpointURL: URL(fileURLWithPath: path), rgb8: frames,
      frameCount: 5, width: 64, height: 64)
    XCTAssertEqual(latents.shape, [1, 2, 4, 4, 24])
    XCTAssertTrue(MLX.isFinite(latents).all().item(Bool.self))
    let still = try H3VideoVAEEncoder.encodeStill(
      checkpointURL: URL(fileURLWithPath: path),
      rgb8: [UInt8](repeating: 127, count: 64 * 64 * 3),
      width: 64, height: 64)
    for channel in 0..<24 {
      XCTAssertEqual(latents[0, 0, 0, 0, channel].item(Float.self),
        still[0, 0, 0, 0, channel].item(Float.self), accuracy: 0.005,
        "first causal latent channel \(channel)")
    }
    let longer = try H3VideoVAEEncoder.encodeVideo(
      checkpointURL: URL(fileURLWithPath: path),
      rgb8: [UInt8](repeating: 127, count: 22 * 64 * 64 * 3),
      frameCount: 22, width: 64, height: 64)
    XCTAssertEqual(longer.shape, [1, 7, 4, 4, 24])
    XCTAssertTrue(MLX.isFinite(longer).all().item(Bool.self))
  }

  func testInstalledStillReferenceEncodesOneDeterministicPosteriorMean() throws {
    guard let path = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set H3_VIDEO_VAE_CHECKPOINT for installed still-image encoder qualification.")
    }
    let pixels = [UInt8](repeating: 127, count: 64 * 64 * 3)
    let moments = try H3VideoVAEEncoder.encodeStill(
      checkpointURL: URL(fileURLWithPath: path), rgb8: pixels,
      width: 64, height: 64)
    XCTAssertEqual(moments.shape, [1, 1, 4, 4, 24])
    XCTAssertTrue(MLX.isFinite(moments).all().item(Bool.self))
    let pythonOracle: [Float] = [1.4784774, 0.45443344, 0.69458336,
      -1.9458747, -1.2183652, 3.3782234, -1.8070253, -1.7420738]
    for (channel, expected) in pythonOracle.enumerated() {
      XCTAssertEqual(moments[0, 0, 0, 0, channel].item(Float.self),
        expected, accuracy: 0.03, "encoder channel \(channel)")
    }
  }
}
