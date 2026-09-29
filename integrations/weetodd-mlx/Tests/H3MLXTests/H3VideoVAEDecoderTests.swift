import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEDecoderTests: XCTestCase {
  func testInstalledHorizontalTwoTileDecodeMatchesReferenceClip() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and spatial oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("spatial-wide-latent.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 7, 2, 17, 24], type: Float.self) }
    let output = try H3VideoVAEDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent)
    XCTAssertEqual(output.shape, [1, 22, 32, 272, 3])
    let expected = try Data(contentsOf: root.appendingPathComponent("spatial-wide-pixels.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 22, 32, 272, 3], type: Float.self) }
    XCTAssertEqual(max(abs(output - expected)).item(Float.self), 0)
  }

  func testInstalledSpatialTwoTileDecodeMatchesReferenceClip() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and spatial oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("spatial-latent.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 7, 17, 2, 24], type: Float.self) }
    let output = try H3VideoVAEDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent)
    XCTAssertEqual(output.shape, [1, 22, 272, 32, 3])
    let expected = try Data(contentsOf: root.appendingPathComponent("spatial-pixels.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 22, 272, 32, 3], type: Float.self) }
    XCTAssertEqual(max(abs(output - expected)).item(Float.self), 0)
  }

  func testInstalledTemporalDecodeMatchesReferenceClip() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and temporal oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("clip-latent.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 22, 2, 2, 24], type: Float.self) }
    var chunks: [MLXArray] = []
    let output = try H3VideoVAEDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent,
      onChunk: { chunks.append($0) })
    XCTAssertEqual(output.shape, [1, 73, 32, 32, 3])
    XCTAssertGreaterThan(chunks.count, 1)
    XCTAssertEqual(chunks.reduce(0) { $0 + $1.shape[1] }, 73)
    let expected = try Data(contentsOf: root.appendingPathComponent("clip-pixels.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 73, 32, 32, 3], type: Float.self) }
    XCTAssertEqual(max(abs(output - expected)).item(Float.self), 0)
  }
}
