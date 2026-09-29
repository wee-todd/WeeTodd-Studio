import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAETileDecoderTests: XCTestCase {
  func testInstalledFullViTTileMatchesReferenceBoundaries() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and tile oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("latent.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 2, 2, 2, 24], type: Float.self) }
    let output = try H3VideoVAETileDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent) { name, value in
      let shape: [Int]
      switch name {
      case "postquant": shape = [1, 2, 2, 2, 24]
      case "embed": shape = [1, 8, 2048]
      case "positions", "vit0", "vit1", "vit35", "normout":
        shape = name == "positions" ? [1, 13, 3] : [1, 13, 2048]
      case "head": shape = [1, 8, 3072]
      case "pixels": shape = [1, 8, 32, 32, 3]
      default: XCTFail("Unknown decoder observation: \(name)"); return
      }
      let expected = try Data(contentsOf: root.appendingPathComponent("\(name).f32"))
        .withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
      XCTAssertEqual(max(abs(value.asType(.float32) - expected)).item(Float.self),
        0, name)
    }
    XCTAssertEqual(output.shape, [1, 8, 32, 32, 3])
  }

  func testInstalledTwoTileBatchMatchesReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and tile oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("batch2-latent.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 2, 2, 2, 24], type: Float.self) }
    let output = try H3VideoVAETileDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent)
    let expected = try Data(contentsOf: root.appendingPathComponent("batch2-pixels.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 8, 32, 32, 3], type: Float.self) }
    XCTAssertEqual(output.shape, [2, 8, 32, 32, 3])
    XCTAssertEqual(max(abs(output - expected)).item(Float.self), 0)
  }
}
