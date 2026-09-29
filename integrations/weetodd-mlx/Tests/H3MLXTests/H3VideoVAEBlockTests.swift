import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEBlockTests: XCTestCase {
  func testInstalledQuantizedDecoderBlockMatchesReferenceStages() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and decoder-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let inputData = try Data(contentsOf: root.appendingPathComponent("input.f16"))
    let input = inputData.withUnsafeBytes {
      MLXArray($0, [1, 13, 2048], type: Float16.self)
    }
    let positionData = try Data(contentsOf: root.appendingPathComponent("positions.f32"))
    let positions = positionData.withUnsafeBytes {
      MLXArray($0, [1, 13, 3], type: Float.self)
    }
    let output = try H3VideoVAEBlock.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      index: 0, input: input, positions: positions) { name, value in
      let data = try Data(contentsOf: root.appendingPathComponent("\(name).f16"))
      let expected = data.withUnsafeBytes {
        MLXArray($0, [1, 13, 2048], type: Float16.self)
      }
      XCTAssertEqual(max(abs(value.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, name)
    }
    XCTAssertEqual(output.shape, [1, 13, 2048])
  }

  func testInstalledBlockMatchesFloat32DecodePath() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_ORACLE"] else {
      throw XCTSkip("Set installed H3 video VAE Q8 and decoder-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let input = try Data(contentsOf: root.appendingPathComponent("input32.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 13, 2048], type: Float.self) }
    let positions = try Data(contentsOf: root.appendingPathComponent("positions.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 13, 3], type: Float.self) }
    _ = try H3VideoVAEBlock.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      index: 0, input: input, positions: positions) { name, value in
      let expected = try Data(contentsOf: root.appendingPathComponent("\(name)32.f32"))
        .withUnsafeBytes { MLXArray($0, [1, 13, 2048], type: Float.self) }
      XCTAssertEqual(max(abs(value - expected)).item(Float.self), 0, name)
    }
  }
}
