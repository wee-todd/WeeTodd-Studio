import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVisionPositionTests: XCTestCase {
  func testInstalledPositionAndRotaryMatchVisionOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"] else {
      throw XCTSkip("Set compact Qwen and vision-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let data = try Data(contentsOf: root.appendingPathComponent("pixels.u16"))
    let pixels = data.withUnsafeBytes {
      MLXArray($0, [16, 1536], type: UInt16.self).view(dtype: .bfloat16)
    }
    let checkpointURL = URL(fileURLWithPath: checkpoint)
    let projected = try H3QwenVisionPatch.embed(pixels: pixels, checkpointURL: checkpointURL)
    let positions = try H3QwenVisionPosition.make(checkpointURL: checkpointURL,
      grids: [.init(temporal: 1, height: 4, width: 4)])
    let hidden = projected + positions.absolute
    let expectedData = try Data(contentsOf: root.appendingPathComponent("input.f32"))
    let expected = expectedData.withUnsafeBytes { MLXArray($0, [16, 1152], type: Float.self) }
    XCTAssertEqual(max(abs(hidden.asType(.float32) - expected)).item(Float.self), 0)
    let rotaryData = try Data(contentsOf: root.appendingPathComponent("rotary.f32"))
    let expectedRotary = rotaryData.withUnsafeBytes { MLXArray($0, [16, 36], type: Float.self) }
    XCTAssertEqual(max(abs(positions.rotary - expectedRotary)).item(Float.self), 0)
    XCTAssertEqual(positions.boundaries, [0, 16])
  }
}
