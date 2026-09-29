import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVisionPatchTests: XCTestCase {
  func testInstalledPatchProjectionMatchesMLXOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_PATCH"] else {
      throw XCTSkip("Set compact Qwen and vision-patch oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let inputData = try Data(contentsOf: root.appendingPathComponent("pixels.u16"))
    let input = inputData.withUnsafeBytes { bytes in
      MLXArray(bytes, [4, 1536], type: UInt16.self).view(dtype: .bfloat16)
    }
    let actual = try H3QwenVisionPatch.embed(pixels: input,
      checkpointURL: URL(fileURLWithPath: checkpoint))
    eval(actual)
    let expectedData = try Data(contentsOf: root.appendingPathComponent("embedded.f32"))
    let expected = expectedData.withUnsafeBytes { bytes in
      MLXArray(bytes, [4, 1152], type: Float.self)
    }
    let difference = max(abs(actual.asType(.float32) - expected)).item(Float.self)
    XCTAssertEqual(difference, 0)
  }
}
