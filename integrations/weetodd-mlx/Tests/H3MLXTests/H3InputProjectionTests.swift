import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3InputProjectionTests: XCTestCase {
  func testInstalledVideoAudioAndConditionProjectionMatchReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_PATCH_ORACLE"] else {
      throw XCTSkip("Set the installed H3 checkpoint and input projection oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    for (kind, width, isBF16) in [
      (H3InputProjection.Kind.video, 96, false),
      (H3InputProjection.Kind.audio, 32, false),
      (H3InputProjection.Kind.condition, 5120, true),
    ] {
      let stem = kind.rawValue
      let inputData = try Data(contentsOf: root.appendingPathComponent(
        "\(stem)-input.\(isBF16 ? "u16" : "f32")"))
      let input = inputData.withUnsafeBytes { bytes in
        isBF16
          ? MLXArray(bytes, [1, 4, width], type: UInt16.self).view(dtype: .bfloat16)
          : MLXArray(bytes, [1, 4, width], type: Float.self)
      }
      let output = try H3InputProjection.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
        kind: kind, input: input)
      let expectedData = try Data(contentsOf: root.appendingPathComponent(
        "\(stem)-output.\(isBF16 ? "u16" : "f32")"))
      let expected = expectedData.withUnsafeBytes { bytes in
        isBF16
          ? MLXArray(bytes, [1, 4, 5376], type: UInt16.self).view(dtype: .bfloat16)
          : MLXArray(bytes, [1, 4, 5376], type: Float.self)
      }
      XCTAssertEqual(output.shape, [1, 4, 5376], stem)
      XCTAssertEqual(max(abs(output.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, stem)
    }
  }
}
