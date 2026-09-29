import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVisionMergerTests: XCTestCase {
  func testInstalledMainAndDeepstackMergerMatchMLXOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"] else {
      throw XCTSkip("Set compact Qwen and vision-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array(_ name: String, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: root.appendingPathComponent(name))
      return data.withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
    }
    let checkpointURL = URL(fileURLWithPath: checkpoint)
    let deepInput = try array("layer-08.f32", shape: [16, 1152]).asType(.bfloat16)
    let deep = try H3QwenVisionMerger.evaluate(checkpointURL: checkpointURL,
      deepIndex: 0, input: deepInput)
    let expectedDeep = try array("deep-0.f32", shape: [4, 5120])
    XCTAssertEqual(max(abs(deep.asType(.float32) - expectedDeep)).item(Float.self), 0)
    let mainInput = try array("layer-26.f32", shape: [16, 1152]).asType(.bfloat16)
    let main = try H3QwenVisionMerger.evaluate(checkpointURL: checkpointURL,
      deepIndex: nil, input: mainInput)
    let expectedMain = try array("merged.f32", shape: [4, 5120])
    XCTAssertEqual(max(abs(main.asType(.float32) - expectedMain)).item(Float.self), 0)
  }
}
