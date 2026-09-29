import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVisionEncoderTests: XCTestCase {
  func testInstalledTwentySevenLayerVisionTowerMatchesMLXOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"] else {
      throw XCTSkip("Set compact Qwen and vision-block oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array(_ name: String, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: root.appendingPathComponent(name))
      return data.withUnsafeBytes { MLXArray($0, shape, type: Float.self) }
    }
    let bytes = try Data(contentsOf: root.appendingPathComponent("pixels.u16"))
    let pixels = bytes.withUnsafeBytes {
      MLXArray($0, [16, 1536], type: UInt16.self).view(dtype: .bfloat16)
    }
    var maximumLayerError: Float = 0
    let output = try H3QwenVisionEncoder.encode(pixels: pixels,
      grids: [.init(temporal: 1, height: 4, width: 4)],
      checkpointURL: URL(fileURLWithPath: checkpoint)) { index, hidden in
      let expected = try array(String(format: "layer-%02d.f32", index), shape: [16, 1152])
      let error = max(abs(hidden.asType(.float32) - expected)).item(Float.self)
      maximumLayerError = max(maximumLayerError, error)
      XCTAssertEqual(error, 0, "vision layer \(index)")
    }
    XCTAssertEqual(maximumLayerError, 0)
    let expected = try array("merged.f32", shape: [4, 5120])
    XCTAssertEqual(max(abs(output.hidden.asType(.float32) - expected)).item(Float.self), 0)
    XCTAssertEqual(output.deepstack.count, 3)
    for index in 0..<3 {
      let deep = try array("deep-\(index).f32", shape: [4, 5120])
      XCTAssertEqual(max(abs(output.deepstack[index].asType(.float32) - deep)).item(Float.self), 0)
    }
  }
}
