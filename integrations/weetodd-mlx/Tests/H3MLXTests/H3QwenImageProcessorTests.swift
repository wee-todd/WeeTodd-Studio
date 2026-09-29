import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenImageProcessorTests: XCTestCase {
  func testStillPatchRowsMatchInstalledProcessorOracle() throws {
    let root = URL(fileURLWithPath: "/tmp/weetodd-h3-qwen-image-processor")
    guard FileManager.default.fileExists(atPath: root.appendingPathComponent("rgb.u8").path) else {
      throw XCTSkip("Qwen image processor oracle is not installed.")
    }
    let rgb = try Data(contentsOf: root.appendingPathComponent("rgb.u8"))
    let actual = try H3QwenImageProcessor.packRGB8(image: rgb, width: 64, height: 64)
    XCTAssertEqual(actual.grid.temporal, 1)
    XCTAssertEqual(actual.grid.height, 4)
    XCTAssertEqual(actual.grid.width, 4)
    let data = try Data(contentsOf: root.appendingPathComponent("pixels.f32"))
    let expected = data.withUnsafeBytes { MLXArray($0, [16, 1536], type: Float.self) }
    XCTAssertLessThan(max(abs(actual.pixels - expected)).item(Float.self), 0.00001)
  }

  func testRejectsReferenceOutsideAdmittedPatchGeometry() {
    XCTAssertThrowsError(try H3QwenImageProcessor.packRGB8(
      image: Data(count: 16 * 16 * 3), width: 16, height: 16))
  }
}
