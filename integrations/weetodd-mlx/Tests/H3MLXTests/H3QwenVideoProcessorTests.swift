import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenVideoProcessorTests: XCTestCase {
  func testTwoFramePatchPackingMatchesReference() throws {
    guard let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PROCESSOR_ORACLE"] else {
      throw XCTSkip("Set the Qwen video processor oracle path.")
    }
    let root = URL(fileURLWithPath: fixture)
    let frames = try Data(contentsOf: root.appendingPathComponent("frames.u8"))
    let result = try H3QwenVideoProcessor.packRGB8(
      frames: frames, frameCount: 2, width: 64, height: 32)
    XCTAssertEqual(result.grid.temporal, 1)
    XCTAssertEqual(result.grid.height, 2)
    XCTAssertEqual(result.grid.width, 4)
    XCTAssertEqual(result.pixels.shape, [8, 1536])
    let expectedData = try Data(contentsOf: root.appendingPathComponent("pixels.f32"))
    let expected = expectedData.withUnsafeBytes {
      MLXArray($0, [8, 1536], type: Float.self)
    }
    XCTAssertLessThan(max(abs(result.pixels - expected)).item(Float.self), 0.0000002)
  }

  func testInvalidDimensionsFailBeforeAllocatingPatches() {
    XCTAssertThrowsError(try H3QwenVideoProcessor.packRGB8(
      frames: Data(count: 16), frameCount: 1, width: 16, height: 16))
    XCTAssertThrowsError(try H3QwenVideoProcessor.packRGB8(
      frames: Data(count: 32 * 32 * 3), frameCount: 2, width: 32, height: 32))
  }
}
