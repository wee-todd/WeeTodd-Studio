import MLX
import XCTest
@testable import H3MLX

final class H3ComfyInt8ProjectionTests: XCTestCase {
  func testRawSignedProjectionRejectsInstalledConvRotWeight() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_TEST_CHECKPOINT for the optional installed-weight check.")
    }
    XCTAssertThrowsError(try H3ComfyInt8Projection(
      checkpointURL: URL(fileURLWithPath: path),
      name: "model.diffusion_model.blocks.0.attn.out_proj.weight",
      rows: 5376, columns: 7168))
  }

  func testSignedTensorwiseRowsProjectThroughPackedQ8WithoutDenseExpansion() throws {
    let raw = [Int8](repeating: 1, count: 64) + [Int8](repeating: -2, count: 64)
    let projection = try raw.withUnsafeBytes {
      try H3ComfyInt8Projection(weightBytes: $0, rows: 2, columns: 64, rowScales: [0.5, 2])
    }
    let input = MLXArray((1...64).map(Float.init), [1, 64])
    let output = try projection.project(input).asArray(Float.self)
    XCTAssertEqual(output[0], 1040, accuracy: 0.05)
    XCTAssertEqual(output[1], -8320, accuracy: 0.05)
    XCTAssertEqual(projection.storageBytes, 128 + 2 * 4 * 2)
  }

  func testRejectsInvalidScaleOrUnalignedInput() throws {
    let raw = [UInt8](repeating: 0, count: 128)
    XCTAssertThrowsError(try raw.withUnsafeBytes {
      try H3ComfyInt8Projection(weightBytes: $0, rows: 2, columns: 64, rowScales: [0, 1])
    })
    XCTAssertThrowsError(try raw.withUnsafeBytes {
      try H3ComfyInt8Projection(weightBytes: $0, rows: 2, columns: 63, rowScales: [1, 1])
    })
  }
}
