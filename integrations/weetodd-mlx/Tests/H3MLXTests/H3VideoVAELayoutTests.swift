import Foundation
import XCTest
@testable import H3MLX

final class H3VideoVAELayoutTests: XCTestCase {
  func testInstalledAffineQ8DecoderIsAdmittedWithoutReadingWeights() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_Q8"] else {
      throw XCTSkip("Set the installed native-layout H3 video VAE Q8 path.")
    }
    let layout = try H3VideoVAELayout(url: URL(fileURLWithPath: path))
    XCTAssertEqual(layout.blockCount, 36)
    XCTAssertEqual(layout.latentsMean.count, 24)
    XCTAssertEqual(layout.latentsStandardDeviation.count, 24)
    XCTAssertEqual(layout.clipLength, 17)
    XCTAssertEqual(layout.tokenDrop, 3)
  }

  func testIncompatibleMetadataFailsBeforeTensorInspection() {
    let payload: [String: Any] = [
      "format": "minimax-h3-mlx-video-vae",
      "format_version": 2,
      "tensor_layout": "ODHWI",
    ]
    XCTAssertThrowsError(try H3VideoVAELayout.validateMetadata(payload))
  }
}
