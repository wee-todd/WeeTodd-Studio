import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoVAEEncoderTests: XCTestCase {
  func testInstalledStillReferenceEncodesOneDeterministicPosteriorMean() throws {
    guard let path = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set H3_VIDEO_VAE_CHECKPOINT for installed still-image encoder qualification.")
    }
    let pixels = [UInt8](repeating: 127, count: 64 * 64 * 3)
    let moments = try H3VideoVAEEncoder.encodeStill(
      checkpointURL: URL(fileURLWithPath: path), rgb8: pixels,
      width: 64, height: 64)
    XCTAssertEqual(moments.shape, [1, 1, 4, 4, 24])
    XCTAssertTrue(MLX.isFinite(moments).all().item(Bool.self))
    let pythonOracle: [Float] = [1.4784774, 0.45443344, 0.69458336,
      -1.9458747, -1.2183652, 3.3782234, -1.8070253, -1.7420738]
    for (channel, expected) in pythonOracle.enumerated() {
      XCTAssertEqual(moments[0, 0, 0, 0, channel].item(Float.self),
        expected, accuracy: 0.03, "encoder channel \(channel)")
    }
  }
}
