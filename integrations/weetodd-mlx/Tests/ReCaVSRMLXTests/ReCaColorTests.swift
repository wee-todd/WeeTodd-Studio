import XCTest
import MLX
@testable import ReCaVSRMLX

final class ReCaColorTests: XCTestCase {
  func testAdaINUsesSampleVarianceInZeroToOneDomain() throws {
    try Device.withDefaultDevice(.cpu) {
      let x = MLXArray([Float(-1), -1, -1, 1, 1, 1], [1, 1, 2, 3])
      let y = try ReCaColorCorrection.adain(decodedNormalized: x, enlargedLQNormalized: x * 0.5)
      let gain = sqrt(Float((0.125 + 1e-5) / (0.5 + 1e-5)))
      XCTAssertEqual(y.asArray(Float.self)[0], -gain, accuracy: 1e-6)
      let single = try ReCaColorCorrection.adain(decodedNormalized: MLXArray.zeros([1, 1, 1, 3]),
        enlargedLQNormalized: MLXArray.ones([1, 1, 1, 3]) * 0.25)
      XCTAssertEqual(single.asArray(Float.self), [0.25, 0.25, 0.25])
    }
  }
  func testWaveletBuildsReferenceFromOriginalBytesAndReturnsBytes() throws {
    try Device.withDefaultDevice(.cpu) {
      let original = MLXArray([UInt8(20), 100, 200], [1, 1, 1, 3])
      let result = try ReCaColorCorrection.wavelet(decodedNormalized: MLXArray.zeros([1, 3, 5, 3]), originalLQUInt8: original)
      XCTAssertEqual(result.dtype, .uint8)
      XCTAssertEqual(result.shape, [1, 3, 5, 3])
      XCTAssertEqual(result.asArray(UInt8.self), Array(repeating: [UInt8(20), 100, 200], count: 15).flatMap { $0 })
    }
  }
  func testBicubicUsesHalfPixelCoordinatesAndReplicateEdges() throws {
    Device.withDefaultDevice(.cpu) {
      let source = MLXArray([Float(0), 0, 0, 1, 1, 1], [1, 1, 2, 3])
      let result = ReCaImagePreparation.resize(source, width: 4, height: 2, bicubic: true)
      XCTAssertEqual(result[0, 0, 0..<4, 0].asArray(Float.self), [-0.10546875, 0.2265625, 0.7734375, 1.10546875])
    }
  }
}
