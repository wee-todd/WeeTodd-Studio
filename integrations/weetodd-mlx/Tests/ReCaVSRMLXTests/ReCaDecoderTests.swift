import XCTest
import MLX
@testable import ReCaVSRMLX

final class ReCaDecoderTests: XCTestCase {
  func testDecoderPixelShuffleKeepsChannelsSeparate() throws {
    try Device.withDefaultDevice(.cpu) {
      let pixels = MLXArray((0..<12).map(Float.init), [1, 1, 1, 12])
      let result = try ReCaFlashDecoder.pixelShuffle(pixels, factor: 2)
      XCTAssertEqual(result.shape, [1, 2, 2, 3])
      XCTAssertEqual(result.asArray(Float.self), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
    }
  }
  func testDecoderFirstGroupRetainsInternalContextButEmitsOneFrame() throws {
    try Device.withDefaultDevice(.cpu) {
      let source = MLXArray([Float(7)], [1, 1, 1, 1])
      let first = try ReCaFlashDecoder.groupLRFeatures(source, first: true)
      XCTAssertEqual(first.asArray(Float.self), [7, 7, 7, 7])
      let body = try ReCaFlashDecoder.groupLRFeatures(MLXArray([Float(1), 2, 3, 4], [4, 1, 1, 1]), first: false)
      XCTAssertEqual(body.asArray(Float.self), [1, 2, 3, 4])
      XCTAssertThrowsError(try ReCaFlashDecoder.groupLRFeatures(source, first: false))
    }
  }
  func testDecoderTruncatedWindowsDoNotShiftAtEdges() throws {
    try Device.withDefaultDevice(.cpu) {
      let span = 5 * 7
      let plan = ReCaAttentionPlan(height: 5, width: 7, windowHeight: 4, windowWidth: 4, slidesAtEdge: false)
      let q = MLXArray.zeros([span * 4, 8, 4]), k = MLXArray.zeros([span * 8, 2, 4])
      let v = broadcast(MLXArray((0..<(span * 8)).map(Float.init), [span * 8, 1, 1]), to: [span * 8, 2, 4])
      let result = try plan.apply(query: q, key: k, value: v)[0, 0, 0].item(Float.self)
      // Eight planes; at the corner each plane exposes rows/columns 0 and 1.
      XCTAssertEqual(result, 126.5, accuracy: 1e-4)
    }
  }
}
