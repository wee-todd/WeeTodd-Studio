import XCTest
import MLX
@testable import ReCaVSRMLX

final class ReCaTransformerTests: XCTestCase {
  func testCacheCommitsChronologicalPlanesAndRejectsPartialRetry() throws {
    try Device.withDefaultDevice(.cpu) {
      var cache = ReCaKVCache(layer: 9)
      let key = MLXArray((0..<12).map(Float.init), [6, 1, 2])
      try cache.commit(key: key, value: key + 100, frames: 6, start: 0)
      XCTAssertEqual(cache.ids, [4, 5])
      XCTAssertEqual(cache.planes[5]?.key.shape, [1, 1, 2])
      XCTAssertEqual(cache.planes[5]?.key.asArray(Float.self), [10, 11])
      XCTAssertThrowsError(try cache.commit(key: key, value: key, frames: 6, start: 0))
      let next = MLXArray([Float(12), 13, 14, 15], [2, 1, 2])
      try cache.commit(key: next, value: next, frames: 2, start: 6)
      XCTAssertEqual(cache.ids, [5, 6, 7])
    }
  }
  func testOneStepPredictionRemainsFP32AndRecycleUsesPredictedTail() throws {
    Device.withDefaultDevice(.cpu) {
      let noise = MLXArray([Float(1), 2, 3, 4, 5, 6], [6, 1, 1, 1])
      let prediction = MLXArray(Array(repeating: Float(0.010009765625), count: 6), [6, 1, 1, 1]).asType(.bfloat16)
      let result = ReCaTransformerSession.predictedLatents(noise: noise, prediction: prediction)
      XCTAssertEqual(result.dtype, .float32)
      XCTAssertEqual(result.asArray(Float.self)[0], 0.989990234375)
      let tail = ReCaMath.own(result.asType(.bfloat16), range: 4..<6)
      XCTAssertEqual(tail.shape, [2, 1, 1, 1])
      XCTAssertEqual(tail.asArray(Float.self), [5, 6])
    }
  }
}
