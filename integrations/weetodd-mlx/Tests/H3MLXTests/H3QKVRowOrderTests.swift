import MLX
import XCTest
@testable import H3MLX

final class H3QKVRowOrderTests: XCTestCase {
  func testRawFL2VAQKVRemainsPerHeadAndComfyGroupedRowsConvert() {
    // Python's raw pruned loader leaves the row order [h0:q,k,v][h1:q,k,v].
    let raw = MLXArray((0..<12).map(Float.init)).reshaped([12, 1])
    let direct = H3QKVRowOrder.forHeadMajorAttention(raw,
      heads: 2, headWidth: 2, groupedSource: false)
    XCTAssertEqual(direct.reshaped([12]).asArray(Float.self),
      (0..<12).map(Float.init))
    // A Comfy grouped tensor is [q0,q1,k0,k1,v0,v1] by head-width pair.
    let comfy = H3QKVRowOrder.forHeadMajorAttention(raw,
      heads: 2, headWidth: 2, groupedSource: true)
    XCTAssertEqual(comfy.reshaped([12]).asArray(Float.self),
      [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
  }
}
