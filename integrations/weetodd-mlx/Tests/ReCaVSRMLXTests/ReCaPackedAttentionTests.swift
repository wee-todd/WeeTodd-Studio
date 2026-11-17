import XCTest
import MLX
@testable import ReCaVSRMLX

final class ReCaPackedAttentionTests: XCTestCase {
  private func samples(_ shape: [Int], dtype: DType) -> MLXArray {
    MLXArray((0..<shape.reduce(1, *)).map { Float(sin(Double($0) * 0.017) * 0.7) }, shape).asType(dtype)
  }
  func testPackedInputsMasksAndAttentionKeepEveryWord() throws {
    for dtype in [DType.float32, .bfloat16] {
      for shifted in [false, true] {
        let qf = shifted ? 2 : 4, kf = shifted ? 6 : 8, span = 9 * 17
        let q = samples([span * qf, 4, 64], dtype: dtype)
        let k = samples([span * kf, 2, 64], dtype: dtype)
        let v = samples(k.shape, dtype: dtype)
        let plan = try ReCaPackedTilePlan(height: 9, width: 17, windowHeight: 6,
          windowWidth: 8, slidesAtEdge: shifted)
        var parts: [Int: MLXArray] = [:], order: [Int: [Int]] = [:]
        for group in plan.groups {
          let packed = try plan.pack(group, query: q, key: k, value: v)
          for (i, tile) in group.tiles.enumerated() {
            let spatial = (tile.top..<tile.top + tile.qr).flatMap { y in
              (tile.left..<tile.left + tile.qc).map { y * 17 + $0 }
            }
            let keys = (tile.kt..<tile.kt + tile.kr).flatMap { y in
              (tile.kl..<tile.kl + tile.kc).map { y * 17 + $0 }
            }
            let qi = (0..<qf).flatMap { t in spatial.map { t * span + $0 } }
            let ki = (0..<kf).flatMap { t in keys.map { t * span + $0 } }
            func bounds(_ p: Int, _ extent: Int, _ window: Int) -> Range<Int> {
              if shifted {
                let count = min(extent, window), start = min(max(0, p - count / 2), extent - count)
                return start..<start + count
              }
              return max(0, p - window / 2)..<min(extent, p + window / 2)
            }
            var mask: [Bool] = []
            for _ in 0..<qf { for query in spatial { for _ in 0..<kf { for key in keys {
              mask.append(bounds(query / 17, 9, 6).contains(key / 17)
                && bounds(query % 17, 17, 8).contains(key % 17))
            } } } }
            let expected = [q.take(MLXArray(qi.map(Int32.init)), axis: 0),
              k.take(MLXArray(ki.map(Int32.init)), axis: 0),
              v.take(MLXArray(ki.map(Int32.init)), axis: 0), MLXArray(mask, [qi.count, ki.count])]
            for field in packed.indices {
              XCTAssertEqual(packed[field][i].asData(access: .copy).data,
                expected[field].asData(access: .copy).data, "field \(field), tile \(tile.id)")
            }
            parts[tile.id] = ReCaAttentionPlan.attend(expected[0], expected[1], expected[2], mask: expected[3])
            order[tile.id] = qi
          }
        }
        let positions = (0..<plan.tiles.count).flatMap { order[$0]! }
        var inverse = Array(repeating: Int32(0), count: positions.count)
        for (i, token) in positions.enumerated() { inverse[token] = Int32(i) }
        let expected = concatenated((0..<plan.tiles.count).map { parts[$0]! }, axis: 0)
          .take(MLXArray(inverse), axis: 0)
        let actual = try plan.apply(query: q, key: k, value: v)
        XCTAssertEqual(actual.asData(access: .copy).data, expected.asData(access: .copy).data)
      }
    }
  }
  func testPackingAdmissionCancellationAndResetAreBounded() async throws {
    let plan = try ReCaPackedTilePlan(height: 9, width: 17, windowHeight: 6,
      windowWidth: 8, slidesAtEdge: true)
    let q = samples([9 * 17 * 2, 2, 128], dtype: .bfloat16)
    let k = samples([9 * 17 * 6, 2, 128], dtype: .bfloat16)
    let v = samples(k.shape, dtype: .bfloat16)
    XCTAssertThrowsError(try plan.apply(query: q, key: k, value: v, maximumWorkspaceBytes: 1))
    let wrapper = ReCaAttentionPlan(height: 9, width: 17, windowHeight: 6,
      windowWidth: 8, slidesAtEdge: true)
    _ = try wrapper.apply(query: q, key: k, value: v)
    XCTAssertNotNil(wrapper.packed)
    wrapper.reset()
    XCTAssertNil(wrapper.packed)
    XCTAssertThrowsError(try ReCaPackedTilePlan(height: 0, width: 17, windowHeight: 6,
      windowWidth: 8, slidesAtEdge: true))
    let cancelled = try await Task.detached {
      let plan = try ReCaPackedTilePlan(height: 9, width: 17, windowHeight: 6,
        windowWidth: 8, slidesAtEdge: true)
      let q = MLXArray.zeros([9 * 17 * 2, 2, 128])
      let k = MLXArray.zeros([9 * 17 * 6, 2, 128])
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try plan.apply(query: q, key: k, value: k); return false }
      catch is CancellationError { return true }
    }.value
    XCTAssertTrue(cancelled)
  }
}
