import XCTest
import ImageLayout
import MLX
import MLXRandom
@testable import MOSSImageMLX

final class RegionalAttentionTests: XCTestCase {
  func testOwnedPropOutsideTheBodyStillReceivesItsSubjectConditioning() throws {
    let body = NormalizedLayoutRect(x: 0, y: 0, width: 0.5, height: 1), prop = NormalizedLayoutRect(x: 0.5, y: 0.5, width: 0.01, height: 0.5)
    let groups: [MOSSRegionalPrompt] = [.init(role: .background, bounds: nil, prompt: "Wall"),
      .init(role: .subject, bounds: body, prompt: "Woman and held prop", masks: [body,prop])]
    let plan = try MOSSRegionalAttentionPlan(height: 2, width: 4, groups: groups, textLengths: [1,1])
    let outside = plan.partitions.first { $0.queries.contains(6) }!
    XCTAssertTrue(outside.keys.contains(9)); XCTAssertTrue(outside.keys.contains(8))
  }
  func testLetteringOnlyReachesItsAuthoredTextAreas() throws {
    let groups: [MOSSRegionalPrompt] = [.init(role: .background, bounds: nil, prompt: "Paper"),
      .init(role: .lettering, bounds: nil, prompt: "Text", masks: [.init(x: 0, y: 0, width: 1, height: 0.25)])]
    let plan = try MOSSRegionalAttentionPlan(height: 4, width: 4, groups: groups, textLengths: [1,1])
    XCTAssertTrue(plan.partitions.first { $0.queries.contains(0) }!.keys.contains(17))
    XCTAssertFalse(plan.partitions.first { $0.queries.contains(12) }!.keys.contains(17))
  }
  func testRegionalRoPEPreservesEachStreamImageTextDotProduct() {
    let lengths = [2,4], rows = 2
    let combined = MOSSMath.regionalRope(height: 1, width: 2, textLengths: lengths, headDimension: 8)
    let all = MOSSMath.rotate(MLXArray.ones([1,8,1,8]),rope: combined)
    var start = rows
    for length in lengths {
      let local = MOSSMath.rotate(MLXArray.ones([1,rows+length,1,8]),rope: MOSSMath.rope(height: 1,width:2,textLength:length,headDimension:8))
      for j in 0..<length {
        let actual = sum(all[0,0,0,0...] * all[0,start+j,0,0...]).item(Float.self)
        let expected = sum(local[0,0,0,0...] * local[0,rows+j,0,0...]).item(Float.self)
        XCTAssertEqual(actual, expected, accuracy: 1e-5)
      }
      start += length
    }
  }
  func testPartitionsCoverEveryQueryAndRestrictRegionalText() throws {
    let groups: [MOSSRegionalPrompt] = [.init(role: .background, bounds: nil, prompt: "Wall"),
      .init(role: .subject, bounds: .init(x: 0.5, y: 0, width: 0.5, height: 1), prompt: "Woman"),
      .init(role: .lettering, bounds: nil, prompt: "Title")]
    let plan = try MOSSRegionalAttentionPlan(height: 2, width: 4, groups: groups, textLengths: [2,3,1])
    XCTAssertEqual(plan.tokenCount, 14)
    XCTAssertEqual(plan.partitions.flatMap(\.queries).sorted(), Array(0..<14))
    let left = plan.partitions.first { $0.queries.contains(0) }!, right = plan.partitions.first { $0.queries.contains(3) }!
    XCTAssertTrue(left.keys.contains(8)); XCTAssertFalse(left.keys.contains(10)); XCTAssertTrue(left.keys.contains(13))
    XCTAssertTrue(right.keys.contains(10)); XCTAssertTrue(right.keys.contains(8)); XCTAssertTrue(right.keys.contains(13))
    let subject = plan.partitions.first { $0.queries.contains(10) }!
    XCTAssertEqual(subject.keys, [2,3,6,7,10,11,12])
  }
  func testPartitionedAttentionMatchesExplicitDenseMask() throws {
    let groups: [MOSSRegionalPrompt] = [.init(role: .background, bounds: nil, prompt: "Wall"),
      .init(role: .subject, bounds: .init(x: 0.5, y: 0, width: 0.5, height: 1), prompt: "Woman")]
    let plan = try MOSSRegionalAttentionPlan(height: 2, width: 2, groups: groups, textLengths: [2,2])
    let q = MLXRandom.normal([1,2,8,4], key: MLXRandom.key(11)), k = MLXRandom.normal([1,2,8,4], key: MLXRandom.key(12)), v = MLXRandom.normal([1,2,8,4], key: MLXRandom.key(13))
    var mask = [Bool](repeating: false, count: 64)
    for p in plan.partitions { for i in p.queries { for j in p.keys { mask[i*8+j] = true } } }
    let expected = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: MLXArray(mask).reshaped(8,8))
    let actual = MOSSRegionalAttention.apply(queries: q, keys: k, values: v, scale: 0.5, plan: .init(plan))
    XCTAssertLessThan(max(abs(actual-expected)).item(Float.self), 1e-5)
  }
}
