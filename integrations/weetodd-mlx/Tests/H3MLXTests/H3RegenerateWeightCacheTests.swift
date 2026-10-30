import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateWeightCacheTests: XCTestCase {
  func testOneBlockRetainsFourProjectionsButNeverAnotherBlock() throws {
    let cache=H3RegenerateBlockWeightCache()
    var builds=0
    func weight(_ block:Int,_ target:String) throws -> MLXArray {
      try cache.weight(block:block,target:target,shape:[4,3],dtype:.bfloat16) {
        builds += 1
        return MLXArray.ones([4,3],dtype:.bfloat16)*Float(builds)
      }
    }
    for target in ["qkv","out","fc1","fc2"] { _ = try weight(0,target) }
    XCTAssertEqual(builds,4);XCTAssertEqual(cache.count,4)
    XCTAssertEqual(try weight(0,"qkv").asType(.float32)[0,0].item(Float.self),1)
    XCTAssertEqual(builds,4)
    XCTAssertEqual(try weight(1,"qkv").asType(.float32)[0,0].item(Float.self),5)
    XCTAssertEqual(cache.count,1)
    cache.clear();XCTAssertEqual(cache.count,0)
  }
  func testFactoryFailureClearsRetainedWeightsAndBadGeometryCannotHitCache() throws {
    let cache=H3RegenerateBlockWeightCache()
    _ = try cache.weight(block:0,target:"qkv",shape:[4,3],dtype:.bfloat16) { .ones([4,3],dtype:.bfloat16) }
    XCTAssertThrowsError(try cache.weight(block:0,target:"qkv",shape:[3,4],dtype:.bfloat16) { .ones([3,4],dtype:.bfloat16) })
    enum Failure:Error { case injected }
    XCTAssertThrowsError(try cache.weight(block:0,target:"out",shape:[4,3],dtype:.bfloat16) { throw Failure.injected })
    XCTAssertEqual(cache.count,0)
  }
  func testMaterializedWeightMatchesBoundedPublisherProjectionAndCastOrder() throws {
    let base=MLXArray([Float(1.0078125),2,3,4],[2,2]).asType(.bfloat16)
    let pair=H3RegenerateProjection.Pair(a:MLXArray([Float(0.004),0.01],[1,2]),b:MLXArray([Float(1),2],[2,1]),scale:1)
    let input=MLXArray([Float(2),1,1,3],[1,2,2]).asType(.bfloat16)
    let weight=try H3RegenerateProjection.mergedWeight(pairs:[pair],layout:.native,rows:2,columns:2,dtype:.bfloat16,rowChunk:1) { base[$0] }
    let cached=matmul(input,weight.T)
    let bounded=try H3RegenerateProjection.projectMerged(input:input,pairs:[pair],layout:.native,rows:2,columns:2,rowChunk:1) { base[$0] }
    XCTAssertEqual(max(abs(cached.asType(.float32)-bounded.asType(.float32))).item(Float.self),0)
  }
}
