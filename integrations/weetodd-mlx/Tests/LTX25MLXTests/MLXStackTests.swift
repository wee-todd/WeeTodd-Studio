import XCTest
import MLX
import LTX25MLX
import LTX25Engine

final class MLXStackTests: XCTestCase {
  private func configuration() throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:5,audioTokens:3,textTokens:4)
  }
  func testStreamsAcrossBlocksAndReleasesBeforeNextWeightLoad() throws {
    let stack=try MLXAVStack(configuration:configuration(),blockCount:3,cacheBytes:1024)
    let layout=try MLXAVBlock(configuration:configuration())
    var inputs=layout.inputShapes.mapValues { MLXArray.zeros($0) }
    inputs["video"] = .ones([5,32]); inputs["audio"] = .ones([3,16])
    var completed:[Int]=[], firstWeightInBlock:Set<Int>=[]
    let previousLimit=Memory.cacheLimit
    let result=try stack.evaluate(inputs,weights:{ index,name,shape in
      if firstWeightInBlock.insert(index).inserted {
        XCTAssertEqual(stack.residentWeightBytes,0,"Previous block must be released before its replacement")
        XCTAssertLessThanOrEqual(Memory.cacheMemory,1024,"Releasing weights must not leave an oversized cache for the next block")
      }
      return try MLXWeight(dense:.zeros(shape))
    },progress:{ completed.append($0.completedBlocks) })
    XCTAssertEqual(completed,[1,2,3])
    XCTAssertEqual(result["video"]!.asArray(Float.self),[Float](repeating:1,count:160))
    XCTAssertEqual(result["audio"]!.asArray(Float.self),[Float](repeating:1,count:48))
    XCTAssertEqual(stack.residentWeightBytes,0)
    XCTAssertEqual(Memory.cacheLimit,previousLimit)
    XCTAssertLessThanOrEqual(Memory.cacheMemory,1024*1024)
  }
  func testObserverFailureReleasesWeightsAndAllowsRetry() throws {
    let stack=try MLXAVStack(configuration:configuration(),blockCount:2)
    let layout=try MLXAVBlock(configuration:configuration())
    let inputs=layout.inputShapes.mapValues { MLXArray.zeros($0) }
    let weights: (Int,String,[Int]) throws -> MLXWeight = { _,_,shape in try MLXWeight(dense:.zeros(shape)) }
    XCTAssertThrowsError(try stack.evaluate(inputs,weights:weights,progress:{ _ in throw LTXError.invalid("observer") }))
    XCTAssertEqual(stack.residentWeightBytes,0)
    XCTAssertNoThrow(try stack.evaluate(inputs,weights:weights))
    XCTAssertEqual(stack.residentWeightBytes,0)
  }
  func testInvalidInputDoesNotLoadAnyWeights() throws {
    let stack=try MLXAVStack(configuration:configuration())
    var loads=0
    XCTAssertThrowsError(try stack.evaluate([:],weights:{ _,_,shape in
      loads += 1; return try MLXWeight(dense:.zeros(shape))
    }))
    XCTAssertEqual(loads,0)
  }
  func testOneGiBCacheIsAdmittedButUnboundedCacheIsRejected() throws {
    XCTAssertNoThrow(try MLXAVStack(configuration:configuration(),cacheBytes:1024*1024*1024))
    XCTAssertThrowsError(try MLXAVStack(configuration:configuration(),cacheBytes:1024*1024*1024+1))
  }
  @MainActor func testCancellationAfterBlockCleansUp() async throws {
    let c=try configuration()
    let canceled=try await Task {
      let stack=try MLXAVStack(configuration:c,blockCount:3)
      let layout=try MLXAVBlock(configuration:c)
      let inputs=layout.inputShapes.mapValues { MLXArray.zeros($0) }
      do {
        _=try stack.evaluate(inputs,weights:{ _,_,shape in try MLXWeight(dense:.zeros(shape)) },
          progress:{ _ in withUnsafeCurrentTask { $0?.cancel() } })
        return false
      } catch is CancellationError { return stack.residentWeightBytes == 0 }
    }.value
    XCTAssertTrue(canceled)
  }
}
