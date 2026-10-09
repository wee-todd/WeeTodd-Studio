import XCTest
import MLX
import MLXRandom
@testable import H3MLX

final class H3TileSummaryTests: XCTestCase {
  func testCancelledSummaryRejectsWorkBeforeKernelSubmission() async throws {
    let cancelled = Task { () throws -> Void in
      withUnsafeCurrentTask { $0?.cancel() }
      let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[1,1,1])
      let x = MLXArray.ones([1,1,tiles.rows,128],dtype:.bfloat16)
      _ = try H3TileSummary.evaluate(query:x,key:x,value:x,tiles:tiles)
    }
    do { try await cancelled.value; XCTFail("Cancelled summary executed.") }
    catch is CancellationError { }
  }
  private func reference(_ x: MLXArray, tiles: H3FastTiles) -> MLXArray {
    let count = tiles.sizes.count
    let mask = MLXArray(tiles.sizes.flatMap { size in
      (0..<64).map { Float($0 < size ? 1 : 0) }
    }, [1,1,count*64,1]).asType(x.dtype)
    return (take(x, MLXArray(tiles.indices), axis: 2) * mask).asType(.float32)
      .reshaped([1,x.shape[1],count,64,128]).sum(axis:3)
      / MLXArray(tiles.sizes.map(Float.init),[1,1,count,1])
  }

  func testMappedSummariesPreserveNativeReductionWithPartialTilesAndStrides() throws {
    let tiles = try H3FastTiles(prefixSegments:[3,65],videoGrid:[7,5,6])
    let backing = MLXRandom.normal([1,6,tiles.rows*2,256],
      key:MLXRandom.key(961)).asType(.bfloat16)
    let q = backing[0..<1,.stride(by:2),.stride(by:2),0..<128]
    let k = backing[0..<1,.stride(by:2),.stride(by:2),128..<256]
    let fused = MLXRandom.normal([1,tiles.rows,3,3,128],
      key:MLXRandom.key(962)).asType(.bfloat16)
    let v = fused[.ellipsis,2,0..<128].transposed(0,2,1,3)
    for inputs in [[q,k,v], [q[.ellipsis,.stride(by:-1)],k,v]] {
      let result = try H3TileSummary.evaluate(query:inputs[0],key:inputs[1],
        value:inputs[2],tiles:tiles)
      for (actual,input) in zip(result,inputs) {
        XCTAssertEqual(actual.dtype,.float32)
        XCTAssertEqual(actual.shape,[1,3,tiles.sizes.count,128])
        XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),
          reference(input,tiles:tiles).asArray(Float.self).map(\.bitPattern))
      }
    }
  }

  func testWideExponentCancellationAndBroadcastRowsRemainExact() throws {
    let tiles = try H3FastTiles(prefixSegments:[64],videoGrid:[4,4,4])
    let data = (0..<(tiles.rows*128)).map { i -> Float in
      let exponent = (i / 128) % 101 - 50
      let sign: Float = i.isMultiple(of:3) ? -1 : 1
      return sign * pow(Float(2),Float(exponent)) * Float(i%7+1)
    }
    let x = MLXArray(data,[1,1,tiles.rows,128]).asType(.bfloat16)
    let broadcasted = broadcast(x,to:[1,3,tiles.rows,128])
    let result = try H3TileSummary.evaluate(query:broadcasted,key:broadcasted,
      value:broadcasted,tiles:tiles)
    let expected = reference(broadcasted,tiles:tiles).asArray(Float.self).map(\.bitPattern)
    for actual in result { XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),expected) }
  }

  func testCPUFallbackUsesExistingPoolingAndRejectsInvalidShape() throws {
    try Device.withDefaultDevice(.cpu) {
      let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[1,1,1])
      let x = MLXArray.ones([1,1,tiles.rows,128],dtype:.bfloat16)
      let result = try H3TileSummary.evaluate(query:x,key:x,value:x,tiles:tiles)
      XCTAssertEqual(result[0].asArray(Float.self),reference(x,tiles:tiles).asArray(Float.self))
      XCTAssertThrowsError(try H3TileSummary.evaluate(query:x,key:x,
        value:MLXArray.zeros([1,1,tiles.rows,64]),tiles:tiles))
      let excessHeads = MLXArray.ones([1,57,tiles.rows,128],dtype:.bfloat16)
      XCTAssertThrowsError(try H3TileSummary.evaluate(query:excessHeads,key:excessHeads,
        value:excessHeads,tiles:tiles))
    }
  }
}
