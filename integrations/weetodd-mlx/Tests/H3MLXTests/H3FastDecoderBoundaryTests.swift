import XCTest
import MLX
@testable import H3MLX

final class H3FastDecoderBoundaryTests:XCTestCase {
  func testFull2KStateAdmissionAnd4KRejection() throws {
    try H3FastVideoDecoder.validateGeometry(frames:37,height:90,width:160,outputFrames:5)
    XCTAssertLessThan(try H3FastVideoDecoder.estimatedActiveBytes(frames:37,height:90,width:160,outputFrames:37),64*1024*1024*1024)
    let plan=try H3FastDecoderWindows.plan(frames:37,height:90,width:160,layer:1)
    XCTAssertEqual(plan.tokenCount,532800)
    XCTAssertEqual(plan.windows.reduce(0){$0+$1.tokens.count},532800)
    XCTAssertEqual(Set(plan.inversePermutation).count,532800)
    XCTAssertThrowsError(try H3FastVideoDecoder.validateGeometry(frames:37,height:180,width:320,outputFrames:5))
    XCTAssertThrowsError(try H3FastVideoDecoder.validateGeometry(frames:37,height:135,width:240,outputFrames:5))
    XCTAssertThrowsError(try H3FastVideoDecoder.validateGeometry(frames:37,height:90,width:160,outputFrames:37,feedForwardChunkTokens:532800))
  }
  func testChunkedQKVMatchesSingleProjection() throws {
    let x=MLXArray((0..<88).map { Float($0%17-8)/10 },[1,11,8])
    let weights:[String:MLXArray]=[
      "norm1.weight":MLXArray((0..<8).map { Float($0+1)/8 }),
      "attn.to_qkv.weight":MLXArray((0..<192).map { Float($0%23-11)/20 },[24,8]),
      "attn.to_qkv.bias":MLXArray((0..<24).map { Float($0)/50 })]
    let packed=try H3FastVideoDecoder.linear(H3FastVideoDecoder.rms(x,weight:weights["norm1.weight"]!),"attn.to_qkv",weights).reshaped([1,11,2,3,4])
    let expected=[H3FastVideoDecoder.rms(packed[.ellipsis,0,0..<4]),H3FastVideoDecoder.rms(packed[.ellipsis,1,0..<4]),packed[.ellipsis,2,0..<4]]
    let actual=try H3FastVideoDecoder.projectQKV(x,weights:weights,heads:2,headDimension:4,chunkTokens:3)
    for (a,b) in zip([actual.0,actual.1,actual.2],expected) { XCTAssertLessThan(max(abs(a-b)).item(Float.self),1e-6) }
  }
  func testSelectedHeadPreservesGlobalTemporalTrimPhase() throws {
    let head=MLXArray((0..<(12*3072)).map(Float.init),[1,12,1,1,3072])
    let whole=try H3FastVideoDecoder.reconstruct(head)
    let selected=try H3FastVideoDecoder.reconstruct(head[0...,3..<7],latentOffset:3)
    let retained=try H3FastDecoderWindows.retainedFrames(latentFrames:12)
    let rows=retained.enumerated().filter{(3..<7).contains($0.element/4)}.map{Int32($0.offset)}
    XCTAssertEqual(selected.shape,[1,13,16,16,3])
    XCTAssertEqual(selected.asArray(Float.self),take(whole,MLXArray(rows),axis:1).asArray(Float.self))
  }
}
