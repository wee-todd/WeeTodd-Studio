import XCTest
@testable import LTX25Text

final class TextAttentionTests: XCTestCase {
  func testNearLimitNonconstantAttentionAtRealHeadWidths() throws {
    let gpu = try TextMatrixGPU()
    for width in [128,256,512] {
      let q = (0..<(1024*2*width)).map { Float($0 % 29 - 14)/31 }
      let k = (0..<(1024*width)).map { Float($0 % 37 - 18)/41 }
      let v = (0..<(1024*width)).map { Float($0 % 43 - 21)/23 }
      let expected = try TextMath.attentionReference(q: q,k: k,v: v,tokens: 1024,
        heads: 2,kvHeads: 1,width: width,scale: 0.05,window: 257,causal: true,gpu: gpu)
      let actual = try gpu.attention(q: q,k: k,v: v,tokens: 1024,
        heads: 2,kvHeads: 1,width: width,scale: 0.05,window: 257,causal: true)
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,1e-5)
    }
  }
  func testCancellationDuringEncodingAndAfterCompletionAllowsCleanRetry() throws {
    let gpu = try TextMatrixGPU(), q = [Float](repeating: 0.1,count: 64*2*128)
    let k = [Float](repeating: 0.2,count: 64*128)
    func run(_ check: () throws -> Void = {}) throws -> [Float] {
      try gpu.attention(q: q,k: k,v: k,tokens: 64,heads: 2,kvHeads: 1,width: 128,
        scale: 1,window: nil,causal: true,checkCancelled: check)
    }
    let expected = try run(), baseline = gpu.device.currentAllocatedSize
    for stop in [2,4] {
      var checks = 0
      XCTAssertThrowsError(try run { checks += 1; if checks == stop { throw CancellationError() } })
      XCTAssertEqual(checks,stop)
      XCTAssertEqual(gpu.lastAttentionCommands,stop == 4 ? 1 : 0)
      XCTAssertEqual(gpu.lastAttentionReadbacks,0)
      XCTAssertEqual(try run(),expected)
    }
    // Metal may retain allocator pages; this excludes growth by live operands.
    XCTAssertLessThanOrEqual(gpu.device.currentAllocatedSize,baseline+4*1024*1024)
  }
  func testGPUAttentionMatchesReferenceForGroupedCausalAndBidirectionalMasks() throws {
    let gpu = try TextMatrixGPU()
    let q: [Float] = (0..<224).map { Float($0 % 19 - 9) / 13 }
    let k: [Float] = (0..<112).map { Float($0 % 17 - 8) / 11 }
    let v: [Float] = (0..<112).map { Float($0 % 23 - 11) / 7 }
    for causal in [true, false] {
      for window in [nil, 1, 3] as [Int?] {
        let expected = try TextMath.attentionReference(q: q, k: k, v: v, tokens: 7,
          heads: 4, kvHeads: 2, width: 8, scale: 0.3, window: window, causal: causal, gpu: gpu)
        let actual = try gpu.attention(q: q, k: k, v: v, tokens: 7,
          heads: 4, kvHeads: 2, width: 8, scale: 0.3, window: window, causal: causal)
        XCTAssertEqual(actual.count, expected.count)
        XCTAssertLessThan(zip(actual, expected).map { abs($0 - $1) }.max()!, 1e-5)
        XCTAssertEqual(gpu.lastAttentionReadbacks, 1)
        XCTAssertEqual(gpu.lastAttentionCommands, 1)
      }
    }
  }
  func testLongAttentionAndBudgetRejection() throws {
    let gpu = try TextMatrixGPU()
    let q = [Float](repeating: 0.01, count: 1024*2*8)
    let k = [Float](repeating: 0.02, count: 1024*8)
    let v = [Float](repeating: 0.75, count: k.count)
    let actual = try gpu.attention(q: q, k: k, v: v, tokens: 1024,
      heads: 2, kvHeads: 1, width: 8, scale: 1, window: nil, causal: false)
    XCTAssertTrue(actual.allSatisfy { abs($0 - 0.75) < 1e-5 })
    XCTAssertThrowsError(try gpu.attention(q: q, k: k, v: v, tokens: 1024,
      heads: 2, kvHeads: 1, width: 8, scale: 1, window: nil, causal: false, maximumWorkingBytes: 1))
    XCTAssertEqual(gpu.lastAttentionReadbacks, 0)
    XCTAssertEqual(gpu.lastAttentionCommands, 0)
  }
  func testInvalidInputsRejectBeforeEncoding() throws {
    let gpu = try TextMatrixGPU()
    XCTAssertThrowsError(try gpu.attention(q: [.nan], k: [1], v: [1], tokens: 1,
      heads: 1, kvHeads: 1, width: 1, scale: 1, window: nil, causal: true))
    XCTAssertThrowsError(try gpu.attention(q: [1], k: [1], v: [1], tokens: 1,
      heads: 1, kvHeads: 1, width: 1, scale: .infinity, window: nil, causal: true))
    XCTAssertEqual(gpu.lastAttentionCommands, 0)
  }
}
