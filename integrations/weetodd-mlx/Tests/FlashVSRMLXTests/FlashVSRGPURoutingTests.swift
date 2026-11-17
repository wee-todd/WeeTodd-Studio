import XCTest
import MLX
import MLXRandom
import InferenceContracts
@testable import FlashVSRMLX

final class FlashVSRGPURoutingTests: XCTestCase {
  // Losing chronological keys, contracting a multiply-add, or selecting ties
  // changes these independently computed raw words and complete selections.
  func testRawDotsKeepTheHostSerialFloatWords() throws {
    for d in [8, 128] {
      let qs = [2, 3, 8, d], ks = [2, 3, 12, d]
      let qa = (0..<qs.reduce(1, *)).map { Float(sin(Double($0) * 0.113) * 3) }
      let ka = (0..<ks.reduce(1, *)).map { Float(cos(Double($0) * 0.173) * 2) }
      let candidates = [[0, 2, 7], [1], [0, 1, 8, 11], [3, 5], [9], [4, 6, 10], [1, 7], [2, 8, 9]]
      let actual = try FlashVSRGPURouting.rawScores(query: MLXArray(qa, qs),
        key: MLXArray(ka, ks), candidates: candidates).asArray(Float.self)
      var expected: [Float] = []
      for bh in 0..<6 { for row in 0..<8 { for key in candidates[row] {
        var value: Float = 0
        for c in 0..<d { value += qa[(bh * 8 + row) * d + c] * ka[(bh * 12 + key) * d + c] }
        expected.append(value)
      } } }
      XCTAssertEqual(actual.map(\.bitPattern), expected.map(\.bitPattern))
    }
  }

  func testGPUThresholdKeepsStrictTiesAndCapacityBoundary() throws {
    let candidates = [[0, 2], [1, 3]]
    let probabilities = MLXArray([Float(0.5), 0.5, 0.25, 0.75], [1, 4])
    XCTAssertEqual(try FlashVSRGPURouting.select(probabilities: probabilities,
      candidates: candidates, capacity: 4), [[[0, 2], [3]]])
    XCTAssertEqual(try FlashVSRGPURouting.select(probabilities: probabilities,
      candidates: candidates, capacity: 2), [[[0], [3]]])
    XCTAssertEqual(try FlashVSRGPURouting.select(probabilities: probabilities,
      candidates: candidates, capacity: 5), [[[0, 2], [1, 3]]])
  }

  func testCompleteRoutingMatchesOriginalForBothDtypesAndClippedEdges() throws {
    for dtype in [DType.float32, .bfloat16] {
      let q = MLXArray((0..<(2*2*8*4*128)).map { Float(sin(Double($0) * 0.31)) }, [2, 2, 8, 4, 128]).asType(dtype)
      let k = MLXArray((0..<(2*2*12*4*128)).map { Float(cos(Double($0) * 0.17)) }, [2, 2, 12, 4, 128]).asType(dtype)
      for range in [1, 3, 11] { for topK in [0, 7, 95, 200] {
        let expected = try FlashVSRAttention.routeReference(q: q, k: k, rows: 2, columns: 2, localRange: range, topK: topK)
        let actual = try FlashVSRGPURouting.route(q: q, k: k, rows: 2, columns: 2, localRange: range, topK: topK)
        XCTAssertEqual(actual, expected, "\(dtype), range\(range), topK\(topK)")
      } }
    }
    let zeros = MLXArray.zeros([1, 1, 4, 2, 128])
    XCTAssertEqual(try FlashVSRGPURouting.route(q: zeros, k: zeros, rows: 2,
      columns: 2, localRange: 11, topK: 7), [[[[0], [0], [0], [0]]]])
  }

  func testNonfiniteOverflowAndSubnormalsRetainTheOriginalContract() throws {
    for value in [Float.infinity, -.infinity, .nan, .greatestFiniteMagnitude] {
      let q = MLXArray.full([1, 1, 4, 1, 128], values: MLXArray(value))
      let k = MLXArray.full(q.shape, values: MLXArray(Float(2)))
      XCTAssertThrowsError(try FlashVSRGPURouting.route(q: q, k: k, rows: 2, columns: 2, localRange: 3, topK: 100))
    }
    let q = MLXArray.full([1, 1, 4, 1, 128], values: MLXArray(Float.leastNonzeroMagnitude))
    let k = MLXArray.full(q.shape, values: MLXArray(Float(0.5)))
    XCTAssertEqual(try FlashVSRGPURouting.route(q: q, k: k, rows: 2, columns: 2, localRange: 3, topK: 3),
      try FlashVSRAttention.routeReference(q: q, k: k, rows: 2, columns: 2, localRange: 3, topK: 3))
  }

  func testCancelledRoutingRejectsBeforeGPUWork() async throws {
    let cancelled = try await Task.detached {
      let q = MLXArray.ones([1, 1, 4, 1, 128]).asType(.bfloat16)
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try FlashVSRAttention.route(q: q, k: q, rows: 2, columns: 2, localRange: 3, topK: 4); return false }
      catch is CancellationError { return true }
    }.value
    XCTAssertTrue(cancelled)
  }

  func testNativeRoutingBenchmarkWhenRequested() throws {
    guard let output = ProcessInfo.processInfo.environment["WEETODD_FLASH_ROUTE_BENCHMARK"] else {
      throw XCTSkip("Opt in with an external JSON output path.")
    }
    let lease = try NativeInferenceLease.acquire(); defer { lease.release() }
    Memory.memoryLimit = 8 * 1024 * 1024 * 1024
    Memory.cacheLimit = 2 * 1024 * 1024 * 1024
    MLXRandom.seed(42)
    let q = MLXRandom.normal([1, 12, 504, 128, 128]).asType(.bfloat16)
    let k = MLXRandom.normal([1, 12, 1260, 128, 128]).asType(.bfloat16)
    eval(q, k)
    var records: [[String: Any]] = []
    for range in [3, 11] {
      let expected = try FlashVSRAttention.routeReference(q: q, k: k, rows: 12, columns: 21, localRange: range, topK: 158759)
      XCTAssertEqual(try FlashVSRGPURouting.route(q: q, k: k, rows: 12, columns: 21, localRange: range, topK: 158759), expected)
      for pair in 0..<4 {
        for candidate in (pair % 2 == 0 ? [false, true] : [true, false]) {
          let start = ProcessInfo.processInfo.systemUptime
          let result = try candidate
            ? FlashVSRGPURouting.route(q: q, k: k, rows: 12, columns: 21, localRange: range, topK: 158759)
            : FlashVSRAttention.routeReference(q: q, k: k, rows: 12, columns: 21, localRange: range, topK: 158759)
          let seconds = ProcessInfo.processInfo.systemUptime - start
          XCTAssertEqual(result, expected)
          records.append(["range": range, "pair": pair, "candidate": candidate, "seconds": seconds,
            "peakMLXBytes": Memory.peakMemory, "exactSelection": result == expected])
        }
      }
    }
    try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: output))
  }
}
