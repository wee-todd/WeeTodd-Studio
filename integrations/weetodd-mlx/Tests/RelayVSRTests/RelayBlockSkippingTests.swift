import XCTest
import MLX
import MLXRandom
import InferenceContracts
@testable import RelayVSR

final class RelayBlockSkippingTests: XCTestCase {
  func testFusedSafetyGuardRejectsNonfiniteExtremeAndTinyOperands() {
    for dtype in [DType.bfloat16, .float32] {
      let valid = MLXArray([Float(-1), 0, 2, 1e-8]).asType(dtype)
      XCTAssertTrue(RelayInputSafety.evaluate(valid, valid, valid).item(Bool.self))
      for value in [Float.nan, .infinity, -.infinity, 1e30, 1e-20] {
        let invalid = MLXArray([Float(0), value, -1]).asType(dtype)
        XCTAssertFalse(RelayInputSafety.evaluate(invalid, valid, valid).item(Bool.self))
        XCTAssertFalse(RelayInputSafety.evaluate(valid, invalid, valid).item(Bool.self))
        XCTAssertFalse(RelayInputSafety.evaluate(valid, valid, invalid).item(Bool.self))
      }
    }
  }
  func testClassifiesVisibleMixedAndSafelySkippableBlocks() throws {
    let queries = (16..<24).flatMap { y in (16..<32).map { y * 64 + $0 } }
    let plane = RelayBlockPlane(rows: 5..<34, columns: 0..<51,
      window: RelayWindow(height: 22, width: 40, slidesAtEdge: true))
    let plan = try RelayBlockSkippingPlan(queries: queries, height: 64, width: 64,
      planes: [plane], keyBlockSize: 16)
    // Hand checked: Q0 covers y16,17; intersection y6..<27,x11..<40.
    // K block4 = y6,x13..<29; block91 = y33,x28..<44.
    XCTAssertEqual(plan.classes[0][4], 2)
    XCTAssertEqual(plan.classes[0][91], 0)
    XCTAssertEqual(plan.classes[0][92], 1, "A partial final key block keeps the fine mask.")
    XCTAssertEqual(plan.classes[3][0], 1, "Leading all-masked blocks keep finite_min startup arithmetic.")
  }

  func testCustomKernelKeepsEveryStockWordBeforeAndAfterSkipping() throws {
    for d in [64, 128] { for dtype in [DType.bfloat16, .float32] {
      let q = MLXArray((0..<(35 * 4 * d)).map { Float(sin(Double($0) * 0.017) * 0.3) }, [35, 4, d]).asType(dtype)
      let k = MLXArray((0..<(131 * 2 * d)).map { Float(cos(Double($0) * 0.031) * 0.4) }, [131, 2, d]).asType(dtype)
      let v = MLXArray((0..<(131 * 2 * d)).map { Float(sin(Double($0) * 0.071) * 0.7) }, [131, 2, d]).asType(dtype)
      // The second half of the queries sees a later range, exercising startup.
      let mask = MLXArray((0..<35).flatMap { q in (0..<131).map { k in
        q < 16 ? (0..<16).contains(k) : (32..<48).contains(k)
      } }, [35, 131])
      let bk = d == 64 ? 32 : 16, nk = (131 + bk - 1) / bk
      var classes = Array(repeating: UInt8(1), count: 2 * nk)
      // All real queries have seen keys before the last full masked block.
      classes[3] = 0; classes[nk + 3] = 0
      let metadata = MLXArray(classes, [2, nk])
      let expected = RelayAttentionPlan.attend(q, k, v, mask: mask)
      for enabled in [false, true] {
        let actual = try RelaySkippedAttention.apply(query: q, key: k, value: v,
          mask: mask, classes: metadata, allowSkipping: MLXArray(enabled))
        XCTAssertEqual(actual.asData(access: .copy).data, expected.asData(access: .copy).data,
          "D\(d), \(dtype), skip\(enabled)")
      }
    } }
  }

  func testZeroAndUnsafeValuesRetainStockWords() throws {
    let q = MLXArray.ones([32, 2, 128]).asType(.bfloat16)
    let k = MLXArray.ones([80, 1, 128]).asType(.bfloat16)
    let mask = MLXArray((0..<32).flatMap { _ in (0..<80).map { (16..<32).contains($0) } }, [32, 80])
    let classes = MLXArray([UInt8(1), 2, 0, 0, 0], [1, 5])
    for value in [Float(0), -Float.zero, Float.leastNormalMagnitude, Float(1e30), Float.infinity] {
      let v = MLXArray.full(k.shape, values: MLXArray(value)).asType(.bfloat16)
      let safe = RelaySkippedAttention.safeInputs(query: q, key: k, value: v)
      let actual = try RelaySkippedAttention.apply(query: q, key: k, value: v,
        mask: mask, classes: classes, allowSkipping: safe)
      let expected = RelayAttentionPlan.attend(q, k, v, mask: mask)
      XCTAssertEqual(actual.asData(access: .copy).data, expected.asData(access: .copy).data, "value \(value)")
    }
  }

  func testCancelledAttentionDoesNotSubmitOrRetainWork() async throws {
    let cancelled = try await Task.detached {
      let plan = RelayAttentionPlan(height: 9, width: 17,
        windows: [RelayWindow(height: 4, width: 4, slidesAtEdge: true)], headDimension: 64)
      let q = MLXArray.ones([153, 2, 64]).asType(.bfloat16)
      let k = MLXArray.ones([153, 1, 64]).asType(.bfloat16)
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try plan(q, [RelayKV(key: k, value: k)]); return false }
      catch is CancellationError { return true }
    }.value
    XCTAssertTrue(cancelled)
  }

  func testNativeAttentionBenchmarkWhenRequested() throws {
    guard let output = ProcessInfo.processInfo.environment["WEETODD_RELAY_SKIP_BENCHMARK"] else {
      throw XCTSkip("Opt in with an external benchmark output path.")
    }
    let lease = try NativeInferenceLease.acquire(); defer { lease.release() }
    Memory.memoryLimit = 12 * 1024 * 1024 * 1024
    Memory.cacheLimit = 3 * 1024 * 1024 * 1024
    MLXRandom.seed(42)
    var records: [[String: Any]] = []
    for (name, h, w, qh, kh, d) in [("wan2",48,84,24,24,128), ("wan4",96,168,24,24,128),
      ("flash2",96,168,8,2,64), ("flash4",192,336,8,2,64)] {
      try autoreleasepool {
        let window = RelayWindow(height: 22, width: 40, slidesAtEdge: true)
        let reference = RelayWindow(height: 48, width: 48, slidesAtEdge: false)
        let video = RelayWindow(height: 24, width: 24, slidesAtEdge: false)
        let plan = RelayAttentionPlan(height: h, width: w,
          windows: d == 128 ? Array(repeating: window, count: 4) : [reference,reference,video,video], headDimension: d)
        let n = h * w
        let q = MLXRandom.normal([n, qh, d]).asType(.bfloat16)
        let k = MLXRandom.normal([n * 4, kh, d]).asType(.bfloat16)
        let v = MLXRandom.normal([n * 4, kh, d]).asType(.bfloat16)
        eval(q,k,v)
        let planes = (0..<4).map { RelayKV(key: k[($0*n)..<(($0+1)*n)], value: v[($0*n)..<(($0+1)*n)]) }
        let expected = try plan.evaluate(q, planes, useBlockSkipping: false).asData(access: .copy).data
        XCTAssertEqual(try plan.evaluate(q, planes, useBlockSkipping: true).asData(access: .copy).data, expected)
        for pair in 0..<4 { for candidate in (pair % 2 == 0 ? [false,true] : [true,false]) {
          let start = ProcessInfo.processInfo.systemUptime
          let result = try plan.evaluate(q, planes, useBlockSkipping: candidate)
          eval(result); Stream.gpu.synchronize()
          let seconds = ProcessInfo.processInfo.systemUptime - start
          let exact = result.asData(access: .copy).data == expected
          XCTAssertTrue(exact, name)
          records.append(["case":name,"pair":pair,"candidate":candidate,"seconds":seconds,"exactWords":exact,
            "peakMLXBytes":Memory.peakMemory])
        } }
      }
      Memory.clearCache()
    }
    try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted,.sortedKeys])
      .write(to: URL(fileURLWithPath: output))
  }
}
