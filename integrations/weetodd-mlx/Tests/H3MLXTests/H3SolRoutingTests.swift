import Foundation
import CryptoKit
import MLX
import XCTest
@testable import H3MLX

final class H3SolRoutingTests: XCTestCase {
  func testProtectedCrossingsRadiusAndTrueTailCounts() throws {
    let g = try H3SolGeometry(rows: 193, heads: 1, approximationRange: 65..<193)
    XCTAssertEqual(g.keyBlocks, 4)
    XCTAssertEqual(g.queryBlocks, 4)
    XCTAssertEqual(g.routeWords, 1)
    XCTAssertEqual((0..<4).map(g.keyCount), [64, 64, 64, 1])
    XCTAssertTrue(g.requiresExact(queryBlock: 1, keyBlock: 3)) // crossing query
    XCTAssertTrue(g.requiresExact(queryBlock: 3, keyBlock: 1)) // crossing key
    XCTAssertTrue(g.requiresExact(queryBlock: 2, keyBlock: 3)) // local
    let all = try H3SolGeometry(rows: 257, heads: 1, approximationRange: 0..<257)
    XCTAssertFalse(all.requiresExact(queryBlock: 0, keyBlock: 2))
    XCTAssertTrue(all.requiresExact(queryBlock: 0, keyBlock: 1))
  }

  func testInvalidGeometryFailsWithoutAnArrayOrCheckpoint() {
    for rows in [0, -1, 40001, Int.max] {
      XCTAssertThrowsError(try H3SolGeometry(rows: rows, heads: 1, approximationRange: 0..<1))
    }
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 0, approximationRange: 0..<64))
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 1, dimension: 64, approximationRange: 0..<64))
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 1, blockSize: 32, approximationRange: 0..<64))
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 1, approximationRange: 0..<65))
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 1, approximationRange: 0..<64, localRadius: 0))
    XCTAssertThrowsError(try H3SolGeometry(rows: 64, heads: 1, approximationRange: 0..<64, scale: .nan))
  }

  func testStrictThresholdAndBaseTwoEpsilon() {
    XCTAssertFalse(H3SolRouting.select(scoreBase2: 2, thresholdBase2: 2, forced: false))
    XCTAssertTrue(H3SolRouting.select(scoreBase2: Float(2).nextUp, thresholdBase2: 2, forced: false))
    XCTAssertTrue(H3SolRouting.select(scoreBase2: -10, thresholdBase2: 2, forced: true))
    XCTAssertEqual(H3SolRouting.threshold(mu: 0, projectedVariance: 0, scale: 0, tau: 0.5), 0.0005, accuracy: 1e-9)
  }

  func testEqualBlockStatisticsAndRoundedCoarseMeansIncludePartialTail() throws {
    let g = try H3SolGeometry(rows: 65, heads: 1, approximationRange: 0..<65)
    let q = [Float](repeating: 1, count: 65 * 128)
    var k = [Float](repeating: 2, count: 65 * 128)
    for d in 0..<128 { k[64 * 128 + d] = 6 }
    let v = [Float](repeating: 3, count: k.count)
    let r = try H3SolRouting.cpuOracle(query: q, key: k, value: v, geometry: g)
    XCTAssertEqual(r.keyMeans[0], 2)
    XCTAssertEqual(r.keyMeans[128], 6)
    XCTAssertEqual(r.keyStats[0], 4) // equal two block means, not 64:1 weights
    XCTAssertEqual(r.keyStats[1], 4)
    XCTAssertEqual(r.valueMeans[128], 3)
    XCTAssertEqual(r.exactCounts, [2, 2]) // radius forces both real blocks
    XCTAssertEqual(r.exactRouteBits, [3, 3])
  }

  func testVarianceOracleMatchesSingleRoundedMultiplySubtractWords() throws {
    let g = try H3SolGeometry(rows: 129, heads: 1, approximationRange: 0..<129)
    let count = 129 * 128
    let zeros = [Float](repeating: 0, count: count)
    let key = (0..<count).map { Float(($0 % 7) - 3) / 4 }
    let result = try H3SolRouting.cpuOracle(query: zeros, key: key, value: zeros, geometry: g)
    // Captured tiny-fixture features: the moment and mean divisions are
    // already Float32 rounded. Only the final multiplication/subtraction
    // differs between two roundings and one fused rounding.
    let witnesses: [(feature: Int, mean: UInt32, moment: UInt32,
      fused: UInt32, separate: UInt32)] = [
      (3, 0xbe7d5555, 0x3e400555, 0x3e0158e3, 0x3e0158e4),
      (5, 0xbdad5555, 0x3cab3555, 0x3c610e39, 0x3c610e38)
    ]
    for witness in witnesses {
      let mean = Float(bitPattern: witness.mean)
      let moment = Float(bitPattern: witness.moment)
      let fused = moment.addingProduct(-mean, mean)
      // Double represents these two Float32 operands and their product
      // exactly, giving an independent single-rounding reference here.
      let singleRounded = Float(Double(moment) - Double(mean) * Double(mean))
      XCTAssertEqual(fused.bitPattern, witness.fused)
      XCTAssertEqual(singleRounded.bitPattern, witness.fused)
      XCTAssertEqual(result.keyStats[witness.feature * 2].bitPattern, witness.mean)
      XCTAssertEqual(result.keyStats[witness.feature * 2 + 1].bitPattern, witness.fused)
      XCTAssertEqual(abs(Int64(witness.fused) - Int64(witness.separate)), 1)
    }
  }

  func testOracleRejectsNonfiniteAndInvalidLengthsAndRoundsBF16TiesEven() throws {
    let g = try H3SolGeometry(rows: 1, heads: 1, approximationRange: 0..<1)
    let valid = [Float](repeating: 1, count: 128)
    var invalid = valid; invalid[5] = .infinity
    XCTAssertThrowsError(try H3SolRouting.cpuOracle(query: invalid, key: valid, value: valid, geometry: g))
    XCTAssertThrowsError(try H3SolRouting.cpuOracle(query: [], key: valid, value: valid, geometry: g))
    XCTAssertEqual(H3SolRouting.roundBF16(Float(bitPattern: 0x3f808000)).bitPattern, 0x3f800000)
    XCTAssertEqual(H3SolRouting.roundBF16(Float(bitPattern: 0x3f818000)).bitPattern, 0x3f820000)
  }

  func testOracleChecksActualTaskCancellationBeforePooling() async throws {
    let g = try H3SolGeometry(rows: 65, heads: 1, approximationRange: 0..<65)
    let values = [Float](repeating: 1, count: 65 * 128)
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return try H3SolRouting.cpuOracle(query: values, key: values, value: values, geometry: g)
    }
    do {
      _ = try await task.value
      XCTFail("A cancelled oracle must not complete pooling.")
    } catch { XCTAssertTrue(error is CancellationError) }
  }

  func testBoundedNumericAdmissionRejectsEveryInvalidSummaryFieldWithoutGPU() throws {
    try H3SolRouting.validatePooledAdmission([0,1_000_000,1_000_000,1_000_000], heads: 1, blocks: 1)
    for malformed in [[], [Float](repeating: 0, count: 3), [Float](repeating: 0, count: 5)] {
      XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission(malformed, heads: 1, blocks: 1))
    }
    XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission([], heads: 0, blocks: 1))
    XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission([], heads: 1, blocks: 626))
    for flag in [Float(1), -1, .nan, .infinity] {
      XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission([flag,0,0,0], heads: 1, blocks: 1))
    }
    for field in 1...3 {
      let bound: Float = 1_000_000
      for invalid in [Float(-1), .nan, .infinity, bound.nextUp] {
        var summary: [Float] = [0,0,0,0]; summary[field] = invalid
        XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission(summary, heads: 1, blocks: 1))
      }
    }
    // A bad final tuple cannot be missed by a leading valid tuple.
    XCTAssertThrowsError(try H3SolRouting.validatePooledAdmission([0,0,0,0,1,0,0,0], heads: 2, blocks: 1))
  }

  func testAdmissionFailureReportsOffendingHeadBlockAndTuple() {
    do {
      try H3SolRouting.validatePooledAdmission([0,0,0,0,0,0,0,0,0,0,0,0,1,32,64,128], heads: 2, blocks: 2)
      XCTFail("A nonfinite flag must be rejected.")
    } catch {
      let detail = String(describing: error)
      XCTAssertTrue(detail.contains("head 1, key block 1"))
      XCTAssertTrue(detail.contains("nonfinite flag=1.0"))
      XCTAssertTrue(detail.contains("max|Q|=32.0"))
      XCTAssertTrue(detail.contains("max|K|=64.0"))
      XCTAssertTrue(detail.contains("max|V|=128.0"))
    }
  }

  func testInstalledFiniteOperandsAboveLegacy16RemainUnchanged() throws {
    try admitPinnedMetal()
    let g = try H3SolGeometry(rows: 65, heads: 1, approximationRange: 0..<65)
    let result = try H3SolRouting.prepare(
      query: MLXArray([Float](repeating: 32, count: 65 * 128), [1,1,65,128]).asType(.bfloat16),
      key: MLXArray([Float](repeating: 64, count: 65 * 128), [1,1,65,128]).asType(.bfloat16),
      value: MLXArray([Float](repeating: 128, count: 65 * 128), [1,1,65,128]).asType(.bfloat16), geometry: g)
    XCTAssertEqual(result.queryMeansF32.asArray(Float.self), [Float](repeating: 32, count: 256))
    XCTAssertEqual(result.keyMeansBF16.asType(.float32).asArray(Float.self), [Float](repeating: 64, count: 256))
    XCTAssertEqual(result.valueMeansBF16.asType(.float32).asArray(Float.self), [Float](repeating: 128, count: 256))
    let expected: [Float] = (0..<128).flatMap { _ -> [Float] in [64, 0] }
    XCTAssertEqual(result.keyStatsF32.asArray(Float.self), expected)
    XCTAssertEqual(result.exactRouteBits.asArray(UInt32.self), [3,3])
    XCTAssertEqual(result.exactCounts.asArray(UInt32.self), [2,2])
  }

  func testInstalledBoundedPoolingRejectsNonfiniteAndOutOfRangeInEveryOperand() throws {
    try admitPinnedMetal()
    let g = try H3SolGeometry(rows: 129, heads: 2, approximationRange: 0..<129)
    let count = 2 * 129 * 128
    for operand in 0..<3 {
      let overflow: Float = 1_048_576
      for invalid in [Float.nan, .infinity, -.infinity, overflow, -overflow] {
        var words = [[Float]](repeating: [Float](repeating: 0, count: count), count: 3)
        words[operand][count-1] = invalid // last head, ragged row, final feature
        XCTAssertThrowsError(try H3SolRouting.prepare(
          query: MLXArray(words[0], [1,2,129,128]).asType(.bfloat16),
          key: MLXArray(words[1], [1,2,129,128]).asType(.bfloat16),
          value: MLXArray(words[2], [1,2,129,128]).asType(.bfloat16), geometry: g))
      }
    }
  }

  private func admitPinnedMetal() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_SOL_ROUTING_TEST"] == "1" else {
      throw XCTSkip("Opt in to pinned-library Sol GPU routing explicitly.")
    }
    guard let path = env["WEETODD_H3_SOL_TEST_METALLIB"],
      let digest = env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"], path.hasPrefix("/") else {
      throw H3CheckpointError.invalid("Missing explicit absolute Sol test Metal library binding.")
    }
    let url = URL(fileURLWithPath: path)
    guard try url.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true,
      FileManager.default.isReadableFile(atPath: path) else {
      throw H3CheckpointError.invalid("Sol test Metal library is not a readable regular file.")
    }
    let bytes = try Data(contentsOf: url)
    let actual = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(actual, digest)
    guard actual == digest, digest.count == 64, digest.allSatisfy({ $0.isHexDigit }) else {
      throw H3CheckpointError.invalid("Sol routing requires the pinned product Metal library.")
    }
    GPU.metallib = url // before any device query or array
  }

  func testInstalledTinyRoutingMatchesIndependentCPUOracle() throws {
    try admitPinnedMetal()
    for rows in [1, 65, 129, 257] {
      let g = try H3SolGeometry(rows: rows, heads: 2, approximationRange: 0..<rows)
      let count = 2 * rows * 128
      let q = (0..<count).map { Float(($0 % 5) - 2) / 8 }
      let k = (0..<count).map { Float(($0 % 7) - 3) / 4 }
      let v = (0..<count).map { Float(($0 % 9) - 4) / 8 }
      let expected = try H3SolRouting.cpuOracle(query: q, key: k, value: v, geometry: g)
      let result = try H3SolRouting.prepare(query: MLXArray(q, [1,2,rows,128]).asType(.bfloat16),
        key: MLXArray(k, [1,2,rows,128]).asType(.bfloat16),
        value: MLXArray(v, [1,2,rows,128]).asType(.bfloat16), geometry: g)
      XCTAssertEqual(result.exactRouteBits.asArray(UInt32.self), expected.exactRouteBits)
      XCTAssertEqual(result.exactCounts.asArray(UInt32.self), expected.exactCounts)
      XCTAssertEqual(result.queryMeansF32.asArray(Float.self), expected.queryMeans)
      XCTAssertEqual(result.keyMeansBF16.asType(.float32).asArray(Float.self), expected.keyMeans)
      XCTAssertEqual(result.valueMeansBF16.asType(.float32).asArray(Float.self), expected.valueMeans)
      XCTAssertEqual(result.keyStatsF32.asArray(Float.self), expected.keyStats)
    }
  }
}
