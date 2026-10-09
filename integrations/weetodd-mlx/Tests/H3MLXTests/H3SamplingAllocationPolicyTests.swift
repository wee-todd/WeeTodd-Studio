import XCTest
@testable import H3MLX

final class H3SamplingAllocationPolicyTests: XCTestCase {
  private let gb = 1024 * 1024 * 1024

  func testPreparedBlockInheritsAdmittedPoolInsteadOfOverridingIt() {
    for existing in [0, H3SamplingAllocationPolicy.minimum, 2 * gb, 4 * gb] {
      XCTAssertEqual(H3SamplingAllocationPolicy.blockLimit(previous: existing,
        prepared: true), existing)
    }
    XCTAssertEqual(H3SamplingAllocationPolicy.blockLimit(previous: 4 * gb,
      prepared: false), H3SamplingAllocationPolicy.minimum)
  }

  func testPoolScalesWithRAMAndRespectsExistingLowerLimit() {
    for (ram, expected) in [(64, 1), (128, 2), (256, 4), (512, 4)] {
      XCTAssertEqual(H3SamplingAllocationPolicy.limit(previous: 8 * gb,
        physical: ram * gb, recommended: ram * 3 / 4 * gb,
        available: ram / 2 * gb, eligible: true), expected * gb)
    }
    for existing in [0, 1, 64 * 1024 * 1024, 128 * 1024 * 1024] {
      XCTAssertEqual(H3SamplingAllocationPolicy.limit(previous: existing,
        physical: 256 * gb, recommended: 192 * gb, available: 160 * gb,
        eligible: true), existing)
    }
  }

  func testPressureAndUnqualifiedPathsRetainSmallPool() {
    for available in [0, 16 * gb, 32 * gb, 39 * gb] {
      XCTAssertEqual(H3SamplingAllocationPolicy.limit(previous: 8 * gb,
        physical: 256 * gb, recommended: 192 * gb, available: available,
        eligible: true), H3SamplingAllocationPolicy.minimum)
    }
    XCTAssertEqual(H3SamplingAllocationPolicy.limit(previous: 8 * gb,
      physical: 32 * gb, recommended: 24 * gb, available: 24 * gb,
      eligible: true), H3SamplingAllocationPolicy.minimum)
    XCTAssertEqual(H3SamplingAllocationPolicy.limit(previous: 8 * gb,
      physical: 256 * gb, recommended: 192 * gb, available: 160 * gb,
      eligible: false), H3SamplingAllocationPolicy.minimum)
  }

  func testReportSeparatesSoftLimitFromObservedPoolAndWeightCache() {
    let report = H3SamplingAllocationReport(softLimitBytes: 4 * gb,
      maximumObservedCachedBytes: 5 * gb, weightPreparationSeconds: 12.5,
      clearedAtEvaluationBoundary: true)
    let metadata = H3BackendReport(samplingAllocationPool: report).metadata
    let pool = metadata["samplingAllocationPool"] as? [String: Any]
    XCTAssertEqual(pool?["softLimitBytes"] as? Int, 4 * gb)
    XCTAssertEqual(pool?["maximumObservedCachedBytes"] as? Int, 5 * gb)
    XCTAssertEqual(pool?["weightPreparationSeconds"] as? Double, 12.5)
    XCTAssertEqual(pool?["clearedAtEvaluationBoundary"] as? Bool, true)
    XCTAssertNil(metadata["transformerWeightCache"])
  }
}
