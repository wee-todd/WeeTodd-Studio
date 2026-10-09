import Foundation
import XCTest
@testable import H3MLX

final class H3NativeBlockReportTests: XCTestCase {
  func testPhysicalSamplesStaySeparateFromIndependentLifetimeEnvelopes() throws {
    let monitor = try H3NativeMemoryMonitor(parentPID: 10, childPID: 20,
      sampler: { pid in .available(.init(currentBytes: pid == 10 ? 100 : 50,
        lifetimeHighWaterBytes: pid == 10 ? 1000 : 200, processStartAbsoluteTime: 1)) },
      automaticSampling: false)
    try monitor.start(); monitor.stop()
    let report = H3NativeBlockReport(evaluations: 4, predictionSeconds: 700,
      bridgeSeconds: 708, childReaped: true, physicalMemory: monitor.report)
    let memory = try XCTUnwrap(report.metadata["physicalMemory"] as? [String: Any])
    XCTAssertEqual(memory["sampledNonAtomicCombinedCurrentPeakEstimateBytes"] as? UInt64, 150)
    XCTAssertEqual(memory["sumIndependentHighWaterEnvelopeBytes"] as? UInt64, 1200)
    XCTAssertNil(memory["measuredCombinedPeakBytes"])
    XCTAssertTrue((memory["scope"] as? String)?.contains("neither exact") == true)
    XCTAssertEqual((memory["parent"] as? [String: Any])?["reportedLifetimeHighWaterBytes"] as? UInt64, 1000)
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: report.metadata))
  }
  func testNestedTimingsAndPrecisionContractRemainExplicit() throws {
    let report = H3NativeBlockReport(evaluations: 4, predictionSeconds: 700,
      bridgeSeconds: 708, childReaped: true)
    let metadata = report.metadata
    XCTAssertEqual(metadata["evaluations"] as? Int, 4)
    XCTAssertEqual(metadata["predictionSeconds"] as? Double, 700)
    XCTAssertEqual(metadata["inclusiveBridgeSeconds"] as? Double, 708)
    XCTAssertEqual(metadata["transferAndManagementSeconds"] as? Double, 8)
    XCTAssertEqual(metadata["childReaped"] as? Bool, true)
    XCTAssertEqual(metadata["precision"] as? String, "fp16_projections_fp32_attention_and_residual")
    XCTAssertTrue((metadata["timingScope"] as? String)?.contains("nested") == true)
    XCTAssertEqual(metadata["physicalObservationStatus"] as? String, "not_sampled")
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: metadata))
  }

  func testBackendReceiptDoesNotCallNativeBlocksMLXOrDoubleCountTiming() throws {
    let report = H3BackendReport(nativeBlocks: H3NativeBlockReport(evaluations: 4,
      predictionSeconds: 700, bridgeSeconds: 708, childReaped: true))
    let metadata = report.metadata
    XCTAssertEqual(metadata["projectionBackend"] as? String, "nnc_experimental")
    XCTAssertTrue((metadata["allocationCounterScope"] as? String)?.contains("excludes") == true)
    let child = try XCTUnwrap(metadata["nativeBlocks"] as? [String: Any])
    XCTAssertEqual(child["inclusiveBridgeSeconds"] as? Double, 708)
    XCTAssertNil(child["totalSeconds"])
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: metadata))
  }
}
