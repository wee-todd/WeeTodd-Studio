import XCTest
@testable import H3MLX

final class H3RenderResourceUsageTests: XCTestCase {
  func testReceiptCapturesProcessFootprintAndSuppliedMLXPeak() throws {
    let usage = try H3RenderResourceUsage.capture(peakMLXBytes: 1234)
    XCTAssertEqual(usage.peakMLXBytes, 1234)
    XCTAssertGreaterThan(usage.currentPhysicalBytes, 0)
    XCTAssertGreaterThanOrEqual(usage.peakPhysicalBytes,
      usage.currentPhysicalBytes)
  }
}
