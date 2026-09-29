import XCTest
@testable import LTX25Text

final class TextEncodingPlanTests: XCTestCase {
  func testAdmissionCountsBothAggregationCopiesAndOverlappingOutputs() throws {
    let plan = try TextEncodingPlan(promptTokens: 1024)
    XCTAssertEqual(plan.interleavedBytes, 1024*3840*49*4)
    XCTAssertGreaterThan(plan.aggregationBytes, 2*plan.interleavedBytes + 1024*6144*4)
    XCTAssertGreaterThanOrEqual(plan.ownedBufferBytes, plan.gemmaBytes)
    XCTAssertGreaterThanOrEqual(plan.ownedBufferBytes, plan.connectorBytes)
    XCTAssertEqual(plan.ownedBufferBytes,plan.metadataReserveBytes + max(plan.gemmaBytes,plan.aggregationBytes,plan.connectorBytes))
    var config = TextEncodingConfiguration()
    config.maximumOwnedBufferBytes = plan.ownedBufferBytes
    XCTAssertNoThrow(try TextEncodingPlan(promptTokens: 1024, configuration: config))
    config.maximumOwnedBufferBytes -= 1
    XCTAssertThrowsError(try TextEncodingPlan(promptTokens: 1024, configuration: config))
    // The old accumulator-only admission cannot admit the complete stage.
    config.maximumOwnedBufferBytes = 1024*3840*50*4
    XCTAssertThrowsError(try TextEncodingPlan(promptTokens: 1024, configuration: config))
  }
  func testShortPromptStillBudgetsFullConnectorAndRejectsInvalidCounts() throws {
    let short = try TextEncodingPlan(promptTokens: 1)
    let long = try TextEncodingPlan(promptTokens: 1024)
    XCTAssertEqual(short.connectorBytes, long.connectorBytes)
    XCTAssertLessThan(short.aggregationBytes, long.aggregationBytes)
    for count in [0,-1,1025,Int.max] {
      XCTAssertThrowsError(try TextEncodingPlan(promptTokens: count))
    }
    var config = TextEncodingConfiguration(); config.maximumOwnedBufferBytes = 0
    XCTAssertThrowsError(try TextEncodingPlan(promptTokens: 1, configuration: config))
  }
}
