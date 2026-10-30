import XCTest
@testable import StudioCore

final class H3RegenerateProgressTests: XCTestCase {
  func testLongRefinementShowsItsStepAndBlockCount() throws {
    let event = try XCTUnwrap(H3RegenerateService.progressEvent(["stage": "sampling_block", "step": 2, "completed": 17, "total": 50], mode: "render"))
    XCTAssertTrue(event.message.contains("step 2"), event.message)
    XCTAssertTrue(event.message.contains("17/50"), event.message)
    XCTAssertEqual(event.fraction ?? -1, 0.34, accuracy: 0.00001)
    XCTAssertNil(H3RegenerateService.progressEvent(["status": "success"], mode: "render"))
    XCTAssertNotNil(H3RegenerateService.progressEvent(["completed": 1e100, "total": 1e200], mode: "render"))
  }
}
