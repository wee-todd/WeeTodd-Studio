import XCTest
@testable import InferenceContracts

final class NativeTAEPreviewPolicyTests: XCTestCase {
  func testDisplayUpdatesAreBoundedByStepsAndMonotonicTime() throws {
    var policy = NativeTAEPreviewPolicy(), rendered = 0, shown: [Int] = []
    for (step, now) in [(1, 0.0), (2, 3), (3, 1), (3, 3), (4, 6), (5, 6)] {
      try policy.attempt(completed: step, total: 8, uptime: now,
        render: { rendered += 1; return step }, publish: { shown.append($0) },
        onFailure: { _ in XCTFail("A valid preview cannot warn") })
    }
    XCTAssertEqual(shown, [1, 3, 5])
    XCTAssertEqual(rendered, 3)
    XCTAssertEqual(policy.emittedCount, 3)
    XCTAssertFalse(policy.disabled)
  }

  func testOptionalDisplayFailureWarnsOnceWithoutFailingGeneration() throws {
    enum Failure: Error { case display }
    var policy = NativeTAEPreviewPolicy(), warnings = 0, renders = 0
    for step in [1, 3, 5] {
      try policy.attempt(completed: step, total: 5, uptime: Double(step) * 3,
        render: { renders += 1; throw Failure.display }, publish: { (_: Int) in XCTFail("No image") },
        onFailure: { _ in warnings += 1 })
    }
    XCTAssertEqual(renders, 1)
    XCTAssertEqual(warnings, 1)
    XCTAssertTrue(policy.disabled)
    XCTAssertEqual(policy.emittedCount, 0)
  }

  func testCancellationPropagatesWithoutBecomingOptionalFailure() throws {
    var policy = NativeTAEPreviewPolicy(), warnings = 0
    XCTAssertThrowsError(try policy.attempt(completed: 1, total: 2, uptime: 0,
      render: { throw CancellationError() }, publish: { (_: Int) in },
      onFailure: { _ in warnings += 1 })) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertFalse(policy.disabled)
    XCTAssertEqual(warnings, 0)
    XCTAssertEqual(policy.emittedCount, 0)
  }
}
