import Foundation
import XCTest
@testable import LTX25NNC

final class WeightPreparationQueueTests: XCTestCase {
  func testSingleSlotBudgetAndDrain() throws {
    let queue = try WeightPreparationQueue(layout: [("x", [2, 3])], maximumBytes: 24)
    try queue.submit(index: 4) { i, _, shape in [Float](repeating: Float(i), count: shape.reduce(1, *)) }
    XCTAssertThrowsError(try queue.submit(index: 5) { _, _, _ in [] })
    let prepared = try queue.take()
    XCTAssertEqual(prepared.index, 4)
    XCTAssertEqual(prepared.tensors["x"], [4, 4, 4, 4, 4, 4])
    XCTAssertEqual(prepared.bytes, 24)
    XCTAssertFalse(queue.pending)
    XCTAssertThrowsError(try queue.take())
    try queue.submit(index: 5) { _, _, _ in [Float](repeating: 1, count: 6) }
    queue.drain()
    XCTAssertFalse(queue.pending)
    XCTAssertThrowsError(try WeightPreparationQueue(layout: [("x", [2, 3])], maximumBytes: 23))
  }
  func testCancellationWhileProviderIsActuallyPending() throws {
    let entered = DispatchSemaphore(value: 0), unblock = DispatchSemaphore(value: 0)
    let queue = try WeightPreparationQueue(layout: [("x", [1])], maximumBytes: 4)
    try queue.submit(index: 0) { _, _, _ in
      entered.signal(); unblock.wait(); return [1]
    }
    XCTAssertEqual(entered.wait(timeout: .now() + 2), .success)
    XCTAssertTrue(queue.pending)
    queue.cancelPending()
    unblock.signal()
    XCTAssertThrowsError(try queue.take()) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertFalse(queue.pending)
  }

  func testErrorAndCancellationDiscardPreparedStorage() throws {
    enum Stop: Error { case now }
    let queue = try WeightPreparationQueue(layout: [("x", [1])], maximumBytes: 4)
    try queue.submit(index: 0) { _, _, _ in throw Stop.now }
    XCTAssertThrowsError(try queue.take())
    XCTAssertFalse(queue.pending)
    try queue.submit(index: 1) { _, _, _ in [1] }
    queue.drain()
    XCTAssertThrowsError(try queue.take())
    try queue.submit(index: 2) { _, _, _ in [2] }
    XCTAssertEqual(try queue.take().tensors["x"], [2])
  }
}
