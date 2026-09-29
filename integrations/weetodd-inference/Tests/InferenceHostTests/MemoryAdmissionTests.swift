import Foundation
import XCTest
@testable import InferenceHost

final class MemoryAdmissionTests: XCTestCase, @unchecked Sendable {
  func testIncludesWeightsActivationsScratchAndPrefetch() async throws {
    let budget = try MemoryAdmission(capacityBytes: 100, reserveBytes: 10)
    let estimate = try StageMemory(weights: 30, activations: 20, scratch: 10, prefetch: 20)
    let lease = try await budget.acquire(jobID: UUID(), stage: "transformer", estimate: estimate)
    let used = await budget.reservedBytes
    XCTAssertEqual(used, 80)
    do {
      _ = try await budget.acquire(jobID: UUID(), stage: "assistant",
        estimate: StageMemory(weights: 11, activations: 0, scratch: 0, prefetch: 0))
      XCTFail("Concurrent jobs must share one budget")
    } catch {
      XCTAssertEqual(error as? HostError, .insufficientMemory(required: 11, available: 10))
    }
    await budget.release(lease)
    await budget.release(lease)
    let remaining = await budget.reservedBytes
    XCTAssertEqual(remaining, 0)
  }

  func testRejectsForeignLeaseWithoutReleasingAnotherJobsMemory() async throws {
    let first = try MemoryAdmission(capacityBytes: 100, reserveBytes: 0)
    let second = try MemoryAdmission(capacityBytes: 100, reserveBytes: 0)
    let estimate = try StageMemory(weights: 20, activations: 0, scratch: 0, prefetch: 0)
    let a = try await first.acquire(jobID: UUID(), stage: "a", estimate: estimate)
    let b = try await second.acquire(jobID: UUID(), stage: "b", estimate: estimate)
    await first.release(b)
    let used = await first.reservedBytes
    XCTAssertEqual(used, 20)
    await first.release(a)
    await second.release(b)
  }

  func testScopedReservationReleasedOnSuccessAndFailure() async throws {
    let budget = try MemoryAdmission(capacityBytes: 100, reserveBytes: 0)
    let estimate = try StageMemory(weights: 75, activations: 0, scratch: 0, prefetch: 0)
    let result = try await budget.withReservation(jobID: UUID(), stage: "text", estimate: estimate) {
      await budget.reservedBytes
    }
    XCTAssertEqual(result, 75)
    let afterSuccess = await budget.reservedBytes
    XCTAssertEqual(afterSuccess, 0)
    enum Failure: Error { case decode }
    do {
      _ = try await budget.withReservation(jobID: UUID(), stage: "decode", estimate: estimate) {
        throw Failure.decode
      } as Int
      XCTFail("Stage failure must propagate")
    } catch Failure.decode { }
    let afterFailure = await budget.reservedBytes
    XCTAssertEqual(afterFailure, 0)
  }

  func testCancellationDoesNotLeakReservation() async throws {
    let budget = try MemoryAdmission(capacityBytes: 100, reserveBytes: 0)
    let estimate = try StageMemory(weights: 75, activations: 0, scratch: 0, prefetch: 0)
    let started = AsyncStream<Void>.makeStream()
    let work = Task {
      try await budget.withReservation(jobID: UUID(), stage: "transformer", estimate: estimate) {
        started.continuation.yield(())
        try await Task.sleep(for: .seconds(30))
      }
    }
    for await _ in started.stream { break }
    work.cancel()
    do { try await work.value; XCTFail("Cancellation must propagate") }
    catch is CancellationError { }
    let afterCancellation = await budget.reservedBytes
    XCTAssertEqual(afterCancellation, 0)
    started.continuation.finish()
  }

  func testRejectsInvalidCapacityAndOverflowBeforeAdmission() throws {
    XCTAssertThrowsError(try MemoryAdmission(capacityBytes: 100, reserveBytes: 101))
    XCTAssertThrowsError(try MemoryAdmission(capacityBytes: 0, reserveBytes: 0))
    XCTAssertThrowsError(try StageMemory(weights: .max, activations: 1, scratch: 0, prefetch: 0))
  }
}
