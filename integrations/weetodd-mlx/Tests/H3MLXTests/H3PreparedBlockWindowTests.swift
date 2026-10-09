import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3PreparedBlockWindowTests: XCTestCase {
  private let missing = URL(fileURLWithPath: "/nonexistent/weetodd-window.safetensors")

  func testPlanRejectsInvalidCountsAndWindowSizes() {
    for (count, size) in [(0, 1), (-1, 1), (51, 1), (Int.max, 1),
      (2, 0), (2, 3), (2, Int.max)] {
      XCTAssertThrowsError(try H3PreparedBlockWindow.Plan(blockCount: count, windowSize: size))
      XCTAssertThrowsError(try H3PreparedBlockWindow(checkpointURL: missing,
        blockCount: count, windowSize: size))
    }
  }

  func testPurePlansCoverEveryBlockOnceAndClampTheFinalWindow() throws {
    for count in [1, 2, 5, 50] {
      for size in [1, 2] {
        var plan = try H3PreparedBlockWindow.Plan(blockCount: count, windowSize: size)
        for index in 0..<count {
          let start = (index / size) * size
          XCTAssertEqual(try plan.begin(index), start..<min(start + size, count))
          XCTAssertEqual(plan.outstandingIndex, index)
          try plan.finish(index)
          XCTAssertNil(plan.outstandingIndex)
          XCTAssertEqual(plan.nextIndex, index + 1)
        }
        XCTAssertTrue(plan.isClosed)
        XCTAssertThrowsError(try plan.begin(count))
        plan.close()
        XCTAssertTrue(plan.isClosed)
      }
    }
  }

  func testPlanRequiresOrderedNonrepeatedRequestsAndMatchingCompletion() throws {
    var plan = try H3PreparedBlockWindow.Plan(blockCount: 3, windowSize: 2)
    XCTAssertThrowsError(try plan.begin(-1))
    XCTAssertThrowsError(try plan.begin(1))
    XCTAssertThrowsError(try plan.finish(0))
    XCTAssertEqual(try plan.begin(0), 0..<2)
    XCTAssertThrowsError(try plan.begin(0))
    XCTAssertThrowsError(try plan.begin(1))
    XCTAssertThrowsError(try plan.finish(1))
    try plan.finish(0)
    XCTAssertThrowsError(try plan.finish(0))
    XCTAssertThrowsError(try plan.begin(0))
    XCTAssertEqual(try plan.begin(1), 0..<2)
    plan.close()
    XCTAssertNil(plan.outstandingIndex)
    XCTAssertThrowsError(try plan.finish(1))
    XCTAssertThrowsError(try plan.begin(1))
  }

  func testConstructorRejectsUnsupportedModesAndWindowsBeforeCheckpointWork() {
    XCTAssertThrowsError(try H3PreparedBlockWindow(checkpointURL: missing,
      blockCount: 2, windowSize: 1, projectionMode: .activationRotated)) {
      XCTAssertEqual($0 as? H3CheckpointError,
        .invalid("Prepared H3 blocks require weight-decoded projections."))
    }
    XCTAssertThrowsError(try H3PreparedBlockWindow(checkpointURL: missing,
      blockCount: 2, windowSize: 1, rowWindow: 0)) {
      XCTAssertEqual($0 as? H3CheckpointError,
        .invalid("Prepared H3 decode row window is unsupported."))
    }
  }

  func testMissingCheckpointFailureClosesTheUnloadedWindow() throws {
    let window = try H3PreparedBlockWindow(checkpointURL: missing, blockCount: 2, windowSize: 1)
    XCTAssertFalse(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertEqual(window.residentIndices, [])
    XCTAssertThrowsError(try window.weights(for: 0))
    XCTAssertTrue(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertEqual(window.residentIndices, [])
  }

  func testOrderFailureAndIdempotentCloseNeedNoCheckpoint() throws {
    let wrongOrder = try H3PreparedBlockWindow(checkpointURL: missing, blockCount: 2, windowSize: 2)
    XCTAssertThrowsError(try wrongOrder.weights(for: 1)) {
      XCTAssertEqual($0 as? H3CheckpointError,
        .invalid("Prepared H3 blocks must be requested once in order."))
    }
    XCTAssertTrue(wrongOrder.isClosed)
    let window = try H3PreparedBlockWindow(checkpointURL: missing, blockCount: 2, windowSize: 1)
    window.close()
    window.close()
    XCTAssertTrue(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertThrowsError(try window.weights(for: 0))
    XCTAssertThrowsError(try window.finish(index: 0))
  }

  func testCancellationBeforeLoadingClosesTheUnloadedWindow() async throws {
    let task = Task { () throws -> Bool in
      let window = try H3PreparedBlockWindow(
        checkpointURL: URL(fileURLWithPath: "/nonexistent/weetodd-cancelled-window.safetensors"),
        blockCount: 2, windowSize: 1)
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try window.weights(for: 0)
        return false
      } catch is CancellationError {
        return window.isClosed && window.storageBytes == 0 && window.residentIndices.isEmpty
      }
    }
    let closedAfterCancellation = try await task.value
    XCTAssertTrue(closedAfterCancellation)
  }

  private func installed() throws -> URL {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_PREPARED_BLOCK_TESTS"] == "1",
      let checkpoint = env["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed H3 preparation-window ownership qualification.")
    }
    return URL(fileURLWithPath: checkpoint)
  }

  private func installedPackedFast() throws -> URL {
    let checkpoint = try installed()
    let layout = try H3CheckpointLayout(url: checkpoint)
    let file = try SafeTensorFile(url: H3CheckpointSource.fileURL(checkpoint, block: 0))
    guard layout.fastVariant != nil,
      ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"].allSatisfy({
        file.tensors[layout.prefix + "blocks.0." + $0 + ".weight"]?.dtype == "U32"
      }) else { throw XCTSkip("Two-owner preparation requires a packed FastH3 checkpoint.") }
    return checkpoint
  }

  func testInstalledSingleOwnerClosesBeforeTheNextBlockLoads() throws {
    let checkpoint = try installed()
    let window = try H3PreparedBlockWindow(checkpointURL: checkpoint, blockCount: 2, windowSize: 1)
    defer { window.close() }
    let first = try window.weights(for: 0)
    XCTAssertEqual(window.residentIndices, [0])
    XCTAssertGreaterThan(window.storageBytes, 0)
    try window.finish(index: 0)
    XCTAssertTrue(first.isClosed)
    XCTAssertFalse(window.isClosed)
    XCTAssertEqual(window.residentIndices, [])
    XCTAssertEqual(window.storageBytes, 0)
    let second = try window.weights(for: 1)
    XCTAssertEqual(window.residentIndices, [1])
    XCTAssertFalse(second.isClosed)
    try window.finish(index: 1)
    XCTAssertTrue(second.isClosed)
    XCTAssertTrue(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
  }

  func testInstalledPackedWindowRetiresFirstOwnerWhilePendingOwnerStaysResident() throws {
    let checkpoint = try installedPackedFast()
    Stream.gpu.synchronize()
    Memory.clearCache()
    let before = Memory.activeMemory
    let window = try H3PreparedBlockWindow(checkpointURL: checkpoint, blockCount: 2, windowSize: 2)
    defer { window.close() }
    let first = try window.weights(for: 0)
    let pendingBytes = window.storageBytes - first.storageBytes
    let bothActive = Memory.activeMemory
    XCTAssertEqual(window.residentIndices, [0, 1])
    XCTAssertGreaterThan(pendingBytes, 0)
    try window.finish(index: 0)
    XCTAssertTrue(first.isClosed)
    XCTAssertEqual(first.storageBytes, 0)
    XCTAssertFalse(window.isClosed)
    XCTAssertEqual(window.residentIndices, [1])
    XCTAssertEqual(window.storageBytes, pendingBytes)
    XCTAssertLessThan(Memory.activeMemory, bothActive)
    let second = try window.weights(for: 1)
    XCTAssertEqual(second.storageBytes, pendingBytes)
    XCTAssertEqual(window.residentIndices, [1])
    try window.finish(index: 1)
    XCTAssertTrue(second.isClosed)
    XCTAssertTrue(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertEqual(window.residentIndices, [])
    Memory.clearCache()
    XCTAssertEqual(Memory.activeMemory, before,
      "Closed owner objects may remain referenced without retaining their GPU arrays.")
  }

  func testInstalledQueuedPairRetainsBothOwnersUntilCompletionThenReleasesActualArrays() throws {
    let checkpoint = try installedPackedFast()
    Stream.gpu.synchronize(); Memory.clearCache()
    let before = Memory.activeMemory
    let window = try H3PreparedBlockWindow(checkpointURL: checkpoint, blockCount: 2, windowSize: 2)
    defer { window.close() }
    let first = try window.weights(for: 0)
    let bytes = window.storageBytes
    try window.finish(index: 0, deferRetirement: true)
    XCTAssertFalse(first.isClosed)
    XCTAssertEqual(window.residentIndices, [0, 1])
    XCTAssertEqual(window.storageBytes, bytes)
    let second = try window.weights(for: 1)
    try window.finish(index: 1, deferRetirement: true)
    XCTAssertTrue(first.isClosed); XCTAssertTrue(second.isClosed)
    XCTAssertTrue(window.isClosed)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertEqual(window.residentIndices, [])
    Memory.clearCache()
    XCTAssertEqual(Memory.activeMemory, before)
  }

  func testInstalledOrderFailureClosesCurrentAndPendingOwners() throws {
    let checkpoint = try installedPackedFast()
    Stream.gpu.synchronize()
    Memory.clearCache()
    let before = Memory.activeMemory
    let window = try H3PreparedBlockWindow(checkpointURL: checkpoint, blockCount: 2, windowSize: 2)
    let first = try window.weights(for: 0)
    XCTAssertEqual(window.residentIndices, [0, 1])
    XCTAssertThrowsError(try window.weights(for: 0))
    XCTAssertTrue(window.isClosed)
    XCTAssertTrue(first.isClosed)
    XCTAssertEqual(window.residentIndices, [])
    XCTAssertEqual(window.storageBytes, 0)
    window.close()
    Memory.clearCache()
    XCTAssertEqual(Memory.activeMemory, before)
  }
}
