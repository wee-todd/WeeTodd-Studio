import Foundation
import XCTest
@testable import InferenceContracts

final class NativeInferenceLeaseTests: XCTestCase {
  func testExclusiveLeaseReleasesForNextWorker() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-native-lease-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let path = root.appendingPathComponent("Runtime/native-inference.lock")
    let first = try NativeInferenceLease.tryAcquire(at: path)
    XCTAssertNotNil(first)
    XCTAssertNil(try NativeInferenceLease.tryAcquire(at: path))
    first?.release()
    let second = try NativeInferenceLease.tryAcquire(at: path)
    XCTAssertNotNil(second)
    second?.release()
  }
}
