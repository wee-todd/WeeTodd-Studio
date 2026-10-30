import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRProcessTests: XCTestCase {
  func testDrainsBothStreamsAndUsesPrivateWorkingDirectory() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "pwd; printf 'error-stream\\n' >&2"], directory: root, log: root.appendingPathComponent("log"))
    XCTAssertEqual(result.exitCode, 0)
    let text = try String(contentsOf: root.appendingPathComponent("log"), encoding: .utf8)
    XCTAssertTrue(text.contains(root.resolvingSymlinksInPath().path)); XCTAssertTrue(text.contains("error-stream"))
  }
  func testCancellationWaitsForChildExit() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let cancellation = NativeHeadlessCancellation()
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { cancellation.cancel() }
    let start = Date()
    XCTAssertThrowsError(try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"], directory: root, log: root.appendingPathComponent("log"), cancellation: cancellation)) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertLessThan(Date().timeIntervalSince(start), 3)
  }
  func testCancellationRemainsResponsiveAfterChildClosesOutput() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let cancellation = NativeHeadlessCancellation()
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { cancellation.cancel() }
    let start = Date()
    XCTAssertThrowsError(try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
      arguments: ["-c", "exec 1>&- 2>&-; trap '' INT; exec /bin/sleep 3"], directory: root,
      log: root.appendingPathComponent("log"), cancellation: cancellation)) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertLessThan(Date().timeIntervalSince(start), 2)
  }
  func testLongProgressStreamKeepsOnlyBoundedCapture() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let script = "printf '%s\\n' " + Array(repeating: "progress", count: 100).joined(separator: " ")
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script], directory: root, log: root.appendingPathComponent("log"), maximumOutput: 32, captureOutput: false)
    XCTAssertEqual(result.exitCode, 0); XCTAssertLessThanOrEqual(result.output.count, 32)
  }
  func testOversizedOutputCannotGrowMemoryOrHangChild() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    XCTAssertThrowsError(try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "dd if=/dev/zero bs=70000 count=1 2>/dev/null"], directory: root, log: root.appendingPathComponent("log")))
  }
}
