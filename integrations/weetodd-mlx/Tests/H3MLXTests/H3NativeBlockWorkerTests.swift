import Darwin
import Foundation
import XCTest
@testable import H3MLX

final class H3NativeBlockWorkerTests: XCTestCase {
  private static let capabilities = #"{"protocol":1,"backend":"nnc_experimental","blocks":50,"max_rows":40000,"parent_watchdog":true}"#
  private func workspace() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("H3-worker-discovery-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }
  private func worker(_ directory: URL, name: String = "WeeToddH3Worker", body: String) throws -> URL {
    let url = directory.appendingPathComponent(name)
    try Data(("#!/bin/sh\n" + body + "\n").utf8).write(to: url)
    guard chmod(url.path, 0o700) == 0 else { throw NSError(domain: "chmod", code: Int(errno)) }
    return url
  }
  private func capabilityWorker(_ directory: URL, json: String? = nil) throws -> URL {
    try worker(directory, body: "[ \"$#\" = 1 ] && [ \"$1\" = --capabilities ] || exit 9\nprintf '%s\\n' '" + (json ?? Self.capabilities) + "'")
  }
  func testSiblingDiscoveryAndAbsoluteOverrideFollowRegularSymlinkTarget() throws {
    let directory = try workspace(), sibling = try capabilityWorker(directory)
    let current = directory.appendingPathComponent("WeeToddH3MLXWorker")
    XCTAssertEqual(try H3NativeBlockWorker.resolve(executableURL: current, environment: [:]), sibling)
    let alternate = try worker(directory, name: "alternate", body: "exit 0")
    let link = directory.appendingPathComponent("link")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: alternate)
    XCTAssertEqual(try H3NativeBlockWorker.resolve(executableURL: current,
      environment: ["WEETODD_H3_NATIVE_WORKER": link.path]), link)
  }
  func testExplicitInvalidOverrideNeverFallsBackToValidSibling() throws {
    let directory = try workspace(); _ = try capabilityWorker(directory)
    let current = directory.appendingPathComponent("WeeToddH3MLXWorker")
    for value in ["", "./WeeToddH3Worker", "relative", directory.path, directory.appendingPathComponent("missing").path] {
      XCTAssertThrowsError(try H3NativeBlockWorker.resolve(executableURL: current,
        environment: ["WEETODD_H3_NATIVE_WORKER": value]))
    }
    let file = directory.appendingPathComponent("not-executable")
    try Data("text".utf8).write(to: file); _ = chmod(file.path, 0o600)
    XCTAssertThrowsError(try H3NativeBlockWorker.resolve(executableURL: current,
      environment: ["WEETODD_H3_NATIVE_WORKER": file.path]))
    XCTAssertThrowsError(try H3NativeBlockWorker.resolve(executableURL: URL(string: "https://example.com/worker")!, environment: [:]))
  }
  func testCapabilitySchemaRejectsBoolNumbersStringsWrongValuesAndTrailingEvents() throws {
    try H3NativeBlockWorker.validateCapabilities(Data(Self.capabilities.utf8))
    for (old, replacement) in [("\"protocol\":1", "\"protocol\":true"),
      ("\"blocks\":50", "\"blocks\":\"50\""), ("\"max_rows\":40000", "\"max_rows\":40001"),
      ("\"parent_watchdog\":true", "\"parent_watchdog\":1"),
      ("nnc_experimental", "mlx"), ("\"protocol\":1", "\"protocol\":1.5")] {
      XCTAssertThrowsError(try H3NativeBlockWorker.validateCapabilities(Data(Self.capabilities.replacingOccurrences(of: old, with: replacement).utf8)))
    }
    XCTAssertThrowsError(try H3NativeBlockWorker.validateCapabilities(Data((Self.capabilities + "\n{}").utf8)))
    XCTAssertThrowsError(try H3NativeBlockWorker.validateCapabilities(Data("[]".utf8)))
  }
  func testCorrectCapabilityProcessAndExitFailure() throws {
    let directory = try workspace()
    try H3NativeBlockWorker.preflight(workerURL: capabilityWorker(directory))
    let wrongExit = try worker(directory, name: "wrong-exit", body: "printf '%s\\n' '" + Self.capabilities + "'\nexit 3")
    XCTAssertThrowsError(try H3NativeBlockWorker.preflight(workerURL: wrongExit))
  }
  func testOversizedStdoutAndStderrRejectWithoutHanging() throws {
    let directory = try workspace()
    for fd in [1, 2] {
      let file = try worker(directory, name: "large-\(fd)", body: "i=0; while [ \"$i\" -lt 70000 ]; do printf x >&\(fd); i=$((i+1)); done")
      XCTAssertThrowsError(try H3NativeBlockWorker.preflight(workerURL: file, timeout: 3))
    }
  }
  func testDeadlineAdmissionAndTimeoutReapOwnedWorker() throws {
    let directory = try workspace(), pidURL = directory.appendingPathComponent("pid")
    let file = try worker(directory, body: "echo $$ > '" + pidURL.path + "'\nwhile :; do :; done")
    for seconds in [0.0, -1.0, 10.1, .infinity, .nan] {
      XCTAssertThrowsError(try H3NativeBlockWorker.preflight(workerURL: file, timeout: seconds))
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: pidURL.path))
    // A short deadline may expire before the child runs its first instruction.
    // Verify that deadline independently from the child-written PID witness.
    let short = try worker(directory, name: "short-deadline", body: "while :; do :; done")
    for (candidate, seconds) in [(short, 0.15), (file, 2.0)] {
      XCTAssertThrowsError(try H3NativeBlockWorker.preflight(workerURL: candidate, timeout: seconds)) { error in
        guard case H3CheckpointError.invalid(let message) = error else {
          return XCTFail("Expected capability deadline failure, got \(error)")
        }
        XCTAssertTrue(message.contains("timed out"))
      }
    }
    let pid = try XCTUnwrap(Int32(String(contentsOf: pidURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
    XCTAssertEqual(kill(pid, 0), -1); XCTAssertEqual(errno, ESRCH)
  }
  func testActualTaskCancellationReapsOwnedWorker() async throws {
    let directory = try workspace(), pidURL = directory.appendingPathComponent("cancel-pid")
    let file = try worker(directory, body: "echo $$ > '" + pidURL.path + "'\nwhile :; do :; done")
    let task = Task.detached { () throws -> Bool in
      do { try H3NativeBlockWorker.preflight(workerURL: file); return false }
      catch is CancellationError { return true }
    }
    // File creation can precede the shell's PID write. Await the parsed value,
    // with a separate bounded fixture-startup deadline, before cancelling.
    let readinessDeadline = DispatchTime.now().uptimeNanoseconds + 2_000_000_000
    var workerPID: Int32?
    while workerPID == nil && DispatchTime.now().uptimeNanoseconds < readinessDeadline {
      workerPID = (try? String(contentsOf: pidURL, encoding: .utf8)).flatMap {
        Int32($0.trimmingCharacters(in: .whitespacesAndNewlines))
      }
      if workerPID == nil { try await Task.sleep(nanoseconds: 10_000_000) }
    }
    task.cancel()
    let cancelled = try await task.value
    XCTAssertTrue(cancelled)
    let pid = try XCTUnwrap(workerPID, "Worker fixture did not publish its PID before the startup deadline.")
    XCTAssertEqual(kill(pid, 0), -1); XCTAssertEqual(errno, ESRCH)
  }
  func testOptionalInstalledWorkerCapabilitiesWithoutInference() throws {
    guard let value = ProcessInfo.processInfo.environment["WEETODD_H3_NATIVE_WORKER_CAPABILITY_TEST"] else {
      throw XCTSkip("Installed native worker capability preflight is opt-in.")
    }
    let worker = try H3NativeBlockWorker.resolve(executableURL: URL(fileURLWithPath: "/unused/WeeToddH3MLXWorker"),
      environment: ["WEETODD_H3_NATIVE_WORKER": value])
    try H3NativeBlockWorker.preflight(workerURL: worker)
  }
}
