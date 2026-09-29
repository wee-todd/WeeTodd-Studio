import Darwin
import Foundation
import XCTest
@testable import InferenceHost

final class WorkerProcessTests: XCTestCase, @unchecked Sendable {
  func testDrainsBothPipesWithoutKeepingUnboundedLogs() async throws {
    let lines = LineCollection()
    let result = try await WorkerProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
      arguments: ["-c", "i=0; while [ $i -lt 2000 ]; do printf 'abcdefghijklmnopqrstuvxyz0123456789\\n' >&2; i=$((i+1)); done; printf 'first\\nsecond\\n'"],
      onLine: { lines.append($0) })
    XCTAssertEqual(result.exitCode, 0)
    XCTAssertEqual(lines.strings, ["first", "second"])
    XCTAssertLessThanOrEqual(result.stderrTail.count, 65536)
    XCTAssertTrue(String(decoding: result.stderrTail, as: UTF8.self).hasSuffix("0123456789\n"))
  }

  func testCancellationKillsUncooperativeWorkerAndReapsItBeforeReturning() async throws {
    let started = AsyncStream<Int32>.makeStream()
    let task = Task {
      try await WorkerProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: ["-c", "trap '' TERM; printf '%s\\n' $$; while :; do :; done"],
        terminationGraceMilliseconds: 100, onLine: { data in
          if let pid = Int32(String(decoding: data, as: UTF8.self)) { started.continuation.yield(pid) }
        })
    }
    var pid: Int32?
    for await value in started.stream { pid = value; break }
    task.cancel()
    do { _ = try await task.value; XCTFail("Cancellation must propagate after cleanup") }
    catch is CancellationError { }
    XCTAssertNotNil(pid)
    if let pid { XCTAssertEqual(kill(pid, 0), -1); XCTAssertEqual(errno, ESRCH) }
    started.continuation.finish()
  }

  func testRejectsOverlongProtocolLineAndReportsNonzeroExit() async throws {
    do {
      _ = try await WorkerProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: ["-c", "i=0; while [ $i -lt 8000 ]; do printf 0123456789; i=$((i+1)); done"],
        terminationGraceMilliseconds: 100, onLine: { _ in })
      XCTFail("An unterminated protocol line cannot grow without limit")
    } catch WorkerProcess.Error.outputLineTooLong { }
    let failed = try await WorkerProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
      arguments: ["-c", "printf 'model failed\\n' >&2; exit 7"], onLine: { _ in })
    XCTAssertEqual(failed.exitCode, 7)
    XCTAssertEqual(String(decoding: failed.stderrTail, as: UTF8.self), "model failed\n")
  }

  func testReceiverFailureStopsAndReapsWorker() async throws {
    enum Rejected: Error { case event }
    do {
      _ = try await WorkerProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: ["-c", "printf 'invalid\\n'; while :; do :; done"],
        terminationGraceMilliseconds: 100, onLine: { _ in throw Rejected.event })
      XCTFail("A protocol receiver error must propagate")
    } catch Rejected.event { }
  }
}

private final class LineCollection: @unchecked Sendable {
  private let lock = NSLock()
  private var values: [String] = []
  func append(_ data: Data) { lock.withLock { values.append(String(decoding: data, as: UTF8.self)) } }
  var strings: [String] { lock.withLock { values } }
}
