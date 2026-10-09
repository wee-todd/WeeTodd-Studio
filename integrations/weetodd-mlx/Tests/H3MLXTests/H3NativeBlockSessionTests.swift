import Darwin
import Foundation
import XCTest
@testable import H3MLX

final class H3NativeBlockSessionTests: XCTestCase {
  private func json(_ value: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: value) }
  private var readyFields: [String: Any] { ["event": "ready", "protocol": 1, "rows": 1,
    "precision": "fp16", "start": 0, "count": 50, "residency": "block", "resident_blocks": 0,
    "projections": "input-scaled", "modulation_spans": 0, "weight_prefetch": true,
    "prefetch_slot_capacity": 1, "buffer_io": "bounded", "qkv_schedule": "serial",
    "load_seconds": 0, "metal_allocated_bytes": 0] }
  func testQualifiedReadyAndStrictTypes() throws {
    XCTAssertEqual(try H3NativeBlockSession.validateReady(json(readyFields), rows: 1).rows, 1)
    let invalidFields: [(String, Any)] = [("protocol", true), ("rows", 2), ("weight_prefetch", 1),
      ("qkv_schedule", "parallel"), ("resident_blocks", 1), ("count", 49)]
    for (key, value) in invalidFields {
      var fields = readyFields; fields[key] = value
      XCTAssertThrowsError(try H3NativeBlockSession.validateReady(json(fields), rows: 1))
    }
  }
  func testProgressIsOrderedAllFiftyAndBooleanIsNotAnInteger() throws {
    for completed in 1...50 {
      let parsed = try H3NativeBlockSession.validateProgress(json(["event": "progress",
        "completed": completed, "total": 50, "resident_blocks": 1,
        "weight_load_seconds": 0, "weight_preparation_seconds": 0, "compute_seconds": 0,
        "metal_allocated_bytes": 0, "model_scratch_bytes": 0, "cpu_prefetch_blocks": 0]), previous: completed - 1)
      XCTAssertEqual(parsed.completed, completed); XCTAssertEqual(parsed.total, 50)
    }
    for completed: Any in [true, 0, 2, 51] {
      XCTAssertThrowsError(try H3NativeBlockSession.validateProgress(json(["event": "progress",
        "completed": completed, "total": 50, "resident_blocks": 1]), previous: 0))
    }
  }
  func testBoundedLineReaderHandlesSplitsUTF8LimitAndEOF() throws {
    var reader = H3NativeBlockSession.LineBuffer()
    try reader.append(Data("{\"event\":\"re".utf8)); XCTAssertNil(try reader.next())
    try reader.append(Data("ady\"}\n{}\n".utf8))
    XCTAssertEqual(try reader.next(), Data("{\"event\":\"ready\"}".utf8))
    XCTAssertEqual(try reader.next(), Data("{}".utf8)); XCTAssertNil(try reader.next()); XCTAssertNoThrow(try reader.finish())
    var oversized = H3NativeBlockSession.LineBuffer()
    XCTAssertThrowsError(try oversized.append(Data(repeating: 65, count: 65537)))
    var invalid = H3NativeBlockSession.LineBuffer(); try invalid.append(Data([255, 10]))
    XCTAssertThrowsError(try invalid.next())
    var partial = H3NativeBlockSession.LineBuffer(); try partial.append(Data("{}".utf8))
    XCTAssertThrowsError(try partial.finish())
  }
  func testPredictionIDPathDtypeCompleteProgressAndExactByteCount() throws {
    let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: output) }
    try Data(repeating: 0, count: 5376 * 4).write(to: output)
    let fields: [String: Any] = ["event": "prediction", "id": "1", "output": output.path,
      "dtype": "F32", "resident_blocks": 1, "seconds": 0.1, "compute_seconds": 0.1,
      "weight_load_seconds": 0, "weight_preparation_seconds": 0, "metal_allocated_bytes": 0]
    XCTAssertEqual(try H3NativeBlockSession.validatePrediction(json(fields), id: "1", output: output,
      rows: 1, completed: 50).byteCount, 21504)
    for (key, value) in [("id", "wrong"), ("output", "/different"), ("dtype", "BF16")] {
      var invalid = fields; invalid[key] = value
      XCTAssertThrowsError(try H3NativeBlockSession.validatePrediction(json(invalid), id: "1", output: output, rows: 1, completed: 50))
    }
    XCTAssertThrowsError(try H3NativeBlockSession.validatePrediction(json(fields), id: "1", output: output, rows: 1, completed: 49))
    try Data([0]).write(to: output)
    XCTAssertThrowsError(try H3NativeBlockSession.validatePrediction(json(fields), id: "1", output: output, rows: 1, completed: 50))
  }
  private struct FakeWorker: Sendable {
    let root: URL; let worker: URL; let initial: URL; let checkpoint: URL; let adapter: URL; let workspace: URL
  }
  private func fakeWorker(ready: Bool = true, prediction: Bool = false, closeAcknowledgement: Bool = true) throws -> FakeWorker {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let worker = root.appendingPathComponent("worker.sh")
    let initial = root.appendingPathComponent("initial"), checkpoint = root.appendingPathComponent("checkpoint"), adapter = root.appendingPathComponent("adapter")
    for file in [initial, checkpoint, adapter] { try Data([0]).write(to: file) }
    let readyLine = String(decoding: try json(readyFields), as: UTF8.self)
    var response = ""
    if prediction {
      let payload = root.appendingPathComponent("result-source")
      try Data(repeating: 0, count: 5376 * 4).write(to: payload)
      let output = root.appendingPathComponent("workspace/prediction.f32")
      response += "/bin/cp '" + payload.path + "' '" + output.path + "'\n"
      for completed in 1...50 {
        let fields: [String: Any] = ["event": "progress", "completed": completed, "total": 50,
          "resident_blocks": 1, "weight_load_seconds": 0, "weight_preparation_seconds": 0,
          "compute_seconds": 0, "metal_allocated_bytes": 0, "model_scratch_bytes": 0, "cpu_prefetch_blocks": 0]
        response += "printf '%s\\n' '" + String(decoding: try json(fields), as: UTF8.self) + "'\n"
      }
      let fields: [String: Any] = ["event": "prediction", "id": "1", "output": output.path,
        "dtype": "F32", "resident_blocks": 1, "seconds": 0.1, "compute_seconds": 0.1,
        "weight_load_seconds": 0, "weight_preparation_seconds": 0, "metal_allocated_bytes": 0]
      response += "printf '%s\\n' '" + String(decoding: try json(fields), as: UTF8.self) + "'\n"
    }
    let earlyClose = closeAcknowledgement
      ? "case \"$request\" in *'\"op\":\"close\"'*) printf '%s\\n' '{\"event\":\"closed\",\"id\":\"close-0\"}'; exit 0;; esac\n" : ""
    let finalClose = closeAcknowledgement
      ? "printf '%s\\n' '{\"event\":\"closed\",\"id\":\"close-1\"}'\n" : ""
    let body = "#!/bin/sh\n" + (ready ? "printf '%s\\n' '" + readyLine + "'\n" : "")
      + "IFS= read -r request || exit 0\n" + earlyClose + response
      + "IFS= read -r blocked || exit 0\n" + finalClose
    try body.write(to: worker, atomically: true, encoding: .utf8)
    guard chmod(worker.path, 0o700) == 0 else { throw H3CheckpointError.invalid("Cannot prepare pure fake worker.") }
    return FakeWorker(root: root, worker: worker, initial: initial, checkpoint: checkpoint,
      adapter: adapter, workspace: root.appendingPathComponent("workspace"))
  }
  func testPureChildCloseReapsBeforeWorkspaceCleanupAndIsIdempotent() throws {
    let fake = try fakeWorker(); defer { try? FileManager.default.removeItem(at: fake.root) }
    let session = try H3NativeBlockSession(initialURL: fake.initial, checkpointURL: fake.checkpoint,
      adapterURL: fake.adapter, workspaceURL: fake.workspace, rows: 1, workerURL: fake.worker,
      readyTimeout: 2, predictionTimeout: 2, shutdownGrace: 0.05)
    XCTAssertNotNil(session.processIdentifier)
    try session.close(); XCTAssertTrue(session.isClosed); XCTAssertTrue(session.childReaped)
    try session.close()
    // Cleanup follows the explicit joined/reaped boundary, never merely SIGTERM.
    try FileManager.default.removeItem(at: fake.workspace)
  }
  func testCancelledPredictionTerminatesAndReapsPureChild() async throws {
    let fake = try fakeWorker(); defer { try? FileManager.default.removeItem(at: fake.root) }
    let session = try H3NativeBlockSession(initialURL: fake.initial, checkpointURL: fake.checkpoint,
      adapterURL: fake.adapter, workspaceURL: fake.workspace, rows: 1, workerURL: fake.worker,
      readyTimeout: 2, predictionTimeout: 2, shutdownGrace: 0.05)
    let task = Task.detached { try session.predict(input: fake.initial,
      output: fake.workspace.appendingPathComponent("prediction.f32")) }
    task.cancel()
    do { _ = try await task.value; XCTFail("Cancelled prediction unexpectedly succeeded.") }
    catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertTrue(session.isClosed); XCTAssertTrue(session.childReaped)
  }
  func testPureChildCompletePredictionAndCallbackFailureBothRetire() throws {
    enum Stop: Error { case requested }
    for stop in [false, true] {
      let fake = try fakeWorker(prediction: true)
      defer { try? FileManager.default.removeItem(at: fake.root) }
      let session = try H3NativeBlockSession(initialURL: fake.initial, checkpointURL: fake.checkpoint,
        adapterURL: fake.adapter, workspaceURL: fake.workspace, rows: 1, workerURL: fake.worker,
        readyTimeout: 2, predictionTimeout: 2, shutdownGrace: 0.05)
      defer { try? session.close() }
      var progress: [Int] = []
      do {
        let result = try session.predict(input: fake.initial,
          output: fake.workspace.appendingPathComponent("prediction.f32")) { completed, total in
          XCTAssertEqual(total, 50); progress.append(completed)
          if stop { throw Stop.requested }
        }
        XCTAssertFalse(stop); XCTAssertEqual(result.byteCount, 21504)
        XCTAssertEqual(progress, Array(1...50))
        try session.close()
      } catch {
        XCTAssertTrue(stop, "Unexpected session error: \(error)"); XCTAssertTrue(error is Stop, "Unexpected session error: \(error)"); XCTAssertEqual(progress, [1])
      }
      XCTAssertTrue(session.childReaped); XCTAssertTrue(session.isClosed)
    }
  }
  func testMissingCloseAcknowledgementThrowsAfterReaping() throws {
    let fake = try fakeWorker(closeAcknowledgement: false)
    defer { try? FileManager.default.removeItem(at: fake.root) }
    let session = try H3NativeBlockSession(initialURL: fake.initial, checkpointURL: fake.checkpoint,
      adapterURL: fake.adapter, workspaceURL: fake.workspace, rows: 1, workerURL: fake.worker,
      readyTimeout: 2, predictionTimeout: 2, shutdownGrace: 0.05)
    XCTAssertThrowsError(try session.close())
    XCTAssertTrue(session.childReaped); XCTAssertTrue(session.isClosed)
    XCTAssertNoThrow(try session.close())
  }
  func testPreflightRejectsRowBoundsWithoutLaunchingChild() throws {
    let missing = URL(fileURLWithPath: "/no-such-worker")
    for rows in [0, 40001] {
      XCTAssertThrowsError(try H3NativeBlockSession(initialURL: missing, checkpointURL: missing,
        adapterURL: missing, workspaceURL: missing, rows: rows, workerURL: missing))
    }
  }
}
