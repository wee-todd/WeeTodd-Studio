import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeWorkerEventArchiveTests: XCTestCase {
  @MainActor func testNativeWorkerArchivesFullProgressBeyondUITailAndKeepsCancelledPartialLog() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("NativeArchive-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe = root.appendingPathComponent("recipe.json")
    try Data("fixture recipe".utf8).write(to: recipe)
    let prefix = "{\"event\":\"progress\",\"message\":\"text released\",\"fraction\":0.1}\n"
      + String(repeating: "{\"event\":\"progress\",\"message\":\"sampling evidence survives the display tail\",\"fraction\":0.5}\n", count: 500)
      + "{\"event\":\"progress\",\"message\":\"transformer released\",\"fraction\":0.8}\n"
    let events = root.appendingPathComponent("events")
    try Data(prefix.utf8).write(to: events)
    var archives = Set<String>()
    for (engine, action, mode) in [("ltx", "render", "success"), ("h3", "preflight", "success"),
      ("h3", "render", "failure"), ("ltx", "render", "cancel")] {
      let worker = root.appendingPathComponent("worker-" + UUID().uuidString)
      let release = root.appendingPathComponent("release-" + UUID().uuidString)
      let terminal = mode == "success" ? "{\"status\":\"success\",\"result\":{\"nativeRuntime\":\"swift-mlx\"}}"
        : "{\"status\":\"error\",\"error\":\"fixture failed\"}"
      let script = """
      #!/bin/sh
      trap 'echo "{\\"status\\":\\"cancelled\\",\\"error\\":\\"fixture cancelled\\"}"; exit 130' INT TERM
      /bin/cat '\(events.path)'
      while ! test -f '\(release.path)'; do sleep 0.05; done
      echo '{"event":"progress","message":"audio released","fraction":1}'
      echo '\(terminal)'
      exit \(mode == "failure" ? "1" : "0")
      """
      try script.write(to: worker, atomically: true, encoding: .utf8)
      try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
      var runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
      runtime.h3WorkerPath = worker.path; runtime.ltx25WorkerPath = worker.path
      let settings = runtime
      let bridge = Bridge(), progressed = expectation(description: "Early and late native progress")
      let observation = bridge.$message.sink { if $0 == "transformer released" { progressed.fulfill() } }
      defer { observation.cancel() }
      let output = root.appendingPathComponent("atomic-worker-output")
      let run = Task { try await bridge.invoke(engine + "-native-" + action, runtime: settings,
        payload: ["recipePath": recipe.path], output: output) }
      await fulfillment(of: [progressed], timeout: 5)
      XCTAssertFalse(bridge.log.contains("text released")); XCTAssertLessThanOrEqual(bridge.log.count, 30000)
      if mode == "cancel" { bridge.cancel() }
      else { try Data().write(to: release) }
      do {
        let result = try await run.value
        XCTAssertEqual(mode, "success")
        XCTAssertEqual(result["workerEventsPath"] as? String, bridge.workerEventsPath)
        XCTAssertEqual(result["workerEventsTruncated"] as? Bool, false)
      } catch { XCTAssertNotEqual(mode, "success") }
      let archivePath = try XCTUnwrap(bridge.workerEventsPath)
      XCTAssertTrue(archives.insert(archivePath).inserted, "Repeated destinations must retain distinct logs")
      XCTAssertFalse(bridge.workerEventsTruncated); XCTAssertNil(bridge.workerEventsArchiveError)
      let bytes = try Data(contentsOf: URL(fileURLWithPath: archivePath))
      XCTAssertTrue(bytes.starts(with: Data(prefix.utf8))); XCTAssertGreaterThan(bytes.count, 30000)
      let archived = String(decoding: bytes, as: UTF8.self)
      XCTAssertTrue(archived.contains("text released")); XCTAssertTrue(archived.contains("transformer released"))
      XCTAssertTrue(archived.contains(mode == "cancel" ? "fixture cancelled" : "audio released"))
      XCTAssertFalse(FileManager.default.fileExists(atPath: output.path), "Archive must not create atomic worker output")
    }
  }

  func testArchiveBoundAndDiskFailureAreExplicitWithoutStoppingDrain() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("NativeArchive-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let archive = try NativeWorkerEventArchive(output: root.appendingPathComponent("take"), jobID: UUID().uuidString, maximumBytes: 128)
    archive.append(Data(repeating: 65, count: 100)); archive.append(Data(repeating: 66, count: 100))
    archive.append(Data(repeating: 67, count: 100)); archive.finish()
    XCTAssertTrue(archive.truncated); XCTAssertEqual(archive.bytesWritten, 128)
    XCTAssertEqual(try Data(contentsOf: archive.url), Data(repeating: 65, count: 100) + Data(repeating: 66, count: 28))
    XCTAssertTrue(archive.errorMessage?.contains("limit") == true)
    let failed = try NativeWorkerEventArchive(output: root.appendingPathComponent("take"), jobID: UUID().uuidString,
      writer: { _, _ in throw CocoaError(.fileWriteOutOfSpace) })
    failed.append(Data("first".utf8)); failed.append(Data("last".utf8)); failed.finish()
    XCTAssertTrue(failed.truncated); XCTAssertEqual(failed.bytesWritten, 0); XCTAssertNotNil(failed.errorMessage)
  }
}
