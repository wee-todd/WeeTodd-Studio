import XCTest
import Combine
import StudioCore
@testable import WeeToddStudio

final class BridgeProgressTests: XCTestCase {
  @MainActor func testPreviewIsLiveRequestScopedAndClearedOnSuccessFailureAndCancel() async throws {
    for command in ["dt-generate-image", "image-generate"] {
    for mode in ["success", "failure", "cancel"] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      try FileManager.default.createDirectory(at: root.appendingPathComponent("scripts"), withIntermediateDirectories: true)
      let script = """
      import json, time, sys
      from pathlib import Path
      root = Path(__file__).parents[1]
      preview = root / 'live-preview.png'
      preview.write_bytes(b'fixture')
      print(json.dumps(dict(event='progress', message='Preview ready', previewPath=str(preview), previewRevision=1)), flush=True)
      print(json.dumps(dict(event='progress', message='Foreign path', previewPath='/tmp/foreign.png', previewRevision=999)), flush=True)
      deadline = time.monotonic() + 10
      while not (root / 'release').exists():
          if time.monotonic() > deadline: raise RuntimeError('handshake timeout')
          time.sleep(0.01)
      if '\(mode)' == 'failure': sys.exit(1)
      print('{"status":"success","result":{}}', flush=True)
      """
      try script.write(to: root.appendingPathComponent("scripts/studio_bridge.py"), atomically: true, encoding: .utf8)
      let bridge = Bridge()
      let settings = RuntimeSettings(root: root.path, pythonPath: "/usr/bin/python3", profilesDirectory: root.path)
      let delivered = expectation(description: "Preview before completion: \(mode)")
      let observation = bridge.$message.sink { if $0 == "Foreign path" { delivered.fulfill() } }
      defer { observation.cancel() }
      let task = Task { try await bridge.invoke(command, runtime: settings, payload: [:], output: root) }
      await fulfillment(of: [delivered], timeout: 5)
      XCTAssertTrue(bridge.busy)
      XCTAssertEqual(bridge.livePreview?.previewRevision, 1)
      XCTAssertEqual(bridge.livePreview?.previewPath, root.appendingPathComponent("live-preview.png").path)
      if mode == "cancel" { bridge.cancel() }
      else { try Data().write(to: root.appendingPathComponent("release")) }
      do { _ = try await task.value; XCTAssertEqual(mode, "success") }
      catch { XCTAssertNotEqual(mode, "success") }
      XCTAssertNil(bridge.livePreview)
      XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("live-preview.png").path))
    }
    }
  }

  @MainActor func testLiveChildProgressAndCompletedResponseAreBothDelivered() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    let script = """
    import sys, time
    from pathlib import Path
    sys.stdout.write('{"event":"progress","message":"Sampling ')
    sys.stdout.flush()
    sys.stdout.write('3/16","fraction":0.1875}\\n')
    sys.stdout.flush()
    deadline = time.monotonic() + 10
    while not (Path(__file__).parents[1] / 'release').exists():
        if time.monotonic() > deadline: raise RuntimeError('test handshake timed out')
        time.sleep(0.01)
    print('{"status":"success","result":{"fixture":true}}', flush=True)
    """
    try script.write(to: root.appendingPathComponent("scripts/studio_bridge.py"), atomically: true, encoding: .utf8)
    let bridge = Bridge()
    let settings = RuntimeSettings(root: root.path, pythonPath: "/usr/bin/python3", profilesDirectory: root.path)
    let delivered = expectation(description: "Progress arrives before the child completes")
    let subscription = bridge.$message.sink { if $0 == "Sampling 3/16" { delivered.fulfill() } }
    defer { subscription.cancel() }
    let invocation = Task { try await bridge.invoke("render", runtime: settings, payload: [:]) }
    await fulfillment(of: [delivered], timeout: 5)
    XCTAssertTrue(bridge.busy)
    XCTAssertEqual(bridge.message, "Sampling 3/16")
    XCTAssertEqual(bridge.fraction, 0.1875)
    try Data().write(to: root.appendingPathComponent("release"))
    let result = try await invocation.value
    XCTAssertEqual(result["fixture"] as? Bool, true)
    XCTAssertNotNil(bridge.startedAt)
    XCTAssertNotNil(bridge.lastOutputAt)
    XCTAssertFalse(bridge.busy)
  }
}

final class WorkflowResponseBufferTests: XCTestCase {
  func testLargeWorkflowResultSurvivesProgressBufferTrimming() {
    var buffer = BridgeResponseBuffer(workflow: true)
    buffer.append(Data(repeating: 65, count: 4_000_000))
    let result = Data(("\n{\"status\":\"success\",\"result\":\"" + String(repeating: "z", count: 2_097_152) + "\"}\n").utf8)
    for start in stride(from: 0, to: result.count, by: 65536) {
      buffer.append(result.subdata(in: start..<min(start + 65536, result.count)))
    }
    let last = String(decoding: buffer.data, as: UTF8.self).split(separator: "\n").last!
    XCTAssertNotNil(try? JSONSerialization.jsonObject(with: Data(last.utf8)))
    XCTAssertLessThanOrEqual(buffer.data.count, 17 * 1024 * 1024)
  }
}


extension WorkflowResponseBufferTests {
  func testMaximumCheckpointResponseSurvivesChunkedProgressAndRemainsBounded() throws {
    var buffer = BridgeResponseBuffer(workflow: true)
    let progress = Data(("{\"event\":\"progress\",\"message\":\"" + String(repeating: "p", count: 6000) + "\"}\n").utf8)
    for _ in 0..<4000 { buffer.append(progress) }
    // A near-limit persisted document must survive the bridge's success wrapper.
    let evidence = String(repeating: "x", count: 16 * 1024 * 1024 - 100)
    let result = Data(("{\"status\":\"success\",\"result\":{\"evidence\":\"" + evidence + "\"}}\n").utf8)
    for start in stride(from: 0, to: result.count, by: 16_387) {
      buffer.append(result.subdata(in: start..<min(start + 16_387, result.count)))
      XCTAssertLessThanOrEqual(buffer.data.count, 17 * 1024 * 1024)
    }
    let finalLine = try XCTUnwrap(String(decoding: buffer.data, as: UTF8.self).split(separator: "\n").last)
    let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(finalLine.utf8)) as? [String: Any])
    XCTAssertEqual((decoded["result"] as? [String: Any])?["evidence"] as? String, evidence)
    // One unusually large read must still retain the final complete response.
    let smallResult = Data("\n{\"status\":\"success\",\"result\":{\"revision\":\"latest\"}}\n".utf8)
    buffer.append(Data(repeating: 65, count: 20 * 1024 * 1024) + smallResult)
    XCTAssertLessThanOrEqual(buffer.data.count, 17 * 1024 * 1024)
    XCTAssertTrue(buffer.data.suffix(smallResult.count).elementsEqual(smallResult))
  }

  func testNonWorkflowResponseRetentionIsUnchanged() {
    var buffer = BridgeResponseBuffer(workflow: false)
    buffer.append(Data(repeating: 65, count: 1_500_000))
    buffer.append(Data(repeating: 66, count: 600_000))
    XCTAssertEqual(buffer.data, Data(repeating: 65, count: 400_000) + Data(repeating: 66, count: 600_000))
  }
}

extension BridgeProgressTests {
  @MainActor func testLargeSuccessfulWorkflowResponseIsNotReportedAsFailed() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    let script = """
    import json, sys
    print(json.dumps(dict(event='progress', message='Saving reviewed objects', fraction=0.9)), flush=True)
    result = json.dumps(dict(status='success', result=dict(status='awaiting_approval', evidence='x' * 4_700_000))) + '\\n'
    for start in range(0, len(result), 16387):
        sys.stdout.write(result[start:start+16387])
        sys.stdout.flush()
    """
    try script.write(to: root.appendingPathComponent("scripts/studio_bridge.py"), atomically: true, encoding: .utf8)
    let bridge = Bridge()
    let settings = RuntimeSettings(root: root.path, pythonPath: "/usr/bin/python3", profilesDirectory: root.path)
    let result = try await bridge.invoke("workflow-review", runtime: settings, payload: [:])
    XCTAssertEqual(result["status"] as? String, "awaiting_approval")
    XCTAssertEqual((result["evidence"] as? String)?.count, 4_700_000)
    XCTAssertFalse(bridge.busy)
  }
}
