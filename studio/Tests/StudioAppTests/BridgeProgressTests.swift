import XCTest
import Combine
import StudioCore
@testable import WeeToddStudio

final class BridgeProgressTests: XCTestCase {
  @MainActor func testInstalledNativeH3PreflightWithoutPython() async throws {
    guard let source = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_PREFLIGHT_RECIPE"],
      let worker = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_PREFLIGHT_WORKER"] else {
      throw XCTSkip("Opt-in installed H3 worker preflight")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let profile = root.appendingPathComponent("h3.json")
    try FileManager.default.copyItem(at: URL(fileURLWithPath: source), to: profile)
    let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: profile)) as! [String: Any]
    let config = recipe["config"] as! [String: Any]
    var project = StudioProject(); var clip = Clip(engine: .h3)
    clip.profileID = profile.path; clip.prompt = recipe["prompt"] as? String ?? "A quiet lake."
    clip.duration = config["duration_seconds"] as! Double
    clip.generationWidth = config["width"] as! Int; clip.generationHeight = config["height"] as! Int
    clip.seed = config["seed"] as! Int; project.clips = [clip]
    var settings = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
    settings.h3WorkerPath = worker
    settings.ffmpegPath = "/opt/homebrew/bin/ffmpeg"
    let bridge = Bridge()
    let payload: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": clip.id.uuidString]
    let prepared = try await bridge.invoke("h3-native-prepare", runtime: settings, payload: payload,
      output: root.appendingPathComponent("prepared"))
    let path = try XCTUnwrap(prepared["recipePath"] as? String)
    let result = try await bridge.invoke("h3-native-preflight", runtime: settings,
      payload: ["recipePath": path], output: root.appendingPathComponent("render"))
    XCTAssertEqual(result["nativeRuntime"] as? String, "swift-mlx")
    XCTAssertEqual(result["productionQualified"] as? Bool, false)
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("render").path))
  }
  @MainActor func testH3NativePreparationAndWorkerUseNoPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "old",
      "components": ["task": "t2va", "transformer": "/model/t", "text_encoder": "/model/q",
        "tokenizer": "/model/tokenizer", "video_vae": "/model/v", "audio_vae": "/model/a"],
      "config": ["width": 768, "height": 448, "duration_seconds": 5.0,
        "steps": 5, "seed": 1, "drop_adaln": true],
      "conditioning": ["version": 1, "task": "t2v", "inputs": []]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("h3.json"))
    let worker = root.appendingPathComponent("worker")
    try """
    #!/bin/sh
    test "$1" = preflight || exit 5
    test "$2" = --request || exit 6
    test "$4" = --output || exit 7
    cp "$3" '\(root.path)/envelope.json'
    echo '{"status":"success","result":{"nativeRuntime":"swift-mlx"}}'
    """.write(to: worker, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    var settings = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
    settings.ffmpegPath = "/usr/bin/true"; settings.h3WorkerPath = worker.path
    let bridge = Bridge()
    let catalog = try await bridge.invoke("h3-native-catalog", runtime: settings, payload: [:])
    XCTAssertEqual((catalog["profiles"] as? [Any])?.count, 1)
    var project = StudioProject(); var clip = Clip(engine: .h3)
    clip.prompt = "A quiet lake."; clip.profileID = root.appendingPathComponent("h3.json").path
    project.clips = [clip]
    let payload: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
      "clipID": clip.id.uuidString]
    let prepared = try await bridge.invoke("h3-native-prepare", runtime: settings,
      payload: payload, output: root.appendingPathComponent("prepared"))
    let path = try XCTUnwrap(prepared["recipePath"] as? String)
    _ = try await bridge.invoke("h3-native-preflight", runtime: settings,
      payload: ["recipePath": path], output: root.appendingPathComponent("render"))
    let envelope = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("envelope.json"))) as! [String: Any]
    XCTAssertEqual(envelope["engine"] as? String, "h3")
    XCTAssertEqual(envelope["recipePath"] as? String, path)
  }
  @MainActor func testNativeCatalogDescribeAndPrepareWithMissingPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "config": ["pipeline_mode": "distilled", "stage1_steps": 8, "stage2_steps": 3, "frame_rate": 24],
      "components": [:], "prompt": "old"]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("model.json"))
    var settings = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
    settings.ffmpegPath = "/usr/bin/true"
    var project = StudioProject(); var clip = Clip(); clip.prompt = "A still landscape."; project.clips = [clip]
    let payload: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)), "clipID": clip.id.uuidString]
    let bridge = Bridge()
    let catalog = try await bridge.invoke("ltx-native-catalog", runtime: settings, payload: [:])
    XCTAssertEqual((catalog["profiles"] as? [Any])?.count, 1)
    let description = try await bridge.invoke("ltx-native-describe", runtime: settings, payload: payload)
    XCTAssertEqual(description["readinessErrors"] as? [String], [])
    let result = try await bridge.invoke("ltx-native-prepare", runtime: settings, payload: payload, output: root.appendingPathComponent("prepared"))
    XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(result["recipePath"] as? String)))
    XCTAssertFalse(bridge.busy)
  }

  @MainActor func testNativeWorkerRunsWithoutPythonAndCleansItsPreview() async throws {
    for mode in ["success", "failure", "cancel"] {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let output=root.appendingPathComponent("render"),preview=output.appendingPathExtension("preview.png")
    let helper=root.appendingPathComponent("worker")
    let recipe=root.appendingPathComponent("recipe.json")
    try Data("abc".utf8).write(to:recipe)
    try """
    #!/bin/sh
    trap 'exit 130' INT TERM
    test "$1" = render || exit 5
    test "$2" = --request || exit 6
    test "$4" = --output || exit 7
    cp "$3" '\(root.path)/envelope.json'
    touch '\(preview.path)'
    echo '{"event":"progress","message":"Native decoded preview","fraction":0.9,"previewPath":"\(preview.path)","previewRevision":1}'
    i=0
    while ! test -f '\(root.path)/release'; do
      test "$i" -lt 200 || exit 8
      i=$((i+1))
      sleep 0.05
    done
    test '\(mode)' != failure || exit 1
    echo '{"status":"success","result":{"nativeRuntime":"swift-mlx"}}'
    """.write(to:helper,atomically:true,encoding:.utf8)
    try FileManager.default.setAttributes([.posixPermissions:0o755],ofItemAtPath:helper.path)
    var settings=RuntimeSettings(root:"/missing",pythonPath:"/missing/python",profilesDirectory:root.path)
    settings.ltx25WorkerPath=helper.path
    let bridge=Bridge(),delivered=expectation(description:"Native progress")
    let observation=bridge.$message.sink { if $0 == "Native decoded preview" { delivered.fulfill() } }
    defer { observation.cancel() }
    let job=Task { try await bridge.invoke("ltx-native-render",runtime:settings,payload:["recipePath":recipe.path],output:output) }
    await fulfillment(of:[delivered],timeout:5)
    let envelope=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("envelope.json"))) as! [String:Any]
    XCTAssertEqual(envelope["version"] as? Int,1)
    XCTAssertEqual(envelope["engine"] as? String,"ltx25")
    XCTAssertEqual(envelope["recipePath"] as? String,recipe.path)
    XCTAssertEqual(envelope["outputDirectory"] as? String,output.path)
    XCTAssertEqual(envelope["recipeSHA256"] as? String,
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    XCTAssertEqual(bridge.livePreview?.previewRevision,1)
    if mode == "cancel" { bridge.cancel() }
    else { try Data().write(to:root.appendingPathComponent("release")) }
    do {
      let result=try await job.value
      XCTAssertEqual(mode,"success")
      XCTAssertEqual(result["nativeRuntime"] as? String,"swift-mlx")
    } catch { XCTAssertNotEqual(mode,"success") }
    XCTAssertNil(bridge.livePreview)
    XCTAssertFalse(FileManager.default.fileExists(atPath:preview.path))
    }
  }

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
