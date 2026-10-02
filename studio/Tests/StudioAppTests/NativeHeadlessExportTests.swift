import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeHeadlessExportTests: XCTestCase {
  private func directory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NativeExport-\(UUID())")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    return directory
  }
  @MainActor func testStudioNativeExporterPreparesThroughSharedServicesWithoutPython() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = try directory()
      var commands: [String] = [], preparedBytes = Data()
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, runtime, payload, output in
        commands.append(command)
        XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
        if command.hasSuffix("-prepare") {
          let target = try XCTUnwrap(output)
          try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
          let recipe = target.appendingPathComponent("recipe.json")
          preparedBytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2",
            "engine": engine.rawValue, "prompt": "Exact composed AV prompt", "components": [:],
            "config": ["seed": 7744, "width": 64, "height": 64]])
          try preparedBytes.write(to: recipe)
          return ["recipePath": recipe.path, "report": ["nativeRuntime": "swift-mlx"]]
        }
        XCTAssertTrue(command.hasSuffix("-preflight")); return ["nativeRuntime": "swift-mlx"]
      })
      store.runtime.root = "/unavailable"; store.runtime.pythonPath = "/unavailable/python"
      store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
      store.runtime.h3WorkerPath = "/usr/bin/true"; store.runtime.ltx25WorkerPath = "/usr/bin/true"
      store.runtime.ffmpegPath = "/usr/bin/true"
      var clip = Clip(engine: engine); clip.duration = 1; clip.seed = 7744; clip.prompt = "Original editor prompt"
      clip.generationWidth = 64; clip.generationHeight = 64
      store.project.clips = [clip]; store.selectedClipID = clip.id
      XCTAssertTrue(store.nativeHeadlessEligible)
      var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
      let url = root.appendingPathComponent("clip.weetodd-job.json")
      try await store.exportNativeHeadlessJob(body: body, to: url, clipOnly: true)
      let job = try NativeHeadlessJob.read(from: url)
      XCTAssertEqual(commands, [engine == .h3 ? "h3-native-prepare" : "ltx-native-prepare",
        engine == .h3 ? "h3-native-preflight" : "ltx-native-preflight"])
      XCTAssertEqual(job.recipes[clip.id.uuidString]?.bytes, preparedBytes)
      XCTAssertEqual(job.project.clips[0].prompt, clip.prompt); XCTAssertEqual(job.project.clips[0].seed, 7744)
      XCTAssertEqual(store.project.clips, [clip]); XCTAssertNil(store.error)
    }
  }
  @MainActor func testNativeExporterRejectsUnsupportedFinishingBeforePreparation() async throws {
    let root = try directory()
    var calls = 0
    let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { _, _, _, _ in calls += 1; return [:] })
    store.runtime.nativeLTX25Enabled = true; store.runtime.ltx25WorkerPath = "/usr/bin/true"
    store.runtime.ffmpegPath = "/usr/bin/true"
    var clip = Clip(); clip.sourcePan = 0.5; store.project.clips = [clip]; store.selectedClipID = clip.id
    var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
    do { try await store.exportNativeHeadlessJob(body: body, to: root.appendingPathComponent("job.json"), clipOnly: false); XCTFail("Pan cannot disappear") }
    catch { XCTAssertTrue(error.localizedDescription.contains("pan")) }
    XCTAssertEqual(calls, 0)
  }
  @MainActor func testNativeExporterRejectsPendingRippleBeforeOrdinaryPreparation() async throws {
    let root = try directory()
    var calls = 0
    let store = StudioStore(dataDirectory: root, restoreSession: false,
      invocation: { _, _, _, _ in calls += 1; return [:] })
    store.runtime.nativeLTX25Enabled = true; store.runtime.nativeRippleEnabled = true
    store.runtime.pythonPath = "/unavailable/python"
    var clip = Clip(engine: .ltx25)
    clip.sourcePath = root.appendingPathComponent("source.mov").path; clip.duration = 2
    clip.rippleDraft = RippleDraft(clip: clip, frameRate: 24)
    clip.rippleDraft!.references[0].path = root.appendingPathComponent("edited.png").path
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
    let destination = root.appendingPathComponent("ripple.weetodd-job.json")
    do {
      try await store.exportNativeHeadlessJob(body: body, to: destination, clipOnly: true)
      XCTFail("Ripple must never become ordinary LTX generation")
    } catch { XCTAssertTrue(error.localizedDescription.contains("pending Ripple")) }
    XCTAssertEqual(calls, 0)
    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    XCTAssertEqual(store.project.clips, [clip])
  }
  @MainActor func testInstalledStudioNativeExportWithoutPython() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_HEADLESS_EXPORT"] else {
      throw XCTSkip("Opt-in native export/preflight using existing installed Studio inputs; no inference")
    }
    let options = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let root = URL(fileURLWithPath: try XCTUnwrap(options["output"]))
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    let profile = URL(fileURLWithPath: try XCTUnwrap(options["recipe"]))
    let copied = profiles.appendingPathComponent("matched.json")
    if !FileManager.default.fileExists(atPath: copied.path) { try FileManager.default.copyItem(at: profile, to: copied) }
    let request = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(options["editorRequest"])))) as! [String: Any]
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project = try JSONDecoder().decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: request["project"]!))
    store.globalAssets = try JSONDecoder().decode([MediaAsset].self, from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    store.selectedClipID = UUID(uuidString: request["clipID"] as! String)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
    store.runtime.h3WorkerPath = options["h3Worker"]; store.runtime.ltx25WorkerPath = options["ltx25Worker"]
    store.runtime.ffmpegPath = options["ffmpeg"] ?? "/opt/homebrew/bin/ffmpeg"
    let selected = try XCTUnwrap(store.selectedClip)
    if let index = store.project.clips.firstIndex(where: { $0.id == selected.id }) { store.project.clips[index].profileID = copied.path }
    var body = try store.payload(); body["generateIDs"] = [selected.id.uuidString]
    let job = root.appendingPathComponent("exported.weetodd-job.json")
    try await store.exportNativeHeadlessJob(body: body, to: job, clipOnly: options["clipOnly"] != "false")
    let frozen = try NativeHeadlessJob.read(from: job)
    XCTAssertEqual(frozen.project.clips.first(where: { $0.id == selected.id })?.prompt, selected.prompt)
    let preflight = try await NativeHeadlessExecutor.run(job: frozen,
      output: root.appendingPathComponent("CLI-preflight"), preflightOnly: true)
    XCTAssertEqual(preflight["python_inference"] as? Bool, false)
    try JSONSerialization.data(withJSONObject: ["exportedJob": job.path, "pythonAvailable": false,
      "inferenceExecuted": false, "clipID": selected.id.uuidString], options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("export-qualification.json"))
  }
}
