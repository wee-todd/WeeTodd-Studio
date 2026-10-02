import AVFoundation
import Foundation
import XCTest
@testable import StudioCore

final class NativeHeadlessJobTests: XCTestCase {
  private func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("NativeJob-\(UUID())")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }
  private func process(_ executable: String, _ arguments: [String]) throws {
    let child = Process(); child.executableURL = URL(fileURLWithPath: executable)
    child.arguments = arguments; child.standardOutput = FileHandle.nullDevice; child.standardError = FileHandle.nullDevice
    try child.run(); child.waitUntilExit(); XCTAssertEqual(child.terminationStatus, 0)
  }
  private var ffmpeg: String { "/opt/homebrew/bin/ffmpeg" }
  private func movie(_ root: URL, duration: Double = 1) throws -> URL {
    guard FileManager.default.isExecutableFile(atPath: ffmpeg) else { throw XCTSkip("FFmpeg is required for the tiny audiovisual fixture") }
    let url = root.appendingPathComponent("fixture.mp4")
    try process(ffmpeg, ["-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24:d=\(duration)",
      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=\(duration)",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-y", url.path])
    return url
  }
  private func worker(_ root: URL, fixture: URL, wrongID: Bool = false) throws -> URL {
    let url = root.appendingPathComponent("worker")
    let content = #"""
      #!/bin/sh
      set -eu
      mode="$1"; request="$3"; output="$5"
      id=$(/usr/bin/sed -n 's/.*"jobID"[ ]*:[ ]*"\([^"]*\)".*/\1/p' "$request")
      \#(wrongID ? "id=00000000-0000-0000-0000-000000000001" : "")
      printf '{"event":"progress","fraction":0.5,"message":"fixture sampling"}\n'
      if [ "$mode" = render ]; then
        /bin/mkdir "$output"
        /bin/cp '\#(fixture.path)' "$output/render.mp4"
        printf '{"status":"success","result":{"jobID":"%s","nativeRuntime":"swift-mlx","seconds":1.25,"metadata":{"peak_process_footprint_bytes":12345},"video":"%s/render.mp4"}}\n' "$id" "$output"
      else
        printf '{"status":"success","result":{"jobID":"%s","nativeRuntime":"swift-mlx","task":"fixture"}}\n' "$id"
      fi
      """#
    try content.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
    return url
  }
  private func project(_ engine: Engine = .ltx25) -> StudioProject {
    var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
    var clip = Clip(engine: engine); clip.duration = 1; clip.generationWidth = 64; clip.generationHeight = 64
    clip.prompt = "Frozen prompt with quotes: \"bright\" and unicode 🎬"; clip.seed = 7654
    project.clips = [clip]; return project
  }
  private func recipe(_ engine: Engine, task: String = "t2v", extra: [String: Any] = [:]) throws -> Data {
    var value: [String: Any] = ["format": "weetodd-headless-v2", "engine": engine.rawValue,
      "prompt": "Frozen composed audiovisual prompt", "config": ["seed": 7654, "width": 64, "height": 64],
      "conditioning": ["task": task, "inputs": [], "audio_policy": "generated"], "components": [:]]
    value.merge(extra) { _, new in new }
    return try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
  }
  func testNativeManifestRoundTripPreservesExactRecipeAndRejectsTampering() throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    for engine in [Engine.h3, .ltx25] {
      let project = project(engine), bytes = try recipe(engine)
      let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString:
        .init(engine: engine.rawValue, bytes: bytes, signature: "fixture")], workers: [engine.rawValue: worker.path], ffmpeg: ffmpeg)
      let url = root.appendingPathComponent(engine.rawValue + ".json"); try job.write(to: url)
      let reopened = try NativeHeadlessJob.read(from: url)
      XCTAssertEqual(reopened.project, project); XCTAssertEqual(reopened.recipes.values.first?.bytes, bytes)
      XCTAssertThrowsError(try NativeHeadlessJob.read(from: url,
        workerOverrides: [engine == .h3 ? "ltx25" : "h3": worker.path]))
      XCTAssertThrowsError(try job.write(to: url))
      var envelope = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
      envelope["payloadSHA256"] = String(repeating: "0", count: 64)
      try JSONSerialization.data(withJSONObject: envelope).write(to: url)
      XCTAssertThrowsError(try NativeHeadlessJob.read(from: url))
    }
  }
  func testNativeExecutorPreflightsAllRoutesPreservesRecipesAndNeverCallsPython() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    let cases: [(Engine, String, [String: Any])] = [
      (.ltx25, "t2v", [:]), (.ltx25, "i2v", [:]), (.ltx25, "fflf", [:]), (.ltx25, "a2v", [:]),
      (.ltx25, "extension", [:]), (.ltx25, "ref2va", ["candidate": "msr"]),
      (.ltx25, "control", ["candidate": "ingredients"]), (.ltx25, "control", ["candidate": "union"]),
      (.ltx25, "t2v", ["config": ["dfr_enabled": true, "dfr_temporal_rounds": 2, "seed": 7654]]),
      (.h3, "t2v", [:]), (.h3, "fflf", [:]), (.h3, "ref2va", [:]), (.h3, "a2v", [:]),
      (.h3, "t2v", ["continuation": ["mode": "motion"]]), (.h3, "ref2va", ["candidate": "external-after"]),
    ]
    for (index, item) in cases.enumerated() {
      let project = project(item.0), bytes = try recipe(item.0, task: item.1, extra: item.2)
      let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString:
        .init(engine: item.0.rawValue, bytes: bytes, signature: "fixture")], workers: [item.0.rawValue: worker.path], ffmpeg: ffmpeg)
      var calls: [String] = []
      let result = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("route-\(index)"),
        preflightOnly: true, worker: { selected, recipe, output, mode in
          XCTAssertEqual(selected, worker.path); XCTAssertEqual(try Data(contentsOf: recipe), bytes)
          XCTAssertEqual(mode, "preflight"); XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
          calls.append(mode); return ["nativeRuntime": "swift-mlx"]
        })
      XCTAssertEqual(calls, ["preflight"]); XCTAssertEqual(result["python_inference"] as? Bool, false)
    }
  }
  func testNativeActualWorkerReceiptFinishingResumeAndReopen() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    let project = project(), id = project.clips[0].id.uuidString, bytes = try recipe(.ltx25)
    let descriptor: [String: Any] = ["supportedTasks": ["t2v"], "presets": [], "controls": [
      "evaluations": 8, "refinementSteps": 3, "stepsEditable": false, "refinementStepsEditable": false,
      "cfgEditable": false, "shiftEditable": false, "stepsExplanation": "Frozen schedule",
      "cfgExplanation": "Distilled guidance", "shiftExplanation": "No override"]]
    let job = try NativeHeadlessJob(project: project, recipes: [id: .init(engine: "ltx25", bytes: bytes, signature: "frozen",
        report: try JSONSerialization.data(withJSONObject: ["resolvedFingerprint": "fixture-resolved", "generation": descriptor]))],
      workers: ["ltx25": worker.path], ffmpeg: ffmpeg)
    var events = Data()
    let output = root.appendingPathComponent("render")
    let result = try await NativeHeadlessExecutor.run(job: job, output: output, emit: { events.append($0) })
    XCTAssertEqual(result["python_inference"] as? Bool, false)
    XCTAssertEqual(result["generations"] as? Int, 1)
    XCTAssertEqual(result["newlyGenerated"] as? Int, 1); XCTAssertEqual(result["resumedGenerations"] as? Int, 0)
    XCTAssertTrue(String(decoding: events, as: UTF8.self).contains("fixture sampling"))
    let logs = try FileManager.default.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)
      .filter { $0.lastPathComponent.hasSuffix(".events.jsonl") }
    XCTAssertEqual(logs.count, 2, "Both admission and generation retain complete worker events")
    for log in logs {
      let lines = try String(contentsOf: log, encoding: .utf8).split(separator: "\n")
      XCTAssertEqual(lines.count, 2)
      XCTAssertTrue(lines[0].contains("fixture sampling"))
      XCTAssertTrue(lines[1].contains("success"))
    }
    let reopened = try ProjectStorage.read( output.appendingPathComponent("result.weetodd"))
    XCTAssertEqual(reopened.clips[0].seed, 7654); XCTAssertEqual(reopened.clips[0].prompt, project.clips[0].prompt)
    XCTAssertEqual(reopened.clips[0].versions.last?.recipePath, output.appendingPathComponent("recipes/\(id).json").path)
    XCTAssertEqual(reopened.clips[0].versions.last?.prompt, "Frozen composed audiovisual prompt")
    XCTAssertEqual(reopened.clips[0].versions.last?.stats?.elapsedSeconds, 1.25)
    XCTAssertEqual(reopened.clips[0].versions.last?.stats?.processPeakBytes, 12345)
    XCTAssertEqual(reopened.clips[0].versions.last?.resolvedFingerprint, "fixture-resolved")
    XCTAssertEqual(reopened.clips[0].versions.last?.generationSettings?.supportedTasks, ["t2v"])
    XCTAssertEqual(reopened.clips[0].versions.last?.generationSettings?.controls.evaluations, 8)
    XCTAssertEqual(reopened.assets.count, 1)
    XCTAssertEqual(reopened.assets[0].scope, .clip); XCTAssertEqual(reopened.assets[0].owner, project.clips[0].id)
    XCTAssertEqual(reopened.assets[0].path, reopened.clips[0].sourcePath)
    let movie = URL(fileURLWithPath: result["video"] as! String)
    let before = try NativeHeadlessJob.fileHash(movie)
    var calls = 0
    let resumed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }, worker: { _, _, _, _ in
      calls += 1; XCTFail("Completed receipts must not be rendered again"); return [:]
    })
    XCTAssertEqual(resumed["generations"] as? Int, 1)
    XCTAssertEqual(resumed["newlyGenerated"] as? Int, 0); XCTAssertEqual(resumed["resumedGenerations"] as? Int, 1)
    XCTAssertEqual(calls, 0); XCTAssertEqual(try NativeHeadlessJob.fileHash(movie), before)
    try Data("changed worker".utf8).append(to: worker)
    await XCTAssertThrowsErrorAsync { _ = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }) }
  }
  func testAllPendingWorkersPreflightBeforeFirstRender() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    var project = project(); var second = project.clips[0]; second.id = UUID(); second.engine = .h3; project.clips.append(second)
    let recipes = try Dictionary(uniqueKeysWithValues: project.clips.map {
      ($0.id.uuidString, NativeHeadlessJob.Recipe(engine: $0.engine.rawValue, bytes: try recipe($0.engine), signature: "fixture"))
    })
    let job = try NativeHeadlessJob(project: project, recipes: recipes, workers: ["h3": worker.path, "ltx25": worker.path], ffmpeg: ffmpeg)
    var calls: [String] = []
    await XCTAssertThrowsErrorAsync { _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("admission"), worker: { _, _, _, mode in
      calls.append(mode); if calls.count == 2 { throw StudioError.invalid("second checkpoint unavailable") }; return [:]
    }) }
    XCTAssertEqual(calls, ["preflight", "preflight"])
  }
  func testCancellationBeforeFirstEventTerminatesWorkerAndPublishesNoTake() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    try "#!/bin/sh\ntrap 'exit 130' INT TERM\nwhile true; do sleep 1; done\n".write(to: worker, atomically: true, encoding: .utf8)
    let recipeURL = root.appendingPathComponent("recipe.json"); try recipe(.h3).write(to: recipeURL)
    let cancellation = NativeHeadlessCancellation(), output = root.appendingPathComponent("cancelled")
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { cancellation.cancel() }
    XCTAssertThrowsError(try NativeHeadlessExecutor.worker(worker.path, recipe: recipeURL, output: output,
      mode: "render", cancellation: cancellation, emit: { _ in })) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
  }
  func testNativeSceneUsesOneWorkerAndReopensSharedTakeWithExactRanges() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    for mode in ["single_decode_native_latent_chain", "windowed_decode_native_latent_chain"] {
      var project = project(); project.clips[0].duration = 0.5
      var second = project.clips[0]; second.id = UUID(); second.continuity = .init(mode: "scene", sourceClipID: project.clips[0].id)
      project.clips.append(second)
      let members = [ContinuousSceneMember(clipID: project.clips[0].id, sourceIn: 0, duration: 0.5),
        ContinuousSceneMember(clipID: second.id, sourceIn: 0.5, duration: 0.5)]
      let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(members))
      let scene: [String: Any] = ["version": 1, "members": encoded, "frame_rate": 24, "publication_mode": mode]
      let bytes = try recipe(.ltx25, extra: ["scene": ["version": 1, "segments": project.clips.map { ["clip_id": $0.id.uuidString] }]])
      let record = NativeHeadlessJob.Recipe(engine: "ltx25", bytes: bytes, signature: "scene-frozen",
        report: try JSONSerialization.data(withJSONObject: ["scene": scene]))
      let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString: record],
        workers: ["ltx25": worker.path], ffmpeg: ffmpeg)
      var calls: [String] = []
      let output = root.appendingPathComponent(mode)
      let sceneResult = try await NativeHeadlessExecutor.run(job: job, output: output, emit: { _ in }, worker: { _, recipe, target, action in
        calls.append(action); XCTAssertEqual(try Data(contentsOf: recipe), bytes)
        if action == "preflight" { return [:] }
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
        let movie = target.appendingPathComponent("render.mp4"); try FileManager.default.copyItem(at: media, to: movie)
        return ["video": movie.path, "scene": scene]
      })
      XCTAssertEqual(calls, ["preflight", "render"])
      XCTAssertEqual(sceneResult["newlyGenerated"] as? Int, 1)
      let reopened = try ProjectStorage.read(output.appendingPathComponent("result.weetodd"))
      XCTAssertEqual(reopened.clips[0].sourcePath, reopened.clips[1].sourcePath)
      XCTAssertEqual(reopened.assets.count, 1); XCTAssertEqual(reopened.assets[0].scope, .project)
      XCTAssertEqual(reopened.assets[0].duration, 1); XCTAssertEqual(reopened.assets[0].fps, 24)
      XCTAssertEqual(reopened.clips.map(\.sourceIn), [0, 0.5]); XCTAssertEqual(reopened.clips.map(\.duration), [0.5, 0.5])
      XCTAssertEqual(reopened.clips[0].versions.last?.sceneTakeID, reopened.clips[1].versions.last?.sceneTakeID)
      XCTAssertEqual(reopened.clips[0].versions.last?.sceneMembers, members)
      _ = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }, worker: { _, _, _, _ in
        XCTFail("Completed scene must resume without another inference"); return [:]
      })
    }
  }
  func testNativeCLIOptionsRejectUnsupportedDuplicatedAndMissingControls() throws {
    let args = ["--job", "movie.json", "--output-directory", "result"]
    let parsed = try NativeHeadlessCLIArguments(args + ["--resume", "--preflight-only",
      "--h3-swift-worker", "/workers/h3", "--ltx25-swift-worker", "/workers/ltx"])
    XCTAssertEqual(parsed.jobPath, "movie.json"); XCTAssertEqual(parsed.outputDirectory, "result")
    XCTAssertTrue(parsed.resume); XCTAssertTrue(parsed.preflightOnly)
    XCTAssertEqual(parsed.workerOverrides, ["h3": "/workers/h3", "ltx25": "/workers/ltx"])
    for extra in [["--unknown"], ["--force"], ["--output-directory", "other"],
      ["--resume", "--resume"], ["--h3-swift-worker"], ["--h3-swift-worker", "--resume"], ["stray"]] {
      XCTAssertThrowsError(try NativeHeadlessCLIArguments(args + extra))
    }
    XCTAssertThrowsError(try NativeHeadlessCLIArguments(["--job", "movie.json"]))
  }
  func testFinishingCapabilitiesAdmitBeforeAnyWorkerAndRejectMissingDependency() async throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    let binary = root.appendingPathComponent("ffmpeg-capabilities")
    let filters = "trim setpts scale pad crop setsar fps format concat anullsrc atrim asetpts aresample aformat volume apad"
    for missing in ["encoder", "filter", "muxer", "none"] {
      let encoders = missing == "encoder" ? "libx264" : "libx264 aac prores_ks pcm_s16le"
      let offeredFilters = missing == "filter" ? "scale fps" : filters
      let muxers = missing == "muxer" ? "wav" : "mp4 mov"
      let script = "#!/bin/sh\ncase \"$2\" in\n-encoders) for value in " + encoders
        + "; do echo \"V..... $value\"; done;;\n-filters) for value in " + offeredFilters
        + "; do echo \"... $value\"; done;;\n-muxers) for value in " + muxers
        + "; do echo \"E $value\"; done;;\nesac\n"
      try script.write(to: binary, atomically: true, encoding: .utf8)
      try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: binary.path)
      var project = project(); if missing == "none" { project.settings.format = .proRes }
      let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString:
        .init(engine: "ltx25", bytes: try recipe(.ltx25), signature: "fixture")],
        workers: ["ltx25": worker.path], ffmpeg: binary.path)
      var calls = 0
      if missing == "none" {
        _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent(missing),
          preflightOnly: true, worker: { _, _, _, mode in XCTAssertEqual(mode, "preflight"); calls += 1; return [:] })
        XCTAssertEqual(calls, 1)
      } else {
        await XCTAssertThrowsErrorAsync {
          _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent(missing),
            preflightOnly: true, worker: { _, _, _, _ in calls += 1; return [:] })
        }
        XCTAssertEqual(calls, 0, "Missing finishing dependencies must fail before worker admission")
      }
    }
    try "#!/bin/sh\ntrap 'exit 130' INT TERM\nwhile true; do sleep 1; done\n".write(to: binary, atomically: true, encoding: .utf8)
    let project = project(), cancellation = NativeHeadlessCancellation()
    let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString:
      .init(engine: "ltx25", bytes: try recipe(.ltx25), signature: "fixture")],
      workers: ["ltx25": worker.path], ffmpeg: binary.path)
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { cancellation.cancel() }
    var cancelledCalls = 0
    do {
      _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("cancelled-probe"),
        preflightOnly: true, cancellation: cancellation, worker: { _, _, _, _ in cancelledCalls += 1; return [:] })
      XCTFail("Cancelled capability probe must not admit the worker")
    } catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertEqual(cancelledCalls, 0)
  }
  func testWrongWorkerCompletionIdentityIsRejected() throws {
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media, wrongID: true)
    let recipeURL = root.appendingPathComponent("recipe.json"); try recipe(.h3).write(to: recipeURL)
    XCTAssertThrowsError(try NativeHeadlessExecutor.worker(worker.path, recipe: recipeURL,
      output: root.appendingPathComponent("take"), mode: "preflight", cancellation: .init(), emit: { _ in }))
  }
  func testUnsupportedFinishingAndQueuedContinuitySourceAreRejected() throws {
    for quality in [-1, 52] {
      var invalid = project(); invalid.settings.quality = quality
      XCTAssertThrowsError(try NativeHeadlessJob.validateFinishing(invalid))
    }
    let root = try directory(), media = try movie(root), worker = try worker(root, fixture: media)
    var project = project(); project.clips[0].sourcePath = media.path
    for mutate: (inout StudioProject) -> Void in [
      { $0.clips[0].sourcePan = 0.5 }, { $0.clips[0].transition = "dissolve" },
      { $0.settings.upscaling = .metalFX }, { $0.settings.interpolation = .rife },
      { $0.settings.format = .pngSequence },
      { $0.clips[0].settingsOverride = $0.settings; $0.clips[0].settingsOverride?.fit = "fill" },
      { $0.clips[0].settingsOverride = $0.settings; $0.clips[0].settingsOverride?.width = 1 },
    ] {
      var changed = project; mutate(&changed); XCTAssertThrowsError(try NativeHeadlessJob.validateFinishing(changed))
    }
    var second = project.clips[0]; second.id = UUID(); second.continuity = .init(mode: "frame", sourceClipID: project.clips[0].id)
    project.clips.append(second)
    let recipes = try Dictionary(uniqueKeysWithValues: project.clips.map {
      ($0.id.uuidString, NativeHeadlessJob.Recipe(engine: "ltx25", bytes: try recipe(.ltx25), signature: "frozen"))
    })
    XCTAssertThrowsError(try NativeHeadlessJob(project: project, recipes: recipes, workers: ["ltx25": worker.path], ffmpeg: ffmpeg))
  }
  func testFrozenSourceChangeRejectedBeforeWorkerAdmission() async throws {
    let root = try directory(), media = try movie(root)
    var project = project(.movie); project.clips[0].sourcePath = media.path
    let job = try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: ffmpeg)
    try Data("source changed".utf8).append(to: media)
    var calls = 0
    await XCTAssertThrowsErrorAsync { _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("out"),
      worker: { _, _, _, _ in calls += 1; return [:] }) }
    XCTAssertEqual(calls, 0)
  }
}

private extension Data {
  func append(to url: URL) throws {
    let handle = try FileHandle(forWritingTo: url); defer { try? handle.close() }
    try handle.seekToEnd(); try handle.write(contentsOf: self)
  }
}
private func XCTAssertThrowsErrorAsync(_ action: () async throws -> Void,
  file: StaticString = #filePath, line: UInt = #line) async {
  do { try await action(); XCTFail("Expected error", file: file, line: line) } catch { }
}
