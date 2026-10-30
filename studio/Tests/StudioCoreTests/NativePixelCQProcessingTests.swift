import Foundation
import XCTest
@testable import StudioCore

/// Replaces only the weighted worker. Preparation, source clocks, audio, receipts and cache remain real.
final class PixelCQWorkerFixture {
  let root: URL
  let model: URL
  var renders = 0
  var failNext = false
  var badFPS = false
  var slowRender = false
  init(_ root: URL) throws {
    self.root = root; model = root.appendingPathComponent("model.safetensors")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try Data("model-v1".utf8).write(to: model)
  }
  var runtime: NativePixelCQRuntime {
    .init(workerPath: "/usr/bin/true", ffmpegPath: "/opt/homebrew/bin/ffmpeg", ffprobePath: "/opt/homebrew/bin/ffprobe",
      components: Dictionary(uniqueKeysWithValues: ["video_checkpoint", "spatial_upscaler_checkpoint", "gemma_root", "transformer_root",
        "connector_checkpoint", "audio_checkpoint", "pixel_spatial_adapter"].map { ($0, model.path) }), cqAdapterPath: model.path + ".cq")
  }
  func worker(_ executable: String, _ recipe: URL, _ output: URL, _ mode: String) throws -> [String: Any] {
    XCTAssertEqual(executable, "/usr/bin/true")
    let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: recipe)) as? [String: Any])
    let body = try XCTUnwrap(raw["movie_upscale"] as? [String: Any]), source = try XCTUnwrap(body["source"] as? [String: Any])
    XCTAssertEqual(body["mode"] as? String, "pixel_spatial"); XCTAssertEqual(body["audio_policy"] as? String, "source")
    let identity = try NativeHeadlessJob.fileHash(model)
    if mode == "preflight" { return ["componentIdentitySHA256": identity] }
    renders += 1
    if slowRender { for _ in 0..<300 { try Task.checkCancellation(); Thread.sleep(forTimeInterval: 0.01) } }
    if failNext { failNext = false; throw StudioError.invalid("Weighted fixture failed") }
    let fps = source["fps"] as! Double, frames = source["frames"] as! Int
    let width = (source["width"] as! Int) * 2, height = (source["height"] as! Int) * 2
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let video = output.appendingPathComponent("render.mp4")
    try VideoProcessingFixture.ffmpeg(["-i", source["path"] as! String, "-vf", "scale=\(width):\(height)", "-an", "-c:v", "libx264", video.path])
    let adapters = [["path": runtime.components["pixel_spatial_adapter"]!, "strength": Double(Float(body["pixel_strength"] as! Double))] as [String: Any]]
      + ((body["refinement_loras"] as? [[String: Any]] ?? []).map { ["path": $0["path"]!, "strength": Double(Float($0["strength"] as! Double))] })
    let metadata: [String: Any] = ["nativeRuntime": "swift-mlx", "task": "video_upscale", "pythonModelInference": false,
      "frames": frames, "fps": badFPS ? 30 : fps, "width": width, "height": height,
      "source_movie_sha256": source["sha256"]!, "source_rgb_sha256": source["rgb_sha256"]!,
      "applied_refinement_adapters": adapters, "component_identity_sha256": identity, "transformer_evaluations_per_chunk": 3]
    try JSONSerialization.data(withJSONObject: metadata).write(to: output.appendingPathComponent("metadata.json"))
    return ["nativeRuntime": "swift-mlx", "video": video.path, "metadata": metadata, "use_complete_duration": true,
      "usable_source_in": 0.0, "usable_duration": Double(frames)/fps]
  }
  func plan(_ settings: NativePixelCQSettings = .init()) async throws -> NativeVideoProcessingPlan {
    let recipe = VideoProcessingRecipe(pixelCQ: settings)
    let resources = try await NativePixelCQService.preflight(recipe: recipe, runtime: runtime, worker: worker)
    return .init(recipe: recipe, native: resources)
  }
}

final class NativePixelCQProcessingTests: XCTestCase {
  func test25FPSAudioAndStrengthChangesReuseOnlyUpscale() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root)
    let source = try media.source(fps: 25), cache = root.appendingPathComponent("cache")
    let plan = try await fixture.plan()
    let first = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertEqual(first.receipt.frames, 25); XCTAssertEqual(first.receipt.fps.value, 25)
    XCTAssertEqual(first.receipt.width, 128); XCTAssertEqual(first.receipt.height, 128)
    XCTAssertEqual(first.receipt.label, "Native Pixel Spatial + CQ")
    let repeated = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertEqual(repeated.reusedSteps, 1); XCTAssertEqual(repeated.receipt.video, first.receipt.video)
    XCTAssertEqual(fixture.renders, 1)
    var settings = NativePixelCQSettings(); settings.cqStrength = 0.6
    let changed = try await NativeVideoProcessingExecutor.run(plan: fixture.plan(settings), source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertNotEqual(changed.receipt.video.path, first.receipt.video.path)
    settings.pixelStrength = 0.8
    _ = try await NativeVideoProcessingExecutor.run(plan: fixture.plan(settings), source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertEqual(fixture.renders, 3); try source.verify(); try first.receipt.video.verify()
    let native = try XCTUnwrap(first.receipt.native)
    XCTAssertNotNil(native.sourceAudioSHA256)
    XCTAssertEqual(native.sourceAudioSHA256, native.outputAudioSHA256)
  }
  func testFailedAndDamagedNativeOutputsRetainOriginalForRetry() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root)
    let source = try media.source(), cache = root.appendingPathComponent("cache"), plan = try await fixture.plan()
    fixture.failNext = true
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker); XCTFail("Failure swallowed") }
    catch { XCTAssertTrue(error.localizedDescription.contains("Weighted fixture")) }
    let first = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker)
    try Data("damaged".utf8).write(to: URL(fileURLWithPath: first.receipt.video.path))
    let repaired = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertEqual(fixture.renders, 3); XCTAssertNotEqual(repaired.receipt.video.path, first.receipt.video.path)
    try source.verify()
    try Data("model-v2".utf8).write(to: fixture.model)
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, nativeWorker: fixture.worker); XCTFail("Changed model reused") }
    catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
    let newPlan = try await fixture.plan()
    let updated = try await NativeVideoProcessingExecutor.run(plan: newPlan, source: source, directory: cache, nativeWorker: fixture.worker)
    XCTAssertNotEqual(updated.receipt.video.path, repaired.receipt.video.path)
    XCTAssertEqual(fixture.renders, 4)
  }
  func testNativeH3AndLTXHeadlessRetryDoesNotRegenerateSource() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root), source = try media.source()
      let plan = try await fixture.plan()
      var project = StudioProject(); project.settings.width = 128; project.settings.height = 128
      var clip = Clip(engine: engine); clip.duration = 1; clip.videoProcessing = plan.recipe; project.clips = [clip]
      let recipe = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue, "components": [:], "config": [:]])
      var job = try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: engine.rawValue, bytes: recipe, signature: "generation")],
        workers: [engine.rawValue: "/usr/bin/true"], ffmpeg: "/opt/homebrew/bin/ffmpeg", nativeProcessingRuntime: fixture.runtime)
      var generations = 0
      let generate: NativeHeadlessExecutor.Worker = { _,_,output,mode in
        if mode == "preflight" { return [:] }; generations += 1
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let movie = output.appendingPathComponent("render.mp4")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
        return ["video": movie.path]
      }
      let admit: NativeHeadlessExecutor.NativeProcessingPreflight = { recipe,runtime in
        .init(recipe: recipe, native: try await NativePixelCQService.preflight(recipe: recipe, runtime: runtime, worker: fixture.worker))
      }
      fixture.failNext = true
      let output = root.appendingPathComponent("job")
      do { _ = try await NativeHeadlessExecutor.run(job: job, output: output, worker: generate,
        nativeProcessingPreflight: admit, nativeProcessingWorker: fixture.worker); XCTFail("Failure swallowed") }
      catch { XCTAssertTrue(error.localizedDescription.contains("Weighted fixture")) }
      XCTAssertEqual(generations, 1)
      let first = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: generate,
        nativeProcessingPreflight: admit, nativeProcessingWorker: fixture.worker)
      XCTAssertEqual(first["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
      var accepted = try ProjectStorage.read(URL(fileURLWithPath: first["project"] as! String))
      let original = try XCTUnwrap(accepted.clips[0].versions.last?.processingLineage?.source)
      // Older v2 checkpoints included unused workers. They remain resumable and migrate to generation-only worker identity.
      job.workers[engine == .h3 ? "ltx25" : "h3"] = "/usr/bin/true"
      let stateURL = output.appendingPathComponent("job-state.json")
      var oldState = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: stateURL)) as? [String: Any])
      oldState["generationSHA256"] = try job.legacyGenerationIdentity()
      try JSONSerialization.data(withJSONObject: oldState).write(to: stateURL, options: .atomic)
      let identity = try job.generationIdentity()
      job.project.clips[0].videoProcessing!.steps[0].pixelCQ!.cqStrength = 0.6
      XCTAssertEqual(try job.generationIdentity(), identity)
      let second = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: generate,
        nativeProcessingPreflight: admit, nativeProcessingWorker: fixture.worker)
      XCTAssertEqual(second["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
      accepted = try ProjectStorage.read(URL(fileURLWithPath: second["project"] as! String))
      XCTAssertEqual(accepted.clips[0].versions.last?.processingLineage?.source, original)
      XCTAssertEqual(fixture.renders, 3); try original.verify()
    }
  }
  func testCancellationStopsNativeWorkerAndNeverPublishesCompletedCache() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root)
    let source = try media.source(), plan = try await fixture.plan(), cancellation = NativeHeadlessCancellation()
    fixture.slowRender = true
    let task = Task { try await NativeVideoProcessingExecutor.run(plan: plan, source: source,
      directory: root.appendingPathComponent("cache"), nativeWorker: fixture.worker, cancellation: cancellation) }
    for _ in 0..<250 { if fixture.renders > 0 { break }; try await Task.sleep(nanoseconds: 10_000_000) }
    XCTAssertEqual(fixture.renders, 1)
    let cancelledAt = Date(); cancellation.cancel()
    do { _ = try await task.value; XCTFail("Cancellation was ignored") } catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertLessThan(Date().timeIntervalSince(cancelledAt), 1, "Cancellation must propagate into the detached worker")
    let entries = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("cache"), includingPropertiesForKeys: nil)
    XCTAssertFalse(entries.contains { $0.pathExtension == "json" })
    try source.verify()
  }
  func testSilentSourceStaysSilentAndCQZeroNeedsNoAdapter() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), original = root.appendingPathComponent("silent.mp4")
    try VideoProcessingFixture.ffmpeg(["-f", "lavfi", "-i", "color=s=64x64:r=24:d=1", "-an", "-c:v", "libx264", original.path])
    var settings = NativePixelCQSettings(); settings.cqStrength = 0
    var runtime = fixture.runtime; runtime.cqAdapterPath = nil
    let recipe = VideoProcessingRecipe(pixelCQ: settings)
    let resources = try await NativePixelCQService.preflight(recipe: recipe, runtime: runtime, worker: fixture.worker)
    let result = try await NativeVideoProcessingExecutor.run(plan: .init(recipe: recipe, native: resources),
      source: .capture(original), directory: root.appendingPathComponent("cache"), nativeWorker: fixture.worker)
    let native = try XCTUnwrap(result.receipt.native)
    XCTAssertFalse(native.media.hasAudio); XCTAssertNil(native.sourceAudioSHA256); XCTAssertNil(native.outputAudioSHA256)
    XCTAssertEqual(native.media.frames, 24)
  }
  func testCancellingHeadlessAdmissionInterruptsNativePreflightProcess() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root), source = try media.source()
    let helper = root.appendingPathComponent("preflight-fixture"), marker = root.appendingPathComponent("preflight-started")
    try """
    #!/usr/bin/python3
    import time
    from pathlib import Path
    Path("\(marker.path)").write_text("started")
    time.sleep(3)
    """.write(to: helper, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helper.path)
    var runtime = fixture.runtime; runtime.workerPath = helper.path
    var project = StudioProject(); project.settings.width = 128; project.settings.height = 128
    var clip = Clip(engine: .movie); clip.sourcePath = source.path; clip.duration = 1
    clip.videoProcessing = VideoProcessingRecipe(pixelCQ: .init()); project.clips = [clip]
    let job = try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: runtime.ffmpegPath, nativeProcessingRuntime: runtime)
    let cancellation = NativeHeadlessCancellation()
    let task = Task { try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("job"), preflightOnly: true,
      cancellation: cancellation, emit: { _ in }) }
    for _ in 0..<500 { if FileManager.default.fileExists(atPath: marker.path) { break }; try await Task.sleep(nanoseconds: 10_000_000) }
    XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
    let cancelledAt = Date(); cancellation.cancel()
    do { _ = try await task.value; XCTFail("Admission cancellation ignored") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
    XCTAssertLessThan(Date().timeIntervalSince(cancelledAt), 1)
    try source.verify()
  }
  func testRotatedSourceUsesVisibleCanvasForExact2xAcceptance() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), original = root.appendingPathComponent("upright.mp4"), rotated = root.appendingPathComponent("rotated.mp4")
    try VideoProcessingFixture.ffmpeg(["-f", "lavfi", "-i", "color=s=64x32:r=24:d=1", "-an", "-c:v", "libx264", original.path])
    try VideoProcessingFixture.ffmpeg(["-display_rotation", "90", "-i", original.path, "-c", "copy", rotated.path])
    let visible = try await NativeLTXMoviePreparation.inspectSource(rotated)
    XCTAssertEqual(visible["width"] as? Int, 32); XCTAssertEqual(visible["height"] as? Int, 64)
    let plan = try await fixture.plan()
    let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: .capture(rotated),
      directory: root.appendingPathComponent("cache"), nativeWorker: fixture.worker)
    XCTAssertEqual(result.receipt.width, 64); XCTAssertEqual(result.receipt.height, 128)
    XCTAssertEqual(result.receipt.frames, 24); XCTAssertEqual(result.receipt.fps.value, 24)
  }
  func testWrongNativeTimingCannotBeAccepted() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root)
    let source = try media.source(), plan = try await fixture.plan(); fixture.badFPS = true
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: root.appendingPathComponent("cache"), nativeWorker: fixture.worker); XCTFail("Wrong FPS accepted") }
    catch { XCTAssertTrue(error.localizedDescription.contains("contract") || error.localizedDescription.contains("frame rate")) }
    try source.verify()
  }
}
