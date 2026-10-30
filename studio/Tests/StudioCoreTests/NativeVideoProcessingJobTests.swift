import Foundation
import XCTest
@testable import StudioCore

final class NativeVideoProcessingJobTests: XCTestCase {
  private func job(root: URL, engine: Engine, backend: VideoProcessingFixture) throws -> NativeHeadlessJob {
    var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
    var clip = Clip(engine: engine); clip.duration = 1; clip.generationWidth = 64; clip.generationHeight = 64
    let plan = try backend.plan(); clip.videoProcessing = plan.recipe; project.clips = [clip]
    let bytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue,
      "prompt": "fixture", "config": [:], "components": [:], "conditioning": ["task": "t2v", "inputs": [], "audio_policy": "generated"]])
    return try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: engine.rawValue, bytes: bytes, signature: "generation-only")],
      workers: [engine.rawValue: "/usr/bin/true"], ffmpeg: "/opt/homebrew/bin/ffmpeg", processingRuntime: plan.runtime)
  }
  func testH3AndLTXUpscaleFailureAndDecoderChangeReuseAcceptedGeneration() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let backend = try VideoProcessingFixture(root: root, failures: 1), source = try backend.source()
      var job = try job(root: root, engine: engine, backend: backend)
      let output = root.appendingPathComponent("job"), id = job.project.clips[0].id.uuidString
      var generations = 0
      let worker: NativeHeadlessExecutor.Worker = { _,_,target,mode in
        if mode == "preflight" { return [:] }
        generations += 1
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        let movie = target.appendingPathComponent("render.mp4")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
        return ["video": movie.path, "nativeRuntime": "swift-mlx"]
      }
      let admit: NativeHeadlessExecutor.ProcessingPreflight = { recipe,_ in try backend.plan(decoder: recipe.steps[0].settings.decoder) }
      do {
        _ = try await NativeHeadlessExecutor.run(job: job, output: output, worker: worker,
          processingPreflight: admit, processingBackend: backend)
        XCTFail("Expected restoration failure after generation checkpoint")
      } catch { XCTAssertTrue(error.localizedDescription.contains("Fixture"), error.localizedDescription) }
      XCTAssertEqual(generations, 1)
      let state = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("job-state.json"))) as? [String: Any])
      let raw = try XCTUnwrap((state["completed"] as? [String: String])?[id])
      let original = try FlashVSRFile.capture(URL(fileURLWithPath: raw))
      let accepted = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(accepted["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
      var project = try ProjectStorage.read(URL(fileURLWithPath: accepted["project"] as! String))
      XCTAssertNotEqual(project.clips[0].sourcePath, raw)
      XCTAssertEqual(project.clips[0].versions.last?.processingLineage?.source, original)
      let tiny = try FlashVSRFile.capture(URL(fileURLWithPath: project.clips[0].sourcePath))
      let reused = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(reused["reusedProcessingSteps"] as? Int, 1)
      job.project.clips[0].videoProcessing!.steps[0].settings.decoder = .vae
      let full = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(full["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
      project = try ProjectStorage.read(URL(fileURLWithPath: full["project"] as! String))
      XCTAssertNotEqual(project.clips[0].sourcePath, tiny.path)
      XCTAssertEqual(project.clips[0].versions.last?.processingLineage?.source, original)
      job.project.settings.format = .proRes
      let codec = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(URL(fileURLWithPath: codec["video"] as! String).pathExtension, "mov")
      XCTAssertEqual(codec["reusedProcessingSteps"] as? Int, 1)
      job.project.settings.format = .mp4
      let mp4Again = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(mp4Again["video"] as? String, full["video"] as? String)
      job.project.clips[0].videoProcessing!.steps[0].settings.decoder = .tiny
      let tinyAgain = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
        processingPreflight: admit, processingBackend: backend)
      XCTAssertEqual(tinyAgain["video"] as? String, accepted["video"] as? String)
      let renders = await backend.renders; XCTAssertEqual(renders, 3)
      XCTAssertEqual(generations, 1); try original.verify(); try tiny.verify()
    }
  }
  func testDamagedProcessedOutputRetriesFromRawWithoutGeneration() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root), source = try backend.source()
    let job = try job(root: root, engine: .ltx25, backend: backend), output = root.appendingPathComponent("job")
    var generations = 0
    let worker: NativeHeadlessExecutor.Worker = { _,_,target,mode in
      if mode == "preflight" { return [:] }; generations += 1
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let movie = target.appendingPathComponent("render.mp4")
      try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
      return ["video": movie.path]
    }
    let admit: NativeHeadlessExecutor.ProcessingPreflight = { recipe,_ in try backend.plan(decoder: recipe.steps[0].settings.decoder) }
    let first = try await NativeHeadlessExecutor.run(job: job, output: output, worker: worker,
      processingPreflight: admit, processingBackend: backend)
    let project = try ProjectStorage.read(URL(fileURLWithPath: first["project"] as! String))
    try Data("broken".utf8).write(to: URL(fileURLWithPath: project.clips[0].sourcePath))
    let repaired = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: worker,
      processingPreflight: admit, processingBackend: backend)
    XCTAssertEqual(repaired["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
    let renders = await backend.renders; XCTAssertEqual(renders, 2)
    var changed = job; changed.project.clips[0].prompt = "new generation"
    do { _ = try await NativeHeadlessExecutor.run(job: changed, output: output, resume: true, worker: worker,
      processingPreflight: admit, processingBackend: backend); XCTFail("Changed generation accepted") }
    catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
    XCTAssertEqual(generations, 1)
  }
  func testDownstreamResourceAdmissionFailureNeverStartsGeneration() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root), job = try job(root: root, engine: .h3, backend: backend)
    var calls = 0
    do { _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("job"),
      worker: { _,_,_,_ in calls += 1; return [:] }, processingPreflight: { _,_ in
        throw StudioError.invalid("FlashVSR model missing")
      }, processingBackend: backend); XCTFail("Missing model admitted") }
    catch { XCTAssertTrue(error.localizedDescription.contains("FlashVSR")) }
    XCTAssertEqual(calls, 0)
  }
  func testCancelledUpscaleLeavesGenerationCheckpointForRetry() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root, waitForCancellation: true), source = try backend.source()
    let job = try job(root: root, engine: .h3, backend: backend), output = root.appendingPathComponent("job")
    var generations = 0
    let worker: NativeHeadlessExecutor.Worker = { _,_,target,mode in
      if mode == "preflight" { return [:] }; generations += 1
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let movie = target.appendingPathComponent("render.mp4")
      try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
      return ["video": movie.path]
    }
    let admit: NativeHeadlessExecutor.ProcessingPreflight = { _,_ in try backend.plan() }
    let cancellation = NativeHeadlessCancellation()
    let task = Task { try await NativeHeadlessExecutor.run(job: job, output: output, cancellation: cancellation,
      worker: worker, processingPreflight: admit, processingBackend: backend) }
    for _ in 0..<250 {
      if await backend.renders > 0 { break }; try await Task.sleep(nanoseconds: 20_000_000)
    }
    cancellation.cancel()
    do { _ = try await task.value; XCTFail("Cancellation was ignored") } catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertEqual(generations, 1); try source.verify(); await backend.allowCompletion()
    let resumed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true,
      worker: worker, processingPreflight: admit, processingBackend: backend)
    XCTAssertEqual(resumed["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1)
  }
  func testContinuousLTXSceneRestoresOnceAndKeepsMemberRanges() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root), source = try backend.source()
    let base = try job(root: root, engine: .ltx25, backend: backend)
    var project = base.project; project.clips[0].duration = 0.5
    var second = project.clips[0]; second.id = UUID()
    second.continuity = .init(mode: "scene", sourceClipID: project.clips[0].id); project.clips.append(second)
    let members = [ContinuousSceneMember(clipID: project.clips[0].id, sourceIn: 0, duration: 0.5),
      ContinuousSceneMember(clipID: second.id, sourceIn: 0.5, duration: 0.5)]
    var document = try JSONSerialization.jsonObject(with: base.recipes.values.first!.bytes) as! [String: Any]
    document["scene"] = ["version": 1, "segments": project.clips.map { ["clip_id": $0.id.uuidString] }]
    let scene: [String: Any] = ["version": 1, "members": try JSONSerialization.jsonObject(with: JSONEncoder().encode(members)),
      "frame_rate": 24, "publication_mode": "single_decode_native_latent_chain"]
    let recipe = NativeHeadlessJob.Recipe(engine: "ltx25", bytes: try JSONSerialization.data(withJSONObject: document),
      signature: "scene", report: try JSONSerialization.data(withJSONObject: ["scene": scene]))
    let job = try NativeHeadlessJob(project: project, recipes: [project.clips[0].id.uuidString: recipe],
      workers: base.workers, ffmpeg: base.ffmpeg, processingRuntime: base.processingRuntime)
    var mismatch = project; mismatch.clips[1].videoProcessing = nil
    XCTAssertThrowsError(try NativeHeadlessJob(project: mismatch, recipes: job.recipes, workers: job.workers,
      ffmpeg: job.ffmpeg, processingRuntime: job.processingRuntime))
    let result = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("job"), worker: { _,_,target,mode in
      if mode == "preflight" { return [:] }
      try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
      let movie = target.appendingPathComponent("render.mp4")
      try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
      return ["video": movie.path, "scene": scene]
    }, processingPreflight: { _,_ in try backend.plan() }, processingBackend: backend)
    let accepted = try ProjectStorage.read(URL(fileURLWithPath: result["project"] as! String))
    XCTAssertEqual(accepted.clips.map(\.sourceIn), [0, 0.5]); XCTAssertEqual(accepted.clips.map(\.duration), [0.5, 0.5])
    XCTAssertEqual(accepted.clips[0].sourcePath, accepted.clips[1].sourcePath)
    XCTAssertEqual(accepted.clips[0].versions.last?.sceneTakeID, accepted.clips[1].versions.last?.sceneTakeID)
    XCTAssertEqual(accepted.clips[0].versions.count, 2); XCTAssertEqual(accepted.clips[1].versions.count, 2)
    let renders = await backend.renders; XCTAssertEqual(renders, 1)
  }
  func testMixedJobKeepsDisabledClipsSelectedUpscaleAndDoesNotRequireOldOriginal() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root), original = try backend.source(), plan = try backend.plan()
    let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: original,
      directory: root.appendingPathComponent("first-cache"), backend: backend)
    var project = StudioProject(); project.settings.width = 128; project.settings.height = 128
    var frozen = Clip(engine: .movie); frozen.duration = 1; frozen.sourcePath = result.receipt.video.path
    var version = RenderVersion(path: frozen.sourcePath, seed: 0, prompt: "", recipePath: "")
    version.processingLineage = try result.lineage(sourceVersionID: nil); frozen.versions = [version]
    var pending = Clip(engine: .movie); pending.duration = 1; pending.sourcePath = original.path; pending.videoProcessing = plan.recipe
    project.clips = [frozen, pending]
    let job = try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: plan.runtime.ffmpegPath, processingRuntime: plan.runtime)
    let completed = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("mixed"),
      processingPreflight: { _,_ in plan }, processingBackend: backend)
    let accepted = try ProjectStorage.read(URL(fileURLWithPath: completed["project"] as! String))
    XCTAssertEqual(accepted.clips[0].sourcePath, frozen.sourcePath)
    project.clips = [frozen]
    try FileManager.default.removeItem(at: URL(fileURLWithPath: original.path))
    let noProcessing = try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: plan.runtime.ffmpegPath)
    _ = try await NativeHeadlessExecutor.run(job: noProcessing, output: root.appendingPathComponent("selected-only"))
  }
  func testVersionTwoEnvelopeRetainsProcessingAndLegacyReaderRemainsSupported() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let backend = try VideoProcessingFixture(root: root), job = try job(root: root, engine: .h3, backend: backend)
    let file = root.appendingPathComponent("job.json"); try job.write(to: file)
    let envelope = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
    XCTAssertEqual(envelope["format"] as? String, "weetodd-studio-native-job-v2")
    let restored = try NativeHeadlessJob.read(from: file)
    XCTAssertEqual(restored.project.clips[0].videoProcessing, job.project.clips[0].videoProcessing)
    XCTAssertEqual(restored.processingRuntime, job.processingRuntime)
    var disabled = restored; disabled.project.clips[0].videoProcessing = nil
    let disabledFile = root.appendingPathComponent("disabled.json"); try disabled.write(to: disabledFile)
    XCTAssertNil(try NativeHeadlessJob.read(from: disabledFile).project.clips[0].videoProcessing)
    var legacy = job; legacy.project.clips[0].videoProcessing = nil; legacy.processingRuntime = nil
    let old = root.appendingPathComponent("v1.json"); try legacy.write(to: old)
    XCTAssertNil(try NativeHeadlessJob.read(from: old).project.clips[0].videoProcessing)
  }
  func testMissingUpscalerSetupRejectsBeforeEitherGenerationWorkerRuns() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    for engine in [Engine.h3, .ltx25] {
      var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
      var clip = Clip(engine: engine); clip.duration = 1; clip.generationWidth = 64; clip.generationHeight = 64
      var settings = FlashVSRSettings(); settings.width = 128; settings.height = 128
      clip.videoProcessing = VideoProcessingRecipe(settings: settings); project.clips = [clip]
      let bytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue,
        "prompt": "fixture", "config": [:], "components": [:], "conditioning": ["task": "t2v", "inputs": [], "audio_policy": "generated"]])
      var calls = 0
      do {
        let job = try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: engine.rawValue, bytes: bytes, signature: "fixture")],
          workers: [engine.rawValue: "/usr/bin/true"], ffmpeg: "/opt/homebrew/bin/ffmpeg")
        _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent(engine.rawValue), worker: { _,_,_,_ in
          calls += 1; throw StudioError.invalid("Generation was reached before FlashVSR setup admission")
        })
        XCTFail("Missing upscaler setup was accepted")
      } catch { XCTAssertTrue(error.localizedDescription.contains("FlashVSR"), error.localizedDescription) }
      XCTAssertEqual(calls, 0, "A missing downstream engine must reject before generation")
    }
  }
}
