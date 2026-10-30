import Foundation
import XCTest
@testable import StudioCore

final class NativeVideoProcessingChainTests: XCTestCase {
  private func combined(_ native: NativeVideoProcessingPlan, _ flash: NativeVideoProcessingPlan) throws -> NativeVideoProcessingPlan {
    let recipe = VideoProcessingRecipe(pixelCQ: native.recipe.steps[0].pixelCQ!, flashVSR: flash.recipe.steps[0].settings)
    return .init(recipe: recipe, runtime: flash.runtime, resources: ["flash": try XCTUnwrap(flash.resources["restore"])], native: native.native)
  }
  func testFractionalSourceClockSurvivesBothStages() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root)
    let sourceURL = root.appendingPathComponent("fractional.mp4")
    try VideoProcessingFixture.ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30000/1001:duration=1.001",
      "-f", "lavfi", "-i", "sine=sample_rate=32000:duration=1.001", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", sourceURL.path])
    let source = try FlashVSRFile.capture(sourceURL), plan = try combined(await worker.plan(), backend.plan())
    let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: source,
      directory: root.appendingPathComponent("cache"), backend: backend, nativeWorker: worker.worker)
    XCTAssertEqual(result.receipt.fps.numerator * 1001, result.receipt.fps.denominator * 30000)
    XCTAssertEqual(result.receipt.frames, 30)
    let receipt = try XCTUnwrap(result.receipt.chain)
    XCTAssertEqual(receipt.sourceAudioSHA256, receipt.outputAudioSHA256)
    XCTAssertEqual(receipt.sourcePCMHash, receipt.outputPCMHash)
  }
  func testAppendFlashPreservesNativeCacheAndOriginal25FPSAudio() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root)
    let source = try backend.source(fps: 25), cache = root.appendingPathComponent("cache")
    let nativePlan = try await worker.plan(), flashPlan = try backend.plan()
    let first = try await NativeVideoProcessingExecutor.run(plan: nativePlan, source: source, directory: cache, nativeWorker: worker.worker)
    let plan = try combined(nativePlan, flashPlan)
    XCTAssertEqual(try plan.recipe.nativePrefix?.fingerprint, try nativePlan.recipe.fingerprint)
    let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend, nativeWorker: worker.worker)
    XCTAssertEqual(result.reusedSteps, 1); XCTAssertEqual(worker.renders, 1)
    let chain = try XCTUnwrap(result.receipt.chain)
    XCTAssertEqual(chain.native.video, first.receipt.video)
    XCTAssertEqual(chain.flashVSR.plan.source, chain.native.video)
    XCTAssertEqual(result.receipt.fps.value, 25); XCTAssertEqual(result.receipt.frames, 25)
    XCTAssertEqual(result.receipt.audio, source); XCTAssertEqual(result.receipt.restored, chain.flashVSR.restored)
    XCTAssertEqual(chain.sourceAudioSHA256, chain.outputAudioSHA256)
    XCTAssertEqual(chain.sourcePCMHash, chain.outputPCMHash); XCTAssertNotNil(chain.sourcePCMHash)
    XCTAssertNotEqual(chain.video, chain.flashVSR.video)
    try chain.flashReceiptFile.verify(); try chain.nativeReceiptFile.verify()
    let lineage = try result.lineage(sourceVersionID: nil)
    XCTAssertEqual(lineage.source, source); XCTAssertEqual(lineage.recipe, plan.recipe)
    let reused = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend, nativeWorker: worker.worker)
    XCTAssertEqual(reused.reusedSteps, 2); XCTAssertEqual(reused.receipt.video, result.receipt.video)
    var full = plan
    full = try combined(nativePlan, backend.plan(decoder: .vae))
    let changed = try await NativeVideoProcessingExecutor.run(plan: full, source: source, directory: cache, backend: backend, nativeWorker: worker.worker)
    XCTAssertEqual(changed.reusedSteps, 1); XCTAssertEqual(worker.renders, 1)
    XCTAssertEqual(changed.receipt.native?.video, first.receipt.video)
    let renders = await backend.renders; XCTAssertEqual(renders, 2)
    // Damaged delivery alone repeats only stream-copy finalization, preserving both weighted stages.
    try Data("damaged delivery".utf8).write(to: URL(fileURLWithPath: changed.receipt.video.path))
    let repaired = try await NativeVideoProcessingExecutor.run(plan: full, source: source, directory: cache, backend: backend, nativeWorker: worker.worker)
    XCTAssertEqual(repaired.reusedSteps, 2); XCTAssertNotEqual(repaired.receipt.video, changed.receipt.video)
    let repairedRenders = await backend.renders; XCTAssertEqual(repairedRenders, 2); XCTAssertEqual(worker.renders, 1)
    try source.verify(); try first.receipt.video.verify()
  }
  func testBothHeadlessEnginesReuseGenerationAndNativeStageAfterFlashFailure() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let nativeWorker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root, failures: 1)
      let source = try backend.source(), plan = try combined(await nativeWorker.plan(), backend.plan())
      var project = StudioProject(); project.settings.width = 128; project.settings.height = 128
      var clip = Clip(engine: engine); clip.duration = 1; clip.videoProcessing = plan.recipe; project.clips = [clip]
      let bytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue, "components": [:], "config": [:]])
      var job = try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: engine.rawValue, bytes: bytes, signature: "generation")],
        workers: [engine.rawValue: "/usr/bin/true"], ffmpeg: plan.runtime.ffmpegPath,
        processingRuntime: plan.runtime, nativeProcessingRuntime: nativeWorker.runtime)
      let initialIdentity = try job.generationIdentity()
      var generations = 0, admissions = [String]()
      let generate: NativeHeadlessExecutor.Worker = { _,_,output,mode in
        XCTAssertEqual(admissions, ["native", "flash"])
        if mode == "preflight" { return [:] }; generations += 1
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let movie = output.appendingPathComponent("render.mp4")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: source.path), to: movie)
        return ["video": movie.path]
      }
      let nativeAdmission: NativeHeadlessExecutor.NativeProcessingPreflight = { recipe,runtime in
        admissions.append("native")
        return .init(recipe: recipe, native: try await NativePixelCQService.preflight(recipe: recipe, runtime: runtime, worker: nativeWorker.worker))
      }
      let flashAdmission: NativeHeadlessExecutor.ProcessingPreflight = { recipe,runtime in
        admissions.append("flash")
        return .init(recipe: recipe, runtime: runtime, resources: [recipe.steps[0].id: try backend.resources()])
      }
      let output = root.appendingPathComponent("job")
      do {
        _ = try await NativeHeadlessExecutor.run(job: job, output: output, worker: generate,
          processingPreflight: flashAdmission, processingBackend: backend, nativeProcessingPreflight: nativeAdmission, nativeProcessingWorker: nativeWorker.worker)
        XCTFail("Flash failure was swallowed")
      } catch { XCTAssertTrue(error.localizedDescription.contains("Fixture restoration"), "\(error)") }
      XCTAssertEqual(generations, 1); XCTAssertEqual(nativeWorker.renders, 1)
      admissions = []
      let resumed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: generate,
        processingPreflight: flashAdmission, processingBackend: backend, nativeProcessingPreflight: nativeAdmission, nativeProcessingWorker: nativeWorker.worker)
      XCTAssertEqual(resumed["newlyGenerated"] as? Int, 0); XCTAssertEqual(resumed["reusedProcessingSteps"] as? Int, 1)
      var accepted = try ProjectStorage.read(URL(fileURLWithPath: resumed["project"] as! String))
      let original = try XCTUnwrap(accepted.clips[0].versions.last?.processingLineage?.source)
      XCTAssertEqual(accepted.clips[0].versions.last?.processingLineage?.recipe, plan.recipe)
      job.project.clips[0].videoProcessing!.steps[1].settings.decoder = .vae
      XCTAssertEqual(try job.generationIdentity(), initialIdentity)
      admissions = []
      let changed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, worker: generate,
        processingPreflight: flashAdmission, processingBackend: backend, nativeProcessingPreflight: nativeAdmission, nativeProcessingWorker: nativeWorker.worker)
      XCTAssertEqual(changed["newlyGenerated"] as? Int, 0); XCTAssertEqual(generations, 1); XCTAssertEqual(nativeWorker.renders, 1)
      accepted = try ProjectStorage.read(URL(fileURLWithPath: changed["project"] as! String))
      XCTAssertEqual(accepted.clips[0].versions.last?.processingLineage?.source, original)
      let renders = await backend.renders; XCTAssertEqual(renders, 3); try original.verify()
      let saved = root.appendingPathComponent("job.json"); try job.write(to: saved)
      let loaded = try NativeHeadlessJob.read(from: saved)
      XCTAssertEqual(loaded.processingRuntime, job.processingRuntime); XCTAssertEqual(loaded.nativeProcessingRuntime, job.nativeProcessingRuntime)
    }
  }
  func testChainRejectsMissingRuntimeAndMalformedOrder() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let nativeWorker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root)
    let plan = try combined(await nativeWorker.plan(), backend.plan()), source = try backend.source()
    var project = StudioProject(); var clip = Clip(engine: .movie); clip.sourcePath = source.path; clip.duration = 1; clip.videoProcessing = plan.recipe; project.clips = [clip]
    XCTAssertThrowsError(try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: plan.runtime.ffmpegPath, processingRuntime: plan.runtime))
    XCTAssertThrowsError(try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: plan.runtime.ffmpegPath, nativeProcessingRuntime: nativeWorker.runtime))
    var reversed = plan.recipe; reversed.steps.reverse(); reversed.steps[0].input = "source"; reversed.steps[1].input = reversed.steps[0].id
    XCTAssertThrowsError(try reversed.validate())
    var invalid = plan.recipe; invalid.steps[1].input = "source"; XCTAssertThrowsError(try invalid.validate())
    invalid = plan.recipe; invalid.steps[1].id = invalid.steps[0].id; XCTAssertThrowsError(try invalid.validate())
    invalid = plan.recipe; invalid.steps.append(invalid.steps[1]); XCTAssertThrowsError(try invalid.validate())
    invalid = plan.recipe; invalid.steps[1].settings.preserveAudio = false; XCTAssertThrowsError(try invalid.validate())
    var noResources = plan; noResources = .init(recipe: plan.recipe, runtime: plan.runtime, resources: [:], native: plan.native)
    do { _ = try await NativeVideoProcessingExecutor.run(plan: noResources, source: source, directory: root.appendingPathComponent("cache"), backend: backend, nativeWorker: nativeWorker.worker); XCTFail("Unadmitted Flash stage ran") }
    catch { XCTAssertTrue(error.localizedDescription.contains("admitted")) }
    XCTAssertEqual(nativeWorker.renders, 0)
  }
  func testMissingFlashResourceAdmissionDoesNotGenerateEitherSourceEngine() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let nativeWorker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root)
      let plan = try combined(await nativeWorker.plan(), backend.plan())
      var project = StudioProject(); var clip = Clip(engine: engine); clip.duration = 1; clip.videoProcessing = plan.recipe; project.clips = [clip]
      let bytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": engine.rawValue, "components": [:], "config": [:]])
      let job = try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: engine.rawValue, bytes: bytes, signature: "generation")],
        workers: [engine.rawValue: "/usr/bin/true"], ffmpeg: plan.runtime.ffmpegPath,
        processingRuntime: plan.runtime, nativeProcessingRuntime: nativeWorker.runtime)
      var generationCalls = 0
      do {
        _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("job"),
          worker: { _,_,_,_ in generationCalls += 1; throw StudioError.invalid("Generation reached before downstream admission") },
          processingPreflight: { recipe,runtime in .init(recipe: recipe, runtime: runtime, resources: [:]) },
          processingBackend: backend, nativeProcessingPreflight: { recipe,_ in .init(recipe: recipe, native: plan.native!) },
          nativeProcessingWorker: nativeWorker.worker)
        XCTFail("Missing Flash resources were admitted")
      } catch { XCTAssertTrue(error.localizedDescription.contains("admission"), "\(error)") }
      XCTAssertEqual(generationCalls, 0); XCTAssertEqual(nativeWorker.renders, 0)
    }
  }
  func testCancellationBetweenStagesRetainsNativeForRetry() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let nativeWorker = try PixelCQWorkerFixture(root), backend = try VideoProcessingFixture(root: root, waitForCancellation: true)
    let source = try backend.source(), plan = try combined(await nativeWorker.plan(), backend.plan())
    let cache = root.appendingPathComponent("cache"), cancellation = NativeHeadlessCancellation()
    let task = Task { try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache,
      backend: backend, nativeWorker: nativeWorker.worker, cancellation: cancellation) }
    for _ in 0..<500 { if await backend.renders > 0 { break }; try await Task.sleep(nanoseconds: 10_000_000) }
    XCTAssertEqual(nativeWorker.renders, 1); cancellation.cancel()
    do { _ = try await task.value; XCTFail("Cancellation ignored") } catch { XCTAssertTrue(error is CancellationError) }
    await backend.allowCompletion()
    let resumed = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend, nativeWorker: nativeWorker.worker)
    XCTAssertEqual(resumed.reusedSteps, 1); XCTAssertEqual(nativeWorker.renders, 1); try source.verify()
  }
}
