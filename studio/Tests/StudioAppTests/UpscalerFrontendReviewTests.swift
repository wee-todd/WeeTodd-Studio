import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore
@testable import WeeToddStudio

@MainActor final class UpscalerFrontendReviewTests: XCTestCase {
  private func folder() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root
  }
  private func receipt(_ root: URL) throws -> H3RegenerateReceipt {
    let original = root.appendingPathComponent("original.mov"), output = root.appendingPathComponent("restored.mov")
    try Data([1]).write(to: original); try Data([2]).write(to: output)
    let source = try FlashVSRFile.capture(original), video = try FlashVSRFile.capture(output)
    func component(_ role: NativeComponentRole) -> H3RegenerateJob.Component {
      .init(path: source.path, identity: .init(family: .h3, role: role, sourceSHA256: source.sha256, layout: "fixture"))
    }
    let runtime = H3RegenerateRuntime()
    let resources = H3RegenerateResources(components: .init(transformer: component(.transformer), refiner: component(.refiner),
      sharpCurve: component(.sharpCurve), sharpInput: component(.sharpInput), latentUpscaler: component(.latentUpscaler),
      videoVAE: component(.videoVAE), audioVAE: component(.audioVAE)),
      qwen: .init(checkpointRoot: root.path, tokenizer: .init(path: source.path, sha256: source.sha256),
        files: [.init(path: source.path, sha256: source.sha256)]), modelFiles: [source], engineFiles: [], profile: source, runtime: runtime)
    let media = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":32,\"height\":32,\"avg_frame_rate\":\"24/1\",\"pix_fmt\":\"rgb24\"}]}".utf8), timestamps: Data("0\n".utf8))
    var settings = H3RegenerateSettings(); settings.targetWidth = 64; settings.targetHeight = 64
    let plan = H3RegenerateResolvedPlan(request: .init(sourcePath: source.path, settings: settings, runtime: runtime), source: source,
      media: media, resources: resources, frames: 1, paddedFrames: 5, sourceWidth: 32, sourceHeight: 32,
      temporaryBytes: 1024, preparationPolicy: "fixture")
    return .init(requestFingerprint: try plan.fingerprint, plan: plan, source: source, video: video, restored: video, audio: nil,
      workerResult: video, workerJob: video, refinedLatents: video, refinedLatentsShape: [1, 8, 96], elapsedSeconds: 1)
  }
  func testSelectingAnotherLibraryOrClipSourceClearsH3ComparisonAndPlan() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let completed = try receipt(root)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("state"), restoreSession: false)
    let other = MediaAsset(name: "Other", kind: .video, path: root.appendingPathComponent("other.mov").path, scope: .project)
    store.project.assets = [other]; store.selectedAssetID = other.id
    store.videoUpscaleSourcePath = completed.source.path; store.h3RegeneratePlan = completed.plan; store.h3RegenerateResult = completed
    store.openVideoUpscale()
    XCTAssertEqual(store.videoUpscaleSourcePath, other.path)
    XCTAssertNil(store.h3RegeneratePlan); XCTAssertNil(store.h3RegenerateResult)
    store.selectedAssetID = nil
    var clip = Clip(engine: .movie); clip.sourcePath = other.path
    store.project.clips = [clip]; store.select(clip.id)
    store.videoUpscaleSourcePath = completed.source.path; store.h3RegeneratePlan = completed.plan; store.h3RegenerateResult = completed
    store.openVideoUpscale()
    XCTAssertEqual(store.videoUpscaleSourcePath, other.path)
    XCTAssertNil(store.h3RegeneratePlan); XCTAssertNil(store.h3RegenerateResult)
  }
  func testH3FinishingUsesSavedSetupImmediatelyAfterStartup() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let first = StudioStore(dataDirectory: root, restoreSession: false)
    first.h3RegenerateSettings.baseProfilePath = root.appendingPathComponent("ref2va.json").path
    first.h3RegenerateSettings.modelLibraryRoot = "/shared/model-library"
    first.saveVideoUpscalePreferences()
    let second = StudioStore(dataDirectory: root, restoreSession: false)
    let clip = Clip(engine: .h3); second.project.clips = [clip]; second.select(clip.id)
    second.setVideoProcessingEngine("h3-regenerate")
    XCTAssertEqual(second.selectedClip?.videoProcessing?.steps.first?.h3Regenerate?.baseProfilePath, first.h3RegenerateSettings.baseProfilePath)
    XCTAssertEqual(second.selectedClip?.videoProcessing?.steps.first?.h3Regenerate?.modelLibraryRoot, "/shared/model-library")
  }
  func testSwitchingRestorationFamilyClearsAdmissionAndRetainsCompletedComparisons() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let h3 = try receipt(root)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("state"), restoreSession: false)
    let components = FlashVSRComponents(root: root.path, dit: h3.source, projection: h3.source, context: h3.source, decoder: h3.source)
    let flashPlan = FlashVSRResolvedPlan(request: .init(sourcePath: h3.source.path, runtime: .init()), source: h3.source,
      media: h3.plan.media, components: components, engineFiles: [], frames: 1, temporaryBytes: 1)
    let flash = FlashVSRReceipt(requestFingerprint: try flashPlan.fingerprint, plan: flashPlan,
      video: h3.video, restored: h3.restored, audio: nil, elapsedSeconds: 1)
    store.videoUpscaleSourcePath = h3.source.path
    store.videoUpscaleResult = flash; store.h3RegenerateResult = h3
    store.videoUpscalePlan = flashPlan; store.h3RegeneratePlan = h3.plan
    store.videoUpscaleStatus = "Ready · 17 frames · 24/1 FPS"

    store.videoUpscaleTool = .h3Regenerate
    XCTAssertNil(store.videoUpscalePlan); XCTAssertNil(store.h3RegeneratePlan)
    XCTAssertFalse(store.videoUpscaleStatus.hasPrefix("Ready"))
    XCTAssertTrue(store.videoUpscaleStatus.contains("H3 Regenerate"))
    XCTAssertEqual(store.videoUpscaleResult?.video, flash.video)
    XCTAssertEqual(store.h3RegenerateResult?.video, h3.video)

    store.h3RegeneratePlan = h3.plan; store.videoUpscaleStatus = "Ready · H3 source admitted"
    store.videoUpscaleTool = .h3Regenerate
    XCTAssertEqual(try store.h3RegeneratePlan?.fingerprint, try h3.plan.fingerprint)
    XCTAssertEqual(store.videoUpscaleStatus, "Ready · H3 source admitted")
    store.videoUpscaleTool = .flashVSR
    XCTAssertNil(store.videoUpscalePlan); XCTAssertNil(store.h3RegeneratePlan)
    XCTAssertFalse(store.videoUpscaleStatus.hasPrefix("Ready"))
    XCTAssertTrue(store.videoUpscaleStatus.contains("FlashVSR+"))
    XCTAssertEqual(store.videoUpscaleResult?.video, flash.video)
    XCTAssertEqual(store.h3RegenerateResult?.video, h3.video)
  }
  func testSymlinkedSelectedSourceAcceptsCanonicalH3Receipt() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let completed = try receipt(root), link = root.appendingPathComponent("linked.mov")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: URL(fileURLWithPath: completed.source.path))
    let store = StudioStore(dataDirectory: root.appendingPathComponent("state"), restoreSession: false)
    let asset = MediaAsset(name: "Linked original", kind: .video, path: link.path, scope: .project)
    store.project.assets = [asset]; store.videoUpscaleSourcePath = link.path
    XCTAssertTrue(store.acceptH3Regenerate(completed, directory: root, session: store.documentSessionID,
      projectID: store.project.id, sourceAssetID: asset.id))
    XCTAssertEqual(store.project.assets.count, 2)
    XCTAssertEqual(store.project.assets.last?.path, completed.video.path)
  }
  func testNativeHeadlessExportResolvesVersionTwoPixelResourceMap() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var recipe = VideoProcessingRecipe(pixelCQ: .init(), flashVSR: .init()); recipe.format = "weetodd-video-processing-v2"
    let runtime = NativePixelCQRuntime(workerPath: "/worker/LTX", ffmpegPath: "/tools/ffmpeg", ffprobePath: "/tools/ffprobe",
      components: ["transformer": "/models/ltx"], cqAdapterPath: "/models/cq")
    let resources = NativePixelCQResources(runtime: runtime, workerFiles: [], componentIdentitySHA256: String(repeating: "a", count: 64))
    let plan = NativeVideoProcessingPlan(recipe: recipe, runtime: .init(), resources: [:], pixelResources: [recipe.steps[0].id: resources])
    XCTAssertNil(plan.native)
    XCTAssertEqual(try store.pixelRuntimeForExport(plan, previous: nil), runtime)
    let legacy = NativeVideoProcessingPlan(recipe: .init(pixelCQ: .init()), native: resources)
    XCTAssertEqual(try store.pixelRuntimeForExport(legacy, previous: nil), runtime)
  }
  func testHeadlessExportDoesNotSubstitutePriorModelsForAnAutomaticPixelStage() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var recipe = VideoProcessingRecipe(pixelCQ: .init(), flashVSR: .init()); recipe.format = "weetodd-video-processing-v2"
    let automatic = NativePixelCQRuntime(workerPath: "/worker/LTX", ffmpegPath: "/tools/ffmpeg", ffprobePath: "/tools/ffprobe",
      components: ["transformer": "/models/A"], cqAdapterPath: "/models/cq")
    var previous = automatic; previous.components["transformer"] = "/models/B"
    let resources = NativePixelCQResources(runtime: automatic, workerFiles: [], componentIdentitySHA256: String(repeating: "a", count: 64))
    let plan = NativeVideoProcessingPlan(recipe: recipe, runtime: .init(), resources: [:], pixelResources: [recipe.steps[0].id: resources])
    XCTAssertThrowsError(try store.pixelRuntimeForExport(plan, previous: previous))
  }
  func testHeadlessExportAcceptsExplicitCQAdapterForAnAutomaticPixelStage() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var settings = NativePixelCQSettings(); settings.cqAdapterPath = "/models/chosen-cq"
    var recipe = VideoProcessingRecipe(pixelCQ: settings, flashVSR: .init()); recipe.format = "weetodd-video-processing-v2"
    let shared = NativePixelCQRuntime(workerPath: "/worker/LTX", ffmpegPath: "/tools/ffmpeg", ffprobePath: "/tools/ffprobe",
      components: ["transformer": "/models/A"], cqAdapterPath: "/models/default-cq")
    var selected = shared; selected.cqAdapterPath = "/models/chosen-cq"
    let resources = NativePixelCQResources(runtime: selected, workerFiles: [], componentIdentitySHA256: String(repeating: "a", count: 64))
    let plan = NativeVideoProcessingPlan(recipe: recipe, runtime: .init(), resources: [:], pixelResources: [recipe.steps[0].id: resources])
    XCTAssertEqual(try store.pixelRuntimeForExport(plan, previous: shared), shared)
  }
  func testHeadlessExportIgnoresDifferentCQAdaptersWhenCQIsDisabled() throws {
    let root = try folder(); defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var settings = NativePixelCQSettings(); settings.cqStrength = 0
    var recipe = VideoProcessingRecipe(pixelCQ: settings, flashVSR: .init()); recipe.format = "weetodd-video-processing-v2"
    let shared = NativePixelCQRuntime(workerPath: "/worker/LTX", ffmpegPath: "/tools/ffmpeg", ffprobePath: "/tools/ffprobe",
      components: ["transformer": "/models/A"], cqAdapterPath: "/models/unused-cq-A")
    var selected = shared; selected.cqAdapterPath = "/models/unused-cq-B"
    let resources = NativePixelCQResources(runtime: selected, workerFiles: [], componentIdentitySHA256: String(repeating: "a", count: 64))
    let plan = NativeVideoProcessingPlan(recipe: recipe, runtime: .init(), resources: [:], pixelResources: [recipe.steps[0].id: resources])
    XCTAssertEqual(try store.pixelRuntimeForExport(plan, previous: shared), shared)
  }
}
