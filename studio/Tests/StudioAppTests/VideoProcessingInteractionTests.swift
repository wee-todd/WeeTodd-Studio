import Foundation
import XCTest
@testable import StudioCore
@testable import WeeToddStudio

@MainActor final class VideoProcessingInteractionTests: XCTestCase {
  func testEnableDefaultsTinyAndGenerationKeyDoesNotChange() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    for engine in [Engine.h3, .ltx23, .ltx25] {
      let clip = Clip(engine: engine); store.project.clips = [clip]; store.select(clip.id)
      let key = store.generationRequestKey(for: clip)
      store.setVideoProcessingEnabled(true)
      store.setVideoProcessingEngine("flashvsr-turbo")
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.settings.decoder, .tiny)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.settings.width, store.project.settings.width)
      XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), key)
      store.editVideoProcessing { $0.decoder = .vae }
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.settings.decoder, .vae)
      store.setVideoProcessingEnabled(false); XCTAssertNil(store.selectedClip?.videoProcessing)
    }
  }
  func testAdmissionCannotAdoptChangedSourceOrReplacementDocumentWithSameIDs() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var clip = Clip(engine: .movie); clip.videoProcessing = VideoProcessingRecipe(); clip.sourcePath = "original.mp4"
    for replaceDocument in [false, true] {
      store.project.clips = [clip]; let frozen = store.project
      do {
        _ = try await store.preflightClipVideoProcessing(clip, admission: { recipe,runtime in
          await Task.yield()
          if replaceDocument { try store.replaceDocument(frozen, url: nil, isDirty: false) }
          else { store.project.clips[0].sourcePath = "changed.mp4" }
          return NativeVideoProcessingPlan(recipe: recipe, runtime: runtime, resources: [:])
        })
        XCTFail("Admission adopted a changed destination")
      } catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
    }
  }
  func testAcceptChecksDocumentSourceAndRecipeAndRetryUsesOriginal() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let fileURL = root.appendingPathComponent("source.mp4"); try Data([1,2,3]).write(to: fileURL)
    let source = try FlashVSRFile.capture(fileURL)
    let output = root.appendingPathComponent("output.mp4"); try Data([4,5,6]).write(to: output)
    let video = try FlashVSRFile.capture(output)
    var clip = Clip(engine: .ltx25); clip.sourcePath = source.path
    clip.videoProcessing = VideoProcessingRecipe(settings: .init())
    clip.versions = [RenderVersion(path: source.path, seed: 1, prompt: "original", recipePath: "recipe")]
    store.project.clips = [clip]; store.select(clip.id)
    let info = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":128,\"height\":128,\"pix_fmt\":\"yuv420p\",\"avg_frame_rate\":\"24/1\"}]}".utf8), timestamps: Data("0\n".utf8))
    let components = FlashVSRComponents(root: root.path, dit: source, projection: source, context: source, decoder: source)
    let request = VideoUpscaleRequest(sourcePath: source.path, runtime: .init())
    let plan = FlashVSRResolvedPlan(request: request, source: source, media: info, components: components, engineFiles: [source], frames: 1, temporaryBytes: 1)
    let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: video, restored: video, audio: nil, elapsedSeconds: 1)
    let result = NativeVideoProcessingResult(source: source, recipe: clip.videoProcessing!, receipt: receipt,
      receiptPath: root.appendingPathComponent("receipt.json").path, reusedSteps: 0)
    let session = store.documentSessionID, projectID = store.project.id
    XCTAssertFalse(store.acceptClipVideoProcessing(result, clips: [clip], session: UUID(), projectID: projectID))
    store.project.clips[0].sourcePath = "changed"
    XCTAssertFalse(store.acceptClipVideoProcessing(result, clips: [clip], session: session, projectID: projectID))
    store.project.clips[0] = clip; store.editVideoProcessing { $0.decoder = .vae }
    XCTAssertFalse(store.acceptClipVideoProcessing(result, clips: [clip], session: session, projectID: projectID))
    store.project.clips[0] = clip
    XCTAssertTrue(store.acceptClipVideoProcessing(result, clips: [clip], session: session, projectID: projectID))
    XCTAssertEqual(store.project.clips[0].sourcePath, video.path)
    XCTAssertEqual(store.videoProcessingOriginal(for: store.project.clips[0]).path, source.path)
    XCTAssertEqual(store.project.clips[0].versions.count, 2)
    XCTAssertEqual(store.project.clips[0].versions.last?.processingLineage?.sourceVersionID, clip.versions[0].id)
    XCTAssertEqual(store.project.clips[0].sourceIn, clip.sourceIn)
  }
  func testCollectedReceiptCannotReplaceValidLocalAssetWithOldPaths() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let original = root.appendingPathComponent("original.mp4"); try Data([1]).write(to: original)
    let source = try FlashVSRFile.capture(original)
    let media = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":128,\"height\":128,\"avg_frame_rate\":\"24/1\"}]}".utf8), timestamps: Data("0\n".utf8))
    let components = FlashVSRComponents(root: root.path, dit: source, projection: source, context: source, decoder: source)
    let plan = FlashVSRResolvedPlan(request: .init(sourcePath: source.path, runtime: .init()), source: source,
      media: media, components: components, engineFiles: [source], frames: 1, temporaryBytes: 1)
    let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: source, restored: source, audio: nil, elapsedSeconds: 1)
    let metadata = root.appendingPathComponent("receipt.json"); try JSONEncoder().encode(receipt).write(to: metadata)
    let collected = root.appendingPathComponent("collected.mp4"); try Data([1]).write(to: collected)
    try FileManager.default.removeItem(at: original)
    var asset = MediaAsset(name: "Collected", kind: .video, path: collected.path, scope: .project)
    asset.upscaleReceipt = metadata.path; store.project.assets = [asset]; store.selectedAssetID = asset.id
    store.openVideoUpscale()
    XCTAssertEqual(store.videoUpscaleSourcePath, collected.path); XCTAssertNil(store.videoUpscaleResult)
    XCTAssertTrue(store.videoUpscaleStatus.lowercased().contains("comparison"))
  }

}
