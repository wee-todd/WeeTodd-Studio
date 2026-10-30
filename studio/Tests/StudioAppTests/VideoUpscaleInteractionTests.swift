import Foundation
import XCTest
@testable import StudioCore
@testable import WeeToddStudio

@MainActor final class VideoUpscaleInteractionTests: XCTestCase {
  func testPreferencesRetainDecoderAndOptionalRuntime() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.videoUpscaleSettings.decoder = .vae
    store.videoUpscaleRuntime.modelRoot = "/shared/models"
    store.videoUpscalePresetName = "Turbo full"
    store.saveVideoUpscalePreset()
    store.saveVideoUpscalePreferences()
    let restored = StudioStore(dataDirectory: root, restoreSession: false)
    restored.loadVideoUpscalePreferences()
    XCTAssertEqual(restored.videoUpscaleSettings.decoder, .vae)
    XCTAssertEqual(restored.videoUpscaleRuntime.modelRoot, "/shared/models")
    XCTAssertEqual(restored.videoUpscalePresets.first?.name, "Turbo full")
    XCTAssertEqual(restored.videoUpscalePresets.first?.settings.decoder, .vae)
  }
  func testResultCannotMutateReplacedDocument() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let session = store.documentSessionID, project = store.project.id
    let url = root.appendingPathComponent("output.mp4"); try Data([1,2,3]).write(to: url)
    let file = try FlashVSRFile.capture(url)
    let info = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":128,\"height\":128,\"pix_fmt\":\"yuv420p\",\"avg_frame_rate\":\"24/1\"}]}".utf8), timestamps: Data("0\n".utf8))
    let components = FlashVSRComponents(root: root.path, dit: file, projection: file, context: file, decoder: file)
    let request = VideoUpscaleRequest(sourcePath: url.path, runtime: .init())
    let plan = FlashVSRResolvedPlan(request: request, source: file, media: info, components: components, engineFiles: [file], frames: 1, temporaryBytes: 1)
    let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: file, restored: file, audio: nil, elapsedSeconds: 1)
    store.project = StudioProject()
    XCTAssertFalse(store.acceptVideoUpscale(receipt, directory: root, session: session, projectID: project, sourceAssetID: nil))
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    XCTAssertTrue(store.acceptVideoUpscale(receipt, directory: root, session: store.documentSessionID, projectID: store.project.id, sourceAssetID: nil))
    XCTAssertEqual(store.project.assets.count, 1)
    XCTAssertEqual(store.project.assets.first?.upscaleReceipt, root.appendingPathComponent("receipt.json").path)
    store.videoUpscalePlan = plan; store.videoUpscaleResult = receipt
    let next = MediaAsset(name: "Next source", kind: .video, path: root.appendingPathComponent("next.mp4").path, scope: .project)
    store.project.assets.append(next); store.selectedAssetID = next.id
    store.openVideoUpscale()
    XCTAssertEqual(store.videoUpscaleSourcePath, next.path)
    XCTAssertNil(store.videoUpscalePlan)
    XCTAssertNil(store.videoUpscaleResult)
  }
}
