import Foundation
import XCTest
@testable import StudioCore

final class FlashVSRIntermediateTests: XCTestCase {
  func testIntermediateSkipsDeliveryAndCarriesVerifiedLosslessClock() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try VideoProcessingFixture(root: root)
    let source = try fixture.source(), resources = try fixture.resources()
    var settings = FlashVSRSettings(); settings.backend = .swiftMLX; settings.width = 64; settings.height = 64
    var runtime = FlashVSRRuntime(); runtime.ffmpegPath = "/opt/homebrew/bin/ffmpeg"; runtime.ffprobePath = "/opt/homebrew/bin/ffprobe"
    let info = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: source.path), runtime: runtime, directory: root, cancellation: .init())
    let request = VideoUpscaleRequest(sourcePath: source.path, settings: settings, runtime: runtime)
    let plan = FlashVSRResolvedPlan(request: request, source: source, media: info, components: resources.components,
      engineFiles: resources.engineFiles, frames: info.frames, temporaryBytes: 1024)
    let output = root.appendingPathComponent("stage")
    let receipt = try FlashVSRMediaPipeline.execute(plan: plan, directory: output, cancellation: .init(), emit: { _ in }, intermediateOnly: true) { _, selected, restored, _, _, _ in
      try VideoProcessingFixture.ffmpeg(["-i", selected.path, "-an", "-c:v", "rawvideo", "-pix_fmt", "rgb24", restored.path])
    }
    XCTAssertEqual(receipt.video, receipt.restored)
    XCTAssertEqual(URL(fileURLWithPath: receipt.restored.path).pathExtension, "mkv")
    XCTAssertNil(receipt.audio); XCTAssertNil(receipt.sourcePreview)
    XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("output.mov").path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("source-audio.wav").path))
    let actual = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: receipt.restored.path), runtime: runtime, directory: root, cancellation: .init())
    XCTAssertEqual(actual.frames, info.frames); XCTAssertEqual(actual.fps, info.fps); XCTAssertFalse(actual.hasAudio)
    let backend: any FlashVSRIntermediateBackend = NativeFlashVSRBackend()
    XCTAssertTrue(backend is NativeFlashVSRBackend)
  }
}
