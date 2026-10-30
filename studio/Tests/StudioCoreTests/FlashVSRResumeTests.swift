import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRResumeTests: XCTestCase {
  func testResumeRejectsChangedOutputAndChangedDecoder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("output.mp4"); try Data([1,2,3]).write(to: url)
    let file = try FlashVSRFile.capture(url)
    let info = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":128,\"height\":128,\"pix_fmt\":\"yuv420p\",\"avg_frame_rate\":\"24/1\"}]}".utf8), timestamps: Data("0\n".utf8))
    let components = FlashVSRComponents(root: root.path, dit: file, projection: file, context: file, decoder: file)
    let request = VideoUpscaleRequest(sourcePath: url.path, runtime: .init())
    let plan = FlashVSRResolvedPlan(request: request, source: file, media: info, components: components, engineFiles: [file], frames: 1, temporaryBytes: 1)
    let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: file, restored: file, audio: nil, elapsedSeconds: 1)
    try JSONEncoder().encode(receipt).write(to: root.appendingPathComponent("receipt.json"))
    XCTAssertEqual(try FlashVSRService.resume(plan: plan, directory: root).video.sha256, file.sha256)
    var changed = request; changed.settings.decoder = .vae
    let other = FlashVSRResolvedPlan(request: changed, source: file, media: info, components: components, engineFiles: [file], frames: 1, temporaryBytes: 1)
    XCTAssertThrowsError(try FlashVSRService.resume(plan: other, directory: root))
    try Data([4,5,6]).write(to: url)
    XCTAssertThrowsError(try FlashVSRService.resume(plan: plan, directory: root))
  }
}
