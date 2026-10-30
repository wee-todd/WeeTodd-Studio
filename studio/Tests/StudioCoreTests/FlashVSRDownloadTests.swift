import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRDownloadTests: XCTestCase {
  func testDownloadsOnlySelectedDecoderAndOfficialTurboFiles() throws {
    let tiny = try FlashVSRModelDownloader.requirements(decoder: .tiny)
    let full = try FlashVSRModelDownloader.requirements(decoder: .vae)
    XCTAssertTrue(tiny.contains { $0.relativePath == "TCDecoder.ckpt" })
    XCTAssertFalse(tiny.contains { $0.relativePath == "Wan2.1_VAE.pth" })
    XCTAssertTrue(full.contains { $0.relativePath == "Wan2.1_VAE.pth" })
    XCTAssertFalse(full.contains { $0.relativePath == "TCDecoder.ckpt" })
    XCTAssertFalse((tiny + full).contains { $0.relativePath.hasPrefix("base/") })
    XCTAssertTrue((tiny + full).allSatisfy { $0.sha256.count == 64 && $0.bytes > 0 })
    XCTAssertTrue(tiny.first { $0.relativePath == "posi_prompt.pth" }!.url.contains("OpenImagingLab/FlashVSR_Plus/b548a2cc"))
  }
  func testWrongExistingFileIsNeverOverwritten() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("turbo"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("turbo/dit.safetensors")
    try Data([1,2,3]).write(to: file)
    do { try await FlashVSRModelDownloader.download(root: root, decoder: .tiny, logDirectory: root.appendingPathComponent("logs")); XCTFail("Wrong weights must not be replaced") }
    catch { XCTAssertTrue(error.localizedDescription.contains("Existing")) }
    XCTAssertEqual(try Data(contentsOf: file), Data([1,2,3]))
  }
}
