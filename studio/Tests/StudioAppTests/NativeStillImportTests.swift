import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeStillImportTests: XCTestCase {
  @MainActor func testImageImportDoesNotInvokePythonBridge() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let image = directory.appendingPathComponent("reference.png")
    let png = try XCTUnwrap(Data(base64Encoded:
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lWZkAAAAASUVORK5CYII="))
    try png.write(to: image)
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { _, _, _, _ in
        XCTFail("Still-image import must not start the Python bridge.")
        throw StudioError.invalid("Unexpected bridge call")
      })
    store.runtime.pythonPath = "/missing/python"
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.importURLs([image], scope: .clip)
    XCTAssertNil(store.error)
    let asset = try XCTUnwrap(store.project.assets.first)
    XCTAssertEqual(asset.kind, .image)
    XCTAssertEqual(asset.width, 1)
    XCTAssertEqual(asset.height, 1)
    XCTAssertEqual(asset.path, image.path)
  }
}
