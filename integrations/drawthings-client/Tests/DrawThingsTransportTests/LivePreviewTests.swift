import XCTest
import Diffusion
import GRPCImageServiceModels
import NNC
@testable import DrawThingsTransport

final class LivePreviewTests: XCTestCase {
  func testPreviewIsBoundedThrottledAndSeparateFromFinalFrames() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let preview = LivePreview(root: root, version: .v1)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, 8, 8, 4))
    for y in 0..<8 { for x in 0..<8 { for c in 0..<4 { tensor[0,y,x,c] = 0 } } }
    let data = tensor.data(using: [.zip, .fpzip])
    let first = try XCTUnwrap(preview.receive(data, now: 1))
    XCTAssertEqual(first["previewRevision"] as? Int, 1)
    XCTAssertTrue(FileManager.default.fileExists(atPath: first["previewPath"] as! String))
    XCTAssertNil(preview.receive(data, now: 1.1))
    XCTAssertEqual(preview.receive(data, now: 2)?["previewRevision"] as? Int, 2)
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["live-preview.png"])
    XCTAssertNil(preview.receive(Data("invalid".utf8), now: 3))
    XCTAssertNil(LivePreview(root: root, version: .flux2_9b).receive(data, now: 1))
  }
}
