import Foundation
import XCTest
@testable import StudioCore

final class TAEPreviewProgressTests: XCTestCase {
  func testApproximateFrameRemainsDistinctFromFinalDecodedFrame() throws {
    var stream = BridgeProgressStream()
    let common: [String: Any] = ["event": "progress", "message": "Frame ready",
      "previewPath": "/tmp/live-preview.png", "previewRevision": 1]
    var approximate = common
    approximate["previewKind"] = "tae"
    let approximateBytes = try JSONSerialization.data(withJSONObject: approximate)
    XCTAssertTrue(stream.append(approximateBytes.prefix(15)).isEmpty)
    let preview = try XCTUnwrap(stream.append(Data(approximateBytes.dropFirst(15)) + Data([10])).first)
    let decoded = try XCTUnwrap(stream.append(try JSONSerialization.data(withJSONObject: common) + Data([10])).first)
    XCTAssertNotEqual(preview, decoded,
      "The viewport must distinguish an approximate denoising preview from a final VAE frame")
    XCTAssertEqual(preview.previewKind, "tae")
    XCTAssertEqual(preview.previewLabel, "LIVE · APPROXIMATE TAE PREVIEW")
    XCTAssertNil(decoded.previewKind)
    XCTAssertEqual(decoded.previewLabel, "LIVE · DECODED FRAME")
  }

  func testUnknownPreviewKindRetainsTheExistingDecodedFrameMeaning() throws {
    var stream = BridgeProgressStream()
    let events = stream.append(Data("""
      {"event":"progress","message":"Frame ready","previewKind":"future-kind"}
      {"event":"progress","message":"Frame ready"}

      """.utf8))
    XCTAssertEqual(events.count, 2)
    XCTAssertEqual(events.first, events.last)
    XCTAssertEqual(events.first?.previewLabel, "LIVE · DECODED FRAME")
  }
}
