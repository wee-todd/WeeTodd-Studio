import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRResourceTests: XCTestCase {
  func testBudgetIncludesLargeSourcePreparationAndBothPCMCopies() throws {
    let source = try FlashVSRMediaInfo.parse(summary: Data("{\"streams\":[{\"codec_type\":\"video\",\"width\":512,\"height\":512,\"pix_fmt\":\"yuv420p\",\"avg_frame_rate\":\"24/1\"},{\"codec_type\":\"audio\",\"sample_rate\":\"32000\",\"channels\":2}]}".utf8), timestamps: Data("0\n".utf8))
    var settings = FlashVSRSettings(); settings.width = 128; settings.height = 96
    XCTAssertEqual(try FlashVSRResourcePlan.temporaryBytes(media: source, settings: settings, frames: 24), 590073856)
    settings.preserveAudio = false
    XCTAssertEqual(try FlashVSRResourcePlan.temporaryBytes(media: source, settings: settings, frames: 24), 589561856)
  }
}
