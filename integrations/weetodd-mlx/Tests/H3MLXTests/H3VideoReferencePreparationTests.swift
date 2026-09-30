import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoReferencePreparationTests: XCTestCase {
  func testRejectsMalformedVideoBeforeWeights() throws {
    let still = H3StillReference(rgb8: Data(repeating: 16,
      count: 64 * 64 * 3), width: 64, height: 64)
    let malformed = H3VideoReference(rgb8: Data(repeating: 128,
      count: 6 * 64 * 64 * 3), frameCount: 6, width: 64, height: 64)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([
      .image(still), .video(malformed)]))
    let valid = H3VideoReference(rgb8: Data(repeating: 128,
      count: 5 * 64 * 64 * 3), frameCount: 5, width: 64, height: 64)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate(
      Array(repeating: .video(valid), count: 4)))
  }

  func testInstalledMixedVideoReferencePreservesQwenAndLatentOrder() throws {
    guard let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let videoVAE = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set installed H3 tokenizer and video VAE paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 2.5)
    let still = H3StillReference(rgb8: Data(repeating: 16,
      count: 64 * 64 * 3), width: 64, height: 64)
    let video = H3VideoReference(rgb8: Data(repeating: 208,
      count: 22 * 64 * 64 * 3), frameCount: 22, width: 64, height: 64)
    let references: [H3Ref2VAReference] = [.image(still), .video(video)]
    let prepared = try H3VideoReferencePreparation.prepare(
      prompt: "A person walks.", geometry: geometry,
      references: references, tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 32)
    XCTAssertEqual(prepared.qwenRequest.visualRanges.count, 2)
    XCTAssertEqual(prepared.qwenGrids.count, 2)
    XCTAssertEqual(prepared.qwenPixels.shape, [32, 1536])
    XCTAssertLessThan(prepared.qwenPixels[0, 0].item(Float.self),
      prepared.qwenPixels[16, 0].item(Float.self))
    let rows = try H3VideoReferencePreparation.encodeVideoRows(
      references: references, layout: prepared.layout,
      videoVAEURL: URL(fileURLWithPath: videoVAE))
    XCTAssertEqual(rows.shape, [1, 32, 96])
    XCTAssertTrue(rows.asArray(Float.self).allSatisfy(\.isFinite))
  }
}
