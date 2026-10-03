import Foundation
import XCTest
@testable import H3MLX

final class H3CanvasAdmissionTests: XCTestCase {
  private let missing = URL(fileURLWithPath: "/missing")

  func testOneMegapixelLandscapeAndPortraitRequestsReachModelPreflight() throws {
    for (width, height) in [(1376, 768), (768, 1376)] {
      let request = try H3T2VARequest(prompt: "Beowulf trains in a boxing gym",
        width: width, height: height, durationSeconds: 5, seed: 20260922,
        requestedSteps: 5, transformer: missing, qwenPages: missing,
        tokenizer: missing, videoVAE: missing, audioVAE: missing)
      XCTAssertEqual(request.geometry.videoRows, 38_184)
      XCTAssertThrowsError(try H3T2VARunner.preflight(request))
    }
  }

  func testOneMegapixelReferenceRequestRetainsAllNineReferences() throws {
    let image = H3StillReference(rgb8: Data(count: 64 * 64 * 3),
      width: 64, height: 64)
    let request = try H3Ref2VAStillRequest(prompt: "Beowulf trains",
      references: Array(repeating: image, count: 9), width: 1376, height: 768,
      durationSeconds: 5, seed: 20260922, requestedSteps: 5,
      transformer: missing, qwenPages: missing, qwenVision: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing)
    XCTAssertEqual(request.references.count, 9)
    XCTAssertEqual(request.geometry.width, 1376)
  }

  func testLargerCanvasAndOversizedTimelineStillFailBeforeModelIO() throws {
    for (width, height, duration) in [(1408, 768, 5.0), (1376, 768, 15.0)] {
      XCTAssertThrowsError(try H3T2VARequest(prompt: "Beowulf trains",
        width: width, height: height, durationSeconds: duration, seed: 1,
        requestedSteps: 5, transformer: missing, qwenPages: missing,
        tokenizer: missing, videoVAE: missing, audioVAE: missing))
    }
  }

  func testOneMegapixelKeyframeAndControlEncodersReachCheckpointInspection() {
    let pixels = [UInt8](repeating: 0, count: 1376 * 768 * 3)
    XCTAssertThrowsError(try H3VideoVAEEncoder.encodeKeyframe(
      checkpointURL: missing, rgb8: pixels, width: 1376, height: 768)) { error in
      XCTAssertTrue(String(describing: error).contains("Cannot open checkpoint"))
    }
    XCTAssertThrowsError(try H3VideoVAEEncoder.encodeControlVideo(
      checkpointURL: missing, rgb8: Array(repeating: pixels, count: 5).flatMap { $0 },
      frameCount: 5, width: 1376, height: 768)) { error in
      XCTAssertTrue(String(describing: error).contains("Cannot open checkpoint"))
    }
  }
}
