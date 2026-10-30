import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRMediaTests: XCTestCase {
  private func probe(_ extra: [String: Any] = [:]) throws -> Data {
    var video: [String: Any] = ["codec_type": "video", "width": 128, "height": 64, "avg_frame_rate": "30000/1001", "pix_fmt": "yuv420p", "sample_aspect_ratio": "1:1"]
    video.merge(extra) { _, new in new }
    return try JSONSerialization.data(withJSONObject: ["streams": [video, ["codec_type": "audio", "sample_rate": "32000", "channels": 2]]])
  }
  func testRationalCadenceAndAudioSurviveInspection() throws {
    let info = try FlashVSRMediaInfo.parse(summary: probe(), timestamps: Data("0.000000\n0.033367\n0.066733\n".utf8))
    XCTAssertEqual(info.fps.numerator, 30000); XCTAssertEqual(info.fps.denominator, 1001)
    XCTAssertEqual(info.frames, 3); XCTAssertTrue(info.hasAudio); XCTAssertEqual(info.audioRate, 32000)
  }
  func testInternalIntermediateUsesOnlyItsVerifiedFrozenClock() throws {
    let summary = try probe(["avg_frame_rate": "0/0", "r_frame_rate": "48000/1"])
    XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: summary, timestamps: Data("0\n".utf8)))
    let fps = try FlashVSRFrameRate(numerator: 24, denominator: 1)
    let info = try FlashVSRMediaInfo.parse(summary: summary, timestamps: Data("0\n".utf8), expectedFrameRate: fps)
    XCTAssertEqual(info.frames, 1); XCTAssertEqual(info.fps.value, 24)
    XCTAssertEqual(try FlashVSRMediaInfo.parse(summary: summary, timestamps: Data("0\n0.041667\n".utf8), expectedFrameRate: fps).frames, 2)
    XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: summary, timestamps: Data("0\n0.12\n".utf8), expectedFrameRate: fps))
  }
  func testVFRHDRAndAlphaRejectBeforeInference() throws {
    XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: probe(), timestamps: Data("0\n0.033367\n0.12\n".utf8)))
    XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: probe(["color_transfer": "smpte2084"]), timestamps: Data("0\n".utf8)))
    XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: probe(["pix_fmt": "yuva420p"]), timestamps: Data("0\n".utf8)))
  }
  func testNonzeroSourceIntervalKeepsExactCount() throws {
    let info = try FlashVSRMediaInfo.parse(summary: probe(), timestamps: Data("1\n1.033367\n1.066733\n".utf8))
    XCTAssertEqual(info.firstTimestamp, 1, accuracy: 0.000001)
    XCTAssertThrowsError(try FlashVSRFrameRate(numerator: 0, denominator: 1))
  }
  func testMissingTimestampIsNotSilentlyDiscarded() throws {
    for text in ["0\nN/A\n", "0\n0.033367\nN/A\n", "0\ninvalid,side-data\n"] {
      XCTAssertThrowsError(try FlashVSRMediaInfo.parse(summary: probe(), timestamps: Data(text.utf8)))
    }
    let annotated = try FlashVSRMediaInfo.parse(summary: probe(), timestamps: Data("0,User Data Unregistered SEI message\n\n0.033367\n".utf8))
    XCTAssertEqual(annotated.frames, 2)
  }
  func testNativeFailureNeverSubstitutesAnExternalEngine() async throws {
    var settings = FlashVSRSettings(); settings.backend = .swiftMLX
    var runtime = FlashVSRRuntime()
    runtime.vpipePath = "/usr/bin/true"; runtime.swiftWorkerPath = "/does/not/exist/WeeToddFlashVSRWorker"
    runtime.modelRoot = "/models"
    runtime.ffmpegPath = "/usr/bin/true"; runtime.ffprobePath = "/usr/bin/true"
    do { _ = try await FlashVSRService.preflightResources(settings: settings, runtime: runtime); XCTFail("A missing selected native helper must fail") }
    catch { XCTAssertTrue(error.localizedDescription.contains("native FlashVSR+ helper")) }
  }
}
