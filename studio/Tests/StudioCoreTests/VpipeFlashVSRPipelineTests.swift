import Foundation
import XCTest
@testable import StudioCore
final class VpipeFlashVSRPipelineTests: XCTestCase {
  func testGraphPreservesGridAndDecoderSourceReference() throws {
    var settings = FlashVSRSettings(); settings.backend = .vpipe; settings.width = 1920; settings.height = 1080; settings.decoder = .vae
    let bytes = try VpipeFlashVSRPipeline.make(settings: settings, modelRoot: "/models/FlashVSR", input: "/media/selected.mkv", output: "/jobs/result.mkv", fps: 30000.0/1001, frames: 124)
    let graph = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
    let stages = try XCTUnwrap(graph["stages"] as? [[String: Any]])
    func stage(_ name: String) throws -> [String: Any] { try XCTUnwrap(stages.first { $0["id"] as? String == name }) }
    let generation = try XCTUnwrap(stage("generate-video")["config"] as? [String: Any])
    XCTAssertEqual(generation["fps"] as? Double, 30000.0/1001)
    XCTAssertEqual(generation["width"] as? Int, 1920); XCTAssertEqual(generation["height"] as? Int, 1152)
    let encoder = try XCTUnwrap(stage("flashvsr-src-encoder")["config"] as? [String: Any])
    XCTAssertEqual(encoder["decoder"] as? String, "vae"); XCTAssertEqual(encoder["continuity"] as? String, "stream")
    let ports = try XCTUnwrap(stage("vae-decode")["iports"] as? [[String: Any]])
    XCTAssertEqual(ports[2]["src"] as? String, "flashvsr-src-encoder"); XCTAssertEqual(ports[2]["oport"] as? Int, 1)
    let crop = try XCTUnwrap(stage("visible-output")["config"] as? [String: Any])
    XCTAssertEqual(crop["height"] as? Int, 1080); XCTAssertEqual(crop["scale"] as? Double, 1)
    XCTAssertEqual(crop["fit"] as? String, "manual")
    let video = try XCTUnwrap(stage("rgb-to-video")["config"] as? [String: Any])
    XCTAssertEqual(video["pix_fmt"] as? String, "rgb24")
    XCTAssertEqual(video["colorspace"] as? String, "bt709")
    XCTAssertEqual(video["fps"] as? Double, 30000.0/1001)
    let saved = try XCTUnwrap(stage("save-video")["config"] as? [String: Any])
    XCTAssertEqual(saved["video_codec"] as? String, "rawvideo")
    XCTAssertEqual(saved["format"] as? String, "nut")
  }
  func testTinyKeepsConditioningWhenColorCorrectionIsOff() throws {
    var settings = FlashVSRSettings(); settings.backend = .vpipe; settings.colorFix = .off
    let bytes = try VpipeFlashVSRPipeline.make(settings: settings, modelRoot: "/models", input: "/input.mkv", output: "/result.mkv", fps: 24, frames: 49)
    let graph = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
    let stages = graph["stages"] as! [[String: Any]]
    let encoder = stages.first { $0["id"] as? String == "flashvsr-src-encoder" }!["config"] as! [String: Any]
    XCTAssertEqual(encoder["decoder"] as? String, "tiny"); XCTAssertEqual(encoder["color_fix"] as? Bool, false)
    XCTAssertEqual((stages.first { $0["id"] as? String == "vae-decode" }!["iports"] as! [[String: Any]]).count, 3)
  }
}
