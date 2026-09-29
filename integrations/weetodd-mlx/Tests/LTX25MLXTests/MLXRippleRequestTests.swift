import Foundation
import CryptoKit
import XCTest
@testable import LTX25MLX

final class MLXRippleRequestTests: XCTestCase {
  private func base() -> [String: Any] {
    ["version": 1, "engine": "ltx25", "task": "ripple",
     "gemma_root": "/models/gemma", "transformer_root": "/models/transformer",
     "connector_checkpoint": "/models/connector", "video_checkpoint": "/models/video",
     "audio_checkpoint": "/models/audio", "adapter_path": "/models/ripple.safetensors",
     "adapter_strength": 1.35, "guide_path": "/input/guide.rgb24",
     "first_reference_path": "/input/edited-first.png",
     "source_path": "/input/source.mp4", "source_sha256": String(repeating: "a", count: 64),
     "source_start": 0, "duration": 10.0 / 24.0, "editorial_frames": 10,
     "width": 64, "height": 64, "frames": 17, "fps": 24, "seed": 42,
     "prompt": "Propagate the edited first frame.", "reference_strength": 1,
     "anchors": [], "audio_policy": "preserve", "ffmpeg_path": "/bin/ffmpeg",
     "output_directory": "/output/ripple"]
  }

  private func decode(_ value: [String: Any]) throws -> MLXRippleRequest {
    try JSONDecoder().decode(MLXRippleRequest.self,
      from: JSONSerialization.data(withJSONObject: value))
  }

  func testStrictRequestAdmitsOneEditedReferenceAndRoundTrips() throws {
    let request = try decode(base())
    XCTAssertEqual(request.geometry.frames, 17)
    XCTAssertEqual(request.editorialFrames, 10)
    XCTAssertEqual(request.anchors.count, 0)
    XCTAssertEqual(try JSONDecoder().decode(MLXRippleRequest.self,
      from: JSONEncoder().encode(request)).sourceSHA256.count, 64)
  }

  func testTimedAnchorsRemainDistinctAndOrdered() throws {
    var value = base()
    value["anchors"] = [["frame": 4, "path": "/input/4.png", "strength": 0.8],
      ["frame": 8, "path": "/input/8.png", "strength": 1]]
    let request = try decode(value)
    XCTAssertEqual(request.anchors.map(\.frame), [4, 8])
    value["anchors"] = [["frame": 8, "path": "/input/8.png", "strength": 1],
      ["frame": 4, "path": "/input/4.png", "strength": 0.8]]
    XCTAssertThrowsError(try decode(value))
  }

  func testUnsupportedControlsAndMismatchedTimingFailBeforeWeights() throws {
    for (key, replacement): (String, Any) in [
      ("task", "i2v"), ("frames", 9), ("editorial_frames", 18),
      ("duration", 3.0), ("source_sha256", "unverified"),
      ("adapter_strength", 0), ("reference_strength", -0.1),
      ("audio_policy", "generate"), ("guide_path", "relative.rgb24"),
      ("stage2_steps", 3)
    ] {
      var value = base(); value[key] = replacement
      XCTAssertThrowsError(try decode(value), key)
    }
    var value = base(); value.removeValue(forKey: "guide_path")
    XCTAssertThrowsError(try decode(value))
  }

  func testGuideMustBeExactRegularRGB24File() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let guide = root.appendingPathComponent("guide.rgb24")
    var value = base(); value["guide_path"] = guide.path
    let request = try decode(value)
    XCTAssertThrowsError(try request.validateGuide())
    try Data(repeating: 0, count: 17 * 64 * 64 * 3 - 1).write(to: guide)
    XCTAssertThrowsError(try request.validateGuide())
    try Data(repeating: 0, count: 17 * 64 * 64 * 3).write(to: guide)
    XCTAssertNoThrow(try request.validateGuide())
  }

  func testSourceIdentityIsCheckedBeforeGeneration() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mp4")
    let bytes = Data("source A".utf8)
    try bytes.write(to: source)
    var value = base(); value["source_path"] = source.path
    value["source_sha256"] = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let request = try decode(value)
    XCTAssertNoThrow(try request.validateSource())
    try Data("source B".utf8).write(to: source)
    XCTAssertThrowsError(try request.validateSource())
  }

  func testRippleMemoryAdmissionIncludesAppendedGuideAndVAE() throws {
    let request = try decode(base())
    let generous = try MLXStudioMemoryPlan(ripple: request,
      physicalMemory: 128 * 1024 * 1024 * 1024,
      recommendedWorkingSet: 96 * 1024 * 1024 * 1024)
    XCTAssertGreaterThan(generous.transformerActivationBytes, 0)
    XCTAssertGreaterThan(generous.videoActivationBytes, 0)
    XCTAssertThrowsError(try MLXStudioMemoryPlan(ripple: request,
      physicalMemory: 8 * 1024 * 1024 * 1024,
      recommendedWorkingSet: 4 * 1024 * 1024 * 1024))
  }
}
