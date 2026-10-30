import Foundation
import XCTest
@testable import StudioCore

final class NativePixelCQRecipeTests: XCTestCase {
  private func bytes(_ settings: [String: Any] = [:]) throws -> Data {
    try JSONSerialization.data(withJSONObject: ["format": "weetodd-video-processing-v1", "steps": [
      ["id": "upscale", "kind": "native-pixel-cq", "input": "source", "pixelCQ": settings]]])
  }
  func testNativeRecipeRoundTripsBothStrengthsAndKeepsGenerationIdentity() throws {
    let recipe = try JSONDecoder().decode(VideoProcessingRecipe.self, from: bytes())
    var clip = Clip(engine: .h3); let generation = clip.generationFingerprint
    clip.videoProcessing = recipe
    XCTAssertEqual(clip.generationFingerprint, generation)
    let data = try JSONEncoder().encode(recipe)
    let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let step = try XCTUnwrap((raw["steps"] as? [[String: Any]])?.first)
    XCTAssertNil(step["settings"], "Native recipes must not carry ignored FlashVSR controls")
    let settings = try XCTUnwrap(step["pixelCQ"] as? [String: Any])
    XCTAssertEqual(settings["pixelStrength"] as? Double, 1)
    XCTAssertEqual(settings["cqStrength"] as? Double, 0.5)
    XCTAssertEqual(settings["refinementStrength"] as? Double, 0.75)
    XCTAssertNil(settings["fps"], "The source owns FPS")
    XCTAssertEqual(try JSONDecoder().decode(VideoProcessingRecipe.self, from: data), recipe)
  }
  func testInvalidNativeStrengthsAndUnknownFPSCannotBeSilentlyIgnored() throws {
    for settings: [String: Any] in [["pixelStrength": 0], ["pixelStrength": 2.1], ["cqStrength": -0.1],
      ["cqStrength": 2.1], ["refinementStrength": 1], ["fps": 30], ["preserveAudio": false]] {
      XCTAssertThrowsError(try JSONDecoder().decode(VideoProcessingRecipe.self, from: bytes(settings)))
    }
    let first = try JSONDecoder().decode(VideoProcessingRecipe.self, from: bytes(["cqStrength": 0.5]))
    let second = try JSONDecoder().decode(VideoProcessingRecipe.self, from: bytes(["cqStrength": 0.6]))
    XCTAssertNotEqual(try first.fingerprint, try second.fingerprint)
  }
  func testUnusedUpscalerWorkerDoesNotChangeH3GenerationIdentity() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mp4"); try Data([1]).write(to: source)
    var project = StudioProject(); var clip = Clip(engine: .h3); clip.sourcePath = source.path
    clip.videoProcessing = VideoProcessingRecipe(pixelCQ: .init()); project.clips = [clip]
    let runtime = NativePixelCQRuntime(workerPath: "/usr/bin/true", ffmpegPath: "/usr/bin/true", ffprobePath: "/usr/bin/true", components: [:], cqAdapterPath: nil)
    var job = try NativeHeadlessJob(project: project, recipes: [:], workers: ["ltx25": "/usr/bin/true"], ffmpeg: "/usr/bin/true", nativeProcessingRuntime: runtime)
    let before = try job.generationIdentity()
    job.nativeProcessingRuntime?.workerPath = "/usr/bin/false"; job.workers["ltx25"] = "/usr/bin/false"
    XCTAssertEqual(try job.generationIdentity(), before)
  }

  func testSavedNativeLineageLabelShowsBothStrengthsWithoutATurboDecoder() throws {
    let source = FlashVSRFile(path: "/source.mp4", sha256: String(repeating: "a", count: 64), bytes: 1)
    var settings = NativePixelCQSettings(); settings.pixelStrength = 0.8; settings.cqStrength = 0.6
    let native = VideoProcessingLineage(source: source, sourceVersionID: nil, receiptPath: "/receipt.json", fingerprint: "native", recipe: .init(pixelCQ: settings))
    XCTAssertEqual(native.label, "Native Pixel 0.80 · CQ 0.60")
    let restored = try JSONDecoder().decode(VideoProcessingLineage.self, from: JSONEncoder().encode(native))
    XCTAssertEqual(restored.label, "Native Pixel 0.80 · CQ 0.60")
    let flash = VideoProcessingLineage(source: source, sourceVersionID: nil, receiptPath: "/receipt.json", fingerprint: "flash", recipe: .init())
    XCTAssertEqual(flash.label, "Turbo Tiny · faster")
  }

}
