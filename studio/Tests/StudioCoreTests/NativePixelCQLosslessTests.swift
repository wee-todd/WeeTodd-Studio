import Foundation
import XCTest
@testable import StudioCore

final class PixelCQLosslessFixture {
  let base: PixelCQWorkerFixture
  var renders = 0
  init(_ root: URL) throws { base = try .init(root) }
  func worker(_ executable: String, _ recipe: URL, _ output: URL, _ mode: String) throws -> [String: Any] {
    if mode == "preflight" { return try base.worker(executable, recipe, output, mode) }
    renders += 1
    let wrapper = try JSONSerialization.jsonObject(with: Data(contentsOf: recipe)) as! [String: Any]
    let body = wrapper["movie_upscale"] as! [String: Any], source = body["source"] as! [String: Any]
    let intermediateOnly = body["intermediate_only"] as? Bool ?? false
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let restored = output.appendingPathComponent("restored.mkv"), decoded = output.appendingPathComponent("expected.rgb24")
    let width = (source["width"] as! Int)*2, height = (source["height"] as! Int)*2, fps = source["fps"] as! Double
    try VideoProcessingFixture.ffmpeg(["-i", source["path"] as! String, "-vf", "scale=\(width):\(height)", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", restored.path])
    try VideoProcessingFixture.ffmpeg(["-i", restored.path, "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", decoded.path])
    let rgbHash = try NativeHeadlessJob.fileHash(decoded); try FileManager.default.removeItem(at: decoded)
    let adapters = [["path": base.runtime.components["pixel_spatial_adapter"]!, "strength": Double(Float(body["pixel_strength"] as! Double))] as [String: Any]]
      + ((body["refinement_loras"] as? [[String: Any]] ?? []).map { ["path": $0["path"]!, "strength": Double(Float($0["strength"] as! Double))] })
    let proof: [String: Any] = ["file": "restored.mkv", "sha256": try NativeHeadlessJob.fileHash(restored), "rgb_sha256": rgbHash,
      "codec": "ffv1", "pixel_format": "bgr0", "frames": source["frames"]!, "width": width, "height": height,
      "fps_numerator": Int(fps), "fps_denominator": 1, "has_audio": false]
    let metadata: [String: Any] = ["nativeRuntime": "swift-mlx", "task": "video_upscale", "pythonModelInference": false,
      "frames": source["frames"]!, "fps": fps, "width": width, "height": height, "source_movie_sha256": source["sha256"]!,
      "source_rgb_sha256": source["rgb_sha256"]!, "component_identity_sha256": try NativeHeadlessJob.fileHash(base.model),
      "transformer_evaluations_per_chunk": 3, "applied_refinement_adapters": adapters, "restored": proof]
    let video = output.appendingPathComponent(intermediateOnly ? "restored.mkv" : "render.mp4")
    if !intermediateOnly { try VideoProcessingFixture.ffmpeg(["-i", restored.path, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", video.path]) }
    try JSONSerialization.data(withJSONObject: metadata).write(to: output.appendingPathComponent("metadata.json"))
    return ["nativeRuntime": "swift-mlx", "video": video.path, "restored": restored.path, "metadata": metadata,
      "use_complete_duration": true, "usable_source_in": 0.0, "usable_duration": Double(source["frames"] as! Int)/fps]
  }
}

final class NativePixelCQLosslessTests: XCTestCase {
  func testIntermediateSkipsDeliveryAndAudioThenReusesVerifiedLosslessReceipt() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQLosslessFixture(root), media = try VideoProcessingFixture(root: root), source = try media.source()
    let plan = try await fixture.base.plan(), resources = try XCTUnwrap(plan.native), cache = root.appendingPathComponent("cache")
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: false)
    let first = try await NativePixelCQService.execute(resources: resources, recipe: plan.recipe, source: source, directory: cache,
      cancellation: .init(), worker: fixture.worker, emit: { _ in }, intermediateOnly: true)
    let native = try XCTUnwrap(first.receipt.native), restored = try XCTUnwrap(native.restored)
    XCTAssertEqual(native.video, restored); XCTAssertFalse(native.media.hasAudio)
    XCTAssertNil(native.sourceAudioSHA256); XCTAssertNil(native.outputAudioSHA256)
    XCTAssertFalse(FileManager.default.fileExists(atPath: URL(fileURLWithPath: restored.path).deletingLastPathComponent().appendingPathComponent("render.mp4").path))
    let repeated = try await NativePixelCQService.execute(resources: resources, recipe: plan.recipe, source: source, directory: cache,
      cancellation: .init(), worker: fixture.worker, emit: { _ in }, intermediateOnly: true)
    XCTAssertEqual(repeated.reusedSteps, 1); XCTAssertEqual(fixture.renders, 1)
    let delivered = try await NativePixelCQService.execute(resources: resources, recipe: plan.recipe, source: source, directory: cache,
      cancellation: .init(), worker: fixture.worker, emit: { _ in })
    let normal = try XCTUnwrap(delivered.receipt.native)
    XCTAssertNotNil(normal.restored); XCTAssertTrue(normal.media.hasAudio)
    XCTAssertEqual(normal.sourceAudioSHA256, normal.outputAudioSHA256); XCTAssertEqual(fixture.renders, 2)
    var legacy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(normal)) as! [String: Any]
    legacy.removeValue(forKey: "restored")
    XCTAssertNil(try JSONDecoder().decode(NativePixelCQReceipt.self, from: JSONSerialization.data(withJSONObject: legacy)).restored)
  }
  func testLegacyLossyWorkerCannotServeIntermediateMode() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try PixelCQWorkerFixture(root), media = try VideoProcessingFixture(root: root), source = try media.source()
    let plan = try await fixture.plan(), cache = root.appendingPathComponent("cache")
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: false)
    do {
      _ = try await NativePixelCQService.execute(resources: XCTUnwrap(plan.native), recipe: plan.recipe, source: source, directory: cache,
        cancellation: .init(), worker: fixture.worker, emit: { _ in }, intermediateOnly: true)
      XCTFail("Legacy MP4 was silently accepted as lossless")
    } catch { XCTAssertTrue(String(describing: error).contains("lossless RGB"), "\(error)") }
  }
}
