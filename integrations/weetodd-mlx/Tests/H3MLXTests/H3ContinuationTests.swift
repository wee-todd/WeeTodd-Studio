import Foundation
import XCTest
@testable import H3MLX

final class H3ContinuationTests: XCTestCase {
  func testWindowAndTrimAdmission() throws {
    let source = URL(fileURLWithPath: "/tmp/context/manifest.json")
    let hash = String(repeating: "a", count: 64)
    let plan = try H3Continuation.Plan(contextFrames: 22,
      requestedDuration: 85.0 / 24, sourceManifest: source,
      sourceSHA256: hash, saveContext: true)
    XCTAssertEqual(plan.generatedFrames, 107)
    XCTAssertEqual(plan.publishedFrames, 85)
    XCTAssertEqual(plan.overlapFrames, 22)
    XCTAssertEqual(plan.tailTrimFrames, 0)
    XCTAssertThrowsError(try H3Continuation.Plan(contextFrames: 22,
      requestedDuration: 84.0 / 24, sourceManifest: source,
      sourceSHA256: hash, saveContext: true))
    XCTAssertThrowsError(try H3Continuation.Plan(contextFrames: 39,
      requestedDuration: 4, sourceManifest: source,
      sourceSHA256: nil, saveContext: false))
  }

  func testChannelMajorTailRoundTripAndTamperDetection() throws {
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 85.0 / 24)
    let video = (0..<(geometry.videoRows * 96)).map(Float.init)
    let audio = (0..<(geometry.audioRows * 32)).map(Float.init)
    let rows = try H3Continuation.tail(video: video, audio: audio,
      geometry: geometry, contextFrames: 22)
    let audioTail = Int((22.0 / 24 * 40).rounded(.toNearestOrEven)) * 32
    let sourceChannel = geometry.audioLatentFrames * 32
    XCTAssertEqual(rows.audio.first, audio[sourceChannel - audioTail])
    XCTAssertEqual(rows.audio[audioTail], audio[2 * sourceChannel - audioTail])

    let plan = try H3Continuation.Plan(contextFrames: 22,
      requestedDuration: 85.0 / 24, sourceManifest: nil,
      sourceSHA256: nil, saveContext: true)
    let folder = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let identity = String(repeating: "b", count: 64)
    let saved = try H3Continuation.save(rows, plan: plan,
      width: 64, height: 64, identity: identity, directory: folder)
    let loaded = try XCTUnwrap(H3Continuation.load(manifestURL: saved.manifest,
      expectedSHA256: saved.sha256, contextFrames: 22,
      width: 64, height: 64, identity: identity))
    XCTAssertEqual(loaded.video, rows.video)
    XCTAssertEqual(loaded.audio, rows.audio)
    XCTAssertThrowsError(try H3Continuation.load(manifestURL: saved.manifest,
      expectedSHA256: saved.sha256, contextFrames: 22,
      width: 64, height: 64, identity: String(repeating: "c", count: 64)))
    let payload = folder.appendingPathComponent("latents.f32")
    var bytes = try Data(contentsOf: payload)
    bytes[0] ^= 1
    try bytes.write(to: payload)
    XCTAssertThrowsError(try H3Continuation.load(manifestURL: saved.manifest,
      expectedSHA256: saved.sha256, contextFrames: 22,
      width: 64, height: 64, identity: identity))
  }
}
