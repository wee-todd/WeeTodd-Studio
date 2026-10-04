import Foundation
import XCTest
@testable import H3MLX

final class H3ReferencePreparationControlTests: XCTestCase {
  func testOwnedCanvasRoundingBudgetAndDownOnlyPolicy() throws {
    let half = try H3ReferenceCanvasPolicy.image(sourceWidth: 448, sourceHeight: 1344,
      outputWidth: 608, outputHeight: 352, percent: 50)
    XCTAssertEqual(half.width, 192); XCTAssertEqual(half.height, 576)
    let full = try H3ReferenceCanvasPolicy.image(sourceWidth: 448, sourceHeight: 1344,
      outputWidth: 608, outputHeight: 352, percent: 100)
    XCTAssertEqual(full.width, 256); XCTAssertEqual(full.height, 800)
    let large = try H3ReferenceCanvasPolicy.image(sourceWidth: 448, sourceHeight: 1344,
      outputWidth: 608, outputHeight: 352, percent: 400)
    XCTAssertEqual(large.width, 448); XCTAssertEqual(large.height, 1344)
    let video = try H3ReferenceCanvasPolicy.video(sourceWidth: 1920, sourceHeight: 1080,
      outputWidth: 768, outputHeight: 448, policy: .matchOutput)
    XCTAssertEqual(video.width, 768); XCTAssertEqual(video.height, 448)
    let native = try H3ReferenceCanvasPolicy.video(sourceWidth: 1920, sourceHeight: 1080,
      outputWidth: 768, outputHeight: 448, policy: .nativeH3)
    XCTAssertEqual(native.width, 1344); XCTAssertEqual(native.height, 768)
    XCTAssertThrowsError(try H3ReferenceCanvasPolicy.image(sourceWidth: 10, sourceHeight: 400,
      outputWidth: 608, outputHeight: 352, percent: 100))
  }
  func testExplicitDensityIndependentIndicesRetainSourceDuration() throws {
    // Independently calculated NumPy rint(linspace(0,38,5)) ties-to-even witness.
    let pixels = Data(repeating: 12, count: 39 * 64 * 64 * 3)
    let quarter = try H3ReferenceTemporalPolicy.resolve(rgb8: pixels, frames: 39,
      width: 64, height: 64, policy: .quarter)
    XCTAssertEqual(quarter.indices, [0,10,19,28,38])
    XCTAssertEqual(quarter.sourceLatentFrames, 12); XCTAssertEqual(quarter.latentFrames, 2)
    let full = try H3ReferenceTemporalPolicy.resolve(rgb8: pixels, frames: 39,
      width: 64, height: 64, policy: .full)
    XCTAssertEqual(full.indices, Array(0..<39)); XCTAssertEqual(full.latentFrames, 12)
    let half = try H3ReferenceTemporalPolicy.resolve(rgb8: pixels, frames: 39,
      width: 64, height: 64, policy: .half)
    XCTAssertEqual(half.indices.count, 22); XCTAssertEqual(half.indices.last, 38)
    XCTAssertEqual(half.sourceLatentFrames, 12)
  }
  func testAutomaticActivityAndMalformedFrames() throws {
    for (delta, expected) in [(0, 0.25), (1, 0.25), (2, 0.5), (8, 1.0)] {
      var bytes = Data()
      for frame in 0..<39 { bytes.append(Data(repeating: UInt8((frame * delta) % 256), count: 64 * 64 * 3)) }
      let decision = try H3ReferenceTemporalPolicy.resolve(rgb8: bytes, frames: 39,
        width: 64, height: 64, policy: .automatic)
      XCTAssertEqual(decision.density, expected)
    }
    XCTAssertThrowsError(try H3ReferenceTemporalPolicy.resolve(rgb8: Data(), frames: 39,
      width: 64, height: 64, policy: .half))
  }
  func testCancellationPrecedesDensityScan() async throws {
    let task = Task {
      while !Task.isCancelled { await Task.yield() }
      _ = try H3ReferenceTemporalPolicy.resolve(rgb8: Data(repeating: 12, count: 39 * 64 * 64 * 3),
        frames: 39, width: 64, height: 64, policy: .automatic)
    }
    task.cancel()
    do { try await task.value; XCTFail("Cancelled density selection must not proceed.") }
    catch is CancellationError { }
  }
  func testPolicyStrictKindsDefaultsAndBooleanRejection() throws {
    XCTAssertNil(try H3ReferencePreparationControls.parse([:], kind: "image"))
    let partial = try H3ReferencePreparationControls.parse(["temporal_density": "half"], kind: "video")
    XCTAssertEqual(partial?.videoSizePolicy, .matchOutput); XCTAssertEqual(partial?.temporalDensity, .half)
    let size = try H3ReferencePreparationControls.parse(["size_policy": "native_h3"], kind: "video")
    XCTAssertEqual(size?.temporalDensity, .full)
    for value in ([true, 49, 401, 100.5, NSNull(), "100"] as [Any]) {
      XCTAssertThrowsError(try H3ReferencePreparationControls.parse(["image_pixel_budget_percent": value], kind: "image"))
    }
    XCTAssertThrowsError(try H3ReferencePreparationControls.parse(["size_policy": "match_output"], kind: "audio"))
    XCTAssertThrowsError(try H3ReferencePreparationControls.parse(["temporal_density": "quarter"], kind: "image"))
  }
  func testAggregateAdmissionBeforePixelPackingAndDecisionIdentity() throws {
    let image = H3StillReference(rgb8: Data(repeating: 12, count: 1024 * 1024 * 3),
      width: 1024, height: 1024, pixelBudgetPercent: 400)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([.image(image), .image(image)]))
    let pixels = Data(repeating: 12, count: 39 * 64 * 64 * 3)
    let decision = try H3ReferenceTemporalPolicy.resolve(rgb8: pixels, frames: 39,
      width: 64, height: 64, policy: .quarter)
    let controls = try H3ReferencePreparationControls(videoSizePolicy: .matchOutput, temporalDensity: .quarter)
    let video = H3VideoReference(rgb8: pixels, frameCount: 39, width: 64, height: 64,
      controls: controls, temporalDecision: decision)
    XCTAssertNoThrow(try H3VideoReferencePreparation.validate([.video(video)]))
    XCTAssertEqual(video.persistentFrameCount, 5); XCTAssertEqual(video.sourceLatentFrames, 12)
    let mismatched = H3VideoReference(rgb8: pixels, frameCount: 39, width: 64, height: 64,
      controls: try .init(videoSizePolicy: .matchOutput, temporalDensity: .half), temporalDecision: decision)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([.video(mismatched)]))
  }
}
