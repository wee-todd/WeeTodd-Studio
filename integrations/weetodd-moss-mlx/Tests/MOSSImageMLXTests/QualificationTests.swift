import Foundation
import MLX
import XCTest
@testable import MOSSImageMLX

/// Opt-in complete-component checks. Fixtures and weights live outside source.
final class QualificationTests: XCTestCase {
  func testSmallVAEAgainstIndependentSwiftReferenceAndBandSizesWhenProvided() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_MOSS_REFERENCE_ROOT"],
      let shared = ProcessInfo.processInfo.environment["WEETODD_MOSS_MODEL_SHARED"] else {
      throw XCTSkip("Opt-in independent full VAE output comparison")
    }
    let directory = URL(fileURLWithPath: root)
    let packed = try MLX.loadArrays(url: directory.appendingPathComponent("captures/sft-cup-512/final-latents.safetensors"))["latents"]!
    let reference = clip(try MLX.loadArrays(url: directory.appendingPathComponent("oracle-results/decoded.safetensors"))["decoded"]!.transposed(0, 2, 3, 1), min: -1, max: 1)
    let source = MOSSWeightSource(index: try MOSSCheckpointIndex(directory: URL(fileURLWithPath: shared).appendingPathComponent("vae")))
    defer { source.release() }
    let ordinary = try MOSSSmallVAE(weights: source).decode(packedLatents: packed)
    let model = MOSSSmallVAE(weights: source, maximumBandBytes: 1024 * 1024)
    let bounded = try model.decode(packedLatents: packed)
    let referenceDelta = ordinary - reference, bandDelta = ordinary - bounded
    let referenceRMS = sqrt(mean(referenceDelta * referenceDelta)).item(Float.self)
    let bandRMS = sqrt(mean(bandDelta * bandDelta)).item(Float.self)
    print("MOSS VAE reference RMS=\(referenceRMS) max=\(abs(referenceDelta).max().item(Float.self)); band RMS=\(bandRMS) max=\(abs(bandDelta).max().item(Float.self))")
    XCTAssertEqual(ordinary.shape, reference.shape)
    XCTAssertLessThan(referenceRMS, 0.0002); XCTAssertLessThan(abs(referenceDelta).max().item(Float.self), 0.005)
    XCTAssertLessThan(bandRMS, 0.0002); XCTAssertLessThan(abs(bandDelta).max().item(Float.self), 0.005)
  }
}
