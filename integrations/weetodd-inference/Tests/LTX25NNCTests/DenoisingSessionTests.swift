import Foundation
import LTX25Engine
import XCTest
@testable import LTX25NNC

final class DenoisingSessionTests: XCTestCase {
  struct Fixture: Decodable { let configuration: AVBlockConfiguration; let inputs: [String: [Float]] }
  func fixture() throws -> Fixture {
    try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
      Bundle.module.url(forResource: "denoiser-reference", withExtension: "json", subdirectory: "Fixtures")!))
  }
  func weight(_ name: String, _ shape: [Int]) -> [Float] {
    let seed = name.utf8.reduce(0) { $0 + Int($1) }
    let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
  }
  func testSessionMatchesIndependentEvaluationsLoadsAdaptiveWeightsOnceAndReleases() throws {
    let f = try fixture(), sigmas: [Float] = [0.75, 0.5, 0.25]
    var reads: [String: Int] = [:]
    func fixed(_ name: String, _ shape: [Int]) -> [Float] { reads[name, default: 0] += 1; return weight(name, shape) }
    let session = try LTXDenoisingSession(configuration: f.configuration, inputs: f.inputs,
      sigmas: sigmas, blockCount: 1, fixedWeights: fixed)
    let reference = try DenoiserRunner(configuration: f.configuration, blockCount: 1)
    for sigma in sigmas {
      let expected = try reference.evaluate(f.inputs, sigma: sigma, fixedWeights: weight,
        blockWeights: { self.weight("transformer_blocks.\($0)." + $1, $2) })
      let actual = try session.evaluate(video: f.inputs["video_latent"]!, audio: f.inputs["audio_latent"]!,
        sigma: sigma, fixedWeights: fixed, blockWeights: { self.weight("transformer_blocks.\($0)." + $1, $2) })
      XCTAssertEqual(actual.videoVelocity, expected.videoVelocity)
      XCTAssertEqual(actual.audioVelocity, expected.audioVelocity)
    }
    XCTAssertEqual(session.graphBuildCount, 1)
    XCTAssertTrue(reads.filter { $0.key.contains("timestep_embedder") }.values.allSatisfy { $0 == 1 })
    try session.release()
    XCTAssertTrue(session.isReleased)
    XCTAssertThrowsError(try session.evaluate(video: f.inputs["video_latent"]!, audio: f.inputs["audio_latent"]!,
      sigma: 0.5, fixedWeights: weight, blockWeights: { self.weight($1, $2) }))
  }
  func testEightStepSessionEqualsStagedSamplingExactly() throws {
    let f = try fixture()
    let schedule = try SamplingSchedule(sigmas: [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0], eta: 0)
    let sampler = try LTXSamplingRunner(configuration: f.configuration, blockCount: 1)
    let expected = try sampler.evaluate(f.inputs, schedule: schedule, reuseSession: false,
      fixedWeights: weight, blockWeights: { self.weight("transformer_blocks.\($0)." + $1, $2) })
    let actual = try sampler.evaluate(f.inputs, schedule: schedule, reuseSession: true,
      fixedWeights: weight, blockWeights: { self.weight("transformer_blocks.\($0)." + $1, $2) })
    XCTAssertEqual(actual.video, expected.video)
    XCTAssertEqual(actual.audio, expected.audio)
    XCTAssertEqual(sampler.lastGraphBuildCount, 1)
  }

  func testNewSessionWithDifferentPromptAndShapeHasNoPriorState() throws {
    let f = try fixture()
    let first = try LTXDenoisingSession(configuration: f.configuration, inputs: f.inputs,
      sigmas: [0.5], blockCount: 1, fixedWeights: weight)
    _ = try first.evaluate(video: f.inputs["video_latent"]!, audio: f.inputs["audio_latent"]!,
      sigma: 0.5, fixedWeights: weight, blockWeights: { self.weight($1, $2) })
    try first.release()
    let c = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 7, audioTokens: 4, textTokens: 6)
    let reference = try DenoiserRunner(configuration: c, blockCount: 1)
    let inputs = reference.inputShapes.mapValues { [Float](repeating: 0.03125, count: $0.reduce(1, *)) }
    let fresh = try LTXDenoisingSession(configuration: c, inputs: inputs, sigmas: [0.5],
      blockCount: 1, fixedWeights: weight)
    let actual = try fresh.evaluate(video: inputs["video_latent"]!, audio: inputs["audio_latent"]!,
      sigma: 0.5, fixedWeights: weight, blockWeights: { self.weight($1, $2) })
    let expected = try reference.evaluate(inputs, sigma: 0.5, fixedWeights: weight,
      blockWeights: { self.weight($1, $2) })
    XCTAssertEqual(actual.videoVelocity, expected.videoVelocity)
    XCTAssertEqual(actual.audioVelocity, expected.audioVelocity)
    try fresh.release()
  }

  func testObserverFailureUnloadsSessionAndReentryCannotReleaseActiveSlot() throws {
    enum Stop: Error { case now }
    let f = try fixture()
    let session = try LTXDenoisingSession(configuration: f.configuration, inputs: f.inputs,
      sigmas: [0.5], blockCount: 1, fixedWeights: weight)
    XCTAssertThrowsError(try session.evaluate(video: f.inputs["video_latent"]!, audio: f.inputs["audio_latent"]!,
      sigma: 0.5, fixedWeights: weight, blockWeights: { self.weight($1, $2) }) { _ in
        XCTAssertThrowsError(try session.release())
        throw Stop.now
      })
    XCTAssertTrue(session.isReleased)
  }
}
