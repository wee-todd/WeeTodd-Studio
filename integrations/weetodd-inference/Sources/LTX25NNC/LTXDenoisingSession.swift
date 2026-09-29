import Foundation
import TensorIO

/// One transformer stage, bound to immutable conditioning and a finite schedule.
/// Internal to one sampling invocation: its providers cannot be replaced by a
/// public caller between steps. A new invocation always creates a fresh session.
final class LTXDenoisingSession {
  private let runner: DenoiserRunner
  private var inputs: [String: [Float]]
  private var active = false
  private(set) var isReleased = false
  var graphBuildCount: Int { runner.sessionGraphBuildCount }

  init(configuration: AVBlockConfiguration, inputs: [String: [Float]], sigmas: [Float],
    blockCount: Int = 48, maximumPreparationBytes: UInt64 = 0, sequenceAttention: Bool = false, experimentalPrecision: LTXPrecisionPolicy = .float32, fixedWeights: (String, [Int]) throws -> [Float],
    progress: (String) throws -> Void = { _ in }) throws {
    runner = try DenoiserRunner(configuration: configuration, blockCount: blockCount,
      maximumPreparationBytes: maximumPreparationBytes, sequenceAttention: sequenceAttention, experimentalPrecision: experimentalPrecision)
    let shapes = runner.inputShapes
    guard Set(inputs.keys) == Set(shapes.keys), shapes.allSatisfy({ name, shape in
      inputs[name]!.count == shape.reduce(1, *) && FloatValidation.allFinite(inputs[name]!)
    }) else { throw BlockError.invalid("Invalid session conditioning.") }
    self.inputs = inputs
    // Round text once without changing the reference's BF16 input boundary.
    self.inputs["video_text"] = inputs["video_text"]!.map(DenoiserMath.bfloat16)
    self.inputs["audio_text"] = inputs["audio_text"]!.map(DenoiserMath.bfloat16)
    try runner.prepareSession(sigmas: sigmas, fixedWeights: fixedWeights, progress: progress)
  }

  func evaluate(video: [Float], audio: [Float], sigma: Float,
    fixedWeights: (String, [Int]) throws -> [Float],
    blockWeights: @escaping (Int, String, [Int]) throws -> [Float],
    progress: (DenoiserRunner.Progress) throws -> Void = { _ in }) throws -> DenoiserRunner.Output {
    guard !active, !isReleased else { throw BlockError.invalid("Session is active or released.") }
    active = true
    defer { active = false }
    do {
      var current = inputs
      current["video_latent"] = video; current["audio_latent"] = audio
      return try runner.evaluate(current, sigma: sigma, fixedWeights: fixedWeights,
        blockWeights: blockWeights, progress: progress)
    } catch {
      try? runner.releaseSession(); inputs.removeAll(); isReleased = true
      throw error
    }
  }

  func release() throws {
    guard !active else { throw BlockError.invalid("Cannot release an active session.") }
    try runner.releaseSession(); inputs.removeAll(); isReleased = true
  }
}
