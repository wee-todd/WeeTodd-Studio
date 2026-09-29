import Foundation
import LTX25Engine

/// Uniform-timestep audiovisual sampling. Encoders, latent packing and decoding
/// are separate stages; this runner never owns their weights.
public final class LTXSamplingRunner {
  private let configuration: AVBlockConfiguration
  private let blockCount: Int
  private let maximumPreparationBytes: UInt64
  private let sequenceAttention: Bool
  private let precision: LTXPrecisionPolicy
  private var active = false
  public private(set) var lastGraphBuildCount = 0
  private let denoiser: DenoiserRunner
  private let trajectory: EulerTrajectory
  public init(configuration: AVBlockConfiguration, blockCount: Int = 48, maximumPreparationBytes: UInt64 = 0, sequenceAttention: Bool = false, experimentalPrecision: LTXPrecisionPolicy = .float32) throws {
    self.configuration = configuration; self.blockCount = blockCount
    self.maximumPreparationBytes = maximumPreparationBytes; self.sequenceAttention = sequenceAttention; self.precision = experimentalPrecision
    denoiser = try DenoiserRunner(configuration: configuration, blockCount: blockCount,
      maximumPreparationBytes: maximumPreparationBytes, sequenceAttention: sequenceAttention, experimentalPrecision: experimentalPrecision)
    trajectory = try EulerTrajectory()
  }
  public func evaluate(
    _ inputs: [String: [Float]], schedule: SamplingSchedule, reuseSession: Bool = true,
    fixedWeights: (String, [Int]) throws -> [Float],
    blockWeights: @escaping (Int, String, [Int]) throws -> [Float],
    noise: EulerTrajectory.Noise? = nil,
    stageProgress: (Int, DenoiserRunner.Progress) throws -> Void = { _, _ in },
    progress: (EulerTrajectory.Progress) throws -> Void = { _ in },
    preview: ((AVLatents, EulerTrajectory.Progress) throws -> Void)? = nil
  ) throws -> AVLatents {
    guard !active else { throw BlockError.invalid("Sampler is already executing.") }
    active = true; defer { active = false }
    lastGraphBuildCount = 0
    guard noise != nil || !schedule.steps.contains(where: \.ancestral) else {
      throw BlockError.invalid("Ancestral sampling requires explicit noise before loading weights.")
    }
    let shapes = denoiser.inputShapes
    guard Set(inputs.keys) == Set(shapes.keys),
      shapes.allSatisfy({ name, shape in
        inputs[name]!.count == shape.reduce(1, *) && inputs[name]!.allSatisfy(\.isFinite)
      })
    else { throw BlockError.invalid("Invalid sampling inputs.") }
    let session = reuseSession ? try LTXDenoisingSession(configuration: configuration,
      inputs: inputs, sigmas: schedule.sigmas.dropLast().map { DenoiserMath.bfloat16(Float($0)) },
      blockCount: blockCount, maximumPreparationBytes: maximumPreparationBytes, sequenceAttention: sequenceAttention, experimentalPrecision: precision, fixedWeights: fixedWeights) { name in
        try stageProgress(0, DenoiserRunner.Progress(stage: "schedule_" + name, completedBlocks: 0,
          metalAllocatedBytes: denoiser.metalAllocatedBytes, stackMetrics: nil))
      } : nil
    defer {
      lastGraphBuildCount = session?.graphBuildCount ?? schedule.steps.count
      try? session?.release()
    }
    var index = 0
    return try trajectory.evaluate(
      video: inputs["video_latent"]!, audio: inputs["audio_latent"]!,
      schedule: schedule, noise: noise,
      predict: { state, sigma in
        var current = inputs
        current["video_latent"] = state.video
        current["audio_latent"] = state.audio
        let result: DenoiserRunner.Output
        if let session {
          result = try session.evaluate(video: state.video, audio: state.audio, sigma: sigma,
            fixedWeights: fixedWeights, blockWeights: blockWeights) { try stageProgress(index + 1, $0) }
        } else {
          result = try self.denoiser.evaluate(current, sigma: sigma,
            fixedWeights: fixedWeights, blockWeights: blockWeights) { try stageProgress(index + 1, $0) }
        }
        return AVLatents(video: result.videoVelocity, audio: result.audioVelocity)
      },
      progress: { event in
        index = event.completedSteps
        try progress(event)
      }, preview: preview)
  }
}
