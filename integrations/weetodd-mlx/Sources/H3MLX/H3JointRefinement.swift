import Foundation
import MLX

/// Explicit initialized AV sampling. Nil on existing requests preserves every
/// ordinary schedule, seeded draw, and sampler update.
public struct H3JointRefinement: Sendable {
  public let strength: Double
  public let startVideoSigma: Double?
  public let evaluations: Int?
  public let preserveAudio: Bool

  public init(strength: Double, startVideoSigma: Double? = nil,
    evaluations: Int? = nil, preserveAudio: Bool = true) throws {
    guard strength.isFinite, strength > 0, strength <= 1,
      startVideoSigma == nil || (startVideoSigma!.isFinite &&
        startVideoSigma! > 0 && startVideoSigma! <= 1),
      evaluations == nil || (startVideoSigma != nil && (1...64).contains(evaluations!)) else {
      throw H3CheckpointError.invalid("Invalid H3 joint refinement strength or explicit noise schedule.")
    }
    self.strength = strength; self.startVideoSigma = startVideoSigma
    self.evaluations = evaluations; self.preserveAudio = preserveAudio
  }

  public func schedules(requestedSteps: Int) throws -> (video: H3Schedule, audio: H3Schedule) {
    let video = try H3Schedule(requestedSteps: requestedSteps, shift: 12)
    let audio = try H3Schedule(requestedSteps: requestedSteps, shift: 3)
    let active = evaluations ?? max(1, Int(ceil(Double(video.timesteps.count) * strength)))
    guard active <= video.timesteps.count || startVideoSigma != nil else {
      throw H3CheckpointError.invalid("Invalid H3 refinement schedule suffix.")
    }
    if let startVideoSigma {
      // The requested video sigma is a noise fraction. Invert its modality
      // shift and construct both schedules from that common base clock.
      let baseStart = startVideoSigma / (12 - 11 * startVideoSigma)
      var v: [Float] = [], a: [Float] = []
      for index in 0...active {
        // np.linspace's float64 construction, then float32 cast before shift.
        let base = Float(index == active ? 0 : baseStart +
          Double(index) * (-baseStart / Double(active)))
        v.append((Float(12) * base) / (Float(1) + Float(11) * base))
        a.append((Float(3) * base) / (Float(1) + Float(2) * base))
      }
      return (try H3Schedule(sigmas: v), try H3Schedule(sigmas: a))
    }
    if strength == 1 { return (video, audio) }
    return (try H3Schedule(sigmas: Array(video.sigmas.suffix(active + 1))),
      try H3Schedule(sigmas: Array(audio.sigmas.suffix(active + 1))))
  }

  /// Inputs are complete normalized sampler rows, never decoder pixels or
  /// continuation-tail rows. Keep existing condition/target PRNG draws outside
  /// this helper, then apply it only to the target slices.
  public static func initialize(source: MLXArray, noise: MLXArray,
    schedule: H3Schedule, canvasAdmission: H3CanvasAdmission = .ordinary) throws -> MLXArray {
    guard source.shape == noise.shape, source.ndim == 3,
      source.shape[0] == 1, [32, 96].contains(source.shape[2]),
      source.dtype == .float32, noise.dtype == .float32,
      source.shape[1] > 0, source.shape[1] <= canvasAdmission.maximumPackedRows,
      let t = schedule.timesteps.first else {
      throw H3CheckpointError.invalid("H3 refinement requires matching full float32 AV rows.")
    }
    try Task.checkCancellation()
    guard all(isFinite(source)).item(Bool.self), all(isFinite(noise)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Non-finite H3 refinement source or target noise.")
    }
    // Preserve scheduler.scale_noise's float32 complement and multiply order.
    return t * source + (Float(1) - t) * noise
  }
}

extension H3JointRefinement {
  static func targetRows(_ rows: MLXArray, source: [Float],
    prefix: Int, schedule: H3Schedule, canvasAdmission: H3CanvasAdmission = .ordinary) throws -> MLXArray {
    guard rows.ndim == 3, rows.shape[0] == 1,
      prefix >= 0, prefix < rows.shape[1],
      source.count == (rows.shape[1] - prefix) * rows.shape[2] else {
      throw H3CheckpointError.invalid("H3 initialized target rows do not match the condition-prefix layout.")
    }
    let target = try initialize(source: MLXArray(source,
      [1, rows.shape[1] - prefix, rows.shape[2]]),
      noise: rows[0..<1, prefix..<rows.shape[1], 0..<rows.shape[2]], schedule: schedule, canvasAdmission: canvasAdmission)
    return prefix == 0 ? target : concatenated([
      rows[0..<1, 0..<prefix, 0..<rows.shape[2]], target], axis: 1)
  }
}
