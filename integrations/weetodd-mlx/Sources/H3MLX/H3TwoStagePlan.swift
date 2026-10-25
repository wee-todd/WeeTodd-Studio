import Foundation

/// Shape and compute admission for automatic generation. The final canvas is exact.
public struct H3TwoStagePlan: Sendable {
  public let base: H3Geometry
  public let target: H3Geometry
  public let draftScale: Double
  public let strength: Double
  public let evaluations: Int
  public let refinement: H3JointRefinement

  public init(width: Int, height: Int, durationSeconds: Double,
    draftScale: Double = 2, strength: Double = 0.35, evaluations: Int = 3) throws {
    guard draftScale.isFinite, draftScale > 1, draftScale <= 2,
      strength.isFinite, strength > 0, strength <= 1, (1...64).contains(evaluations),
      width >= 64, height >= 64 else {
      throw H3CheckpointError.invalid("H3 two-stage needs a draft scale above 1 through 2, strength above 0 through 1, 1–64 refinement evaluations and final axes of at least 64.")
    }
    let target = try H3Geometry(width: width, height: height,
      durationSeconds: durationSeconds, canvasAdmission: .spatialRefinement)
    try target.canvasAdmission.validatePackedRows(target.videoRows + target.audioRows + 1)
    let w = Int(ceil(Double(width) / draftScale / 32)) * 32
    let h = Int(ceil(Double(height) / draftScale / 32)) * 32
    guard w < width, h < height else {
      throw H3CheckpointError.invalid("The selected H3 draft scale cannot produce a smaller 32-pixel canvas. Increase draft scale toward 2.")
    }
    let base = try H3Geometry(width: w, height: h, durationSeconds: durationSeconds)
    try base.canvasAdmission.validate(width: w, height: h)
    try base.canvasAdmission.validatePackedRows(base.videoRows + base.audioRows + 1)
    self.base = base; self.target = target; self.draftScale = draftScale
    self.strength = strength; self.evaluations = evaluations
    // Strength is a fraction of the common unshifted clock. The existing helper
    // builds both modality schedules from this start, with exactly N intervals.
    self.refinement = try H3JointRefinement(strength: strength,
      startVideoSigma: 12 * strength / (1 + 11 * strength),
      evaluations: evaluations, preserveAudio: true)
  }
}
