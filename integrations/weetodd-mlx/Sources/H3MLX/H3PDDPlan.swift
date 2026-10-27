import MLX

/// The trained 32-interval grid partitioned into eight model evaluations.
/// Coefficients are constructed in Float64; model times are formed after the
/// coarse sigmas are rounded to Float32, matching the native export's clock.
public struct H3PDDPlan: Sendable {
  public enum Modality: Sendable { case video, audio }
  public struct Block: Sendable {
    public let headRange: Range<Int>
    public let normalizedWeights: [Double]
    /// Signed noise difference. Negative along the descending noise clock.
    public let integrationWeight: Double
  }
  public let evaluationCount: Int
  public let fineGrid: [Double]
  private let boundaries: [Int]
  private let video: Coefficients
  private let audio: Coefficients

  private struct Coefficients: Sendable {
    let fine: [Double]
    let coarse: [Double]
    let weights: [Double]
    let times: [Float]
  }

  public init(contract: H3OmniContract) throws {
    evaluationCount = contract.transformerForwards
    boundaries = contract.pddStepIndices
    let maximum = contract.gridMaxT
    let interval = maximum / Double(contract.pddSteps)
    let grid = (0...contract.pddSteps).map { maximum - Double($0) * interval }
    fineGrid = grid
    func coefficients(shift: Double) throws -> Coefficients {
      let fine = grid.map { u in shift * u * maximum / (maximum + (shift - 1) * u) }
      // Algebraic difference of the shifted endpoints, avoiding cancellation
      // near the maximum-time fixed point. This is deliberately separate from
      // ordinary H3's unit-maximum rational shift.
      let weights = zip(grid, grid.dropFirst()).map { a, b in
        shift * maximum * maximum * (b - a)
          / ((maximum + (shift - 1) * a) * (maximum + (shift - 1) * b))
      }
      guard weights.allSatisfy({ $0.isFinite && $0 < 0 }) else {
        throw H3GeometryError.invalid("FastH3 Omni PDD integration coefficients are invalid.")
      }
      let coarse = contract.pddStepIndices.map { fine[$0] }
      return Coefficients(fine: fine, coarse: coarse, weights: weights,
        times: coarse.dropLast().map { Float(1) - Float($0) })
    }
    video = try coefficients(shift: contract.videoSchedulerShift)
    audio = try coefficients(shift: contract.audioSchedulerShift)
  }

  private func coefficients(for modality: Modality) -> Coefficients {
    modality == .video ? video : audio
  }
  public func fineSigmas(for modality: Modality) -> [Double] { coefficients(for: modality).fine }
  public func coarseSigmas(for modality: Modality) -> [Double] { coefficients(for: modality).coarse }
  public func timesteps(for modality: Modality) -> [Float] { coefficients(for: modality).times }
  public func integrationWeights(for modality: Modality) -> [Double] { coefficients(for: modality).weights }

  public func block(step: Int, modality: Modality) throws -> Block {
    guard (0..<evaluationCount).contains(step) else {
      throw H3GeometryError.invalid("FastH3 Omni PDD evaluation index is out of range.")
    }
    let range = boundaries[step]..<boundaries[step + 1]
    let weights = coefficients(for: modality).weights[range]
    let total = weights.reduce(0,+)
    return Block(headRange: range, normalizedWeights: weights.map { $0 / total }, integrationWeight: total)
  }

  /// Consume the fused data-ward model mean, whose sign is opposite the
  /// descending noise-clock coefficients. Initialize target noise using the
  /// first coarse sigma (0.999); this method updates target rows only.
  public func advance(sample: MLXArray, mean: MLXArray, step: Int, modality: Modality) throws -> MLXArray {
    guard (0..<evaluationCount).contains(step), sample.shape == mean.shape,
      sample.size > 0, sample.dtype.isFloatingPoint, mean.dtype.isFloatingPoint else {
      throw H3GeometryError.invalid("FastH3 Omni PDD step or latent/mean shape is invalid.")
    }
    let schedule = coefficients(for: modality)
    let sigma = Double(Float(schedule.coarse[step]))
    let nextSigma = Double(Float(schedule.coarse[step + 1]))
    let ratio = nextSigma / sigma
    // The model emits a data-ward direction. Retain the projection/sample
    // arithmetic dtype for x0, then blend in FP32 before returning to the
    // sample's dtype. Reconstruct sigma from the exact model-time conversion.
    let denoised = sample + (Double(1) - Double(schedule.times[step])) * mean
    return (ratio * sample.asType(.float32)
      + (Double(1) - ratio) * denoised.asType(.float32)).asType(sample.dtype)
  }
}
