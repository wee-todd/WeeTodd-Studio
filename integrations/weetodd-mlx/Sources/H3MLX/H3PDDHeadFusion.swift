import MLX

/// Reduce one trained four-head parameter block before projection. No output
/// tensor containing all 32 heads over the packed sequence is constructed.
public enum H3PDDHeadFusion {
  /// Widened rows are head-major: [32 * outputFeatures, inputFeatures].
  /// The optional bias uses [32 * outputFeatures]. Each parameter independently
  /// returns to its source dtype after an FP32 weighted accumulation.
  public static func fuse(weight: MLXArray, bias: MLXArray?, plan: H3PDDPlan,
    step: Int, modality: H3PDDPlan.Modality) throws -> (weight: MLXArray, bias: MLXArray?) {
    let gridSize = plan.fineGrid.count - 1
    guard weight.ndim == 2, weight.dtype.isFloatingPoint,
      weight.shape[0] > 0, weight.shape[1] > 0, weight.shape[0] % gridSize == 0 else {
      throw H3GeometryError.invalid("FastH3 Omni PDD weight must contain head-major floating projection rows.")
    }
    if let bias {
      guard bias.ndim == 1, bias.shape[0] == weight.shape[0], bias.dtype.isFloatingPoint else {
        throw H3GeometryError.invalid("FastH3 Omni PDD bias must match the widened projection rows.")
      }
    }
    let block = try plan.block(step: step, modality: modality)
    let outputs = weight.shape[0] / gridSize
    let rows = (block.headRange.lowerBound * outputs)..<(block.headRange.upperBound * outputs)
    let normalized = MLXArray(block.normalizedWeights.map(Float.init))
    // Slice before widening precision: only four heads need FP32 workspace.
    let selected = weight[rows].asType(.float32).reshaped([block.headRange.count, outputs, weight.shape[1]])
    let fusedWeight = (selected * normalized.reshaped([block.headRange.count,1,1]))
      .sum(axis: 0).asType(weight.dtype)
    let fusedBias = bias.map { bias in
      (bias[rows].asType(.float32).reshaped([block.headRange.count,outputs])
        * normalized.reshaped([block.headRange.count,1])).sum(axis: 0).asType(bias.dtype)
    }
    return (weight: fusedWeight, bias: fusedBias)
  }
}
