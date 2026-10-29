import Drama3Contracts
import Foundation
import MLX
import MLXFast
import MLXNN

public enum Drama3Math {
  public static func norm(_ x: MLXArray, weight: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: epsilon).asType(
      x.dtype)
  }
}
public enum Drama3Sampler {
  public static func sample(
    logits: MLXArray, temperature: Float, topK: Int, topP: Float, checked: Bool = true
  ) throws -> MLXArray {
    if checked && !MLX.all(MLX.isFinite(logits)).item(Bool.self) {
      throw Drama3Error.invalid("Sampling produced non-finite logits.")
    }
    if temperature == 0 { return argMax(logits, axis: -1).asType(.int32) }
    let x = logits.asType(.float32) / max(temperature, 1e-5)
    let shifted = x - x.max(axis: -1, keepDims: true)
    let k = min(topK, x.dim(-1))
    let ids = argPartition(-shifted, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    let values = takeAlong(shifted, ids, axis: -1)
    let order = argSort(-values, axis: -1)
    let sorted = takeAlong(values, order, axis: -1)
    let sortedIDs = takeAlong(ids, order, axis: -1)
    let probs = exp(sorted - logSumExp(shifted, axis: -1, keepDims: true))
    let mask = (cumsum(probs, axis: -1) .> topP) .&& (MLXArray(0..<k) .> 0)
    let chosen = MLXRandom.categorical(MLX.where(mask, -Float.infinity, sorted))
    return takeAlong(sortedIDs, expandedDimensions(chosen, axis: -1), axis: -1).squeezed(axis: -1)
      .asType(.int32)
  }
}
