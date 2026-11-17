import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum MOSSQwenMath {
  static func norm(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    let f = x.asType(.float32)
    return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + 1e-6) * (1 + weight.asType(.float32))).asType(x.dtype)
  }
  static func delta(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, useKernel: Bool = true) -> MLXArray {
    func normalized(_ x: MLXArray) -> MLXArray {
      let f = x.asType(.float32)
      return f * rsqrt(sum(f * f, axis: -1, keepDims: true) + 1e-6)
    }
    let queries = normalized(q) * Float(1 / sqrt(Double(q.dim(-1))))
    let (out, _) = gatedDeltaUpdate(q: queries, k: normalized(k), v: v.asType(.float32),
      a: a.asType(.float32), b: b.asType(.float32), aLog: aLog.asType(.float32), dtBias: dtBias.asType(.float32), useKernel: useKernel)
    return out
  }
  static func gatedNorm(_ x: MLXArray, weight: MLXArray, gate: MLXArray) -> MLXArray {
    let f = x.asType(.float32)
    let normalized = (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + 1e-6)).asType(x.dtype)
    return ((normalized * weight.asType(x.dtype)).asType(.float32) * silu(gate.asType(.float32))).asType(x.dtype)
  }
}
