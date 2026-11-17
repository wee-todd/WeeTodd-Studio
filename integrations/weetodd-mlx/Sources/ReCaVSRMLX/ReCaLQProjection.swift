import Foundation
import MLX
import MLXNN

struct ReCaCausalState { var history: MLXArray?; var seenFrames = 0 }
struct ReCaLQState { var first = ReCaCausalState(); var second = ReCaCausalState() }

struct ReCaCausalConv {
  let weight: MLXArray, bias: MLXArray
  let temporalStride: Int, spatialStride: Int
  var spatialPadding = 1
  func step(_ input: MLXArray, state: inout ReCaCausalState) throws -> MLXArray {
    guard input.ndim == 4, input.dim(0) > 0, weight.ndim == 5, weight.dim(1) == 4,
      input.dim(3) == weight.dim(4), state.history.map({ Array($0.shape.dropFirst()) == Array(input.shape.dropFirst()) }) ?? true else {
      throw ReCaError.invalid("Causal LQ convolution geometry changed.")
    }
    let seen = state.seenFrames, next = (seen + temporalStride - 1) / temporalStride * temporalStride
    let last = (seen + input.dim(0) - 1) / temporalStride * temporalStride
    guard next <= last else { throw ReCaError.invalid("LQ chunk does not reach the next strided output.") }
    let combined = state.history.map { concatenated([$0, input], axis: 0) } ?? input
    let combinedStart = seen - (state.history?.dim(0) ?? 0), needed = next - 3
    var x = combined[(max(needed, 0) - combinedStart)..<(last - combinedStart + 1)]
    if needed < 0 { x = concatenated([repeated(x[0..<1], count: -needed, axis: 0), x], axis: 0) }
    let result = conv3d(x.expandedDimensions(axis: 0), weight,
      stride: [temporalStride, spatialStride, spatialStride], padding: [0, spatialPadding, spatialPadding])
      .squeezed(axis: 0) + bias
    eval(result)
    state.history = ReCaMath.own(combined, range: max(0, combined.dim(0) - 3)..<combined.dim(0))
    state.seenFrames += input.dim(0)
    return result
  }
}

struct ReCaLQProjection {
  let first: ReCaCausalConv, second: ReCaCausalConv
  let norm1: MLXArray, norm2: MLXArray
  let output: ReCaLinear
  init(checkpoint: ReCaCheckpoint) throws {
    first = try checkpoint.causalConv("lq_proj.conv1", temporalStride: 2, spatialStride: 1)
    second = try checkpoint.causalConv("lq_proj.conv2", temporalStride: 2, spatialStride: 2)
    norm1 = try checkpoint.tensor("lq_proj.norm1.weight").reshaped([-1])
    norm2 = try checkpoint.tensor("lq_proj.norm2.weight").reshaped([-1])
    output = try checkpoint.linear("lq_proj.out_proj")
  }
  static func pixelUnshuffle(_ x: MLXArray, factor: Int = 16) throws -> MLXArray {
    guard x.ndim == 4, factor > 0, x.dim(1) % factor == 0, x.dim(2) % factor == 0 else {
      throw ReCaError.invalid("LQ RGB canvas must divide by the pixel-unshuffle factor.")
    }
    return x.reshaped([x.dim(0), x.dim(1) / factor, factor, x.dim(2) / factor, factor, x.dim(3)])
      .transposed(0, 1, 3, 5, 2, 4).reshaped([x.dim(0), x.dim(1) / factor, x.dim(2) / factor, x.dim(3) * factor * factor])
  }
  func step(_ rgb: MLXArray, state: inout ReCaLQState) throws -> MLXArray {
    var x = try Self.pixelUnshuffle(rgb.asType(.bfloat16))
    x = silu(ReCaMath.rms(try first.step(x, state: &state.first), weight: norm1, epsilon: 1e-6))
    x = silu(ReCaMath.rms(try second.step(x, state: &state.second), weight: norm2, epsilon: 1e-6))
    return output(x.reshaped([-1, 3072]))
  }
}
