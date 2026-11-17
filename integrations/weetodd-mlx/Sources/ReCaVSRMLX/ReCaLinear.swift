import Foundation
import MLX

struct ReCaLinear {
  let master: MLXArray
  let weight: MLXArray, bias: MLXArray?
  let scale: MLXArray?
  let computeDType: DType
  init(weight: MLXArray, bias: MLXArray?, scale: MLXArray? = nil, computeDType: DType = .bfloat16) {
    self.computeDType = computeDType
    self.bias = bias?.asType(computeDType)
    if let scale {
      self.weight = (weight.asType(.int16) + 128).asType(.uint8).view(dtype: .uint32)
      self.master = self.weight
      self.scale = broadcast(scale.asType(.float32), to: [weight.dim(0), weight.dim(1) / 64])
    } else { self.master = weight; self.weight = weight.asType(computeDType); self.scale = nil }
    eval(self.weight); if let bias = self.bias { eval(bias) }; if let scale = self.scale { eval(scale) }
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let result: MLXArray
    if let scale {
      let rotated = matmul(x.asType(computeDType).reshaped([-1, 256]), Self.rotation(dtype: computeDType)).reshaped(x.shape)
      // FP32 row scales are retained. This affine MLX arithmetic is separately
      // qualified from the CUDA dynamic-activation INT8 backend.
      result = quantizedMM(rotated.asType(.float32), weight, scales: scale, biases: -128 * scale,
        groupSize: 64, bits: 8).asType(computeDType)
    } else { result = matmul(x.asType(computeDType), weight.T) }
    return bias.map { result + $0 } ?? result
  }
  static func rotation(dtype: DType) -> MLXArray {
    let values: [Float] = (0..<(256 * 256)).map { i in
      let difference = i / 256 ^ (i % 256)
      let negative = (0..<4).reduce(false) { $0 != ((difference >> ($1 * 2)) & 3 == 3) }
      return negative ? -0.0625 : 0.0625
    }
    return MLXArray(values, [256, 256]).asType(dtype)
  }
}
