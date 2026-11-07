import Foundation
import MLX
import TensorIO

struct RelayWeights {
  let header: SafeTensorFile
  let arrays: [String: MLXArray]

  init(url: URL, format: String) throws {
    let header = try SafeTensorFile(url: url)
    guard header.metadata["format"] == format else {
      throw RelayVSRError.invalid("Expected \(format) checkpoint at \(url.path).")
    }
    let arrays = try loadArrays(url: url)
    try header.checkUnchanged(at: url)
    self.header = header
    self.arrays = arrays
  }

  func tensor(_ name: String, _ shape: [Int], dtype: DType? = .bfloat16) throws -> MLXArray {
    guard let value = arrays[name], value.shape == shape,
      [.float32, .bfloat16, .float16].contains(value.dtype) else {
      throw RelayVSRError.invalid("Missing or incompatible RelayVSR tensor \(name), expected \(shape).")
    }
    return dtype.map { value.asType($0) } ?? value
  }

  func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = true,
    rotation: MLXArray? = nil) throws -> RelayLinear
  {
    let bias = bias ? try tensor(name + ".bias", [output]) : nil
    let key = name + ".weight"
    guard let raw = arrays[key], raw.shape == [output, input] else {
      throw RelayVSRError.invalid("Missing or incompatible RelayVSR matrix \(key).")
    }
    if raw.dtype == .int8 {
      let config = try header.withTensorBytes(named: name + ".comfy_quant") {
        try JSONDecoder().decode(RelayQuantFormat.self, from: Data($0))
      }
      guard let rotation, config.format == "int8_tensorwise", config.convrot,
        config.convrot_groupsize == 256, input.isMultiple(of: 256) else {
        throw RelayVSRError.invalid("Unsupported RelayVSR INT8 layout for \(key).")
      }
      let scale = try tensor(name + ".weight_scale", [output, 1], dtype: nil)
      return RelayLinear(weight: raw, bias: bias, scale: scale, rotation: rotation)
    }
    return RelayLinear(weight: try tensor(key, [output, input]), bias: bias)
  }

  func convolution(_ name: String, _ input: Int, _ output: Int, kernel: Int,
    stride: Int = 1, padding: Int = 0, temporalAxis: Bool = false) throws -> RelayConv
  {
    let shape = temporalAxis ? [output, input, 1, kernel, kernel] : [output, input, kernel, kernel]
    let weight = try tensor(name + ".weight", shape)
      .reshaped([output, input, kernel, kernel]).transposed(0, 2, 3, 1)
    return RelayConv(weight: weight, bias: try tensor(name + ".bias", [output]),
      stride: stride, padding: padding)
  }
}

private struct RelayQuantFormat: Decodable {
  let format: String
  let convrot: Bool
  let convrot_groupsize: Int
}

struct RelayLinear {
  let weight: MLXArray
  let bias: MLXArray?
  private let quantization: (scale: MLXArray, offset: MLXArray, rotation: MLXArray)?

  init(weight: MLXArray, bias: MLXArray?, scale: MLXArray? = nil, rotation: MLXArray? = nil) {
    self.bias = bias
    if let scale, let rotation {
      let columns = weight.dim(1)
      let packed = (weight.asType(.int16) + 128).asType(.uint8).view(dtype: .uint32)
      self.weight = packed
      let scales = broadcast(scale.asType(.bfloat16), to: [weight.dim(0), columns / 64])
      quantization = (scales, -128 * scales, rotation)
    } else {
      self.weight = weight
      quantization = nil
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let y: MLXArray
    if let q = quantization {
      let rotated = matmul(x.reshaped([-1, 256]), q.rotation).reshaped(x.shape)
      y = quantizedMM(rotated, weight, scales: q.scale, biases: q.offset,
        groupSize: 64, bits: 8)
    } else {
      y = matmul(x, weight.T)
    }
    return bias.map { y + $0 } ?? y
  }

  static func rotation() -> MLXArray {
    // The checkpoint uses the fourth Kronecker power of the regular 4×4 Hadamard.
    let size = 256
    let matrix: [Float] = (0..<(size * size)).map { index in
      let difference = index / size ^ (index % size)
      let negative = (0..<4).reduce(false) { odd, digit in
        odd != ((difference >> (digit * 2)) & 3 == 3)
      }
      return negative ? -0.0625 : 0.0625
    }
    return MLXArray(matrix, [size, size]).asType(.bfloat16)
  }
}

struct RelayConv {
  let weight: MLXArray
  let bias: MLXArray
  let stride: Int
  let padding: Int

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv2d(x.expandedDimensions(axis: 0), weight, stride: IntOrPair(stride),
      padding: IntOrPair(padding)).squeezed(axis: 0) + bias
  }
}
