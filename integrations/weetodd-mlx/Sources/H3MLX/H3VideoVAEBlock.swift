import Foundation
import MLX
import MLXNN
import TensorIO

/// One quantized ViT block of the installed H3 video decoder. A block's four
/// affine-Q8 projections are loaded one at a time from the shared checkpoint.
public enum H3VideoVAEBlock {
  private static let inverseFrequencyBits: [UInt32] = [
    0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887,
    0x3dcccccc, 0x3d6655c2, 0x3d0186e3, 0x3c91ad39,
  ]

  public static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray, positions: MLXArray) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index,
      input: input, positions: positions, observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray, positions: MLXArray,
    session: H3VideoVAEDecodeSession? = nil,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<36).contains(index), input.ndim == 3,
      (1...4).contains(input.shape[0]),
      (1...16_384).contains(input.shape[1]),
      input.shape[2] == 2048,
      (input.dtype == .float16 || input.dtype == .float32),
      positions.shape == [input.shape[0], input.shape[1], 3],
      positions.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 video VAE block input.")
    }
    let file: SafeTensorFile?
    if let session {
      guard session.checkpointURL == checkpointURL else {
        throw H3CheckpointError.invalid("H3 video decoder session checkpoint differs.")
      }
      try session.checkUnchanged()
      file = nil
    } else {
      _ = try H3VideoVAELayout(url: checkpointURL)
      file = try SafeTensorFile(url: checkpointURL)
    }
    let prefix = "decoder.transformer_blocks.\(index)."
    let previousCacheLimit = Memory.cacheLimit
    if session == nil { Memory.cacheLimit = 128 * 1024 * 1024 }
    defer {
      if session == nil {
        Stream.gpu.synchronize()
        Memory.clearCache()
        Memory.cacheLimit = previousCacheLimit
      }
    }
    func read(_ suffix: String, shape: [Int]) throws -> MLXArray {
      let name = prefix + suffix
      if let session { return try session.read(name, shape: shape) }
      guard let file else { throw H3CheckpointError.invalid("Missing H3 video decoder reader.") }
      guard let descriptor = file.tensors[name], descriptor.dtype == "F16",
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 video decoder tensor: \(suffix)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: Float16.self)
      }
      eval(value)
      return value
    }
    func project(_ value: MLXArray, _ suffix: String,
      rows: Int) throws -> MLXArray {
      let projection: H3QwenQ8Projection
      if let session {
        projection = try session.projection(prefix + suffix + ".weight")
      } else if let file {
        projection = try H3QwenQ8Projection(file: file,
          name: prefix + suffix + ".weight")
      } else { throw H3CheckpointError.invalid("Missing H3 video decoder reader.") }
      guard projection.rows == rows else {
        throw H3CheckpointError.invalid("H3 video decoder projection rows changed.")
      }
      let bias = try read(suffix + ".bias", shape: [rows])
      let result = try projection.project(value) + bias
      if let session { session.materializeProjection(result) }
      else { eval(result) }
      if session == nil { Memory.clearCache() }
      try Task.checkCancellation()
      return result
    }
    let batch = input.shape[0]
    let count = input.shape[1]
    let inverse = MLXArray(inverseFrequencyBits.map(Float.init(bitPattern:)))
    let angles = Float(2 * Double.pi) * positions.expandedDimensions(axis: 3)
      * inverse.reshaped([1, 1, 1, 8])
    let frequency = angles.reshaped([batch, count, 24])
    let doubled = concatenated([frequency, frequency], axis: -1)
    let cosine = MLX.cos(doubled).asType(input.dtype).reshaped([batch, count, 1, 48])
    let sine = MLX.sin(doubled).asType(input.dtype).reshaped([batch, count, 1, 48])
    func rotate(_ value: MLXArray) -> MLXArray {
      let first = value[.ellipsis, 0..<24]
      let second = value[.ellipsis, 24..<48]
      let rotated = concatenated([-second, first], axis: -1)
      let leading = value[.ellipsis, 0..<48] * cosine + rotated * sine
      return concatenated([leading, value[.ellipsis, 48..<64]], axis: -1)
    }
    func unweightedRMS(_ value: MLXArray) -> MLXArray {
      let x = value.asType(.float32)
      return (x * rsqrt(mean(x * x, axis: -1, keepDims: true) + 1e-5))
        .asType(value.dtype)
    }
    let firstNorm = try read("norm1.weight", shape: [2048])
    let normalized = MLXFast.rmsNorm(input.asType(.float32),
      weight: firstNorm.asType(.float32), eps: 1e-5).asType(input.dtype)
    try observe("norm1", normalized)
    let qkv = try project(normalized, "attn.to_qkv", rows: 6144)
      .reshaped([batch, count, 32, 3, 64])
    let query = rotate(unweightedRMS(qkv[.ellipsis, 0, 0..<64]))
      .transposed(0, 2, 1, 3)
    let key = rotate(unweightedRMS(qkv[.ellipsis, 1, 0..<64]))
      .transposed(0, 2, 1, 3)
    let value = qkv[.ellipsis, 2, 0..<64].transposed(0, 2, 1, 3)
    let attended = MLXFast.scaledDotProductAttention(queries: query,
      keys: key, values: value, scale: 1 / Float(64).squareRoot(),
      mask: nil).transposed(0, 2, 1, 3).reshaped([batch, count, 2048])
    let attention = try project(attended, "attn.to_out", rows: 2048)
    try observe("attention", attention)
    let firstScale = try read("scale1", shape: [2048])
    let residual = input + attention * firstScale
    if let session { session.materializeFirstResidual(residual) }
    else { eval(residual) }
    try observe("residual", residual)
    let secondNorm = try read("norm2.weight", shape: [2048])
    let feedInput = MLXFast.rmsNorm(residual.asType(.float32),
      weight: secondNorm.asType(.float32), eps: 1e-5).asType(input.dtype)
    try observe("norm2", feedInput)
    let fused = try project(feedInput, "ff.w1", rows: 16384)
    let gate = fused[.ellipsis, 0..<8192]
    let feed = try project(silu(gate) * fused[.ellipsis, 8192..<16384],
      "ff.w2", rows: 2048)
    try observe("feed", feed)
    let secondScale = try read("scale2", shape: [2048])
    let output = residual + feed * secondScale
    eval(output)
    try observe("output", output)
    if let session { try session.checkUnchanged() }
    else { try file?.checkUnchanged(at: checkpointURL) }
    try Task.checkCancellation()
    return output
  }
}
