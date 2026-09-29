import Foundation
import MLX
import MLXNN
import TensorIO

/// One independently admitted Qwen3-VL vision block. Video frames and stills
/// attend within each temporal patch group, never across the entire batch.
public enum H3QwenVisionBlock {
  private static func rotate(_ x: MLXArray, rotary: MLXArray) -> MLXArray {
    let cosine = concatenated([MLX.cos(rotary), MLX.cos(rotary)], axis: -1)
      .expandedDimensions(axis: 1).expandedDimensions(axis: 0)
    let sine = concatenated([MLX.sin(rotary), MLX.sin(rotary)], axis: -1)
      .expandedDimensions(axis: 1).expandedDimensions(axis: 0)
    let float = x.asType(.float32)
    let left = float[.ellipsis, 0..<36]
    let right = float[.ellipsis, 36..<72]
    let turned = concatenated([-right, left], axis: -1)
    return (float * cosine + turned * sine).asType(x.dtype)
  }

  public static func evaluate(checkpointURL: URL, index: Int, input: MLXArray,
    rotary: MLXArray, boundaries: [Int]) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index, input: input,
      rotary: rotary, boundaries: boundaries, observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, index: Int, input: MLXArray,
    rotary: MLXArray, boundaries: [Int],
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<27).contains(index), input.ndim == 2, input.shape[1] == 1152,
      input.dtype == .bfloat16, (1...16_384).contains(input.shape[0]),
      rotary.shape == [input.shape[0], 36], rotary.dtype == .float32,
      boundaries.count >= 2, boundaries.first == 0,
      boundaries.last == input.shape[0],
      zip(boundaries, boundaries.dropFirst()).allSatisfy({ $0 < $1 }) else {
      throw H3CheckpointError.invalid("Invalid H3 Qwen vision block input or frame boundaries.")
    }
    try Task.checkCancellation()
    let file = try SafeTensorFile(url: checkpointURL)
    let root = "visual.blocks.\(index)."
    func read(_ name: String, shape: [UInt64]) throws -> MLXArray {
      let full = root + name
      guard file.tensors[full].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
          == H3TensorInfo(dtype: "BF16", shape: shape) else {
        throw H3CheckpointError.invalid("Incomplete H3 Qwen vision block tensor: \(full)")
      }
      return try file.withTensorBytes(named: full) { bytes in
        MLXArray(bytes, shape.map(Int.init), type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let norm1Weight = try read("norm1.weight", shape: [1152])
    let norm1Bias = try read("norm1.bias", shape: [1152])
    let norm2Weight = try read("norm2.weight", shape: [1152])
    let norm2Bias = try read("norm2.bias", shape: [1152])
    let qkv = try H3QwenQ8Projection(file: file, name: root + "attn.qkv.weight")
    let qkvBias = try read("attn.qkv.bias", shape: [3456])
    let output = try H3QwenQ8Projection(file: file, name: root + "attn.proj.weight")
    let outputBias = try read("attn.proj.bias", shape: [1152])
    let fc1 = try H3QwenQ8Projection(file: file, name: root + "mlp.linear_fc1.weight")
    let fc1Bias = try read("mlp.linear_fc1.bias", shape: [4304])
    let fc2Weight = try read("mlp.linear_fc2.weight", shape: [1152, 4304])
    let fc2Bias = try read("mlp.linear_fc2.bias", shape: [1152])

    let n = input.shape[0]
    let attentionInput = MLXFast.layerNorm(input, weight: norm1Weight,
      bias: norm1Bias, eps: 1e-6)
    try observe("norm1", attentionInput)
    let qkvRows = try (qkv.project(attentionInput) + qkvBias)
      .reshaped([n, 3, 16, 72]).transposed(1, 0, 2, 3)
    try observe("qkv", qkvRows)
    let query = rotate(qkvRows[0].expandedDimensions(axis: 0), rotary: rotary)
      .transposed(0, 2, 1, 3)
    try observe("qrope", query.transposed(0, 2, 1, 3))
    let key = rotate(qkvRows[1].expandedDimensions(axis: 0), rotary: rotary)
      .transposed(0, 2, 1, 3)
    try observe("krope", key.transposed(0, 2, 1, 3))
    let value = qkvRows[2].expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
    let widths: [IntOrPair] = [.init(0), .init(0), .init(0), .init((0, 8))]
    let paddedQuery = padded(query, widths: widths)
    let paddedKey = padded(key, widths: widths)
    let paddedValue = padded(value, widths: widths)
    var pieces: [MLXArray] = []
    for (lower, upper) in zip(boundaries, boundaries.dropFirst()) {
      try Task.checkCancellation()
      let attended = MLXFast.scaledDotProductAttention(
        queries: paddedQuery[0..<1, 0..<16, lower..<upper, 0..<80],
        keys: paddedKey[0..<1, 0..<16, lower..<upper, 0..<80],
        values: paddedValue[0..<1, 0..<16, lower..<upper, 0..<80],
        scale: Float(pow(72.0, -0.5)), mask: nil)
      pieces.append(attended[0..<1, 0..<16, 0..<(upper - lower), 0..<72])
    }
    let attended = concatenated(pieces, axis: 2)
      .transposed(0, 2, 1, 3).reshaped([n, 1152])
    try observe("attended", concatenated(pieces, axis: 2)[0..<1, 0..<16, 0..<n, 0..<72])
    let attentionProjection = try output.project(attended) + outputBias
    try observe("attn_proj", attentionProjection)
    let residual = input + attentionProjection
    try observe("residual", residual)
    let feedInput = MLXFast.layerNorm(residual, weight: norm2Weight,
      bias: norm2Bias, eps: 1e-6)
    try observe("norm2", feedInput)
    let expansion = try fc1.project(feedInput) + fc1Bias
    try observe("fc1", expansion)
    let activated = geluApproximate(expansion)
    try observe("gelu", activated)
    let feed = addMM(fc2Bias, activated, fc2Weight.T)
    try observe("feed", feed)
    let result = residual + feed
    try observe("output", result)
    eval(result)
    Stream.gpu.synchronize()
    try file.checkUnchanged(at: checkpointURL)
    return result
  }
}
