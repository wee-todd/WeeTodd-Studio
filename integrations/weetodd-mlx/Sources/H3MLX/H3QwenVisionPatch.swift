import Foundation
import MLX
import TensorIO

/// Qwen3-VL's 2×16×16 patch embedding applied to processor-produced patches.
/// The flattened 3×2×16×16 patch is a dense projection, so it need not expand
/// into a five-dimensional convolution input on the GPU.
public enum H3QwenVisionPatch {
  public static func embed(pixels: MLXArray, checkpointURL: URL) throws -> MLXArray {
    guard pixels.ndim == 2, pixels.shape[1] == 1536,
      pixels.shape[0] > 0, pixels.shape[0] <= 16_384,
      pixels.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("H3 Qwen vision patches require bounded BF16 rows of width 1536.")
    }
    let file = try SafeTensorFile(url: checkpointURL)
    let weightName = "visual.patch_embed.proj.weight"
    let biasName = "visual.patch_embed.proj.bias"
    guard file.tensors[weightName].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
        == H3TensorInfo(dtype: "BF16", shape: [1152, 3, 2, 16, 16]),
      file.tensors[biasName].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
        == H3TensorInfo(dtype: "BF16", shape: [1152]) else {
      throw H3CheckpointError.invalid("H3 Qwen vision patch weights are incomplete.")
    }
    func read(_ name: String, shape: [Int]) throws -> MLXArray {
      try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let weight = try read(weightName, shape: [1152, 1536])
    let bias = try read(biasName, shape: [1152])
    let result = matmul(pixels, weight.T) + bias
    eval(result)
    try file.checkUnchanged(at: checkpointURL)
    return result
  }
}
