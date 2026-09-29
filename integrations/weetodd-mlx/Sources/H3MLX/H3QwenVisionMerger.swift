import Foundation
import MLX
import MLXNN
import TensorIO

/// Spatially merge four Qwen vision patches into one 5120-wide text feature.
public enum H3QwenVisionMerger {
  public static func evaluate(checkpointURL: URL, deepIndex: Int?,
    input: MLXArray) throws -> MLXArray {
    guard input.ndim == 2, input.dtype == .bfloat16,
      input.shape[1] == 1152, input.shape[0] > 0,
      input.shape[0].isMultiple(of: 4), input.shape[0] <= 16_384,
      deepIndex.map({ (0..<3).contains($0) }) ?? true else {
      throw H3CheckpointError.invalid("Invalid H3 Qwen vision merger input.")
    }
    try Task.checkCancellation()
    let file = try SafeTensorFile(url: checkpointURL)
    let root = deepIndex.map({ "visual.deepstack_merger_list.\($0)" }) ?? "visual.merger"
    func read(_ suffix: String, shape: [UInt64]) throws -> MLXArray {
      let name = root + "." + suffix
      guard file.tensors[name].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
        == H3TensorInfo(dtype: "BF16", shape: shape) else {
        throw H3CheckpointError.invalid("Incomplete H3 Qwen merger tensor: \(name)")
      }
      return try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape.map(Int.init), type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let normWidth: UInt64 = deepIndex == nil ? 1152 : 4608
    let normWeight = try read("norm.weight", shape: [normWidth])
    let normBias = try read("norm.bias", shape: [normWidth])
    let fc1 = try H3QwenQ8Projection(file: file, name: root + ".linear_fc1.weight")
    let fc1Bias = try read("linear_fc1.bias", shape: [4608])
    let fc2 = try H3QwenQ8Projection(file: file, name: root + ".linear_fc2.weight")
    let fc2Bias = try read("linear_fc2.bias", shape: [5120])
    let normalized: MLXArray
    if deepIndex == nil {
      normalized = MLXFast.layerNorm(input, weight: normWeight,
        bias: normBias, eps: 1e-6).reshaped([-1, 4608])
    } else {
      normalized = MLXFast.layerNorm(input.reshaped([-1, 4608]),
        weight: normWeight, bias: normBias, eps: 1e-6)
    }
    let expanded = try fc1.project(normalized) + fc1Bias
    let result = try fc2.project(gelu(expanded)) + fc2Bias
    eval(result)
    Stream.gpu.synchronize()
    try file.checkUnchanged(at: checkpointURL)
    return result
  }
}
