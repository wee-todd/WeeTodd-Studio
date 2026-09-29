import Foundation
import MLX
import TensorIO

/// Installed H3's three input projections. Video/audio execute in FP32 while
/// the Qwen condition stream retains the released BF16 projection precision.
public enum H3InputProjection {
  public enum Kind: String, Sendable {
    case video = "video_patch_proj"
    case audio = "audio_patch_proj"
    case condition = "condition_proj"

    var columns: Int {
      switch self {
      case .video: 96
      case .audio: 32
      case .condition: 5120
      }
    }
  }

  public static func evaluate(checkpointURL: URL, kind: Kind,
    input: MLXArray) throws -> MLXArray {
    guard input.ndim == 3, input.shape[0] == 1,
      (1...65536).contains(input.shape[1]), input.shape[2] == kind.columns,
      input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 \(kind.rawValue) input shape.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let stem = layout.prefix + kind.rawValue
    let weightName = stem + ".weight"
    let biasName = stem + ".bias"
    let biasDtype = kind == .condition ? "BF16" : "F32"
    let weightDtype = layout.curveRank != nil && kind != .condition ? "F32" : "BF16"
    guard let weightInfo = file.tensors[weightName],
      weightInfo.dtype == weightDtype,
      weightInfo.shape == [5376, UInt64(kind.columns)],
      let biasInfo = file.tensors[biasName],
      biasInfo.dtype == biasDtype,
      biasInfo.shape == [5376] else {
      throw H3CheckpointError.invalid("Missing H3 input projection: \(kind.rawValue)")
    }
    let weight = try file.withTensorBytes(named: weightName) { bytes in
      weightDtype == "F32"
        ? MLXArray(bytes, [5376, kind.columns], type: Float.self)
        : MLXArray(bytes, [5376, kind.columns], type: UInt16.self).view(dtype: .bfloat16)
    }
    let bias = try file.withTensorBytes(named: biasName) { bytes in
      kind == .condition
        ? MLXArray(bytes, [5376], type: UInt16.self).view(dtype: .bfloat16)
        : MLXArray(bytes, [5376], type: Float.self)
    }
    let result: MLXArray
    if kind == .condition {
      result = addMM(bias, input.asType(.bfloat16), weight.T)
    } else {
      result = addMM(bias, input.asType(.float32), weight.asType(.float32).T)
    }
    eval(result)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return result
  }
}
