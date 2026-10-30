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

  static func regenerateWindows(rows:Int,chunkRows:Int=8192) throws -> [Range<Int>] {
    guard (1...H3RegenerateAdmission.maximumPackedRows).contains(rows),(1...8192).contains(chunkRows),
      rows<=Int(Int32.max)/5376 else {
      throw H3CheckpointError.invalid("Regenerate input projection exceeds its bounded row/index profile.")
    }
    return stride(from:0,to:rows,by:chunkRows).map{$0..<min(rows,$0+chunkRows)}
  }

  static func projectRegenerateRows(kind:Kind,input:MLXArray,chunkRows:Int=8192,
    project:(MLXArray)throws->MLXArray) throws -> MLXArray {
    guard kind != .condition,input.ndim==3,input.shape[0]==1,input.shape[2]==kind.columns,input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid regenerate input projection stream.")
    }
    let windows=try regenerateWindows(rows:input.shape[1],chunkRows:chunkRows)
    var parts=[MLXArray]()
    for rows in windows {
      try Task.checkCancellation()
      let part=try autoreleasepool {
        let value=try project(input[0..<1,rows,0..<kind.columns])
        guard value.shape==[1,rows.count,5376],value.dtype.isFloatingPoint else {
          throw H3CheckpointError.invalid("Regenerate input projection changed its row layout.")
        }
        let cast=value.asType(.bfloat16);eval(cast);return cast
      }
      parts.append(part)
    }
    let result=concatenated(parts,axis:1);eval(result)
    return result
  }

  /// Exact row-independent FP32 projections, cast to model precision per window.
  /// The ordinary loader and its65,536-row admission remain unchanged.
  static func evaluateRegenerate(checkpointURL:URL,kind:Kind,input:MLXArray) throws -> MLXArray {
    try projectRegenerateRows(kind:kind,input:input) { rows in
      try evaluate(checkpointURL:checkpointURL,kind:kind,input:rows)
    }
  }

  public static func evaluate(checkpointURL: URL, kind: Kind,
    input: MLXArray, omni: H3OmniCheckpoint? = nil) throws -> MLXArray {
    guard input.ndim == 3, input.shape[0] == 1,
      (1...65536).contains(input.shape[1]), input.shape[2] == kind.columns,
      input.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 \(kind.rawValue) input shape.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointSource.layout(checkpointURL, omni: omni)
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, omni: omni)
    let file = try SafeTensorFile(url: tensorURL)
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let stem = layout.prefix + kind.rawValue
    if let omni {
      return try omni.withParameters { scope in
        let bias = try scope.read(stem + ".bias")
        let result: MLXArray
        if kind == .condition {
          result = try scope.project(stem, input: input.asType(.bfloat16)) + bias
        } else {
          let weight = try scope.read(stem + ".weight")
          result = addMM(bias.asType(.float32), input.asType(.float32), weight.asType(.float32).T)
        }
        eval(result)
        return result
      }
    }
    let weightName = stem + ".weight"
    let biasName = stem + ".bias"
    let biasDtype = kind == .condition || layout.fastVariant != nil ? "BF16" : "F32"
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
      biasDtype == "BF16"
        ? MLXArray(bytes, [5376], type: UInt16.self).view(dtype: .bfloat16)
        : MLXArray(bytes, [5376], type: Float.self)
    }
    let result: MLXArray
    if kind == .condition {
      result = addMM(bias, input.asType(.bfloat16), weight.T)
    } else {
      result = addMM(bias.asType(.float32), input.asType(.float32), weight.asType(.float32).T)
    }
    eval(result)
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL, omni: omni)
    try Task.checkCancellation()
    return result
  }
}
