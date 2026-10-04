import Foundation
import MLX
import MLXNN
import TensorIO

/// The installed 50-block H3 checkpoint stores each block's 18-way AdaLN
/// projection as a 96768×2688 Comfy INT8 matrix. Evaluate it by bounded rows
/// so the complete decoded weight never occupies unified memory.
public enum H3AdaLNProjection {
  public static func evaluate(checkpointURL: URL, blockIndex: Int,
    timeEmbeddings: MLXArray,
    projectionMode: H3ProjectionMode = .weightDecoded,
    rowWindow: Int = 16384) throws -> MLXArray {
    guard (0..<50).contains(blockIndex), timeEmbeddings.ndim == 2,
      (1...128).contains(timeEmbeddings.shape[0]),
      [64, 2688].contains(timeEmbeddings.shape[1]),
      timeEmbeddings.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 AdaLN block or time embedding shape.")
    }
    let layout = try H3CheckpointLayout(url: checkpointURL)
    guard timeEmbeddings.shape[1] == (layout.curveRank ?? 2688) else {
      throw H3CheckpointError.invalid("H3 AdaLN coordinates differ from the checkpoint.")
    }
    let name = layout.prefix + "blocks.\(blockIndex).adaln_proj.linear.weight"
    if layout.curveRank == 64 {
      let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, block: blockIndex)
    let file = try SafeTensorFile(url: tensorURL)
      let weight = try file.withTensorBytes(named: name) {
        MLXArray($0, [96768, 64], type: Float.self)
      }
      let bias = try file.withTensorBytes(named:
        layout.prefix + "blocks.\(blockIndex).adaln_proj.linear.bias") {
        MLXArray($0, [96768], type: Float.self)
      }
      let result = addMM(bias, timeEmbeddings.asType(.float32), weight.T)
        .asType(.bfloat16)
      eval(result)
      try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
      return result
    }
    let rotated: H3ComfyRotatedProjection?
    if projectionMode == .activationRotated {
      let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, block: blockIndex)
    let file = try SafeTensorFile(url: tensorURL)
      rotated = try H3ComfyRotatedProjection(file: file,
        name: name, rows: 96768, columns: 2688,
        biasName: String(name.dropLast(".weight".count)) + ".bias")
      try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    } else {
      rotated = nil
    }
    var parts: [MLXArray] = []
    for start in stride(from: 0, to: timeEmbeddings.shape[0], by: 16) {
      let stop = min(start + 16, timeEmbeddings.shape[0])
      let activated = silu(timeEmbeddings[start..<stop, 0..<2688]
        .asType(.float32)).asType(.bfloat16)
      eval(activated)
      if let rotated {
        parts.append(try rotated.project(activated))
      } else {
        parts.append(try H3ComfyDecodedProjection.projectStreaming(
          checkpointURL: checkpointURL, name: name, rows: 96768,
          columns: 2688, input: activated, rowWindow: rowWindow))
      }
      try Task.checkCancellation()
    }
    let result = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
    eval(result)
    return result
  }
}
