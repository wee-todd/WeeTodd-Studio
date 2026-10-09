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
    try evaluate(checkpointURL: checkpointURL, blockIndex: blockIndex,
      timeEmbeddings: timeEmbeddings, projectionMode: projectionMode,
      rowWindow: rowWindow, lora: nil)
  }

  static func evaluate(checkpointURL: URL, blockIndex: Int,
    timeEmbeddings: MLXArray, projectionMode: H3ProjectionMode = .weightDecoded,
    rowWindow: Int = 16384, lora: (any H3LoRAApplying)?, loraInput: MLXArray? = nil,
    deferredFastLoading: Bool = true) throws -> MLXArray {
    guard (0..<50).contains(blockIndex), timeEmbeddings.ndim == 2,
      (1...128).contains(timeEmbeddings.shape[0]),
      [64, 2688].contains(timeEmbeddings.shape[1]),
      timeEmbeddings.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 AdaLN block or time embedding shape.")
    }
    try Task.checkCancellation()
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
      let base = addMM(bias, timeEmbeddings.asType(.float32), weight.T)
      let applied = try lora?.apply(base:base,input:loraInput ?? timeEmbeddings,
        target:"diffusion_model.blocks.\(blockIndex).adaln_proj.linear",reorderQKV:false) ?? base
      let result = applied.asType(.bfloat16)
      eval(result)
      try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
      return result
    }
    if layout.fastVariant != nil {
      let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, block: blockIndex)
      let file = try SafeTensorFile(url: tensorURL)
      let stem = String(name.dropLast(".weight".count))
      // Admit all four factors before the first deferred payload request.
      try validateFastFactors(tensors:file.tensors.mapValues {
        H3TensorInfo(dtype:$0.dtype,shape:$0.shape)
      },weightName:name)
      try file.checkUnchanged(at:tensorURL)
      try Task.checkCancellation()
      let page = deferredFastLoading ? H3NativePage(file:file,url:tensorURL) : nil
      defer { page?.clear() }
      let reader: ((String) throws -> MLXArray)? = page.map { native in
        { try native.read($0,materialize:false) }
      }
      let projection = try H3QwenQ8Projection(file:file,name:name,tensor:reader,
        materializeWeights:!deferredFastLoading)
      let bias: MLXArray
      if let page { bias = try page.read(stem + ".bias",materialize:false) }
      else {
        bias = try file.withTensorBytes(named:stem + ".bias") {
          MLXArray($0,[96768],type:UInt16.self).view(dtype:.bfloat16)
        }
      }
      let activated = silu(timeEmbeddings.asType(.float32)).asType(.bfloat16)
      let result = try projection.project(activated) + bias
      eval(result)
      try file.checkUnchanged(at: tensorURL)
      try H3CheckpointSource.checkUnchanged(checkpointURL)
      try Task.checkCancellation()
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
      let base: MLXArray
      if let rotated { base = try rotated.project(activated) }
      else {
        base = try H3ComfyDecodedProjection.projectStreaming(
          checkpointURL: checkpointURL, name: name, rows: 96768,
          columns: 2688, input: activated, rowWindow: rowWindow)
      }
      let value = try lora?.apply(base: base, input: activated,
        target: "diffusion_model.blocks.\(blockIndex).adaln_proj.linear", reorderQKV: false) ?? base
      eval(value)
      parts.append(value)
      try Task.checkCancellation()
    }
    let result = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
    eval(result)
    return result
  }

  /// Header-only admission shared with focused malformed-factor tests.
  static func validateFastFactors(tensors:[String:H3TensorInfo],weightName:String) throws {
    let stem = String(weightName.dropLast(".weight".count))
    for (name,dtype,shape) in [(weightName,"U32",[UInt64(96768),672]),
      (stem + ".scales","BF16",[UInt64(96768),42]),
      (stem + ".biases","BF16",[UInt64(96768),42]),
      (stem + ".bias","BF16",[UInt64(96768)])] {
      guard tensors[name] == H3TensorInfo(dtype:dtype,shape:shape) else {
        throw H3CheckpointError.invalid("Invalid deferred FastH3 AdaLN factor: \(name)")
      }
    }
  }
}
