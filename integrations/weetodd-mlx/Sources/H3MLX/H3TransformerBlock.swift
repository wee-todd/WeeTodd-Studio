import Foundation
import MLX
import MLXNN
import TensorIO

struct H3RotaryAngles {
  let rows: Int
  let cosine: MLXArray
  let sine: MLXArray
}

/// One released H3 diffusion block. The five quantized projections are loaded
/// sequentially from the installed Comfy checkpoint and discarded after use.
public enum H3TransformerBlock {
  static func prepareRotaryAngles(checkpointURL: URL,
    positions: MLXArray) throws -> H3RotaryAngles {
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    let angles = try prepareRotaryAngles(file: file,
      prefix: layout.prefix, positions: positions)
    try file.checkUnchanged(at: checkpointURL)
    return angles
  }

  private static func prepareRotaryAngles(file: SafeTensorFile,
    prefix: String, positions: MLXArray) throws -> H3RotaryAngles {
    guard positions.ndim == 2, (1...40_000).contains(positions.shape[0]),
      positions.shape[1] == 3, positions.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 rotary positions.")
    }
    let name = prefix + "rope.inv_freq"
    guard let descriptor = file.tensors[name], descriptor.dtype == "F32",
      descriptor.shape == [16] else {
      throw H3CheckpointError.invalid("Missing H3 rotary frequencies.")
    }
    // Python constructs rotary frequencies and angles in FP32, then casts
    // sine/cosine to the BF16 query dtype. Rounding frequencies first can
    // amplify phase error at later video positions.
    let inverse = MLXArray(try file.readFloat32(named: name))
    let rows = positions.shape[0]
    let axisAngles = (0..<3).map { axis in
      positions[0..<rows, axis].asType(.float32).expandedDimensions(axis: 1)
        * inverse.expandedDimensions(axis: 0)
    }
    let angles = concatenated(axisAngles, axis: 1)
    let doubledAngles = concatenated([angles, angles], axis: 1)
    let cosine = MLX.cos(doubledAngles).asType(.bfloat16)
      .reshaped([1, 1, rows, 96])
    let sine = MLX.sin(doubledAngles).asType(.bfloat16)
      .reshaped([1, 1, rows, 96])
    eval([cosine, sine])
    return H3RotaryAngles(rows: rows, cosine: cosine, sine: sine)
  }

  public static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray, modulation: MLXArray,
    modulationIndices: MLXArray, positions: MLXArray,
    projectionMode: H3ProjectionMode = .weightDecoded) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index, input: input,
      modulation: modulation, modulationIndices: modulationIndices,
      positions: positions, projectionMode: projectionMode,
      observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray, modulation: MLXArray,
    modulationIndices: MLXArray, positions: MLXArray,
    projectionMode: H3ProjectionMode = .weightDecoded,
    lora: (any H3LoRAApplying)? = nil,
    rotaryAngles: H3RotaryAngles? = nil,
    rowWindow: Int = 16384,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<50).contains(index), input.ndim == 3,
      input.shape[0] == 1, (1...40_000).contains(input.shape[1]),
      input.shape[2] == 5376, input.dtype == .bfloat16,
      modulation.ndim == 2, (1...128).contains(modulation.shape[0]),
      modulation.shape[1] == 96768, modulation.dtype == .bfloat16,
      modulationIndices.shape == [input.shape[1]],
      modulationIndices.dtype == .int32,
      positions.shape == [input.shape[1], 3],
      positions.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 diffusion block inputs.")
    }
    let indices = modulationIndices.asArray(Int32.self)
    guard indices.allSatisfy({ (0..<(modulation.shape[0] * 3)).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("H3 modulation index exceeds the timestep table.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    let prefix = layout.prefix + "blocks.\(index)."
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    func read(_ name: String, shape: [Int], dtype: String = "BF16") throws -> MLXArray {
      guard let descriptor = file.tensors[name], descriptor.dtype == dtype,
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 block tensor: \(name)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        dtype == "F32"
          ? MLXArray(bytes, shape, type: Float.self)
          : MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
      eval(value)
      return value
    }
    func project(_ activation: MLXArray, _ suffix: String,
      rows: Int, columns: Int, qkv: Bool = false) throws -> MLXArray {
      let base: MLXArray
      if layout.curveRank != nil {
        let name = prefix + suffix + ".weight"
        guard let descriptor = file.tensors[name], descriptor.dtype == "BF16",
          descriptor.shape == [UInt64(rows), UInt64(columns)] else {
          throw H3CheckpointError.invalid("Missing H3 FL2VA block projection: \(name)")
        }
        var weight = try file.withTensorBytes(named: name) {
          MLXArray($0, [rows, columns], type: UInt16.self).view(dtype: .bfloat16)
        }
        // The pruned FL2VA file already uses [head, QKV, channel] rows.
        if qkv {
          weight = H3QKVRowOrder.forHeadMajorAttention(weight,
            heads: 56, headWidth: 128, groupedSource: false)
        }
        base = matmul(activation, weight.T)
      } else if projectionMode == .activationRotated {
        let weight = try H3ComfyRotatedProjection(file: file,
          name: prefix + suffix + ".weight",
          rows: rows, columns: columns)
        base = try weight.project(activation, reorderQKV: qkv)
      } else {
        let weight = try H3ComfyDecodedProjection.load(file: file,
          checkpointURL: checkpointURL, name: prefix + suffix + ".weight",
          rows: rows, columns: columns, reorderQKV: qkv,
          rowWindow: rowWindow)
        base = matmul(activation, weight.T)
      }
      let value = try lora?.apply(base: base, input: activation,
        target: "diffusion_model.blocks.\(index).\(suffix)",
        reorderQKV: qkv) ?? base
      eval(value)
      Memory.clearCache()
      try Task.checkCancellation()
      return value
    }
    let count = input.shape[1]
    let angles = try rotaryAngles ?? prepareRotaryAngles(
      file: file, prefix: layout.prefix, positions: positions)
    guard angles.rows == count else {
      throw H3CheckpointError.invalid("H3 rotary rows differ from packed input.")
    }
    let mod = modulation.reshaped([modulation.shape[0] * 3, 6 * 5376])
    let tables = (0..<6).map { slot in
      take(mod[0..<(modulation.shape[0] * 3),
        (slot * 5376)..<((slot + 1) * 5376)],
        modulationIndices, axis: 0)
    }
    func rotate(_ value: MLXArray) -> MLXArray {
      let first = value[.ellipsis, 0..<48]
      let second = value[.ellipsis, 48..<96]
      let rotated = concatenated([-second, first], axis: -1)
      let leading = value[.ellipsis, 0..<96] * angles.cosine
        + rotated * angles.sine
      return concatenated([leading, value[.ellipsis, 96..<128]], axis: -1)
    }
    let firstNorm = try read(prefix + "norm1.weight", shape: [5376])
    let first = MLXFast.rmsNorm(input, weight: firstNorm, eps: 1e-5)
      * (1 + tables[1]) + tables[0]
    try observe("norm1_adaln", first)
    let qkv = try project(first, "attn.qkv_proj",
      rows: 21504, columns: 5376, qkv: true)
      .reshaped([1, count, 56, 3, 128])
    try observe("qkv", qkv)
    let qNorm = try read(prefix + "attn.q_norm.weight", shape: [128])
    let kNorm = try read(prefix + "attn.k_norm.weight", shape: [128])
    let query = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 0, 0..<128],
      weight: qNorm, eps: 1e-5).transposed(0, 2, 1, 3))
    let key = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 1, 0..<128],
      weight: kNorm, eps: 1e-5).transposed(0, 2, 1, 3))
    let value = qkv[.ellipsis, 2, 0..<128].transposed(0, 2, 1, 3)
    try observe("query", query)
    try observe("key", key)
    let attended = MLXFast.scaledDotProductAttention(queries: query,
      keys: key, values: value, scale: 1 / Float(128).squareRoot(),
      mask: nil).transposed(0, 2, 1, 3).reshaped([1, count, 7168])
    try observe("attended", attended)
    let attention = try project(attended, "attn.out_proj",
      rows: 5376, columns: 7168)
    try observe("attention", attention)
    let residual = input + tables[2] * attention
    eval(residual)
    try observe("attention_residual", residual)
    let secondNorm = try read(prefix + "norm2.weight", shape: [5376])
    let feedInput = MLXFast.rmsNorm(residual, weight: secondNorm,
      eps: 1e-5) * (1 + tables[4]) + tables[3]
    try observe("norm2_adaln", feedInput)
    let fused = try project(feedInput, "mlp.fc1",
      rows: 28672, columns: 5376)
    try observe("fused", fused)
    let gate = fused[.ellipsis, 0..<14336]
    let gated = silu(gate) * fused[.ellipsis, 14336..<28672]
    try observe("gated", gated)
    let feed = try project(gated, "mlp.fc2",
      rows: 5376, columns: 14336)
    try observe("feed", feed)
    let output = residual + tables[5] * feed
    eval(output)
    try observe("output", output)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return output
  }
}
