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
  static func trainedCompressionGate(_ input:MLXArray,weight:MLXArray,heads:Int,headWidth:Int) -> MLXArray {
    // The trained correction is signed and unbounded, including an exact zero
    // for zero weights. A sigmoid changes the checkpoint's attention function.
    matmul(input,weight.T).reshaped([1,input.shape[1],heads,headWidth]).transposed(0,2,1,3)
  }
  static func validateTrainedAttentionGeometry(variant:H3FastVariant?,tiles:H3FastTiles?,rows:Int) throws {
    guard variant != .vsaV1 || tiles?.rows == rows,
      tiles == nil || variant == .vsaV1 else {
      throw H3CheckpointError.invalid("Trained FastH3 VSA requires explicit matching tile geometry; use its T2VA runner.")
    }
  }
  static func prepareRotaryAngles(checkpointURL: URL,
    positions: MLXArray, maximumRows: Int = 40_000) throws -> H3RotaryAngles {
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL)
    let file = try SafeTensorFile(url: tensorURL)
    let angles = try H3CheckpointSource.isPaged(checkpointURL)
      ? prepareRotaryAngles(inverse: computedInverseFrequency(), positions: positions, maximumRows: maximumRows)
      : prepareRotaryAngles(file: file, prefix: layout.prefix, positions: positions, maximumRows: maximumRows)
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    return angles
  }

  private static func prepareRotaryAngles(file: SafeTensorFile,
    prefix: String, positions: MLXArray, maximumRows: Int = 40_000) throws -> H3RotaryAngles {
    guard positions.ndim == 2, [40_000,64_000].contains(maximumRows), (1...maximumRows).contains(positions.shape[0]),
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
    let inverse = MLXArray(try file.readFloat32(named: name, access: .buffered))
    return try prepareRotaryAngles(inverse: inverse, positions: positions, maximumRows: maximumRows)
  }

  static func computedInverseFrequency() -> MLXArray {
    let exponent = MLXArray((0..<16).map { Float(2 * $0) }) / Float(32)
    return Float(1) / MLX.pow(MLXArray(Float(10_000)), exponent)
  }

  private static func prepareRotaryAngles(inverse: MLXArray,
    positions: MLXArray, maximumRows: Int = 40_000) throws -> H3RotaryAngles {
    guard positions.ndim == 2, [40_000,64_000].contains(maximumRows), (1...maximumRows).contains(positions.shape[0]),
      positions.shape[1] == 3, positions.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 rotary positions.")
    }
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
    rotaryAngles: H3RotaryAngles? = nil, maximumRows: Int = 40_000,
    rowWindow: Int = 16384, vdn: H3VDNRuntime? = nil,fastTiles:H3FastTiles? = nil,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<50).contains(index), input.ndim == 3,
      input.shape[0] == 1, [40_000,64_000].contains(maximumRows), (1...maximumRows).contains(input.shape[1]),
      input.shape[2] == 5376, input.dtype == .bfloat16,
      modulation.ndim == 2, (1...128).contains(modulation.shape[0]),
      modulation.shape[1] == 96768, modulation.dtype == .bfloat16,
      modulationIndices.shape == [input.shape[1]],
      modulationIndices.dtype == .int32,
      positions.shape == [input.shape[1], 3],
      positions.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 diffusion block inputs.")
    }
    let layout = try H3CheckpointLayout(url: checkpointURL)
    try validateTrainedAttentionGeometry(variant:layout.fastVariant,tiles:fastTiles,rows:input.shape[1])
    let indices = modulationIndices.asArray(Int32.self)
    guard indices.allSatisfy({ (0..<(modulation.shape[0] * 3)).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("H3 modulation index exceeds the timestep table.")
    }
    try Task.checkCancellation()
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, block: index)
    let file = try SafeTensorFile(url: tensorURL)
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
      let value = try H3TensorPayload.withTensorBytes(file: file, name: name,
        maximumBufferedBytes: 4 * 1024 * 1024) { bytes in
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
      let name = prefix + suffix + ".weight"
      if file.tensors[name]?.dtype == "U32" {
        let weight = try H3QwenQ8Projection(file: file, name: name)
        guard weight.rows == rows, weight.columns == columns else {
          throw H3CheckpointError.invalid("Paged H3 affine projection changed after admission.")
        }
        // Owned FL pages preserve head-major QKV rows, including quantized rows.
        base = try weight.project(activation)
      } else if layout.curveRank != nil {
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
    let angles: H3RotaryAngles
    if let rotaryAngles { angles = rotaryAngles }
    else if H3CheckpointSource.isPaged(checkpointURL) {
      angles = try prepareRotaryAngles(inverse: computedInverseFrequency(), positions: positions, maximumRows: maximumRows)
    } else { angles = try prepareRotaryAngles(file: file, prefix: layout.prefix, positions: positions, maximumRows: maximumRows) }
    guard angles.rows == count else {
      throw H3CheckpointError.invalid("H3 rotary rows differ from packed input.")
    }
    let hybrid: ((MLXArray,MLXArray,MLXArray,MLXArray,MLXArray) throws -> MLXArray)?
    if let fastTiles {
      guard layout.fastVariant == .vsaV1,vdn == nil else {
        throw H3CheckpointError.invalid("FastH3 VSA geometry requires its trained checkpoint.")
      }
      hybrid = { first,_,query,key,value in
        let weight = try read(prefix + "attn.gate_compress.weight",shape:[7168,5376])
        let gate = trainedCompressionGate(first,weight:weight,heads:56,headWidth:128)
        let attended = try H3FastAttention.evaluate(query:query,key:key,value:value,gate:gate,tiles:fastTiles)
          .transposed(0,2,1,3).reshaped([1,count,7168])
        return try project(attended,"attn.out_proj",rows:5376,columns:7168)
      }
    } else if let runtime = vdn {
      hybrid = { first,qkv,query,key,value in
        try runtime.attention(block:index,input:first,qkv:qkv,query:query,key:key,value:value,
          project:{ try project($0,"attn.out_proj",rows:5376,columns:7168) })
      }
    } else { hybrid = nil }
    let output = try evaluateKernel(input: input, modulation: modulation,
      modulationIndices: modulationIndices, angles: angles,
      read: { try read(prefix + $0, shape: $1) },
      project: { try project($0, $1, rows: $2, columns: $3, qkv: $4) },
      hybridAttention:hybrid,
      observe: { name,value in
        if maximumRows == 64_000, UInt64(Memory.activeMemory) > H3CanvasAdmission.maximumStageBytes {
          throw H3CheckpointError.invalid("H3 spatial block exceeded its 32 GiB active-memory budget.")
        }
        try observe(name,value)
      })
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    try Task.checkCancellation()
    return output
  }

  /// Shared dense H3 block arithmetic. ControlNet supplies its own checked
  /// tensor names; normalization, AdaLN, RoPE, attention and gated MLP remain
  /// the same implementation as the base model. Smaller dimensions permit
  /// deterministic numerical contract tests without loading model weights.
  static func evaluateKernel(input: MLXArray, modulation: MLXArray,
    modulationIndices: MLXArray, angles: H3RotaryAngles,
    hiddenWidth: Int = 5376, heads: Int = 56, headWidth: Int = 128,
    feedWidth: Int = 14336, rotaryWidth: Int = 96,
    read: (String, [Int]) throws -> MLXArray,
    project: (MLXArray, String, Int, Int, Bool) throws -> MLXArray,
    hybridAttention: ((MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) throws -> MLXArray)? = nil,
    observe: (String, MLXArray) throws -> Void = { _, _ in }) throws -> MLXArray {
    let count = input.shape[1]
    let mod = modulation.reshaped([modulation.shape[0] * 3, 6 * hiddenWidth])
    let tables = (0..<6).map { slot in
      take(mod[0..<(modulation.shape[0] * 3),
        (slot * hiddenWidth)..<((slot + 1) * hiddenWidth)],
        modulationIndices, axis: 0)
    }
    func rotate(_ value: MLXArray) -> MLXArray {
      let first = value[.ellipsis, 0..<(rotaryWidth / 2)]
      let second = value[.ellipsis, (rotaryWidth / 2)..<rotaryWidth]
      let rotated = concatenated([-second, first], axis: -1)
      let leading = value[.ellipsis, 0..<rotaryWidth] * angles.cosine
        + rotated * angles.sine
      return concatenated([leading, value[.ellipsis, rotaryWidth..<headWidth]], axis: -1)
    }
    let firstNorm = try read("norm1.weight", [hiddenWidth])
    let first = MLXFast.rmsNorm(input, weight: firstNorm, eps: 1e-5)
      * (1 + tables[1]) + tables[0]
    try observe("norm1_adaln", first)
    let qkv = try project(first, "attn.qkv_proj",
      (3 * heads * headWidth), hiddenWidth, true)
      .reshaped([1, count, heads, 3, headWidth])
    try observe("qkv", qkv)
    let qNorm = try read("attn.q_norm.weight", [headWidth])
    let kNorm = try read("attn.k_norm.weight", [headWidth])
    let query = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 0, 0..<headWidth],
      weight: qNorm, eps: 1e-5).transposed(0, 2, 1, 3))
    let key = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 1, 0..<headWidth],
      weight: kNorm, eps: 1e-5).transposed(0, 2, 1, 3))
    let value = qkv[.ellipsis, 2, 0..<headWidth].transposed(0, 2, 1, 3)
    try observe("query", query)
    try observe("key", key)
    let attention: MLXArray
    if let hybridAttention {
      attention = try hybridAttention(first, qkv, query, key, value)
    } else {
      let attended = MLXFast.scaledDotProductAttention(queries: query,
        keys: key, values: value, scale: 1 / Float(headWidth).squareRoot(),
        mask: nil).transposed(0, 2, 1, 3).reshaped([1, count, (heads * headWidth)])
      try observe("attended", attended)
      attention = try project(attended, "attn.out_proj",
        hiddenWidth, (heads * headWidth), false)
    }
    try observe("attention", attention)
    let residual = input + tables[2] * attention
    eval(residual)
    try observe("attention_residual", residual)
    let secondNorm = try read("norm2.weight", [hiddenWidth])
    let feedInput = MLXFast.rmsNorm(residual, weight: secondNorm,
      eps: 1e-5) * (1 + tables[4]) + tables[3]
    try observe("norm2_adaln", feedInput)
    let fused = try project(feedInput, "mlp.fc1",
      (2 * feedWidth), hiddenWidth, false)
    try observe("fused", fused)
    let gate = fused[.ellipsis, 0..<feedWidth]
    let gated = silu(gate) * fused[.ellipsis, feedWidth..<(2 * feedWidth)]
    try observe("gated", gated)
    let feed = try project(gated, "mlp.fc2",
      hiddenWidth, feedWidth, false)
    try observe("feed", feed)
    let output = residual + tables[5] * feed
    eval(output)
    try observe("output", output)
    return output
  }
}
