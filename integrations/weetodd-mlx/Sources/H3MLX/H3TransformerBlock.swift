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
  static func validateAttendedOverrideLayout(fastVariant: H3FastVariant?, curveRank: Int?) throws {
    guard fastVariant == nil, curveRank == nil || curveRank == 64 else {
      throw H3CheckpointError.invalid("Attended-only override requires ordinary Ref2VA or rank-64 FL attention.")
    }
  }
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
    preparedWeights: H3PreparedBlock? = nil,
    preparedFeedRowChunk: Int = 8192,
    attendedOverride: ((MLXArray, MLXArray, MLXArray) throws -> MLXArray)? = nil,
    onAttentionBackend: @escaping (H3FastAttention.Consumer) -> Void = { _ in },
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard (0..<50).contains(index), input.ndim == 3,
      input.shape[0] == 1, [40_000,64_000].contains(maximumRows), (1...maximumRows).contains(input.shape[1]),
      input.shape[2] == 5376, input.dtype == .bfloat16,
      modulation.ndim == 2, (1...128).contains(modulation.shape[0]),
      modulation.shape[1] == 96768, modulation.dtype == .bfloat16,
      modulationIndices.shape == [input.shape[1]],
      modulationIndices.dtype == .int32,
      positions.shape == [input.shape[1], 3],
      positions.dtype == .float32, [8192,16_384].contains(preparedFeedRowChunk) else {
      throw H3CheckpointError.invalid("Invalid H3 diffusion block inputs.")
    }
    if let preparedWeights {
      guard preparedWeights.checkpointURL == checkpointURL, preparedWeights.index == index,
        projectionMode == .weightDecoded else {
        throw H3CheckpointError.invalid("Prepared H3 weights differ from the requested block.")
      }
      try preparedWeights.checkUnchanged()
    }
    if attendedOverride != nil {
      guard preparedWeights != nil, projectionMode == .weightDecoded,
        fastTiles == nil, vdn == nil, maximumRows == 40_000 else {
        throw H3CheckpointError.invalid("Experimental attended-only override requires ordinary prepared attention.")
      }
    }
    let blockLoRA = try preparedWeights?.prepareLoRA(lora) ?? lora
    let layout = try preparedWeights?.layout ?? H3CheckpointLayout(url: checkpointURL)
    try validateTrainedAttentionGeometry(variant:layout.fastVariant,tiles:fastTiles,rows:input.shape[1])
    if attendedOverride != nil {
      try validateAttendedOverrideLayout(fastVariant: layout.fastVariant, curveRank: layout.curveRank)
    }
    let indices = modulationIndices.asArray(Int32.self)
    guard indices.allSatisfy({ (0..<(modulation.shape[0] * 3)).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("H3 modulation index exceeds the timestep table.")
    }
    try Task.checkCancellation()
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL, block: index)
    let file = try preparedWeights?.file ?? SafeTensorFile(url: tensorURL)
    let prefix = layout.prefix + "blocks.\(index)."
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = H3SamplingAllocationPolicy.blockLimit(
      previous: previousCacheLimit, prepared: preparedWeights != nil)
    let nativePage = layout.fastVariant == nil || preparedWeights != nil ? nil : H3NativePage(file: file, url: tensorURL)
    defer {
      nativePage?.clear()
      if preparedWeights == nil {
        Stream.gpu.synchronize()
        Memory.clearCache()
      }
      Memory.cacheLimit = previousCacheLimit
    }
    func read(_ name: String, shape: [Int], dtype: String = "BF16") throws -> MLXArray {
      if let preparedWeights { return try preparedWeights.read(name,shape:shape,dtype:dtype) }
      guard let descriptor = file.tensors[name], descriptor.dtype == dtype,
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 block tensor: \(name)")
      }
      if let nativePage { return try nativePage.read(name) }
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
      if let preparedWeights {
        base = try preparedWeights.project(suffix,activation:activation,
          rows:rows,columns:columns,qkv:qkv)
      } else if file.tensors[name]?.dtype == "U32" {
        let reader: ((String) throws -> MLXArray)? = nativePage.map { page in { try page.read($0) } }
        let weight = try H3QwenQ8Projection(file: file, name: name, tensor: reader)
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
      let value: MLXArray
      if let lora = blockLoRA {
        let target="diffusion_model.blocks.\(index).\(suffix)"
        value = try preparedWeights == nil
          ? lora.apply(base:base,input:activation,target:target,reorderQKV:qkv)
          : lora.applyQueued(base:base,input:activation,target:target,reorderQKV:qkv)
      } else { value=base }
      if preparedWeights != nil { asyncEval(value) }
      else { eval(value); Memory.clearCache() }
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
        let attended = try H3FastAttention.evaluate(query:query,key:key,value:value,gate:gate,tiles:fastTiles,onConsumer:onAttentionBackend)
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
      hybridAttention:hybrid, attendedOverride: attendedOverride,
      feedRowChunk: preparedWeights != nil && count >= 16_384 ? preparedFeedRowChunk:nil,
      drainAttentionInputs: preparedWeights != nil && fastTiles == nil,
      queueBranches: preparedWeights != nil && layout.fastVariant != nil && count <= 16_384,
      retireQKV: { preparedWeights?.retireProjection("attn.qkv_proj") },
      retireAttention: {
        preparedWeights?.retireProjection("attn.qkv_proj")
        preparedWeights?.retireProjection("attn.out_proj")
      },
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
    attendedOverride: ((MLXArray, MLXArray, MLXArray) throws -> MLXArray)? = nil,
    feedRowChunk: Int? = nil,
    drainAttentionInputs: Bool = false,
    queueBranches: Bool = false,
    retireQKV: () -> Void = {},
    retireAttention: () -> Void = {},
    observe: (String, MLXArray) throws -> Void = { _, _ in }) throws -> MLXArray {
    guard attendedOverride == nil || (hybridAttention == nil && drainAttentionInputs && !queueBranches) else {
      throw H3CheckpointError.invalid("Attended-only override cannot retain hybrid attention inputs.")
    }
    let count = input.shape[1]
    let mod = modulation.reshaped([modulation.shape[0] * 3, 6 * hiddenWidth])
    func rotate(_ value: MLXArray) -> MLXArray {
      H3Rotary.apply(value, cosine: angles.cosine, sine: angles.sine, rotaryWidth: rotaryWidth)
    }
    func runAttention() throws -> MLXArray {
      func prepareInputs() throws -> (MLXArray?, MLXArray?, MLXArray, MLXArray, MLXArray) {
        let firstNorm = try read("norm1.weight", [hiddenWidth])
        let first = H3Modulation.scaleShift(MLXFast.rmsNorm(input, weight: firstNorm, eps: 1e-5),
          table:mod,indices:modulationIndices,shift:0,scale:1)
        try observe("norm1_adaln", first)
        let qkv = try project(first, "attn.qkv_proj",
          (3 * heads * headWidth), hiddenWidth, true)
          .reshaped([1, count, heads, 3, headWidth])
        try observe("qkv", qkv)
        let qNorm = try read("attn.q_norm.weight", [headWidth])
        let kNorm = try read("attn.k_norm.weight", [headWidth])
        let query = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 0, 0..<headWidth],
          weight: qNorm, eps: 1e-5).transposed(0, 2, 1, 3))
        // Bound in-flight QKV/LoRA and rotary scratch before submitting SDPA.
        // The GPU completion fence retires their Metal buffers; detaching an
        // async graph alone does not release command-buffer input storage.
        if drainAttentionInputs { eval(query) }
        let key = rotate(MLXFast.rmsNorm(qkv[.ellipsis, 1, 0..<headWidth],
          weight: kNorm, eps: 1e-5).transposed(0, 2, 1, 3))
        if drainAttentionInputs { eval(key) }
        var value = qkv[.ellipsis, 2, 0..<headWidth].transposed(0, 2, 1, 3)
        if drainAttentionInputs && hybridAttention == nil {
          // The strided V view otherwise retains the three-times-larger QKV
          // allocation throughout SDPA. Own V and retire QKV before attention.
          value = contiguous(value)
          eval(value)
          retireQKV()
        }
        try observe("query", query)
        try observe("key", key)
        return (hybridAttention == nil ? nil:first, hybridAttention == nil ? nil:qkv, query,key,value)
      }
      let (first,qkv,query,key,value) = try prepareInputs()
      let attention: MLXArray
      if let hybridAttention {
        guard let first, let qkv else { throw H3CheckpointError.invalid("Missing H3 hybrid attention inputs.") }
        attention = try hybridAttention(first, qkv, query, key, value)
      } else {
        let attended: MLXArray
        if let attendedOverride {
          let headMajor = try attendedOverride(query, key, value)
          guard headMajor.shape == [1, heads, count, headWidth], headMajor.dtype == input.dtype else {
            throw H3CheckpointError.invalid("Experimental attention output differs from admitted geometry or precision.")
          }
          attended = headMajor.transposed(0, 2, 1, 3).reshaped([1, count, heads * headWidth])
        } else {
          attended = MLXFast.scaledDotProductAttention(queries: query,
            keys: key, values: value, scale: 1 / Float(headWidth).squareRoot(),
            mask: nil).transposed(0, 2, 1, 3).reshaped([1, count, (heads * headWidth)])
        }
        try observe("attended", attended)
        attention = try project(attended, "attn.out_proj",
          hiddenWidth, (heads * headWidth), false)
      }
      return attention
    }
    let attention = try runAttention()
    try observe("attention", attention)
    let residual = H3Modulation.residual(input,branch:attention,table:mod,
      indices:modulationIndices,gate:2)
    if queueBranches { asyncEval(residual) } else { eval(residual) }
    try observe("attention_residual", residual)
    retireAttention()
    func runFeed() throws -> MLXArray {
      let secondNorm = try read("norm2.weight", [hiddenWidth])
      let feedInput = H3Modulation.scaleShift(MLXFast.rmsNorm(residual, weight: secondNorm,eps:1e-5),
        table:mod,indices:modulationIndices,shift:3,scale:4)
      try observe("norm2_adaln", feedInput)
      func part(_ partInput: MLXArray) throws -> MLXArray {
        let fused = try project(partInput, "mlp.fc1", (2 * feedWidth), hiddenWidth, false)
        try observe("fused", fused)
        let gate = fused[.ellipsis, 0..<feedWidth]
        let gated = silu(gate) * fused[.ellipsis, feedWidth..<(2 * feedWidth)]
        try observe("gated", gated)
        let feed = try project(gated, "mlp.fc2", hiddenWidth, feedWidth, false)
        try observe("feed", feed)
        return feed
      }
      let feed:MLXArray
      if let feedRowChunk, count > feedRowChunk {
        guard feedRowChunk > 0 else { throw H3CheckpointError.invalid("Invalid H3 feed-forward row chunk.") }
        var parts:[MLXArray]=[]
        for start in stride(from:0,to:count,by:feedRowChunk) {
          try Task.checkCancellation()
          let value=try part(feedInput[0..<1,start..<min(start+feedRowChunk,count),0..<hiddenWidth])
          eval(value) // Retire this chunk's wide gate/up graph before the next.
          parts.append(value)
        }
        feed=concatenated(parts,axis:1);eval(feed)
      } else { feed=try part(feedInput) }

      return feed
    }
    let feed = try runFeed()
    let output = H3Modulation.residual(residual,branch:feed,table:mod,
      indices:modulationIndices,gate:5)
    if queueBranches { asyncEval(output) } else { eval(output) }
    try observe("output", output)
    return output
  }
}
