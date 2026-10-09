import Foundation
import MLX
import TensorIO

/// One streamed Qwen3-VL decoder layer. H3 reads its unnormalized output
/// after layer 49, so no final norm or language-model head is loaded.
public struct H3QwenLayer {
  private static let mixedRotaryKernel = MLXFast.metalKernel(
    name: "weetodd_h3_qwen_mixed_rotary",
    inputNames: ["x", "position_ids", "inverse", "selector"],
    outputNames: ["rotated"],
    source: """
      uint element = thread_position_in_grid.x;
      int length = x_shape[2];
      int heads = x_shape[1];
      if (element >= uint(heads * length * 64)) return;
      int block = int(element / 64);
      int slot = int(element) - block * 64;
      int head = block / length;
      int token = block - head * length;
      int base = (head * length + token) * 128;
      int axis = selector[slot];
      float angle = float(position_ids[axis * length + token]) * float(inverse[slot]);
      float cosine = metal::cos(angle);
      float sine = metal::sin(angle);
      float a = float(x[base + slot]);
      float b = float(x[base + slot + 64]);
      rotated[base + slot] = static_cast<T>(a * cosine - b * sine);
      rotated[base + slot + 64] = static_cast<T>(b * cosine + a * sine);
      """)

  private static func mixedRotary(_ input: MLXArray, positions: MLXArray,
    frequencies: MLXArray, selectors: MLXArray) -> MLXArray {
    mixedRotaryKernel([input, positions, frequencies, selectors],
      template: [("T", input.dtype)],
      grid: (input.shape[1] * input.shape[2] * 64, 1, 1),
      threadGroup: (256, 1, 1), outputShapes: [input.shape],
      outputDTypes: [input.dtype])[0]
  }

  private let inputNorm: MLXArray
  private let postAttentionNorm: MLXArray
  private let queryNorm: MLXArray
  private let keyNorm: MLXArray
  private let query: H3QwenQ8Projection
  private let key: H3QwenQ8Projection
  private let value: H3QwenQ8Projection
  private let output: H3QwenQ8Projection
  private let gate: H3QwenQ8Projection
  private let up: H3QwenQ8Projection
  private let down: H3QwenQ8Projection

  public init(file: SafeTensorFile, index: Int) throws {
    try self.init(file: file, index: index, tensor: nil)
  }

  private init(file: SafeTensorFile, index: Int,
    tensor: ((String) throws -> MLXArray)?) throws {
    try Task.checkCancellation()
    try file.checkUnchanged()
    let headers = file.tensors.mapValues { H3TensorInfo(dtype: $0.dtype, shape: $0.shape) }
    try H3QwenCheckpointLayout.validateLayer(index: index, tensors: headers)
    let root = "model.layers.\(index)."
    func norm(_ suffix: String) throws -> MLXArray {
      try Task.checkCancellation()
      let name = root + suffix
      let descriptor = file.tensors[name]!
      if let tensor {
        let result = try tensor(name)
        guard result.shape == descriptor.shape.map(Int.init),
          result.dtype == (descriptor.dtype == "BF16" ? .bfloat16 : .float16) else {
          throw H3CheckpointError.invalid("H3 Qwen norm changed after admission.")
        }
        return result
      }
      let result: MLXArray = try file.withTensorBytes(named: name) { bytes in
        let shape = descriptor.shape.map(Int.init)
        switch descriptor.dtype {
        case "BF16": return MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
        case "F16": return MLXArray(bytes, shape, type: Float16.self)
        default: throw H3CheckpointError.invalid("Unsupported H3 Qwen norm dtype.")
        }
      }
      return result
    }
    inputNorm = try norm("input_layernorm.weight")
    postAttentionNorm = try norm("post_attention_layernorm.weight")
    queryNorm = try norm("self_attn.q_norm.weight")
    keyNorm = try norm("self_attn.k_norm.weight")
    query = try H3QwenQ8Projection(file: file, name: root + "self_attn.q_proj.weight",
      tensor: tensor, materializeWeights: false)
    key = try H3QwenQ8Projection(file: file, name: root + "self_attn.k_proj.weight",
      tensor: tensor, materializeWeights: false)
    value = try H3QwenQ8Projection(file: file, name: root + "self_attn.v_proj.weight",
      tensor: tensor, materializeWeights: false)
    output = try H3QwenQ8Projection(file: file, name: root + "self_attn.o_proj.weight",
      tensor: tensor, materializeWeights: false)
    gate = try H3QwenQ8Projection(file: file, name: root + "mlp.gate_proj.weight",
      tensor: tensor, materializeWeights: false)
    up = try H3QwenQ8Projection(file: file, name: root + "mlp.up_proj.weight",
      tensor: tensor, materializeWeights: false)
    down = try H3QwenQ8Projection(file: file, name: root + "mlp.down_proj.weight",
      tensor: tensor, materializeWeights: false)
    try Task.checkCancellation()
    try file.checkUnchanged()
    // A single admitted layer uses MLX-owned packed arrays. A native page
    // lets MLX read the tensors through its bounded parallel file reader;
    // the direct initializer has already copied each scoped tensor mapping.
    // Materialize the full layer once. No future layer is requested.
    eval([inputNorm, postAttentionNorm, queryNorm, keyNorm] +
      [query, key, value, output, gate, up, down].flatMap(\.parametersToMaterialize))
    try file.checkUnchanged()
    try Task.checkCancellation()
  }

  private static func rotary(_ x: MLXArray, cosine: MLXArray,
    sine: MLXArray) -> MLXArray {
    // The Qwen fused rotary path keeps the rotation in Float32 and rounds
    // only the rotated result back to the attention activation dtype.
    let floatInput = x.asType(.float32)
    let left = floatInput[.ellipsis, 0..<64]
    let right = floatInput[.ellipsis, 64..<128]
    return concatenated([left * cosine - right * sine,
      right * cosine + left * sine], axis: -1).asType(x.dtype)
  }

  public func callAsFunction(_ input: MLXArray) throws -> MLXArray {
    try callAsFunction(input, positions: nil, observe: { _, _ in })
  }

  func callAsFunction(_ input: MLXArray,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    try callAsFunction(input, positions: nil, observe: observe)
  }

  func callAsFunction(_ input: MLXArray, positions: [[Int32]]?,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard input.ndim == 2, (1...1024).contains(input.shape[0]),
      input.shape[1] == 5120, input.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("H3 Qwen layer expects 1–1024 BF16 rows of width 5120.")
    }
    try Task.checkCancellation()
    let n = input.shape[0]
    let normalized = MLXFast.rmsNorm(input, weight: inputNorm, eps: 1e-6)
    try observe("normalized", normalized)
    var q = try query.project(normalized).reshaped([1, n, 64, 128])
    try observe("qproj", q)
    var k = try key.project(normalized).reshaped([1, n, 8, 128])
    try observe("kproj", k)
    let rawV = try value.project(normalized).reshaped([1, n, 8, 128])
    try observe("vproj", rawV)
    let v = rawV
      .transposed(0, 2, 1, 3)
    q = MLXFast.rmsNorm(q, weight: queryNorm, eps: 1e-6)
      .transposed(0, 2, 1, 3)
    try observe("qnorm", q)
    k = MLXFast.rmsNorm(k, weight: keyNorm, eps: 1e-6)
      .transposed(0, 2, 1, 3)
    try observe("knorm", k)
    // The released Qwen transformer's Float32 inverse-frequency table. MLX
    // power implementations differ by one ULP between installed runtimes;
    // that becomes visible after BF16 rotation over long mixed sequences.
    let frequencyBits: [UInt32] = [
      0x3f800000, 0x3f492c28, 0x3f1e165e, 0x3ef875a5, 0x3ec33f3a, 0x3e996e51, 0x3e712429, 0x3e3d7efc,
      0x3e14e962, 0x3dea09d9, 0x3db7ea1a, 0x3d908687, 0x3d63251b, 0x3d327f4f, 0x3d0c44be, 0x3cdc7456,
      0x3cad3d5d, 0x3c88230f, 0x3c55f605, 0x3c282310, 0x3c042087, 0x3bcfa8a8, 0x3ba32f3e, 0x3b803c3c,
      0x3b498ad2, 0x3b1e60c3, 0x3af8ea93, 0x3ac39b1b, 0x3a99b686, 0x3a7195a5, 0x3a3dd828, 0x3a152f76,
      0x39ea77fd, 0x39b840a7, 0x3990ca8b, 0x39638ffe, 0x3932d34f, 0x390c86c1, 0x38dcdc14, 0x38ad8ee4,
      0x38886320, 0x38565ab5, 0x38287230, 0x38045eb5, 0x37d00a62, 0x37a37c09, 0x37807896, 0x3749e9ab,
      0x371eab4a, 0x36f95fb6, 0x36c3f72a, 0x3699fedc, 0x36720755, 0x363e317f, 0x361575ab, 0x35eae653,
      0x35b8975c, 0x35910eae, 0x3563fb16, 0x35332776, 0x350cc8e3, 0x34dd4405, 0x34ade091, 0x3488a34f,
    ]
    let frequencies = frequencyBits.map(Float.init(bitPattern:))
    if let positions {
      guard positions.count == 3,
        positions.allSatisfy({ $0.count == n && $0.allSatisfy({ (0...16_384).contains($0) }) }) else {
        throw H3CheckpointError.invalid("H3 Qwen M-RoPE positions differ from the token rows.")
      }
      let positionIDs = MLXArray(positions.flatMap { $0 }, [3, n])
      let selectorValues: [Int32] = (0..<64).map { slot in
        slot < 60 && slot % 3 != 0 ? Int32(slot % 3) : 0
      }
      let selectors = MLXArray(selectorValues)
      let inverse = MLXArray(frequencies)
      q = Self.mixedRotary(q, positions: positionIDs,
        frequencies: inverse, selectors: selectors)
      try observe("qrope", q)
      k = Self.mixedRotary(k, positions: positionIDs,
        frequencies: inverse, selectors: selectors)
      try observe("krope", k)
    } else {
      let positionArray = MLXArray((0..<n).map(Float.init), [1, 1, n, 1])
      let angles = positionArray * MLXArray(frequencies, [1, 1, 1, 64])
      let cosine = MLX.cos(angles)
      let sine = MLX.sin(angles)
      q = Self.rotary(q, cosine: cosine, sine: sine)
      try observe("qrope", q)
      k = Self.rotary(k, cosine: cosine, sine: sine)
      try observe("krope", k)
    }
    let ids = MLXArray((0..<n).map(Int32.init))
    let mask = lessEqual(ids.reshaped([1, n]), ids.reshaped([n, 1]))
    let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k,
      values: v, scale: 1 / Float(128).squareRoot(), mask: mask)
      .transposed(0, 2, 1, 3).reshaped([n, 8192])
    try observe("attended", attended)
    let attention = try output.project(attended)
    try observe("attention", attention)
    let residual = input + attention
    try observe("residual", residual)
    let feedInput = MLXFast.rmsNorm(residual, weight: postAttentionNorm, eps: 1e-6)
    try observe("feed_input", feedInput)
    let gated = try gate.project(feedInput)
    let value = try up.project(feedInput)
    let feed = try down.project((gated * sigmoid(gated)) * value)
    try observe("feed", feed)
    let result = residual + feed
    try observe("output", result)
    return result
  }

  public static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index, input: input,
      positions: nil, observe: { _, _ in })
  }

  public static func evaluate(checkpointURL: URL, index: Int,
    input: MLXArray, positions: [[Int32]]) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index, input: input,
      positions: positions, observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, index: Int, input: MLXArray,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, index: index, input: input,
      positions: nil, observe: observe)
  }

  static func evaluate(checkpointURL: URL, index: Int, input: MLXArray,
    positions: [[Int32]]?,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    try Task.checkCancellation()
    let file = try SafeTensorFile(url: checkpointURL)
    try file.checkUnchanged(at: checkpointURL)
    let page = H3NativePage(file: file, url: checkpointURL)
    defer { page.clear() }
    let layer = try Self(file: file, index: index,
      tensor: { try page.read($0, materialize: false) })
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    let result = try layer.callAsFunction(input, positions: positions,
      observe: observe)
    eval(result)
    Stream.gpu.synchronize()
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return result
  }
}
