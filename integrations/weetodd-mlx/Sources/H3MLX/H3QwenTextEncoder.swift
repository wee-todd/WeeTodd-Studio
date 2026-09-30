import Foundation
import MLX
import TensorIO

/// Text-only H3 conditioner. It streams 50 existing Qwen3-VL Q8 decoder pages
/// and returns hidden_states[50] before the model's final norm. Vision blocks
/// are a separate path and are never silently omitted from a visual request.
public enum H3QwenTextEncoder {
  public struct Output {
    public let hidden: MLXArray
    public let tokenIDs: [Int32]
    public let tags: [Int32]
  }

  public static func encode(prompt: String, checkpointRoot: URL,
    tokenizerURL: URL, progress: (Int, Int) throws -> Void = { _, _ in }) throws -> Output {
    try encodeWithTrace(prompt: prompt, checkpointRoot: checkpointRoot,
      tokenizerURL: tokenizerURL, progress: progress, observe: { _, _ in })
  }

  static func encodeWithTrace(prompt: String, checkpointRoot: URL,
    tokenizerURL: URL, progress: (Int, Int) throws -> Void,
    observe: (Int, MLXArray) throws -> Void) throws -> Output {
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let request = try H3QwenRequest.text(prompt, tokenizer: tokenizer)
    return try encodeTextRequest(request: request, checkpointRoot: checkpointRoot,
      progress: progress, observe: observe)
  }

  private static func encodeTextRequest(request: H3QwenRequest,
    checkpointRoot: URL, progress: (Int, Int) throws -> Void,
    observe: (Int, MLXArray) throws -> Void) throws -> Output {
    guard Device.defaultDevice().deviceType == .gpu else {
      throw H3CheckpointError.invalid("H3 Qwen conditioning requires an MLX Metal device.")
    }
    try Task.checkCancellation()
    let layout = try H3QwenCheckpointLayout.inspect(root: checkpointRoot)
    let ids = request.tokenIDs
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    let fixed = try SafeTensorFile(url: layout.embeddingFile)
    let rowBytes: UInt64 = 5120 * 2
    var rows: [MLXArray] = []
    rows.reserveCapacity(ids.count)
    for id in ids {
      try Task.checkCancellation()
      let lower = UInt64(id) * rowBytes
      let row = try fixed.withTensorBytes(named: "model.embed_tokens.weight",
        range: lower..<(lower + rowBytes)) { bytes in
        MLXArray(bytes, [1, 5120], type: UInt16.self).view(dtype: .bfloat16)
      }
      rows.append(row)
    }
    var hidden = concatenated(rows, axis: 0)
    eval(hidden)
    rows.removeAll()
    try fixed.checkUnchanged(at: layout.embeddingFile)
    try observe(0, hidden)
    try progress(0, 50)
    for index in 0..<50 {
      try Task.checkCancellation()
      hidden = try H3QwenLayer.evaluate(checkpointURL: layout.layerFiles[index],
        index: index, input: hidden)
      try observe(index + 1, hidden)
      if Memory.cacheMemory > 128 * 1024 * 1024 { Memory.clearCache() }
      try progress(index + 1, 50)
    }
    try Task.checkCancellation()
    return Output(hidden: hidden, tokenIDs: ids, tags: request.tags)
  }

  /// Encode FL2VA keyframes that have already been converted to Qwen's BF16
  /// patch rows. The vision tower is staged and released before language pages
  /// are loaded; compact and paged checkpoints may remain in separate installed
  /// locations without copying either model.
  public static func encodeKeyframes(prompt: String, pixels: MLXArray,
    grids: [H3QwenRequest.Grid], checkpointRoot: URL,
    visionCheckpointURL: URL, tokenizerURL: URL,
    progress: (Int, Int) throws -> Void = { _, _ in }) throws -> Output {
    try encodeKeyframesWithTrace(prompt: prompt, pixels: pixels, grids: grids,
      checkpointRoot: checkpointRoot, visionCheckpointURL: visionCheckpointURL,
      tokenizerURL: tokenizerURL, progress: progress, observe: { _, _ in })
  }

  /// Reference media arrives as one ordered Qwen patch tensor. The same
  /// vision tower, MRoPE, deepstack and language pages used by keyframes run
  /// here; Ref2VA differs only in its presentation and video-pad tokens.
  public static func encodeReferences(prompt: String, pixels: MLXArray,
    references: [H3QwenRequest.Reference], checkpointRoot: URL,
    visionCheckpointURL: URL, tokenizerURL: URL,
    progress: (Int, Int) throws -> Void = { _, _ in }) throws -> Output {
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let request = try H3QwenRequest.references(prompt: prompt,
      references: references, tokenizer: tokenizer)
    let grids: [H3QwenRequest.Grid] = references.flatMap { reference in
      switch reference {
      case .image(let grid): return [grid]
      case .video(let blocks, _): return blocks.map(\.grid)
      case .audio: return []
      }
    }
    if request.visualRanges.isEmpty {
      return try encodeTextRequest(request: request,
        checkpointRoot: checkpointRoot, progress: progress,
        observe: { _, _ in })
    }
    return try encodeVisualRequest(request: request, pixels: pixels,
      grids: grids, checkpointRoot: checkpointRoot,
      visionCheckpointURL: visionCheckpointURL,
      progress: progress, observe: { _, _ in })
  }

  static func encodeKeyframesWithTrace(prompt: String, pixels: MLXArray,
    grids: [H3QwenRequest.Grid], checkpointRoot: URL,
    visionCheckpointURL: URL, tokenizerURL: URL,
    progress: (Int, Int) throws -> Void = { _, _ in },
    observe: (Int, MLXArray) throws -> Void) throws -> Output {
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let request = try H3QwenRequest.keyframes(prompt: prompt, grids: grids,
      tokenizer: tokenizer)
    return try encodeVisualRequest(request: request, pixels: pixels,
      grids: grids, checkpointRoot: checkpointRoot,
      visionCheckpointURL: visionCheckpointURL,
      progress: progress, observe: observe)
  }

  private static func encodeVisualRequest(request: H3QwenRequest,
    pixels: MLXArray, grids: [H3QwenRequest.Grid], checkpointRoot: URL,
    visionCheckpointURL: URL,
    progress: (Int, Int) throws -> Void,
    observe: (Int, MLXArray) throws -> Void) throws -> Output {
    guard Device.defaultDevice().deviceType == .gpu else {
      throw H3CheckpointError.invalid("H3 Qwen conditioning requires an MLX Metal device.")
    }
    try Task.checkCancellation()
    let layout = try H3QwenCheckpointLayout.inspect(root: checkpointRoot)
    guard !request.visualRanges.isEmpty else {
      throw H3CheckpointError.invalid("H3 visual conditioning requires an image or video reference.")
    }
    let positions = try H3QwenMRoPE.positions(request: request, grids: grids)
    let vision = try H3QwenVisionEncoder.encode(pixels: pixels, grids: grids,
      checkpointURL: visionCheckpointURL)
    let visualTokens = request.visualRanges.flatMap {
      Array(($0.lowerBound + 1)..<($0.upperBound - 1))
    }
    guard vision.hidden.shape == [visualTokens.count, 5120],
      vision.deepstack.count == 3,
      vision.deepstack.allSatisfy({ $0.shape == [visualTokens.count, 5120] }) else {
      throw H3CheckpointError.invalid("H3 Qwen vision features differ from image-pad rows.")
    }
    let visualIndex = Dictionary(uniqueKeysWithValues:
      visualTokens.enumerated().map { ($0.element, $0.offset) })
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    let fixed = try SafeTensorFile(url: layout.embeddingFile)
    let rowBytes: UInt64 = 5120 * 2
    var rows: [MLXArray] = []
    rows.reserveCapacity(request.tokenIDs.count)
    for (index, id) in request.tokenIDs.enumerated() {
      try Task.checkCancellation()
      if let feature = visualIndex[index] {
        rows.append(vision.hidden[feature..<(feature + 1), 0..<5120])
      } else {
        let lower = UInt64(id) * rowBytes
        let row = try fixed.withTensorBytes(named: "model.embed_tokens.weight",
          range: lower..<(lower + rowBytes)) { bytes in
          MLXArray(bytes, [1, 5120], type: UInt16.self).view(dtype: .bfloat16)
        }
        rows.append(row)
      }
    }
    var hidden = concatenated(rows, axis: 0)
    eval(hidden)
    rows.removeAll()
    try fixed.checkUnchanged(at: layout.embeddingFile)
    try observe(0, hidden)
    try progress(0, 50)
    for index in 0..<50 {
      try Task.checkCancellation()
      hidden = try H3QwenLayer.evaluate(checkpointURL: layout.layerFiles[index],
        index: index, input: hidden, positions: positions)
      if index < 3 {
        var combined: [MLXArray] = []
        combined.reserveCapacity(request.tokenIDs.count)
        for token in 0..<request.tokenIDs.count {
          let row = hidden[token..<(token + 1), 0..<5120]
          if let feature = visualIndex[token] {
            combined.append(row + vision.deepstack[index][feature..<(feature + 1), 0..<5120])
          } else {
            combined.append(row)
          }
        }
        hidden = concatenated(combined, axis: 0)
        eval(hidden)
      }
      try observe(index + 1, hidden)
      if Memory.cacheMemory > 128 * 1024 * 1024 { Memory.clearCache() }
      try progress(index + 1, 50)
    }
    try Task.checkCancellation()
    return Output(hidden: hidden, tokenIDs: request.tokenIDs, tags: request.tags)
  }
}
