import Foundation
import MLX

/// Stream Qwen3-VL's vision blocks and three deepstack mergers from installed
/// weights, materializing only the current block and its small feature output.
public enum H3QwenVisionEncoder {
  public struct Output {
    public let hidden: MLXArray
    public let deepstack: [MLXArray]
  }

  public static func encode(pixels: MLXArray, grids: [H3QwenRequest.Grid],
    checkpointURL: URL) throws -> Output {
    try encode(pixels: pixels, grids: grids, checkpointURL: checkpointURL,
      observe: { _, _ in })
  }

  static func encode(pixels: MLXArray, grids: [H3QwenRequest.Grid],
    checkpointURL: URL,
    observe: (Int, MLXArray) throws -> Void) throws -> Output {
    guard Device.defaultDevice().deviceType == .gpu else {
      throw H3CheckpointError.invalid("H3 Qwen vision requires an MLX Metal device.")
    }
    try Task.checkCancellation()
    let layout = try H3QwenCheckpointLayout.inspect(root: checkpointURL)
    guard let file = layout.visionFile else {
      throw H3CheckpointError.invalid("The selected H3 Qwen checkpoint has no complete vision tower.")
    }
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      Memory.cacheLimit = previousCacheLimit
    }
    let position = try H3QwenVisionPosition.make(checkpointURL: file, grids: grids)
    guard pixels.ndim == 2, pixels.shape == [position.rotary.shape[0], 1536],
      pixels.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("H3 Qwen vision pixel rows differ from the processor grid.")
    }
    var hidden = try H3QwenVisionPatch.embed(pixels: pixels, checkpointURL: file)
      + position.absolute
    eval(hidden)
    var deepstack: [MLXArray] = []
    deepstack.reserveCapacity(3)
    for index in 0..<27 {
      try Task.checkCancellation()
      hidden = try H3QwenVisionBlock.evaluate(checkpointURL: file, index: index,
        input: hidden, rotary: position.rotary, boundaries: position.boundaries)
      try observe(index, hidden)
      if let deepIndex = [8, 16, 24].firstIndex(of: index) {
        let merged = try H3QwenVisionMerger.evaluate(checkpointURL: file,
          deepIndex: deepIndex, input: hidden)
        deepstack.append(merged)
      }
      if Memory.cacheMemory > 128 * 1024 * 1024 { Memory.clearCache() }
    }
    let merged = try H3QwenVisionMerger.evaluate(checkpointURL: file,
      deepIndex: nil, input: hidden)
    try Task.checkCancellation()
    return Output(hidden: merged, deepstack: deepstack)
  }
}
