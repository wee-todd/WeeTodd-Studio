import Foundation
import MLX

/// One weighted denoiser for standard and reference-conditioned H3 layouts.
/// Only packing differs; projection, LoRA, blocks and final heads stay shared.
final class H3WeightedDiTState {
  enum Layout {
    case audiovisual(H3PackedLayout)
    case references(H3ReferenceLayout)

    var tags: [Int32] {
      switch self {
      case .audiovisual(let value): value.tags
      case .references(let value): value.tags
      }
    }
    var textRows: Int {
      switch self {
      case .audiovisual(let value): value.audioStart - value.conditionVideoRows
      case .references(let value):
        value.tags.count - value.videoIndices.count - value.audioIndices.count
      }
    }
    var videoRows: Int {
      switch self {
      case .audiovisual(let value):
        value.conditionVideoRows + value.tags.count - value.videoStart
      case .references(let value): value.videoIndices.count
      }
    }
    var audioRows: Int {
      switch self {
      case .audiovisual(let value): value.videoStart - value.audioStart
      case .references(let value): value.audioIndices.count
      }
    }
    func pack(text: MLXArray, video: MLXArray, audio: MLXArray,
      timestepIndices: [Int32]) throws -> H3PackedSequence {
      switch self {
      case .audiovisual(let value):
        try H3PackedSequence(layout: value, text: text, video: video,
          audio: audio, timestepIndices: timestepIndices)
      case .references(let value):
        try H3PackedSequence(layout: value, text: text, video: video,
          audio: audio, timestepIndices: timestepIndices)
      }
    }
  }
  private let checkpointURL: URL
  private let layout: Layout
  private let blockCount: Int
  private let projectionMode: H3ProjectionMode
  private let lora: H3LoRAFile?
  private var text: MLXArray?
  private var timeEmbeddings: MLXArray?
  private var modulations: [MLXArray]?

  var isResident: Bool {
    text != nil && timeEmbeddings != nil && modulations != nil
  }

  /// Prepared activations held across steps; block weights are streamed and
  /// therefore intentionally absent from this resident count.
  var residentActivationBytes: Int {
    (text?.nbytes ?? 0) + (timeEmbeddings?.nbytes ?? 0)
      + (modulations?.reduce(0) { $0 + $1.nbytes } ?? 0)
  }

  init(checkpointURL: URL, layout: Layout,
    textEmbeddings: MLXArray, timestepTable: [Float],
    blockCount: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    let textRows = layout.textRows
    guard (1...50).contains(blockCount),
      textRows > 0, textEmbeddings.shape == [1, textRows, 5120],
      textEmbeddings.dtype.isFloatingPoint,
      (1...128).contains(timestepTable.count),
      timestepTable.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      zip(timestepTable, timestepTable.dropFirst()).allSatisfy({ $0 < $1 }),
      (1...40_000).contains(layout.tags.count) else {
      throw H3CheckpointError.invalid("Invalid H3 denoiser preparation request.")
    }
    self.checkpointURL = checkpointURL
    self.layout = layout
    self.blockCount = blockCount
    self.projectionMode = projectionMode
    self.lora = try turboLoRAURL.map { try H3LoRAFile(url: $0,
      strength: turboLoRAStrength) }
    self.text = nil
    self.timeEmbeddings = nil
    self.modulations = nil
    try Task.checkCancellation()
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let projected = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .condition, input: textEmbeddings)
    let refined = try H3TokenRefiner.evaluate(checkpointURL: checkpointURL,
      input: projected, lora: lora)
    let time = try H3TimeEmbedding.evaluate(checkpointURL: checkpointURL,
      timesteps: MLXArray(timestepTable))
    var tables: [MLXArray] = []
    tables.reserveCapacity(blockCount)
    for index in 0..<blockCount {
      tables.append(try H3AdaLNProjection.evaluate(
        checkpointURL: checkpointURL, blockIndex: index,
        timeEmbeddings: time, projectionMode: projectionMode))
      progress(index + 1, blockCount)
      try Task.checkCancellation()
    }
    text = refined
    timeEmbeddings = time
    modulations = tables
  }

  public func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void = { _, _ in }) throws
    -> H3FinalLayer.Output {
    guard let text, let timeEmbeddings, let modulations else {
      throw H3CheckpointError.invalid("H3 denoiser state was unloaded.")
    }
    guard videoLatents.shape == [1, layout.videoRows, 96],
      audioLatents.shape == [1, layout.audioRows, 32],
      videoLatents.dtype.isFloatingPoint,
      audioLatents.dtype.isFloatingPoint,
      timestepIndices.count == layout.tags.count,
      timestepIndices.allSatisfy({ (0..<timeEmbeddings.shape[0]).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("Invalid H3 denoiser latent or timestep rows.")
    }
    try Task.checkCancellation()
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let video = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .video, input: videoLatents).asType(.bfloat16)
    let audio = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .audio, input: audioLatents).asType(.bfloat16)
    let packed = try layout.pack(text: text, video: video, audio: audio,
      timestepIndices: timestepIndices)
    let rotaryAngles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: checkpointURL, positions: packed.positions)
    var value = packed.embeddings
    for index in 0..<blockCount {
      value = try H3TransformerBlock.evaluate(checkpointURL: checkpointURL,
        index: index, input: value, modulation: modulations[index],
        modulationIndices: packed.modulationIndices,
        positions: packed.positions, projectionMode: projectionMode,
        lora: lora, rotaryAngles: rotaryAngles,
        observe: { _, _ in })
      progress(index + 1, blockCount)
      try Task.checkCancellation()
    }
    return try H3FinalLayer.evaluate(checkpointURL: checkpointURL,
      input: value, timeEmbeddings: timeEmbeddings,
      timestepIndices: packed.timestepIndices,
      videoIndices: packed.videoIndices,
      audioIndices: packed.audioIndices)
  }

  public func unload() {
    text = nil
    timeEmbeddings = nil
    modulations = nil
    Stream.gpu.synchronize()
    Memory.clearCache()
  }

  deinit { unload() }
}
