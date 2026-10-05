import Foundation
import MLX

/// One weighted denoiser for standard and reference-conditioned H3 layouts.
/// Only packing differs; projection, LoRA, blocks and final heads stay shared.
final class H3WeightedDiTState {
  enum Layout {
    case audiovisual(H3PackedLayout)
    case references(H3ReferenceLayout)

    var maximumPackedRows: Int {
      switch self {
      case .audiovisual(let value): value.maximumPackedRows
      case .references(let value): value.maximumPackedRows
      }
    }
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
  private let lora: H3LoRAStack?
  private var text: MLXArray?
  private var timeEmbeddings: MLXArray?
  private var modulations: [MLXArray]?
  private var modulationLoRAInputs: MLXArray?
  private var funControl: H3FunControlState?
  private let vdn:H3VDNRuntime?

  var isResident: Bool {
    text != nil && timeEmbeddings != nil && modulations != nil
  }

  /// Prepared activations held across steps; block weights are streamed and
  /// therefore intentionally absent from this resident count.
  var residentActivationBytes: Int {
    (text?.nbytes ?? 0) + (timeEmbeddings?.nbytes ?? 0)
      + (modulations?.reduce(0) { $0 + $1.nbytes } ?? 0)
      + (funControl?.residentActivationBytes ?? 0) + (modulationLoRAInputs?.nbytes ?? 0)
  }

  init(checkpointURL: URL, layout: Layout,
    textEmbeddings: MLXArray, timestepTable: [Float],
    blockCount: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    loRAAdapters: [H3LoRAAdapter]? = nil,
    funControl: H3FunControlCondition? = nil,
    vdn:H3VDNSelection? = nil,
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    let textRows = layout.textRows
    guard (1...50).contains(blockCount),
      textRows > 0, textEmbeddings.shape == [1, textRows, 5120],
      textEmbeddings.dtype.isFloatingPoint,
      (1...128).contains(timestepTable.count),
      timestepTable.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      zip(timestepTable, timestepTable.dropFirst()).allSatisfy({ $0 < $1 }),
      [40_000, 64_000].contains(layout.maximumPackedRows),
      (1...layout.maximumPackedRows).contains(layout.tags.count) else {
      throw H3CheckpointError.invalid("Invalid H3 denoiser preparation request.")
    }
    if let funControl {
      guard blockCount == 50, case .audiovisual(let packed) = layout,
        packed.conditionVideoRows == 0,
        funControl.guideRows.shape == [1, layout.videoRows, 96],
        funControl.strength.isFinite, (0...1).contains(funControl.strength) else {
        throw H3CheckpointError.invalid("H3 Fun control requires a compatible dense T2VA transformer and matching guide rows.")
      }
      _ = try H3FunControlLayout(url: funControl.checkpoint,
        base: H3CheckpointLayout(url: checkpointURL))
    }
    self.checkpointURL = checkpointURL
    self.layout = layout
    self.blockCount = blockCount
    self.projectionMode = projectionMode
    var adapters = additionalLoRAs
    if let turboLoRAURL {
      adapters.insert(try H3LoRAAdapter(url: turboLoRAURL,
        strength: turboLoRAStrength), at: 0)
    }
    let effective = loRAAdapters ?? adapters
    if let vdn {
      guard blockCount == 50,funControl == nil,effective.isEmpty,
        case .audiovisual(let packed)=layout,
        (try H3CheckpointLayout(url:checkpointURL).curveRank == nil ? vdn.adalnInputGrid == nil :
          (vdn.variant == .fiftyStep || vdn.adalnInputGrid != nil)) else {
        throw H3CheckpointError.invalid("VDN denoising requires the complete T2VA backbone and correct original-width adapter coordinates without other adapters or controls.")
      }
      self.vdn=try H3VDNRuntime(selection:vdn,packed:packed)
    } else { self.vdn=nil }
    self.lora = try effective.isEmpty ? nil : H3LoRAStack(adapters: effective)
    let application: (any H3LoRAApplying)?
    if let vdn = self.vdn { application = vdn.lora } else { application = self.lora }
    self.text = nil
    self.timeEmbeddings = nil
    self.modulations = nil
    self.modulationLoRAInputs = nil
    try Task.checkCancellation()
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let projected = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .condition, input: textEmbeddings)
    let refined = try H3TokenRefiner.evaluate(checkpointURL: checkpointURL,
      input: projected, lora: application)
    let time = try H3TimeEmbedding.evaluate(checkpointURL: checkpointURL,
      timesteps: MLXArray(timestepTable))
    let loraTime = try vdn?.adalnInputGrid.map {
      try H3VDNInputGrid(url:$0).evaluate(timesteps:timestepTable)
    }
    var tables: [MLXArray] = []
    tables.reserveCapacity(blockCount)
    for index in 0..<blockCount {
      tables.append(try H3AdaLNProjection.evaluate(
        checkpointURL: checkpointURL, blockIndex: index,
        timeEmbeddings: time, projectionMode: projectionMode,lora:application,loraInput:loraTime))
      if index + 1 < blockCount { progress(index + 1, blockCount) }
      try Task.checkCancellation()
    }
    text = refined
    timeEmbeddings = time
    modulations = tables
    modulationLoRAInputs = loraTime
    if let funControl {
      self.funControl = try H3FunControlState(condition: funControl,
        timeEmbeddings: time, base: H3CheckpointLayout(url: checkpointURL))
    }
    progress(blockCount, blockCount)
  }

  private var completedEvaluations = 0

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
    lora?.evaluation = completedEvaluations
    defer { lora?.evaluation = nil }
    let application: (any H3LoRAApplying)?
    if let vdn { application = vdn.lora } else { application = lora }
    let video = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .video, input: videoLatents).asType(.bfloat16)
    let audio = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .audio, input: audioLatents).asType(.bfloat16)
    let packed = try layout.pack(text: text, video: video, audio: audio,
      timestepIndices: timestepIndices)
    let rotaryAngles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: checkpointURL, positions: packed.positions, maximumRows: layout.maximumPackedRows)
    let control = try funControl?.initialize(hidden: packed.embeddings,
      targetIndices: packed.videoIndices)
    let controlBlock: ((Int, MLXArray) throws -> (MLXArray, MLXArray))? = funControl.map { state in
      { index, current in
        try state.step(index: index, control: current,
          modulationIndices: packed.modulationIndices,
          angles: rotaryAngles, audioIndices: packed.audioIndices)
      }
    }
    let value = try H3FunControlMath.runBlocks(input: packed.embeddings,
      blockCount: blockCount, control: control,
      injectionLayers: funControl?.injectionLayers ?? H3FunControlLayout.v1InjectionLayers,
      baseBlock: { index, input in
        try H3TransformerBlock.evaluate(checkpointURL: checkpointURL,
          index: index, input: input, modulation: modulations[index],
          modulationIndices: packed.modulationIndices,
          positions: packed.positions, projectionMode: projectionMode,
          lora: application, rotaryAngles: rotaryAngles, maximumRows: layout.maximumPackedRows,vdn:vdn,observe: { _, _ in })
      }, controlBlock: controlBlock, progress: progress)
    let result = try H3FinalLayer.evaluate(checkpointURL: checkpointURL,
      input: value, timeEmbeddings: timeEmbeddings,
      timestepIndices: packed.timestepIndices,
      videoIndices: packed.videoIndices,
      audioIndices: packed.audioIndices, maximumRows: layout.maximumPackedRows,lora:application,loraInput:modulationLoRAInputs,observe: { _, _ in })
    completedEvaluations += 1
    return result
  }

  public func unload() {
    text = nil
    timeEmbeddings = nil
    modulations = nil
    modulationLoRAInputs = nil
    funControl?.unload()
    funControl = nil
    Stream.gpu.synchronize()
    Memory.clearCache()
  }

  deinit { unload() }
}
