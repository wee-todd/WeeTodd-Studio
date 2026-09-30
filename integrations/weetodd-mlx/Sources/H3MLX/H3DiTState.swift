import Foundation
import MLX

/// Process-local H3 denoiser state for ordinary synchronized AV generation.
public final class H3DiTState {
  public let layout: H3PackedLayout
  private let core: H3WeightedDiTState

  public var isResident: Bool { core.isResident }
  public var residentActivationBytes: Int { core.residentActivationBytes }

  public convenience init(checkpointURL: URL, layout: H3PackedLayout,
    textEmbeddings: MLXArray, timestepTable: [Float],
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    try self.init(checkpointURL: checkpointURL, layout: layout,
      textEmbeddings: textEmbeddings, timestepTable: timestepTable,
      blockCount: 50, projectionMode: projectionMode,
      turboLoRAURL: turboLoRAURL, turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs,
      progress: progress)
  }

  init(checkpointURL: URL, layout: H3PackedLayout,
    textEmbeddings: MLXArray, timestepTable: [Float], blockCount: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    self.layout = layout
    core = try H3WeightedDiTState(checkpointURL: checkpointURL,
      layout: .audiovisual(layout), textEmbeddings: textEmbeddings,
      timestepTable: timestepTable, blockCount: blockCount,
      projectionMode: projectionMode, turboLoRAURL: turboLoRAURL,
      turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs, progress: progress)
  }

  public func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void = { _, _ in }) throws
    -> H3FinalLayer.Output {
    try core.predict(videoLatents: videoLatents, audioLatents: audioLatents,
      timestepIndices: timestepIndices, progress: progress)
  }

  public func unload() { core.unload() }
}

/// Reference-conditioned H3 shares the same weighted transformer stage.
public final class H3ReferenceDiTState: H3ReferenceVelocityPredictor {
  public let layout: H3ReferenceLayout
  private let core: H3WeightedDiTState

  public var isResident: Bool { core.isResident }
  public var residentActivationBytes: Int { core.residentActivationBytes }

  public convenience init(checkpointURL: URL, layout: H3ReferenceLayout,
    textEmbeddings: MLXArray, timestepTable: [Float],
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    try self.init(checkpointURL: checkpointURL, layout: layout,
      textEmbeddings: textEmbeddings, timestepTable: timestepTable,
      blockCount: 50, projectionMode: projectionMode,
      turboLoRAURL: turboLoRAURL, turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs,
      progress: progress)
  }

  init(checkpointURL: URL, layout: H3ReferenceLayout,
    textEmbeddings: MLXArray, timestepTable: [Float], blockCount: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    self.layout = layout
    core = try H3WeightedDiTState(checkpointURL: checkpointURL,
      layout: .references(layout), textEmbeddings: textEmbeddings,
      timestepTable: timestepTable, blockCount: blockCount,
      projectionMode: projectionMode, turboLoRAURL: turboLoRAURL,
      turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs, progress: progress)
  }

  public func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void = { _, _ in }) throws
    -> H3FinalLayer.Output {
    try core.predict(videoLatents: videoLatents, audioLatents: audioLatents,
      timestepIndices: timestepIndices, progress: progress)
  }

  public func unload() { core.unload() }
}
