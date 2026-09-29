import Foundation
import Metal

/// One unguided model evaluation from packed latents and prepared text embeddings
/// to audiovisual velocity. Text encoding, latent packing, masks and sampling are
/// separate contracts; unsupported controls cannot be passed to this entry point.
public final class DenoiserRunner {
  public struct Output { public let videoVelocity: [Float]; public let audioVelocity: [Float] }
  public struct Progress {
    public let stage: String
    public let completedBlocks: Int
    public let metalAllocatedBytes: UInt64
    public let stackMetrics: ExecutionMetrics?
  }
  private let configuration: AVBlockConfiguration
  private let blockCount: Int
  private let maximumPreparationBytes: UInt64
  private let sequenceAttention: Bool
  private let precision: LTXPrecisionPolicy
  private var active = false
  private var sessionHeads: [UInt32: [String: [[Float]]]]?
  private var sessionRotary: [String: [Float]]?
  private var sessionStack: AVStackRunner?
  var sessionGraphBuildCount: Int { sessionStack?.graphBuildCount ?? 0 }

  func prepareSession(sigmas: [Float], fixedWeights: (String, [Int]) throws -> [Float],
    progress: (String) throws -> Void) throws {
    guard !active, sessionHeads == nil, !sigmas.isEmpty, sigmas.count <= 256 else {
      throw BlockError.invalid("Invalid or already prepared denoising session.")
    }
    active = true; defer { active = false }
    let times = try sigmas.map { [try DenoiserMath.timestep($0)] }
    var cache: [UInt32: [String: [[Float]]]] = [:]
    for head in DenoiserLayout.heads(configuration) {
      let output = try autoreleasepool {
        try FixedStage.adaptive(head.name, dimension: head.dimension, parameters: head.parameters)
          .evaluateMany(times, weights: fixedWeights)
      }
      for (index, sigma) in sigmas.enumerated() { cache[sigma.bitPattern, default: [:]][head.name] = output[index] }
      try progress(head.name)
      try Task.checkCancellation()
    }
    sessionHeads = cache
  }

  func releaseSession() throws {
    guard !active else { throw BlockError.invalid("Cannot release an active denoiser.") }
    try sessionStack?.release()
    sessionStack = nil; sessionHeads = nil; sessionRotary = nil
  }
  public var metalAllocatedBytes: UInt64 { UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0) }
  public var inputShapes: [String: [Int]] { DenoiserLayout.inputShapes(configuration) }
  public init(configuration: AVBlockConfiguration, blockCount: Int = 48, maximumPreparationBytes: UInt64 = 0, sequenceAttention: Bool = false, experimentalPrecision: LTXPrecisionPolicy = .float32) throws {
    try AVBlockRunner.validateAllocation(configuration: configuration)
    guard (1...48).contains(blockCount) else { throw BlockError.invalid("Invalid denoiser block count.") }
    try AVStackRunner.validatePreparationBudget(configuration: configuration, maximumBytes: maximumPreparationBytes)
    self.configuration = configuration; self.blockCount = blockCount
    self.maximumPreparationBytes = maximumPreparationBytes; self.sequenceAttention = sequenceAttention; self.precision = experimentalPrecision
  }

  /// Providers must be preflighted before invocation. Adaptive heads, the block
  /// slot and output heads are separately owned. Standalone calls release all of
  /// them on return; the internal sampling session retains only its one block slot.
  public func evaluate(_ inputs: [String: [Float]], sigma: Float,
    fixedWeights: (String, [Int]) throws -> [Float],
    blockWeights: @escaping (Int, String, [Int]) throws -> [Float],
    progress: (Progress) throws -> Void = { _ in }) throws -> Output {
    guard !active else { throw BlockError.invalid("Denoiser is already evaluating.") }
    active = true; defer { active = false }
    if let sessionHeads, sessionHeads[sigma.bitPattern] == nil {
      throw BlockError.invalid("Sigma was not admitted by the denoising session.")
    }
    let c = configuration, shapes = inputShapes
    guard Set(inputs.keys) == Set(shapes.keys), shapes.allSatisfy({ name, shape in
      inputs[name]!.count == shape.reduce(1, *) && inputs[name]!.allSatisfy(\.isFinite)
    }) else { throw BlockError.invalid("Invalid packed-latent, text or position inputs.") }
    let time = try DenoiserMath.timestep(sigma)
    var prepared: [String: [Float]] = sessionRotary ?? [:], embedded: [String: [Float]] = [:]
    func report(_ stage: String, _ blocks: Int = 0, _ metrics: ExecutionMetrics? = nil) throws {
      try Task.checkCancellation()
      try progress(Progress(stage: stage, completedBlocks: blocks,
        metalAllocatedBytes: UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0), stackMetrics: metrics))
      try Task.checkCancellation()
    }
    if sessionRotary == nil {
    let videoPositions = inputs["video_positions"]!, audioPositions = inputs["audio_positions"]!
    let videoTemporal = stride(from: 0, to: videoPositions.count, by: 3).map { videoPositions[$0] }
    for (name, positions, axes, tokens, width, maximum): (String, [Float], Int, Int, Int, [Float]) in [
      ("video", videoPositions, 3, c.videoTokens, c.videoHeadDimension, [20, 2048, 2048]),
      ("audio", audioPositions, 1, c.audioTokens, c.audioHeadDimension, [20]),
      ("video_cross", videoTemporal, 1, c.videoTokens, c.audioHeadDimension, [20]),
      ("audio_cross", audioPositions, 1, c.audioTokens, c.audioHeadDimension, [20])] {
      let rotary = try DenoiserMath.rotary(positions: positions, axes: axes, tokens: tokens,
        heads: c.heads, headWidth: width, maximumPositions: maximum)
      prepared[name + "_rope_cos"] = rotary.cos; prepared[name + "_rope_sin"] = rotary.sin
    }
    if sessionHeads != nil { sessionRotary = prepared }
    }
    for head in DenoiserLayout.heads(c) {
      let values: [[Float]]
      if let cached = sessionHeads?[sigma.bitPattern]?[head.name] { values = cached }
      else {
        values = try autoreleasepool {
          try FixedStage.adaptive(head.name, dimension: head.dimension, parameters: head.parameters)
            .evaluate([time], weights: fixedWeights)
        }
      }
      prepared[head.input] = values[0]
      if head.name == "adaln_single" { embedded["video"] = values[1] }
      if head.name == "audio_adaln_single" { embedded["audio"] = values[1] }
      try report(head.name)
    }
    for (name, prefix, rows, dim) in [("video", "", c.videoTokens, c.videoDimension),
      ("audio", "audio_", c.audioTokens, c.audioDimension)] {
      prepared[name] = try autoreleasepool {
        try FixedStage.projection(prefix + "patchify_proj", rows: rows, width: 128, output: dim)
          .evaluate([inputs[name + "_latent"]!.map(DenoiserMath.bfloat16)], weights: fixedWeights)[0]
      }
      prepared[name + "_text"] = sessionHeads == nil
        ? inputs[name + "_text"]!.map(DenoiserMath.bfloat16) : inputs[name + "_text"]!
      try report(prefix + "patchify_proj")
    }
    let hidden = try autoreleasepool {
      let stack: AVStackRunner
      if let existing = sessionStack { stack = existing }
      else {
        stack = try AVStackRunner(configuration: c, blockCount: blockCount, sequenceAttention: sequenceAttention, experimentalPrecision: precision)
        if sessionHeads != nil { sessionStack = stack }
      }
      return try stack.evaluate(prepared, retainWeights: sessionHeads != nil, maximumPreparationBytes: maximumPreparationBytes, weights: blockWeights) { try report("transformer", $0.completedBlocks, $0.metrics) }
    }
    try report(sessionHeads == nil ? "transformer_released" : "transformer_retained", blockCount)
    var outputs: [String: [Float]] = [:]
    for (name, prefix, rows, dim, value) in [("video", "", c.videoTokens, c.videoDimension, hidden.video),
      ("audio", "audio_", c.audioTokens, c.audioDimension, hidden.audio)] {
      outputs[name] = try autoreleasepool {
        try FixedStage.projection(prefix + "proj_out", rows: rows, width: dim, output: 128,
          table: prefix + "scale_shift_table").evaluate([value, embedded[name]!], weights: fixedWeights)[0]
      }
      try report(prefix + "proj_out", blockCount)
    }
    return Output(videoVelocity: outputs["video"]!, audioVelocity: outputs["audio"]!)
  }
}
