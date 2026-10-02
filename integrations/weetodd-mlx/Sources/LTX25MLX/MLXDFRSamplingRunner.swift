import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// Spatial DFR uses the shared streamed denoiser for both stages. The low-pass
/// generated keyframes are carried into stage two; stage-one video is also
/// appended as a clean half-resolution reference. Audio comes from stage one.
public final class MLXDFRSamplingRunner {
  private let recipe: DistilledTwoStageRecipe
  private let low: MLXDFRLayout
  private let high: MLXDFRLayout
  private let weights: [MLXDenoiserWeights]
  private let upscaler: MLXLatentUpscaler
  private let maximumActivationBytes: Int
  private let requiresEndpoints:Bool
  private let gate = NSLock()
  public private(set) var stageSeconds: [String: Double] = [:]

  public init(recipe: DistilledTwoStageRecipe, transformerRoot: URL,
    upscalerCheckpoint: URL, statisticsCheckpoint: URL,
    slotFrames: [Int], detailingAdapter: LoRAAdapter,
    firstStrength:Float?=nil,lastStrength:Float?=nil,
    maximumActivationBytes: Int = 2 * 1024 * 1024 * 1024) throws {
    self.recipe = recipe
    self.maximumActivationBytes = maximumActivationBytes
    self.requiresEndpoints=firstStrength != nil
    low = try MLXDFRLayout(geometry: recipe.low, slotFrames: slotFrames,
      firstStrength:firstStrength,lastStrength:lastStrength)
    high = try MLXDFRLayout(geometry: recipe.high, slotFrames: slotFrames, reference: recipe.low,
      firstStrength:firstStrength,lastStrength:lastStrength)
    for layout in [low, high] {
      let block = try MLXAVBlock(configuration: Self.configuration(layout),
        maximumActivationBytes: maximumActivationBytes)
      if layout.referenceTokens > 0 || firstStrength != nil { try block.admitPerTokenVideo() }
    }
    _ = try MLXLatentUpscaler.admit(shape: [recipe.low.latentFrames,
      recipe.low.latentHeight, recipe.low.latentWidth, 128],
      maximumActivationBytes: maximumActivationBytes)
    _ = try MLXLatentUpscaler.admit(shape: [slotFrames.count,
      recipe.low.latentHeight, recipe.low.latentWidth, 128],
      maximumActivationBytes: maximumActivationBytes)
    weights = try [(low, [LoRAAdapter]()), (high, [detailingAdapter])].map { layout, adapters in
      let source = try MLXDenoiserWeights(root: transformerRoot,
        configuration: Self.configuration(layout), adapters: adapters,
        maximumActivationBytes: maximumActivationBytes, requireKeyframeMarker: true,
        pixelSpatialDFRAdapterPath:adapters.first?.path)
      guard source.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
        throw LTXError.invalid("DFR requires the released distilled transformer checkpoint.")
      }
      return source
    }
    upscaler = try MLXLatentUpscaler(checkpoint: upscalerCheckpoint,
      statisticsCheckpoint: statisticsCheckpoint)
  }

  private static func configuration(_ layout: MLXDFRLayout) throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoTokens: layout.videoTokens,
      audioTokens: layout.geometry.audioFrames, textTokens: 1024)
  }

  public func evaluate(videoContext: MLXArray, audioContext: MLXArray,
    references:[(first:MLXArray,last:MLXArray?)]=[],
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> [String: MLXArray] {
    guard gate.try() else { throw LTXError.invalid("DFR sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    for (value, width) in [(videoContext, 4096), (audioContext, 2048)] {
      guard value.dtype == .float32, value.shape == [1024, width],
        MLX.isFinite(value).all().item(Bool.self) else {
        throw LTXError.invalid("DFR text context differs from the trained layout.")
      }
    }
    guard references.count == (requiresEndpoints ? 2 : 0) else {
      throw LTXError.invalid("DFR endpoint references must be encoded at both resolutions.")
    }
    if !references.isEmpty {
      for (index,pair) in references.enumerated() {
        let layout=index == 0 ? low : high
        let count=layout.geometry.latentHeight*layout.geometry.latentWidth
        guard pair.first.dtype == .float32,pair.first.shape == [count,128],
          pair.last.map({ $0.dtype == .float32 && $0.shape == [count,128] }) ?? true else {
          throw LTXError.invalid("DFR endpoint latent differs from stage geometry.")
        }
      }
    }
    stageSeconds = [:]
    let stageOne = try autoreleasepool { () throws -> (MLXArray, MLXArray, MLXArray) in
      let start = Date()
      let g = recipe.low
      let video = MLXNoisePolicy.seeded(recipe.seed, tokens: g.videoTokens).asType(.float32)
      let audio = MLXNoisePolicy.seeded(recipe.seed &+ 1, tokens: g.audioFrames).asType(.float32)
      let pair=references.first
      let prepared = try low.prepare(generated: video,first:pair?.first,last:pair?.last)
      let sampled = try sample(stage: 1, layout: low, video: prepared.latent,
        audio: audio, condition: prepared.condition,
        videoContext: videoContext, audioContext: audioContext,
        schedule: recipe.first, bfloat16: ["video", "audio"], progress: progress)
      let primary = sampled["video"]![0..<g.videoTokens].reshaped([g.videoTokens, 128])
      let slots = sampled["video"]![low.endpointTokens..<low.videoTokens]
        .reshaped([low.slotTokens, 128])
      let resultAudio = sampled["audio"]!.reshaped([g.audioFrames, 128])
      eval(primary, slots, resultAudio)
      stageSeconds["stage1"] = Date().timeIntervalSince(start)
      try progress("stage1_weights_released", 1, 2)
      return (primary, slots, resultAudio)
    }
    Memory.clearCache()
    let upscaled = try autoreleasepool { () throws -> (MLXArray, MLXArray) in
      let start = Date()
      let main = try upscaler.upscale(stageOne.0.reshaped([recipe.low.latentFrames,
        recipe.low.latentHeight, recipe.low.latentWidth, 128]),
        maximumActivationBytes: maximumActivationBytes,
        progress: { try progress("upscale:" + $0, 0, 1) })
      let slots = try upscaler.upscale(stageOne.1.reshaped([low.slotFrames.count,
        recipe.low.latentHeight, recipe.low.latentWidth, 128]),
        maximumActivationBytes: maximumActivationBytes,
        progress: { try progress("slot_upscale:" + $0, 0, 1) })
      let primary = main.reshaped([recipe.high.videoTokens, 128])
      let seededSlots = slots.reshaped([high.slotTokens, 128])
      eval(primary, seededSlots)
      stageSeconds["latent_upscale_mlx"] = Date().timeIntervalSince(start)
      try progress("upscaler_weights_released", 1, 1)
      return (primary, seededSlots)
    }
    Memory.clearCache()
    let sigma = Float(recipe.second.sigmas[0])
    let videoNoise = MLXNoisePolicy.seeded(recipe.seed &+ 2,
      tokens: recipe.high.videoTokens).asType(.float32)
    let audioNoise = MLXNoisePolicy.seeded(recipe.seed &+ 2,
      tokens: recipe.high.audioFrames).asType(.float32)
    let video = videoNoise * sigma + upscaled.0 * (1 - sigma)
    let audio = audioNoise * sigma + stageOne.2 * (1 - sigma)
    let pair=references.count == 2 ? references[1] : nil
    let prepared = try high.prepare(generated: video, first:pair?.first,last:pair?.last,
      reference: stageOne.0,
      slots: upscaled.1)
    let started = Date()
    let sampled = try sample(stage: 2, layout: high, video: prepared.latent,
      audio: audio, condition: prepared.condition,
      videoContext: videoContext, audioContext: audioContext,
      schedule: recipe.second, bfloat16: ["audio"], progress: progress)
    let primary = sampled["video"]![0..<recipe.high.videoTokens]
      .reshaped([recipe.high.videoTokens, 128])
    eval(primary)
    stageSeconds["stage2"] = Date().timeIntervalSince(started)
    try progress("stage2_weights_released", 2, 2)
    return ["video": primary, "audio": stageOne.2]
  }

  private func sample(stage: Int, layout: MLXDFRLayout, video: MLXArray,
    audio: MLXArray, condition: MLXVideoDenoiseCondition,
    videoContext: MLXArray, audioContext: MLXArray,
    schedule: SamplingSchedule, bfloat16: Set<String>,
    progress: @escaping (String, Int, Int) throws -> Void) throws -> [String: MLXArray] {
    let g = layout.geometry
    let source = weights[stage - 1]
    let runner = try MLXSamplingRunner(configuration: Self.configuration(layout),
      maximumActivationBytes: maximumActivationBytes,
      keyframeMarkerRows: layout.slotTokens)
    let inputs: [String: MLXArray] = [
      "video_latent": video, "audio_latent": audio,
      "video_text": videoContext, "audio_text": audioContext,
      "video_positions": MLXArray(layout.positions, [layout.videoTokens, 3]),
      "audio_positions": MLXArray(g.audioPositions, [g.audioFrames, 1])]
    var key = MLXRandom.key(recipe.seed &+ 10000)
    let sampled = try runner.evaluate(inputs, schedule: schedule,
      videoConditioning: condition, bfloat16State: bfloat16,
      fixedWeights: source.readFixed, blockWeights: source.readBlock,
      fixedAdapters: source.fixedAdapters, blockAdapters: source.blockAdapters,
      noise: { _, _, shape in
        let (next, draw) = MLXRandom.split(key: key)
        key = next
        return MLXRandom.normal([1] + shape, key: draw).reshaped(shape)
      }, stageProgress: { _, event in
        try progress("stage\(stage):" + event.stage, event.completedBlocks, 48)
      }, progress: { event in
        try progress("sampling", (stage == 1 ? 0 : 8) + event.completedSteps, 11)
      })
    return sampled
  }
}
