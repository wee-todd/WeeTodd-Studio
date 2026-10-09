import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// Shared single-stage LTX 2.5 reference sampling for Ripple and Ingredients.
/// Both enter as a full-length VAE guide; ancestral Ingredients is explicit.
public final class MLXSingleStageRipple {
  static func schedule(sampling:MLXIngredientsSampling) throws -> SamplingSchedule {
    try SamplingSchedule(sigmas:[1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0],
      eta:sampling == .deterministic ? 0 : 1)
  }
  static func ancestralNoise(seed:UInt64) -> MLXSamplingRunner.Noise {
    var key=MLXRandom.key(seed &+ 10000)
    return { _,_,shape in
      let (next,draw)=MLXRandom.split(key:key);key=next
      return MLXRandom.normal([1]+shape,key:draw).reshaped(shape)
    }
  }
  static func cfgppReserveBytes(geometry:AVGeometry) -> Int {
    (geometry.videoTokens*2+geometry.audioFrames)*128*4*8+1024*6144*4
  }
  public enum AdapterTask:Sendable,Equatable { case ripple, ingredients }
  static func leadingMarkerRows(geometry:AVGeometry,task:AdapterTask,
    sampling:MLXIngredientsSampling) -> Int {
    task == .ingredients && sampling == .ancestralCFGPP
      ? geometry.latentHeight*geometry.latentWidth : 0
  }
  static func guidedSchedule(geometry:AVGeometry,sampling:MLXGuidedSampling) throws -> SamplingSchedule {
    // Shift by the canvas being generated. Frozen reference rows enlarge
    // transformer admission, but do not enlarge the noisy target latent.
    try sampling.schedule(videoTokens:geometry.videoTokens)
  }
  public struct Plan: Sendable {
    public let layout: MLXReferenceVideoLayout
    public let configuration: AVBlockConfiguration
    public let schedule: SamplingSchedule
  }

  public static func plan(geometry: AVGeometry, strength: Float,
    anchors: [RippleImageAnchor] = [],
    maximumActivationBytes: Int) throws -> Plan {
    let layout = try MLXReferenceVideoLayout(geometry: geometry, strength: strength,
      anchors: anchors)
    let configuration = try AVBlockConfiguration(videoTokens: layout.videoTokens,
      audioTokens: geometry.audioFrames, textTokens: 1024)
    let schedule = try Self.schedule(sampling:.deterministic)
    let block = try MLXAVBlock(configuration: configuration,
      maximumActivationBytes: maximumActivationBytes)
    try block.admitPerTokenVideo()
    _ = try MLXDenoiser.admitRotary(configuration: configuration,
      maximumActivationBytes: maximumActivationBytes)
    return Plan(layout: layout, configuration: configuration, schedule: schedule)
  }

  private let geometry: AVGeometry
  private let plan: Plan
  private let weights: MLXDenoiserWeights
  private let maximumActivationBytes: Int
  private let gate = NSLock()
  private let task:AdapterTask
  private let ingredientsSampling:MLXIngredientsSampling
  private let guidedSampling:MLXGuidedSampling?

  public init(geometry: AVGeometry, referenceStrength: Float,
    imageAnchors: [RippleImageAnchor] = [], transformerRoot: URL,
    adapters: [LoRAAdapter], task:AdapterTask = .ripple,
    ingredientsSampling:MLXIngredientsSampling = .deterministic,
    guidedSampling:MLXGuidedSampling? = nil,
    maximumActivationBytes: Int = 2 * 1024 * 1024 * 1024) throws {
    guard adapters.count == 1, adapters[0].enabled,
      adapters[0].strength.isFinite, adapters[0].strength > 0 else {
      throw LTXError.invalid("Single-stage reference sampling requires exactly one enabled task IC-LoRA.")
    }
    guard task == .ripple || imageAnchors.isEmpty else {
      throw LTXError.invalid("Ingredients cannot combine with timed Ripple image anchors.")
    }
    guard ingredientsSampling == .deterministic ||
      (task == .ingredients && referenceStrength == 1) else {
      throw LTXError.invalid("Authored ancestral sampling requires Ingredients with reference strength 1.")
    }
    let adapterURL = URL(fileURLWithPath: adapters[0].path)
      .resolvingSymlinksInPath().standardizedFileURL
    if task == .ripple { try MLXRippleAdapterIdentity.verify(adapterURL) }
    self.geometry = geometry
    self.task = task
    self.ingredientsSampling=ingredientsSampling
    self.guidedSampling=guidedSampling
    guard guidedSampling == nil || (task == .ingredients && referenceStrength == 1 &&
      imageAnchors.isEmpty && ingredientsSampling == .deterministic &&
      guidedSampling?.singleStage == true && guidedSampling?.mode == .guided) else {
      throw LTXError.invalid("Dev reference guidance requires single-stage Ingredients with a frozen full-strength sheet.")
    }
    // Reserve additional Float32 predictions/state and the negative text
    // contexts before admitting any transformer block. This is an owned-array
    // budget, not a prediction of process footprint or Metal allocator peak.
    let cfgReserve=ingredientsSampling == .ancestralCFGPP
      ? Self.cfgppReserveBytes(geometry:geometry) : guidedSampling == nil ? 0 :
        MLXGuidedSampling.reserveBytes(videoTokens:geometry.videoTokens*2,audioTokens:geometry.audioFrames)
    guard cfgReserve < maximumActivationBytes else {
      throw LTXError.invalid("CFG++ exceeds the configured activation budget before model loading.")
    }
    let blockBudget=maximumActivationBytes-cfgReserve
    self.maximumActivationBytes = blockBudget
    plan = try Self.plan(geometry: geometry, strength: referenceStrength,
      anchors: imageAnchors,
      maximumActivationBytes: blockBudget)
    weights = try MLXDenoiserWeights(root: transformerRoot,
      configuration: plan.configuration,
      adapters: [LoRAAdapter(path: adapterURL.path, strength: adapters[0].strength)],
      ingredientsAdapterPath:task == .ingredients ? adapterURL.path : nil,
      maximumActivationBytes: blockBudget,requireKeyframeMarker:Self.leadingMarkerRows(geometry:geometry,task:task,
        sampling:ingredientsSampling)>0 || guidedSampling != nil)
    let expected=guidedSampling == nil ? "ltx-2.5-22b-distilled-transformer-bf16.safetensors" : "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    guard weights.sourceCheckpoint == expected else {
      throw LTXError.invalid("Single-stage reference sampling requires matching released Dev/distilled LTX 2.5 transformer provenance.")
    }
  }

  public func evaluate(videoContext: MLXArray, audioContext: MLXArray,
    unconditionalVideoContext:MLXArray?=nil,unconditionalAudioContext:MLXArray?=nil,
    referenceVideo: MLXArray, imageAnchors: [MLXArray] = [], seed: UInt64,
    progress: (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> [String: MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Ripple sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    for (context, width) in [(videoContext, 4096), (audioContext, 2048)] {
      guard context.dtype == .float32, context.shape == [1024, width],
        MLX.isFinite(context).all().item(Bool.self) else {
        throw LTXError.invalid("Ripple text context shape, dtype or values are invalid.")
      }
    }
    let videoNoise = MLXNoisePolicy.seeded(seed, tokens: geometry.videoTokens).asType(.float32)
    let audioNoise = MLXNoisePolicy.seeded(seed &+ 1, tokens: geometry.audioFrames).asType(.float32)
    let prepared = try plan.layout.prepare(generated: videoNoise, reference: referenceVideo,
      anchors: imageAnchors)
    let inputs: [String: MLXArray] = [
      "video_text": videoContext, "audio_text": audioContext,
      "video_latent": prepared.latent, "audio_latent": audioNoise,
      "video_positions": MLXArray(plan.layout.positions, [plan.layout.videoTokens, 3]),
      "audio_positions": MLXArray(geometry.audioPositions, [geometry.audioFrames, 1])]
    let cfgpp=ingredientsSampling == .ancestralCFGPP
    let unconditional:[String:MLXArray]?
    if cfgpp || guidedSampling != nil {
      guard let unconditionalVideoContext,let unconditionalAudioContext else {
        throw LTXError.invalid("CFG++ requires learned unconditional audiovisual text contexts.")
      }
      unconditional=["video_text":unconditionalVideoContext,"audio_text":unconditionalAudioContext]
    } else {
      guard unconditionalVideoContext == nil,unconditionalAudioContext == nil else {
        throw LTXError.invalid("Positive-only reference sampling does not evaluate unconditional text contexts.")
      }
      unconditional=nil
    }
    if let guidedSampling {
      // Use the same serial guided denoiser as ordinary Dev jobs, without
      // adding another transformer sampler or a refinement model.
      let reserve=MLXGuidedSampling.reserveBytes(videoTokens:plan.configuration.videoTokens,audioTokens:geometry.audioFrames)
      let sampler=try MLXGuidedSamplingRunner(configuration:plan.configuration,sampling:guidedSampling,
        maximumActivationBytes:maximumActivationBytes+reserve,
        leadingKeyframeMarkerRows:geometry.latentHeight*geometry.latentWidth)
      let sampled=try sampler.evaluate(inputs,schedule:Self.guidedSchedule(geometry:geometry,sampling:guidedSampling),
        negativeContexts:unconditional!,videoConditioning:prepared.condition,frozenAudio:false,
        fixedWeights:weights.readFixed,blockWeights:weights.readBlock,
        fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,
        stageProgress:{ _,event in try progress("ingredients:"+event.stage,event.completedBlocks,48) },
        progress:{ event in try progress("sampling",event.completedSteps,event.totalSteps) })
      return ["video":sampled["video"]![0..<geometry.videoTokens],"audio":sampled["audio"]!]
    }
    let sampler = try MLXSamplingRunner(configuration: plan.configuration,
      maximumActivationBytes: maximumActivationBytes,
      leadingKeyframeMarkerRows:Self.leadingMarkerRows(geometry:geometry,task:task,sampling:ingredientsSampling))
    let schedule=try Self.schedule(sampling:ingredientsSampling)
    let noise:MLXSamplingRunner.Noise?
    if cfgpp {
      noise={ index,name,shape in
        let offset=UInt64(index*2+(name == "audio" ? 1 : 0))
        return MLXNoisePolicy.seeded(seed &+ 10000 &+ offset,tokens:shape[0]).asType(.float32)
      }
    } else { noise=ingredientsSampling == .ancestral ? Self.ancestralNoise(seed:seed) : nil }
    let sampled = try sampler.evaluate(inputs, schedule: schedule,
      videoConditioning: prepared.condition, bfloat16State: cfgpp ? [] : ["video", "audio"],
      unconditionalContexts:unconditional,
      fixedWeights: weights.readFixed, blockWeights: weights.readBlock,
      fixedAdapters: weights.fixedAdapters, blockAdapters: weights.blockAdapters,
      noise:noise,
      stageProgress: { evaluation, event in
        let branch=cfgpp && evaluation > 0 ? (evaluation%2 == 0 ? "unconditional:" : "conditional:") : ""
        try progress((task == .ripple ? "ripple:" : "ingredients:") + branch + event.stage, event.completedBlocks, 48)
      }, progress: { event in
        try progress("sampling", event.completedSteps, event.totalSteps)
      })
    return ["video": sampled["video"]![0..<geometry.videoTokens],
      "audio": sampled["audio"]!]
  }
}
