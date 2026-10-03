import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// The single-stage, deterministic LTX 2.5 reference schedule used by Ripple
/// and static Ingredients sheets. Both enter as a full-length VAE guide.
public final class MLXSingleStageRipple {
  static func cfgppReserveBytes(geometry:AVGeometry) -> Int {
    (geometry.videoTokens*2+geometry.audioFrames)*128*4*8+1024*6144*4
  }
  public enum AdapterTask:Sendable,Equatable { case ripple, ingredients }
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
    let schedule = try SamplingSchedule(
      sigmas: [1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0], eta: 0)
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

  public init(geometry: AVGeometry, referenceStrength: Float,
    imageAnchors: [RippleImageAnchor] = [], transformerRoot: URL,
    adapters: [LoRAAdapter], task:AdapterTask = .ripple,
    ingredientsSampling:MLXIngredientsSampling = .deterministic,
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
      throw LTXError.invalid("Authored CFG++ requires Ingredients with reference strength 1.")
    }
    let adapterURL = URL(fileURLWithPath: adapters[0].path)
      .resolvingSymlinksInPath().standardizedFileURL
    if task == .ripple { try MLXRippleAdapterIdentity.verify(adapterURL) }
    self.geometry = geometry
    self.task = task
    self.ingredientsSampling=ingredientsSampling
    // Reserve additional Float32 predictions/state and the negative text
    // contexts before admitting any transformer block. This is an owned-array
    // budget, not a prediction of process footprint or Metal allocator peak.
    let cfgReserve=ingredientsSampling == .ancestralCFGPP
      ? Self.cfgppReserveBytes(geometry:geometry) : 0
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
      maximumActivationBytes: blockBudget)
    guard weights.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw LTXError.invalid("Single-stage reference sampling requires the released distilled LTX 2.5 transformer.")
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
    let sampler = try MLXSamplingRunner(configuration: plan.configuration,
      maximumActivationBytes: maximumActivationBytes)
    let inputs: [String: MLXArray] = [
      "video_text": videoContext, "audio_text": audioContext,
      "video_latent": prepared.latent, "audio_latent": audioNoise,
      "video_positions": MLXArray(plan.layout.positions, [plan.layout.videoTokens, 3]),
      "audio_positions": MLXArray(geometry.audioPositions, [geometry.audioFrames, 1])]
    let cfgpp=ingredientsSampling == .ancestralCFGPP
    let unconditional:[String:MLXArray]?
    if cfgpp {
      guard let unconditionalVideoContext,let unconditionalAudioContext else {
        throw LTXError.invalid("CFG++ requires learned unconditional audiovisual text contexts.")
      }
      unconditional=["video_text":unconditionalVideoContext,"audio_text":unconditionalAudioContext]
    } else {
      guard unconditionalVideoContext == nil,unconditionalAudioContext == nil else {
        throw LTXError.invalid("Deterministic sampling does not evaluate unconditional text contexts.")
      }
      unconditional=nil
    }
    let schedule=try cfgpp ? SamplingSchedule(sigmas:plan.schedule.sigmas,eta:1) : plan.schedule
    let sampled = try sampler.evaluate(inputs, schedule: schedule,
      videoConditioning: prepared.condition, bfloat16State: cfgpp ? [] : ["video", "audio"],
      unconditionalContexts:unconditional,
      fixedWeights: weights.readFixed, blockWeights: weights.readBlock,
      fixedAdapters: weights.fixedAdapters, blockAdapters: weights.blockAdapters,
      noise:cfgpp ? { index,name,shape in
        let offset=UInt64(index*2+(name == "audio" ? 1 : 0))
        return MLXNoisePolicy.seeded(seed &+ 10000 &+ offset,tokens:shape[0]).asType(.float32)
      } : nil,
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
