import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// The single-stage, deterministic LTX 2.5 reference schedule used by Ripple
/// and static Ingredients sheets. Both enter as a full-length VAE guide.
public final class MLXSingleStageRipple {
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

  public init(geometry: AVGeometry, referenceStrength: Float,
    imageAnchors: [RippleImageAnchor] = [], transformerRoot: URL,
    adapters: [LoRAAdapter], task:AdapterTask = .ripple,
    maximumActivationBytes: Int = 2 * 1024 * 1024 * 1024) throws {
    guard adapters.count == 1, adapters[0].enabled,
      adapters[0].strength.isFinite, adapters[0].strength > 0 else {
      throw LTXError.invalid("Single-stage reference sampling requires exactly one enabled task IC-LoRA.")
    }
    guard task == .ripple || imageAnchors.isEmpty else {
      throw LTXError.invalid("Ingredients cannot combine with timed Ripple image anchors.")
    }
    let adapterURL = URL(fileURLWithPath: adapters[0].path)
      .resolvingSymlinksInPath().standardizedFileURL
    if task == .ripple { try MLXRippleAdapterIdentity.verify(adapterURL) }
    self.geometry = geometry
    self.task = task
    self.maximumActivationBytes = maximumActivationBytes
    plan = try Self.plan(geometry: geometry, strength: referenceStrength,
      anchors: imageAnchors,
      maximumActivationBytes: maximumActivationBytes)
    weights = try MLXDenoiserWeights(root: transformerRoot,
      configuration: plan.configuration,
      adapters: [LoRAAdapter(path: adapterURL.path, strength: adapters[0].strength)],
      ingredientsAdapterPath:task == .ingredients ? adapterURL.path : nil,
      maximumActivationBytes: maximumActivationBytes)
    guard weights.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw LTXError.invalid("Single-stage reference sampling requires the released distilled LTX 2.5 transformer.")
    }
  }

  public func evaluate(videoContext: MLXArray, audioContext: MLXArray,
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
    let sampled = try sampler.evaluate(inputs, schedule: plan.schedule,
      videoConditioning: prepared.condition, bfloat16State: ["video", "audio"],
      fixedWeights: weights.readFixed, blockWeights: weights.readBlock,
      fixedAdapters: weights.fixedAdapters, blockAdapters: weights.blockAdapters,
      stageProgress: { _, event in
        try progress((task == .ripple ? "ripple:" : "ingredients:") + event.stage, event.completedBlocks, 48)
      }, progress: { event in
        try progress("sampling", event.completedSteps, event.totalSteps)
      })
    return ["video": sampled["video"]![0..<geometry.videoTokens],
      "audio": sampled["audio"]!]
  }
}
