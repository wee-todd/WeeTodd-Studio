import Foundation
import MLX
import LTX25Engine

/// Shared Euler coefficients, MLX-resident state and explicit per-step noise.
/// Each call owns its conditioning cache; no model state survives return, failure
/// or cancellation. Preview hooks receive latents after all weights are released.
public final class MLXSamplingRunner {
  public struct Progress {
    public let completedSteps:Int
    public let totalSteps:Int
    public let sigma:Double
    public let nextSigma:Double
  }
  public typealias Noise = (Int,String,[Int]) throws -> MLXArray
  private let denoiser:MLXDenoiser
  private var active=false
  public var residentWeightBytes:Int { denoiser.residentWeightBytes }
  public init(configuration:AVBlockConfiguration,blockCount:Int=48,cacheBytes:Int=128*1024*1024,
    maximumActivationBytes:Int=2*1024*1024*1024,videoAttentionGroups:[Int]=[]) throws {
    denoiser=try MLXDenoiser(configuration:configuration,blockCount:blockCount,cacheBytes:cacheBytes,
      maximumActivationBytes:maximumActivationBytes,videoAttentionGroups:videoAttentionGroups)
  }
  func preflightWithoutText(_ inputs:[String:MLXArray],schedule:SamplingSchedule) throws -> [String:MLXArray] {
    try denoiser.preflightWithoutText(inputs,schedule:schedule)
  }
  public func evaluate(_ inputs:[String:MLXArray],schedule:SamplingSchedule,
    videoConditioning:MLXVideoDenoiseCondition?=nil,audioConditioning:MLXAudioDenoiseCondition?=nil,
    frozenAudio:Bool=false,bfloat16State:Set<String>=[],
    fixedWeights:MLXDenoiser.FixedProvider,blockWeights:MLXDenoiser.BlockProvider,
    fixedAdapters:MLXDenoiser.FixedAdapters = { _ in [] },
    blockAdapters:(Int) throws -> [String:[MLXLoRA]] = { _ in [:] },
    noise:Noise?=nil,
    stageProgress:(Int,MLXDenoiser.Progress) throws -> Void = { _,_ in },
    progress:(Progress) throws -> Void = { _ in },
    preview:(([String:MLXArray],Progress) throws -> Void)?=nil) throws -> [String:MLXArray] {
    guard !active else { throw LTXError.invalid("MLX sampler already executing.") }
    guard noise != nil || !schedule.steps.contains(where:\.ancestral) else {
      throw LTXError.invalid("Ancestral sampling requires explicit noise before loading weights.")
    }
    guard videoConditioning == nil || videoConditioning!.clean.shape == denoiser.inputShapes["video_latent"] else {
      throw LTXError.invalid("Reference conditioning differs from the admitted video shape.")
    }
    guard audioConditioning == nil || audioConditioning!.clean.shape == denoiser.inputShapes["audio_latent"] else {
      throw LTXError.invalid("Reference conditioning differs from the admitted audio shape.")
    }
    guard audioConditioning == nil || !frozenAudio else {
      throw LTXError.invalid("Frozen audio cannot also use per-token conditioning.")
    }
    guard !frozenAudio || !bfloat16State.contains("audio") else {
      throw LTXError.invalid("Frozen source audio cannot be rounded as generated BF16 state.")
    }
    let videoTokenTimesteps=videoConditioning.map { !$0.mask.allSatisfy { $0 == 1 } } ?? false
    let audioTokenTimesteps=audioConditioning.map { !$0.mask.allSatisfy { $0 == 1 } } ?? false
    let videoClean=videoConditioning?.clean,audioClean=audioConditioning?.clean
    let videoMask=videoConditioning.map { MLXArray($0.mask,[$0.mask.count,1]) }
    let audioMask=audioConditioning.map { MLXArray($0.mask,[$0.mask.count,1]) }
    active=true
    defer { Stream.gpu.synchronize(); Memory.clearCache(); active=false }
    // MLXArray is a mutable Swift reference wrapper. Reshape creates a distinct
    // handle over the same immutable MLX value without copying its GPU payload.
    // Snapshot before invoking any caller callback, including head preparation.
    var current=inputs.mapValues { $0.reshaped($0.shape) }
    guard bfloat16State.isSubset(of:["video","audio"]) else { throw LTXError.invalid("Unknown BF16 state modality.") }
    for name in bfloat16State {
      guard let latent=current[name+"_latent"] else { throw LTXError.invalid("Missing BF16 source latent: \(name).") }
      current[name+"_latent"]=latent.asType(.bfloat16).asType(.float32)
    }
    let sigmas=schedule.sigmas.dropLast().map { DenoiserMath.bfloat16(Float($0)) }
    let preparation=try denoiser.prepare(current,sigmas:schedule.sigmas.dropLast().map(Float.init),
      videoDenoiseMask:videoTokenTimesteps ? videoConditioning?.mask : nil,
      audioDenoiseMask:audioTokenTimesteps ? audioConditioning?.mask : nil,
      frozenAudio:frozenAudio,weights:fixedWeights,adapters:fixedAdapters) {
      try stageProgress(0,$0)
    }
    for (index,step) in schedule.steps.enumerated() {
      try Task.checkCancellation()
      var noises:[String:MLXArray]=[:]
      if step.ancestral {
        for name in frozenAudio ? ["video"] : ["video","audio"] {
          try Task.checkCancellation()
          let shape=denoiser.inputShapes[name+"_latent"]!
          let value=try noise!(index,name,shape)
          try Task.checkCancellation()
          guard value.dtype == .float32, value.shape == shape,
            MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Invalid ancestral noise for \(name).") }
          noises[name]=value.reshaped(shape)
        }
      }
      let velocity=try denoiser.evaluatePrepared(current,sigma:Float(schedule.sigmas[index]),preparation:preparation,
        fixedWeights:fixedWeights,blockWeights:blockWeights,fixedAdapters:fixedAdapters,blockAdapters:blockAdapters) {
        try stageProgress(index+1,$0)
      }
      for name in ["video","audio"] {
        if name == "audio" && frozenAudio { continue }
        let key=name+"_latent", state=current[key]!.reshaped(denoiser.inputShapes[key]!)
        // X0Model uses raw per-token time for conditioned video, otherwise
        // the global BF16 scalar. Its source-dtype cast precedes anchor blending.
        let conditionedMask=name == "video" ? videoMask : audioMask
        let conditionedClean=name == "video" ? videoClean : audioClean
        let tokenTimesteps=name == "video" ? videoTokenTimesteps : audioTokenTimesteps
        let sigma:MLXArray = tokenTimesteps
          ? conditionedMask!*Float(schedule.sigmas[index]) : MLXArray(sigmas[index])
        var clean=state-sigma*velocity[name]!
        if bfloat16State.contains(name) { clean=clean.asType(.bfloat16).asType(.float32) }
        if let conditionedClean,let conditionedMask {
          clean=clean*conditionedMask+conditionedClean*(1-conditionedMask)
        }
        var next:MLXArray
        if step.terminal { next=clean }
        else {
          next=state*step.sampleScale+clean*step.predictionScale
          if step.ancestral {
            next=next+noises[name]!*step.noiseScale
            if bfloat16State.contains(name) { next=next.asType(.bfloat16).asType(.float32) }
            // The working distilled pipeline reapplies conditioning after
            // ancestral re-noising. Otherwise even mask-zero endpoint tokens
            // become noisy inputs at the next step. Deterministic Euler must
            // not apply this second blend to fractional-strength references.
            if let conditionedClean,let conditionedMask {
              next=next*conditionedMask+conditionedClean*(1-conditionedMask)
            }
          }
        }
        if bfloat16State.contains(name) { next=next.asType(.bfloat16).asType(.float32) }
        eval(next)
        guard MLX.isFinite(next).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite sampled latents.") }
        current[key]=next
      }
      try Task.checkCancellation()
      let event=Progress(completedSteps:index+1,totalSteps:schedule.steps.count,
        sigma:schedule.sigmas[index],nextSigma:schedule.sigmas[index+1])
      try progress(event)
      try Task.checkCancellation()
      if let preview {
        let snapshot=["video":current["video_latent"]!,"audio":current["audio_latent"]!]
          .mapValues { $0.reshaped($0.shape) }
        try preview(snapshot,event)
      }
      try Task.checkCancellation()
    }
    return ["video":current["video_latent"]!,"audio":current["audio_latent"]!]
  }
}
