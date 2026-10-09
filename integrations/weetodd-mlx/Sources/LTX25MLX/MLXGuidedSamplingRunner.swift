import Foundation
import MLX
import LTX25Engine

/// Model-independent AV trajectory; prediction owns and releases every weighted
/// pass before returning. CPU fixtures exercise this same production integrator.
enum MLXGuidedTrajectory {
  typealias Prediction = ([String:MLXArray], Float) throws -> [String:MLXArray]
  typealias Noise = (Int, Bool, String, [Int]) throws -> MLXArray
  static func seededNoise(index: Int, substep: Bool, modality: String, shape: [Int]) -> MLXArray {
    let seed=UInt64(index*10000+(substep ? 1 : 2))
    let (next,video)=MLXRandom.split(key:MLXRandom.key(seed))
    let (_,audio)=MLXRandom.split(key:next)
    return MLXRandom.normal([1]+shape,key:modality == "video" ? video : audio).reshaped(shape)
  }
  static func evaluate(_ initial: [String:MLXArray], sampling: MLXGuidedSampling,
    schedule: SamplingSchedule, frozenAudio: Bool, noise: Noise,
    predict: Prediction, progress: (MLXSamplingRunner.Progress) throws -> Void) throws -> [String:MLXArray] {
    guard Set(initial.keys) == ["video","audio"] else { throw LTXError.invalid("Guided trajectory needs both AV states.") }
    let shapes=initial.mapValues(\.shape)
    var state=initial.mapValues { $0.reshaped($0.shape) }
    func checked(_ value: MLXArray, name: String) throws -> MLXArray {
      try Task.checkCancellation()
      guard value.dtype == .float32, value.shape == shapes[name] else { throw LTXError.invalid("Guided trajectory returned a changed shape or dtype.") }
      eval(value)
      guard MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Guided trajectory returned nonfinite latents.") }
      return value
    }
    for name in ["video","audio"] { state[name]=try checked(state[name]!,name:name) }
    let names=frozenAudio ? ["video"] : ["video","audio"]
    if sampling.mode == .guided {
      for name in names { state[name]=state[name]!.asType(.bfloat16).asType(.float32) }
      for index in schedule.steps.indices {
        try Task.checkCancellation()
        let sigma=schedule.sigmas[index],nextSigma=schedule.sigmas[index+1]
        let clean=try predict(state,Float(sigma))
        guard Set(clean.keys) == ["video","audio"] else { throw LTXError.invalid("Guided prediction omitted a modality.") }
        for name in names {
          let prediction=try checked(clean[name]!,name:name)
          let x=state[name]!.asType(.bfloat16), x0=prediction.asType(.bfloat16)
          // Preserve the released Euler arithmetic at sigma_next=0 too: a
          // BF16 subtract/update is not bit-identical to returning x0 directly.
          let next=x+(Float(nextSigma-sigma))*((x-x0)/Float(sigma))
          state[name]=try checked(next.asType(.float32),name:name)
        }
        try progress(.init(completedSteps:index+1,totalSteps:schedule.steps.count,
          sigma:sigma,nextSigma:nextSigma))
      }
      return state
    }
    var sigmas=schedule.sigmas
    sigmas[sigmas.count-1]=0.0011
    let total=schedule.steps.count+1
    for index in 0..<schedule.steps.count {
      try Task.checkCancellation()
      let sigma=sigmas[index], next=sigmas[index+1]
      let h=log(sigma/next), middle=sqrt(sigma*next)
      let coefficients=MLXGuidanceMath.res2Coefficients(h)
      let first=try predict(state,Float(sigma))
      var anchor=state, epsilon: [String:MLXArray]=[:], midpoint=state
      for name in names {
        let prediction=try checked(first[name]!,name:name)
        epsilon[name]=prediction-anchor[name]!
        let interpolated=anchor[name]!+Float(h*coefficients.a21)*epsilon[name]!
        let draw=try checked(noise(index,true,name,shapes[name]!),name:name)
        midpoint[name]=MLXGuidanceMath.res2Noise(sample:anchor[name]!,prediction:interpolated,
          sigma:sigma,nextSigma:middle,noise:MLXGuidanceMath.normalizedNoise(draw))
        if h < 0.5 && sigma > 0.03 {
          for _ in 0..<100 {
            try Task.checkCancellation()
            anchor[name]=midpoint[name]!-Float(h*coefficients.a21)*epsilon[name]!
            epsilon[name]=prediction-anchor[name]!
            eval(anchor[name]!,epsilon[name]!)
          }
          eval(anchor[name]!,epsilon[name]!)
        }
        midpoint[name]=try checked(midpoint[name]!,name:name)
      }
      let second=try predict(midpoint,Float(middle))
      for name in names {
        let prediction=try checked(second[name]!,name:name)
        let nextPrediction=anchor[name]!+Float(h)*(Float(coefficients.b1)*epsilon[name]!+Float(coefficients.b2)*(prediction-anchor[name]!))
        let draw=try checked(noise(index,false,name,shapes[name]!),name:name)
        state[name]=try checked(MLXGuidanceMath.res2Noise(sample:anchor[name]!,prediction:nextPrediction,
          sigma:sigma,nextSigma:next,noise:MLXGuidanceMath.normalizedNoise(draw)),name:name)
      }
      try progress(.init(completedSteps:index+1,totalSteps:total,sigma:Double(sigma),nextSigma:Double(next)))
    }
    let terminal=try predict(state,0.0011)
    for name in names { state[name]=try checked(terminal[name]!.asType(.bfloat16).asType(.float32),name:name) }
    try progress(.init(completedSteps:total,totalSteps:total,sigma:0.0011,nextSigma:0))
    return state
  }
}

/// Dev CFG/STG/modality predictions use the shared streamed denoiser. Branches
/// are serial; no additional model copies or resident transformer stacks.
final class MLXGuidedSamplingRunner {
  private let denoiser: MLXDenoiser
  private let sampling: MLXGuidedSampling
  private let gate=NSLock()
  init(configuration: AVBlockConfiguration, sampling: MLXGuidedSampling,
    maximumActivationBytes: Int,keyframeMarkerRows:Int=0,leadingKeyframeMarkerRows:Int=0) throws {
    self.sampling=sampling
    let reserve=MLXGuidedSampling.reserveBytes(videoTokens:configuration.videoTokens,audioTokens:configuration.audioTokens)
    guard reserve < maximumActivationBytes else { throw LTXError.invalid("Dev guidance latents and contexts exceed the admitted workspace.") }
    denoiser=try MLXDenoiser(configuration:configuration,maximumActivationBytes:maximumActivationBytes-reserve,
      keyframeMarkerRows:keyframeMarkerRows,leadingKeyframeMarkerRows:leadingKeyframeMarkerRows)
  }
  func evaluate(_ inputs: [String:MLXArray], schedule: SamplingSchedule,
    negativeContexts: [String:MLXArray], videoConditioning: MLXVideoDenoiseCondition?, frozenAudio: Bool,
    fixedWeights: MLXDenoiser.FixedProvider, blockWeights: MLXDenoiser.BlockProvider,
    fixedAdapters: MLXDenoiser.FixedAdapters, blockAdapters: (Int) throws -> [String:[MLXLoRA]],
    stageProgress: (Int,MLXDenoiser.Progress) throws -> Void,
    progress: (MLXSamplingRunner.Progress) throws -> Void) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Guided sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    guard Set(negativeContexts.keys) == ["video_text","audio_text"] else { throw LTXError.invalid("Dev CFG requires both encoded negative contexts.") }
    for (name,value) in negativeContexts {
      guard value.dtype == .float32, value.shape == denoiser.inputShapes[name],
        MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Invalid Dev negative text context.") }
    }
    guard videoConditioning == nil || videoConditioning!.clean.shape == denoiser.inputShapes["video_latent"] else {
      throw LTXError.invalid("Dev reference conditioning differs from admitted video tokens.")
    }
    var predictionSigmas=schedule.sigmas.dropLast().map(Float.init)
    if sampling.mode == .guidedHQ {
      var full=schedule.sigmas; full[full.count-1]=0.0011
      predictionSigmas += zip(full,full.dropFirst()).map { Float(sqrt($0*$1)) }
      predictionSigmas.append(0.0011)
    }
    let tokenTimes=videoConditioning.map { !$0.mask.allSatisfy { $0 == 1 } } ?? false
    let preparation=try denoiser.prepare(inputs,sigmas:predictionSigmas,
      videoDenoiseMask:tokenTimes ? videoConditioning?.mask : nil,frozenAudio:frozenAudio,
      weights:fixedWeights,adapters:fixedAdapters) { try stageProgress(0,$0) }
    let mask=videoConditioning.map { MLXArray($0.mask,[$0.mask.count,1]) }
    var invocation=0
    func predict(_ state:[String:MLXArray],_ sigma:Float) throws -> [String:MLXArray] {
      var modelInputs=inputs
      // Both Dev solvers evaluate the model through a BF16 input boundary,
      // even though HQ retains the trajectory itself in FP32.
      modelInputs["video_latent"]=state["video"]!.asType(.bfloat16).asType(.float32)
      modelInputs["audio_latent"]=state["audio"]!.asType(.bfloat16).asType(.float32)
      func branch(_ contexts:[String:MLXArray]?=nil,_ perturbation:MLXGuidancePerturbation = .none) throws -> [String:MLXArray] {
        try Task.checkCancellation(); invocation+=1
        var branchInputs=modelInputs
        if let contexts { branchInputs.merge(contexts) { _,new in new } }
        let velocity=try denoiser.evaluatePrepared(branchInputs,sigma:sigma,preparation:preparation,
          fixedWeights:fixedWeights,blockWeights:blockWeights,fixedAdapters:fixedAdapters,blockAdapters:blockAdapters,
          progress:{ try stageProgress(invocation,$0) },perturbation:perturbation)
        var clean:[String:MLXArray]=[:]
        for name in ["video","audio"] {
          let time:MLXArray=name == "audio" && frozenAudio ? MLXArray(Float(0)) :
            (name == "video" && tokenTimes ? mask!*sigma : MLXArray(DenoiserMath.bfloat16(sigma)))
          let value=modelInputs[name+"_latent"]!-time*velocity[name]!
          clean[name]=value.asType(.bfloat16)
        }
        eval(Array(clean.values)); return clean
      }
      let conditional=try branch()
      let negative=sampling.videoCFG != 1 || sampling.audioCFG != 1 ? try branch(negativeContexts) : nil
      let perturbed=sampling.stg != 0 ? try branch(nil,
        MLXGuidancePerturbation(videoSelfAttentionBlocks:Set(sampling.stgBlocks),audioSelfAttentionBlocks:sampling.stgAudio ? Set(sampling.stgBlocks) : [])) : nil
      let isolated=sampling.modality != 1 ? try branch(nil,MLXGuidancePerturbation(skipCrossModality:true)) : nil
      var clean:[String:MLXArray]=[:]
      for name in ["video","audio"] {
        if name == "audio" && frozenAudio { clean[name]=state[name]; continue }
        var value=MLXGuidanceMath.combine(conditional[name]!,negative:negative?[name],perturbed:perturbed?[name],
          isolated:isolated?[name],cfg:name == "video" ? sampling.videoCFG : sampling.audioCFG,
          stg:name == "video" || sampling.stgAudio ? sampling.stg : 0,modality:sampling.modality,rescale:name == "video" ? sampling.videoRescale : sampling.audioRescale).asType(.float32)
        if name == "video",let mask,let condition=videoConditioning {
          value=value*mask+condition.clean*(1-mask)
        }
        eval(value); clean[name]=value
      }
      return clean
    }
    let initial=["video":inputs["video_latent"]!,"audio":inputs["audio_latent"]!]
    return try MLXGuidedTrajectory.evaluate(initial,sampling:sampling,schedule:schedule,frozenAudio:frozenAudio,
      noise:{ MLXGuidedTrajectory.seededNoise(index:$0,substep:$1,modality:$2,shape:$3) },
      predict:predict,progress:progress)
  }
}
