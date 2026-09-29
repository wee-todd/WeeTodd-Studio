import Foundation
import MLX
import LTX25Engine

/// A staged text-to-latent call. The encoder releases all weighted stages before
/// the sampler invokes any weight provider. One worker owns the MLX process.
public final class MLXPromptSamplingRunner {
  private let sampler:MLXSamplingRunner
  private let encode:(String) throws -> [String:MLXArray]
  private let gate=NSLock()

  public convenience init(configuration:AVBlockConfiguration,encoder:MLXTextEncoder,
    textProgress:@escaping (MLXTextEncoder.Progress) throws -> Void = { _ in }) throws {
    guard configuration.videoDimension == 4096, configuration.audioDimension == 2048,
      configuration.textTokens == 1024 else { throw LTXError.invalid("Trained LTX2.5 text requires 1024 tokens and 4096/2048 features.") }
    try self.init(configuration:configuration,encode:{ prompt in
      let output=try encoder.encode(prompt:prompt,progress:textProgress)
      return ["video_text":output.video,"audio_text":output.audio]
    })
  }

  // Dependency injection is package-scoped so consumers cannot select a substitute
  // text model with an unqualified contract. Tests exercise orchestration cheaply.
  package init(configuration:AVBlockConfiguration,blockCount:Int=48,
    encode:@escaping (String) throws -> [String:MLXArray]) throws {
    sampler=try MLXSamplingRunner(configuration:configuration,blockCount:blockCount)
    self.encode=encode
  }

  public func evaluate(prompt:String,inputs:[String:MLXArray],schedule:SamplingSchedule,
    fixedWeights:MLXDenoiser.FixedProvider,blockWeights:MLXDenoiser.BlockProvider,
    fixedAdapters:MLXDenoiser.FixedAdapters = { _ in [] },
    blockAdapters:(Int) throws -> [String:[MLXLoRA]] = { _ in [:] },
    noise:MLXSamplingRunner.Noise?=nil,
    stageProgress:(Int,MLXDenoiser.Progress) throws -> Void = { _,_ in },
    progress:(MLXSamplingRunner.Progress) throws -> Void = { _ in },
    preview:(([String:MLXArray],MLXSamplingRunner.Progress) throws -> Void)?=nil) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Prompt sampler is already executing.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    guard noise != nil || !schedule.steps.contains(where:\.ancestral) else {
      throw LTXError.invalid("Ancestral sampling requires explicit noise before text encoding.")
    }
    // Validate and snapshot handles before text callbacks can mutate caller data.
    var current=try sampler.preflightWithoutText(inputs,schedule:schedule)
    try Task.checkCancellation()
    let text=try encode(prompt)
    try Task.checkCancellation()
    guard Set(text.keys) == ["video_text","audio_text"] else { throw LTXError.invalid("Missing text contexts.") }
    current.merge(text) { _,new in new }
    return try sampler.evaluate(current,schedule:schedule,fixedWeights:fixedWeights,blockWeights:blockWeights,
      fixedAdapters:fixedAdapters,blockAdapters:blockAdapters,noise:noise,
      stageProgress:stageProgress,progress:progress,preview:preview)
  }
}
