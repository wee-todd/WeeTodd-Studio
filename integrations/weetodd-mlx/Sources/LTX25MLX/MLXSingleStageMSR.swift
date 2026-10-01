import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// Full-resolution single-stage MSR sampling through the shared LTX denoiser.
/// Reference LoRA factors stream per block; no duplicate Python/Swift sampler.
final class MLXSingleStageMSR {
  let layout:MLXMSRLayout
  let configuration:AVBlockConfiguration
  private let weights:MLXDenoiserWeights
  private let maximumActivationBytes:Int
  private let gate=NSLock()

  init(layout:MLXMSRLayout,transformerRoot:URL,adapter:LoRAAdapter,
    maximumActivationBytes:Int) throws {
    try adapter.validate()
    guard adapter.enabled,adapter.strength>0 else {
      throw LTXError.invalid("MSR needs one enabled task adapter.")
    }
    let config=try AVBlockConfiguration(videoTokens:layout.videoTokens,
      audioTokens:layout.target.audioFrames,textTokens:1024)
    let block=try MLXAVBlock(configuration:config,maximumActivationBytes:maximumActivationBytes,
      videoAttentionGroups:layout.groupRows)
    try block.admitPerTokenVideo()
    _ = try MLXDenoiser.admitRotary(configuration:config,maximumActivationBytes:maximumActivationBytes)
    let checked=try MLXDenoiserWeights(root:transformerRoot,configuration:config,
      adapters:[adapter],msrAdapterPath:adapter.path,
      maximumActivationBytes:maximumActivationBytes)
    guard checked.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw LTXError.invalid("MSR requires the released distilled LTX 2.5 transformer.")
    }
    self.layout=layout;configuration=config;weights=checked
    self.maximumActivationBytes=maximumActivationBytes
  }

  func evaluate(videoContext:MLXArray,audioContext:MLXArray,references:[MLXArray],seed:UInt64,
    progress:(String,Int,Int) throws -> Void) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("MSR sampler is already active.") }
    defer { Stream.gpu.synchronize();Memory.clearCache();gate.unlock() }
    for (context,width) in [(videoContext,4096),(audioContext,2048)] {
      guard context.dtype == .float32,context.shape == [1024,width],
        MLX.isFinite(context).all().item(Bool.self) else {
        throw LTXError.invalid("MSR text context shape or values are invalid.")
      }
    }
    let videoNoise=MLXNoisePolicy.seeded(seed,tokens:layout.target.videoTokens).asType(.float32)
    let audioNoise=MLXNoisePolicy.seeded(seed &+ 1,tokens:layout.target.audioFrames).asType(.float32)
    let prepared=try layout.prepare(generated:videoNoise,references:references)
    let runner=try MLXSamplingRunner(configuration:configuration,
      maximumActivationBytes:maximumActivationBytes,videoAttentionGroups:layout.groupRows)
    let schedule=try SamplingSchedule(sigmas:[1,0.99375,0.9875,0.98125,0.975,
      0.909375,0.725,0.421875,0],eta:0)
    let inputs:[String:MLXArray]=[
      "video_text":videoContext,"audio_text":audioContext,
      "video_latent":prepared.latent,"audio_latent":audioNoise,
      "video_positions":MLXArray(layout.positions,[layout.videoTokens,3]),
      "audio_positions":MLXArray(layout.target.audioPositions,[layout.target.audioFrames,1]),
      "video_attention_templates":prepared.attentionTemplates]
    let sampled=try runner.evaluate(inputs,schedule:schedule,
      videoConditioning:prepared.condition,bfloat16State:["video","audio"],
      fixedWeights:weights.readFixed,blockWeights:weights.readBlock,
      fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,
      stageProgress:{ _,event in try progress("msr:"+event.stage,event.completedBlocks,48) },
      progress:{ event in try progress("sampling",event.completedSteps,event.totalSteps) })
    return ["video":sampled["video"]![0..<layout.target.videoTokens],
      "audio":sampled["audio"]!]
  }
}
