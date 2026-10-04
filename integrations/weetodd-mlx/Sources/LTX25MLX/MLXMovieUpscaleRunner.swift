import Foundation
import MLX
import LTX25Engine
import AdapterRuntime
import LTX25Video

/// The sole source-movie weighted trajectory. Callers stage source VAE/audio/
/// text encoding before this call and publish only its normalized video result.
/// There is no distilled first pass, audio sampling or learned keyframe slot.
public final class MLXMovieUpscaleRunner {
  public struct Result { public let video:MLXArray,geometry:AVGeometry,visibleFrames:Int,stageSeconds:[String:Double] }
  public let plan:MLXMovieUpscalePlan
  public let layout:MLXMovieUpscaleLayout?
  private let seed:UInt64,maximumActivationBytes:Int
  private let upscaler:MLXLatentUpscaler,weights:MLXDenoiserWeights?
  private let gate=NSLock()
  public init(request:MLXMovieUpscaleRequest,chunk:MLXMovieUpscalePlan.Chunk,
    firstStrength:Float?,lastStrength:Float?,maximumActivationBytes:Int=2*1024*1024*1024) throws {
    guard chunk.startFrame>=0,chunk.endFrame<=request.plan.frames,chunk.frames>0,
      chunk.paddedFrames == (try MLXMovieUpscalePlan.paddedFrameCount(chunk.frames)),
      maximumActivationBytes>0 else { throw LTXError.invalid("Movie runner differs from its frozen chunk admission.") }
    plan=try MLXMovieUpscalePlan(mode:request.plan.mode,width:request.plan.size.width,height:request.plan.size.height,
      frames:chunk.frames,fps:request.plan.fps,sizePolicy:.strict,refinementStrength:request.plan.refinementStrength)
    self.seed=request.seed;self.maximumActivationBytes=maximumActivationBytes
    _ = try MLXLatentUpscaler.admit(shape:[plan.paddedFrames/8+1,plan.size.height/32,plan.size.width/32,128],
      maximumActivationBytes:maximumActivationBytes)
    upscaler=try MLXLatentUpscaler(checkpoint:URL(fileURLWithPath:request.components["spatial_upscaler_checkpoint"]!),
      statisticsCheckpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!))
    if plan.mode == .latentOnly {
      guard firstStrength == nil,lastStrength == nil else { throw LTXError.invalid("Latent-only cannot consume unused movie anchors.") }
      layout=nil;weights=nil
      _ = try AVGeometry(width:plan.size.outputWidth,height:plan.size.outputHeight,frames:plan.paddedFrames,fps:plan.fps)
    } else {
      let layout=try MLXMovieUpscaleLayout(plan:plan,firstStrength:firstStrength,lastStrength:lastStrength)
      self.layout=layout
      let config=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:layout.geometry.audioFrames,textTokens:1024)
      let block=try MLXAVBlock(configuration:config,maximumActivationBytes:maximumActivationBytes,
        videoAttentionGroups:layout.groupRows.count>1 ? layout.groupRows : [])
      if layout.mask.contains(where:{ $0 != 1 }) { try block.admitPerTokenVideo() }
      _ = try MLXDenoiser.admitRotary(configuration:config,maximumActivationBytes:maximumActivationBytes)
      let adapter=plan.mode == .pixelSpatial ? LoRAAdapter(path:request.components["pixel_spatial_adapter"]!,strength:request.pixelStrength) : nil
      let source=try MLXDenoiserWeights(root:URL(fileURLWithPath:request.components["transformer_root"]!),
        configuration:config,adapters:adapter.map { [$0] } ?? [],maximumActivationBytes:maximumActivationBytes,
        pixelSpatialDFRAdapterPath:adapter?.path)
      guard source.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
        throw LTXError.invalid("Movie refinement requires the released distilled native transformer pages.")
      }
      weights=source
    }
  }
  /// All arguments are normalized FHWC/packed latent values from shared VAEs.
  /// Frozen clean source audio is padded by repeating its final encoded token.
  public func run(source:MLXArray,audio:MLXArray?=nil,videoContext:MLXArray?=nil,audioContext:MLXArray?=nil,
    first:MLXArray?=nil,last:MLXArray?=nil,
    progress:@escaping(String,Int,Int)throws->Void={ _,_,_ in },
    preview:(([String:MLXArray],MLXSamplingRunner.Progress)throws->Void)?=nil) throws -> Result {
    guard gate.try() else { throw LTXError.invalid("Movie runner is already active.") }
    defer { Stream.gpu.synchronize();Memory.clearCache();gate.unlock() }
    try Task.checkCancellation()
    let f=plan.paddedFrames/8+1,h=plan.size.height/32,w=plan.size.width/32
    guard source.dtype == .float32,source.shape == [f,h,w,128],MLX.isFinite(source).all().item(Bool.self) else {
      throw LTXError.invalid("Movie source needs the normalized shared video-encoder latent, without a second normalization.")
    }
    let g=try AVGeometry(width:plan.size.outputWidth,height:plan.size.outputHeight,frames:plan.paddedFrames,fps:plan.fps)
    if plan.mode == .latentOnly {
      guard audio == nil,videoContext == nil,audioContext == nil,first == nil,last == nil else {
        throw LTXError.invalid("Latent-only cannot silently ignore weighted context or references.")
      }
    } else {
      guard let audio,let videoContext,let audioContext,
        audio.dtype == .float32,audio.shape.count==2,audio.shape[1]==128,(1...1501).contains(audio.shape[0]),
        videoContext.dtype == .float32,videoContext.shape == [1024,4096],
        audioContext.dtype == .float32,audioContext.shape == [1024,2048],
        [audio,videoContext,audioContext].allSatisfy({ MLX.isFinite($0).all().item(Bool.self) }) else {
        throw LTXError.invalid("Movie refinement needs frozen finite source audio and shared positive text contexts.")
      }
    }
    var seconds:[String:Double]=[:]
    let started=Date()
    let upscaled=try upscaler.upscale(source,maximumActivationBytes:maximumActivationBytes) {
      try progress("movie_upscale:"+$0,0,1)
    }.reshaped([g.videoTokens,128])
    eval(upscaled);seconds["latent_upscale"]=Date().timeIntervalSince(started)
    try progress("movie_upscaler_weights_released",1,1);try Task.checkCancellation()
    guard let layout,let weights else { return Result(video:upscaled,geometry:g,visibleFrames:plan.frames,stageSeconds:seconds) }
    let originalAudio=audio!
    let frozen:MLXArray
    if originalAudio.shape[0]<g.audioFrames {
      frozen=concatenated([originalAudio,broadcast(originalAudio[(originalAudio.shape[0]-1)..<originalAudio.shape[0]],to:[g.audioFrames-originalAudio.shape[0],128])],axis:0)
    } else { frozen=originalAudio[0..<g.audioFrames] }
    eval(frozen)
    let sigma=Float(plan.sigmas[0])
    let noise=MLXNoisePolicy.seeded(seed &+ 2,tokens:g.videoTokens).asType(.float32)
    let initialized=upscaled*(1-sigma)+noise*sigma
    let prepared=try layout.prepare(generated:initialized,source:layout.pixelReference ? source.reshaped([layout.sourceGeometry.videoTokens,128]) : nil,
      first:first,last:last)
    var inputs:[String:MLXArray]=["video_latent":prepared.latent,"audio_latent":frozen,
      "video_text":videoContext!,"audio_text":audioContext!,
      "video_positions":MLXArray(layout.positions,[layout.videoTokens,3]),"audio_positions":MLXArray(g.audioPositions,[g.audioFrames,1])]
    if layout.groupRows.count>1 {
      inputs["video_attention_templates"]=MLXArray(layout.attentionTemplates,[layout.groupRows.count,layout.videoTokens])
    }
    let config=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:g.audioFrames,textTokens:1024)
    let runner=try MLXSamplingRunner(configuration:config,maximumActivationBytes:maximumActivationBytes,
      videoAttentionGroups:layout.groupRows.count>1 ? layout.groupRows : [])
    let start=Date()
    let sampled=try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:plan.sigmas,eta:0),
      videoConditioning:prepared.condition,frozenAudio:true,bfloat16State:[],fixedWeights:weights.readFixed,
      blockWeights:weights.readBlock,fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,
      stageProgress:{ _,event in try progress("movie_refine:"+event.stage,event.completedBlocks,48) },
      progress:{ try progress("movie_sampling",$0.completedSteps,$0.totalSteps) },preview:preview)
    let result=sampled["video"]![0..<g.videoTokens].reshaped([g.videoTokens,128]);eval(result)
    seconds["refinement"]=Date().timeIntervalSince(start)
    try progress("movie_transformer_weights_released",3,3);try Task.checkCancellation()
    return Result(video:result,geometry:g,visibleFrames:plan.frames,stageSeconds:seconds)
  }
}
