import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// Explicit full-resolution distilled execution. A selected CFG++ schedule
/// controls actual negative forwards; the terminal update never needs one.
public struct MLXSingleStageSampling:Codable,Sendable,Equatable {
  public enum Method:String,Codable,Sendable { case euler, ancestral="euler_ancestral", cfgpp="euler_ancestral_cfg_pp" }
  public enum NegativeSchedule:String,Codable,Sendable { case full,balanced,speed
    public var indices:Set<Int> { switch self {
      case .full:return Set(0...6)
      case .balanced:return [0,2,4,6]
      case .speed:return [0,4]
    } }
  }
  public let method:Method
  public let negativeSchedule:NegativeSchedule
  public let negativePrompt:String
  enum CodingKeys:String,CodingKey,CaseIterable {
    case method,negativeSchedule="negative_schedule",negativePrompt="negative_prompt"
  }
  public init(method:Method,negativeSchedule:NegativeSchedule = .full,negativePrompt:String="") throws {
    guard negativePrompt.utf8.count<=65536,
      method == .cfgpp || (negativeSchedule == .full && negativePrompt.isEmpty) else {
      throw LTXError.invalid("Negative text and schedule require single-stage CFG++.")
    }
    self.method=method;self.negativeSchedule=negativeSchedule;self.negativePrompt=negativePrompt
  }
  public init(from decoder:Decoder) throws {
    let all=try decoder.container(keyedBy:StrictKey.self)
    guard Set(all.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Single-stage sampling requires its exact method and negative fields.")
    }
    let c=try decoder.container(keyedBy:CodingKeys.self)
    try self.init(method:c.decode(Method.self,forKey:.method),
      negativeSchedule:c.decode(NegativeSchedule.self,forKey:.negativeSchedule),
      negativePrompt:c.decode(String.self,forKey:.negativePrompt))
  }
  private struct StrictKey:CodingKey {
    let stringValue:String;var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { return nil }
  }
  public var transformerEvaluations:Int { 8+(method == .cfgpp ? negativeSchedule.indices.count : 0) }
  public func schedule() throws -> SamplingSchedule {
    try SamplingSchedule(sigmas:[1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0],eta:method == .euler ? 0 : 1)
  }
}

/// Shares the staged denoiser, conditioning and reproducible noise implementation
/// with two-stage jobs. No spatial upscaler or second transformer stage is loaded.
public final class MLXSingleStageSamplingRunner {
  private let geometry:AVGeometry
  private let policy:MLXSingleStageSampling
  private let layout:MLXSingleStageControlLayout
  private let weights:MLXDenoiserWeights
  private let blockBudget:Int
  private let lock=NSLock()
  public init(geometry:AVGeometry,policy:MLXSingleStageSampling,
    anchors:[MLXOrdinaryKeyframeLayout.Anchor],generatedCount:Int,
    transformerRoot:URL,adapters:[LoRAAdapter],icControl:MLXICControl?=nil,
    unionControlGuide:MLXUnionControlGuide?=nil,maximumActivationBytes:Int) throws {
    self.geometry=geometry;self.policy=policy
    layout=try MLXSingleStageControlLayout(geometry:geometry,anchors:anchors,generatedCount:generatedCount,
      icControl:icControl,unionStrength:unionControlGuide?.referenceStrength)
    let reserve=try policy.method == .cfgpp ? layout.cfgppReserveBytes(audioTokens:geometry.audioFrames) : 0
    let active=try Self.resolveAdapters(generic:adapters,icControl:icControl,unionControlGuide:unionControlGuide)
    guard reserve<maximumActivationBytes else { throw LTXError.invalid("Single-stage CFG++ state exceeds the activation budget before loading weights.") }
    blockBudget=maximumActivationBytes-reserve
    let configuration=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:geometry.audioFrames,textTokens:1024)
    let block=try MLXAVBlock(configuration:configuration,maximumActivationBytes:blockBudget)
    if layout.requiresPerTokenVideo { try block.admitPerTokenVideo() }
    _ = try MLXDenoiser.admitRotary(configuration:configuration,maximumActivationBytes:blockBudget)
    weights=try MLXDenoiserWeights(root:transformerRoot,configuration:configuration,adapters:active,
      unionControlAdapterPath:unionControlGuide?.adapterPath,
      maximumActivationBytes:blockBudget,requireKeyframeMarker:layout.slotTokens>0,
      icControlFamilies:Dictionary(uniqueKeysWithValues:icControl?.adapters.map { ($0.path,$0.family) } ?? []))
    guard weights.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw LTXError.invalid("Single-stage sampling requires the released distilled transformer provenance.")
    }
  }
  /// Metadata-only resolution; dedicated adapters follow the caller's ordered generic stack.
  static func resolveAdapters(generic:[LoRAAdapter],icControl:MLXICControl?,
    unionControlGuide:MLXUnionControlGuide?) throws -> [LoRAAdapter] {
    guard icControl == nil || unionControlGuide == nil else {
      throw LTXError.invalid("Single-stage IC and Union controls are mutually exclusive.")
    }
    let dedicated=(unionControlGuide.map { [LoRAAdapter(path:$0.adapterPath,strength:$0.adapterStrength)] } ?? []) +
      (icControl?.adapters.map { LoRAAdapter(path:$0.path,strength:$0.strength) } ?? [])
    let paths=Set(dedicated.map(\.path))
    guard generic.count+dedicated.count<=16,!generic.contains(where:{ paths.contains($0.path) }) else {
      throw LTXError.invalid("Single-stage control adapters must appear exactly once after the generic stack.")
    }
    return generic+dedicated
  }
  public func evaluate(videoContext:MLXArray,audioContext:MLXArray,
    negativeContexts:[String:MLXArray]?=nil,references:[MLXArray],guides:[MLXArray]=[],frozenAudio:MLXArray?=nil,
    seed:UInt64,progress:(String,Int,Int) throws -> Void) throws -> [String:MLXArray] {
    guard lock.try() else { throw LTXError.invalid("Single-stage sampler is already active.") }
    defer { Stream.gpu.synchronize();Memory.clearCache();lock.unlock() }
    guard (negativeContexts != nil) == (policy.method == .cfgpp),
      frozenAudio == nil || (policy.method != .cfgpp && frozenAudio!.dtype == .float32 &&
        frozenAudio!.shape == [geometry.audioFrames,128] && MLX.isFinite(frozenAudio!).all().item(Bool.self)) else {
      throw LTXError.invalid("CFG++ requires negative contexts and generated audio; frozen source audio uses Euler sampling.")
    }
    let generated=MLXNoisePolicy.seeded(seed,tokens:geometry.videoTokens).asType(.float32)
    let audio=frozenAudio?.reshaped(frozenAudio!.shape) ?? MLXNoisePolicy.seeded(seed &+ 1,tokens:geometry.audioFrames).asType(.float32)
    let prepared=try layout.prepare(generated:generated,anchors:references,guides:guides)
    let configuration=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:geometry.audioFrames,textTokens:1024)
    let runner=try MLXSamplingRunner(configuration:configuration,maximumActivationBytes:blockBudget,keyframeMarkerRows:layout.slotTokens)
    let inputs:[String:MLXArray]=["video_text":videoContext,"audio_text":audioContext,
      "video_latent":prepared.latent,"audio_latent":audio,
      "video_positions":MLXArray(layout.positions,[layout.videoTokens,3]),
      "audio_positions":MLXArray(geometry.audioPositions,[geometry.audioFrames,1])]
    var key=MLXRandom.key(seed &+ 10000)
    let cfgpp=policy.method == .cfgpp
    let result=try runner.evaluate(inputs,schedule:policy.schedule(),videoConditioning:prepared.condition,
      frozenAudio:frozenAudio != nil,bfloat16State:frozenAudio == nil ? ["video","audio"] : ["video"],
      unconditionalContexts:negativeContexts,cfgppStepIndices:cfgpp ? policy.negativeSchedule.indices : nil,
      fixedWeights:weights.readFixed,blockWeights:weights.readBlock,
      fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,
      noise:policy.method == .euler ? nil : { _,_,shape in
        let (next,draw)=MLXRandom.split(key:key);key=next
        return MLXRandom.normal([1]+shape,key:draw).reshaped(shape)
      },stageProgress:{ _,event in try progress("single_stage:"+event.stage,event.completedBlocks,48) },
      progress:{ try progress("sampling",$0.completedSteps,$0.totalSteps) })
    return ["video":result["video"]![0..<geometry.videoTokens],"audio":result["audio"]!]
  }
}
