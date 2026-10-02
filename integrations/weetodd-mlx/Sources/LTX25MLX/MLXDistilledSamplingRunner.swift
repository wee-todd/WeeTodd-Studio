import Foundation
import MLX
import LTX25Engine
import LTX25Video
import AdapterRuntime

/// Released distilled T2AV recipe. Header-only preflight admits both stages and
/// both ordered adapter stacks before payload reads. Each sampler is stage-local.
public final class MLXDistilledSamplingRunner {
  public struct ExtensionGuides {
    public let stageOneVideo:MLXArray
    public let stageTwoVideo:MLXArray
    public let audio:MLXArray
    public init(stageOneVideo:MLXArray,stageTwoVideo:MLXArray,audio:MLXArray) {
      self.stageOneVideo=stageOneVideo.reshaped(stageOneVideo.shape)
      self.stageTwoVideo=stageTwoVideo.reshaped(stageTwoVideo.shape)
      self.audio=audio.reshaped(audio.shape)
    }
  }
  private let recipe:DistilledTwoStageRecipe
  private let noisePolicy:MLXNoisePolicy
  private let maximumActivationBytes:Int
  private let weights:[MLXDenoiserWeights]
  private let upscaler:MLXLatentUpscaler
  private let gate=NSLock()
  private let layouts:[MLXReferenceLayout?]
  private let guideLayouts:[MLXExtensionGuideLayout?]
  private let unionLayout:MLXUnionControlLayout?
  private let icLayout:MLXICControlLayout?
  public private(set) var stageSeconds:[String:Double]=[:]

  public init(recipe:DistilledTwoStageRecipe,transformerRoot:URL,upscalerCheckpoint:URL,
    statisticsCheckpoint:URL,firstStrength:Float?=nil,lastStrength:Float?=nil,
    firstFrame:Int=0,
    extensionContextFrames:Int?=nil,
    extensionVideoGuideLatentFrames:Int?=nil,extensionAudioGuideTokens:Int?=nil,
    unionControlGuide:MLXUnionControlGuide?=nil,
    icControl:MLXICControl?=nil,
    stageOneLoras:[LoRAAdapter]=[],stageTwoLoras:[LoRAAdapter]=[],
    noisePolicy:MLXNoisePolicy = .native,maximumActivationBytes:Int=2*1024*1024*1024) throws {
    self.recipe=recipe;self.noisePolicy=noisePolicy;self.maximumActivationBytes=maximumActivationBytes
    guard firstStrength != nil || lastStrength == nil else { throw LTXError.invalid("Last-frame reference requires a first frame.") }
    guard extensionContextFrames == nil || (lastStrength == nil && noisePolicy == .releasedMLX) else {
      throw LTXError.invalid("LTX extension needs the released MLX noise policy and no last-frame reference.")
    }
    guard extensionContextFrames != nil ||
      (extensionVideoGuideLatentFrames == nil && extensionAudioGuideTokens == nil) else {
      throw LTXError.invalid("LTX guide overrides require a causal source context.")
    }
    guard unionControlGuide == nil ||
      (firstStrength == nil && extensionContextFrames == nil) else {
      throw LTXError.invalid("Union Control cannot combine with endpoint or extension guides.")
    }
    unionLayout=try unionControlGuide.map {
      try MLXUnionControlLayout(geometry:recipe.low,strength:$0.referenceStrength)
    }
    guard icControl == nil || (unionControlGuide == nil && firstStrength == nil && extensionContextFrames == nil) else {
      throw LTXError.invalid("IC controls cannot combine with endpoint, history or Union guide overrides.")
    }
    icLayout=try icControl.map { try MLXICControlLayout(geometry:recipe.low,control:$0) }
    layouts=try [recipe.low,recipe.high].map { g in
      try firstStrength.map { try MLXReferenceLayout(geometry:g,firstStrength:$0,lastStrength:lastStrength,firstFrame:firstFrame) }
    }
    guideLayouts=try [recipe.low,recipe.high].map { g in
      try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:g,contextFrames:$0,
        videoGuideLatentFrames:extensionVideoGuideLatentFrames,
        audioGuideTokens:extensionAudioGuideTokens) }
    }
    // Reject impossible stage-two geometry before even opening stage-one files.
    for (index,g) in [recipe.low,recipe.high].enumerated() {
      let block=try MLXAVBlock(configuration:Self.configuration(g,layout:layouts[index],guide:guideLayouts[index],union:index == 0 ? unionLayout : nil,ic:index == 0 ? icLayout : nil),maximumActivationBytes:maximumActivationBytes)
      if layouts[index] != nil || (index == 0 && (unionLayout != nil || icLayout != nil)) { try block.admitPerTokenVideo() }
      if guideLayouts[index] != nil { try block.admitPerTokenAV() }
    }
    _ = try MLXLatentUpscaler.admit(shape:[recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128],maximumActivationBytes:maximumActivationBytes)
    var sources:[MLXDenoiserWeights]=[]
    for (index,pair) in [(recipe.low,stageOneLoras),(recipe.high,stageTwoLoras)].enumerated() {
      let (g,adapters)=pair
      let active = adapters + (index == 0 ? unionControlGuide.map {
        [LoRAAdapter(path:$0.adapterPath,strength:$0.adapterStrength)]
      } ?? [] : []) + (index == 0 ? icControl?.adapters.map { LoRAAdapter(path:$0.path,strength:$0.strength) } ?? [] : [])
      let source=try MLXDenoiserWeights(root:transformerRoot,
        configuration:Self.configuration(g,layout:layouts[index],guide:guideLayouts[index],union:index == 0 ? unionLayout : nil,ic:index == 0 ? icLayout : nil),
        adapters:active,unionControlAdapterPath:index == 0 ? unionControlGuide?.adapterPath : nil,
        maximumActivationBytes:maximumActivationBytes,
        icControlFamilies:index == 0 ? Dictionary(uniqueKeysWithValues:icControl?.adapters.map { ($0.path,$0.family) } ?? []) : [:])
      guard source.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
        throw LTXError.invalid("Released two-stage sampling requires the distilled transformer provenance.")
      }
      sources.append(source)
    }
    weights=sources
    upscaler=try MLXLatentUpscaler(checkpoint:upscalerCheckpoint,statisticsCheckpoint:statisticsCheckpoint)
  }
  private static func configuration(_ g:AVGeometry,layout:MLXReferenceLayout?=nil,
    guide:MLXExtensionGuideLayout?=nil,union:MLXUnionControlLayout?=nil,ic:MLXICControlLayout?=nil) throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoTokens:guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? g.videoTokens,
      audioTokens:guide?.audioTokens ?? g.audioFrames,textTokens:1024)
  }
  public func evaluate(videoContext:MLXArray,audioContext:MLXArray,
    references:[(first:MLXArray,last:MLXArray?)]=[],frozenAudio:MLXArray?=nil,
    extensionGuides:ExtensionGuides?=nil,unionGuide:MLXArray?=nil,icGuides:[MLXArray]=[],
    progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> [String:MLXArray] {
    try evaluateWithStageOneCapture(videoContext:videoContext,audioContext:audioContext,
      references:references,frozenAudio:frozenAudio,extensionGuides:extensionGuides,unionGuide:unionGuide,icGuides:icGuides,
      stageOneVideoObserver:nil,progress:progress)
  }

  public func evaluateWithStageOneCapture(videoContext:MLXArray,audioContext:MLXArray,
    references:[(first:MLXArray,last:MLXArray?)]=[],frozenAudio:MLXArray?=nil,
    extensionGuides:ExtensionGuides?=nil,unionGuide:MLXArray?=nil,icGuides:[MLXArray]=[],
    stageOneVideoObserver:((MLXArray) throws -> Void)?=nil,
    progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Two-stage MLX sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    try Task.checkCancellation()
    guard references.count == (layouts[0] == nil ? 0 : 2) else { throw LTXError.invalid("Both stage references must be encoded before sampling.") }
    guard (extensionGuides != nil) == (guideLayouts[0] != nil),
      frozenAudio == nil || (frozenAudio!.dtype == .float32 &&
        frozenAudio!.shape == [recipe.low.audioFrames,128] &&
        recipe.low.audioFrames == recipe.high.audioFrames &&
        MLX.isFinite(frozenAudio!).all().item(Bool.self)) else {
      throw LTXError.invalid("LTX source audio must fit both stages and any audiovisual history guides.")
    }
    guard (unionGuide != nil) == (unionLayout != nil) else {
      throw LTXError.invalid("Union Control requires exactly one encoded guide.")
    }
    guard icGuides.count == (icLayout?.strengths.count ?? 0) else {
      throw LTXError.invalid("IC control requires every admitted guide in the supplied order.")
    }
    if let extensionGuides {
      for (value,count) in [(extensionGuides.stageOneVideo,guideLayouts[0]!.videoGuideTokens),
        (extensionGuides.stageTwoVideo,guideLayouts[1]!.videoGuideTokens),
        (extensionGuides.audio,guideLayouts[0]!.audioGuideTokens)] {
        guard value.dtype == .float32,value.shape == [count,128],MLX.isFinite(value).all().item(Bool.self) else {
          throw LTXError.invalid("LTX extension source guide differs from the admitted audiovisual layout.")
        }
      }
    }
    let references=references.map { (first:$0.first.reshaped($0.first.shape),last:$0.last.map { $0.reshaped($0.shape) }) }
    for (index,pair) in references.enumerated() { try layouts[index]!.validate(first:pair.first,last:pair.last) }
    var text:[String:MLXArray]=[:]
    for (name,x,width) in [("video_text",videoContext,4096),("audio_text",audioContext,2048)] {
      guard x.dtype == .float32, x.shape == [1024,width], MLX.isFinite(x).all().item(Bool.self) else {
        throw LTXError.invalid("Invalid trained text context shape/dtype/values.")
      }
      text[name]=x.reshaped(x.shape)
    }
    stageSeconds=[:]
    func report(_ name:String,_ completed:Int,_ total:Int) throws {
      try Task.checkCancellation(); try progress(name,completed,total); try Task.checkCancellation()
    }
    let sampleStage:MLXTwoStageTrajectory.GuideSample={ stage,g,state,guideNoise,schedule,noise in
      let start=Date()
      let output=try autoreleasepool {
        let source=self.weights[stage-1]
        let layout=self.layouts[stage-1]
        let guideLayout=self.guideLayouts[stage-1]
        let union=stage == 1 ? self.unionLayout : nil
        let ic=stage == 1 ? self.icLayout : nil
        let sampler=try MLXSamplingRunner(configuration:Self.configuration(g,layout:layout,guide:guideLayout,union:union,ic:ic),maximumActivationBytes:self.maximumActivationBytes)
        var inputs=text
        inputs["video_latent"]=state["video"]!; inputs["audio_latent"]=state["audio"]!
        var conditioning:MLXVideoDenoiseCondition?
        var audioConditioning:MLXAudioDenoiseCondition?
        if let layout {
          let pair=references[stage-1]
          let prepared=try layout.prepare(generated:state["video"]!,first:pair.first,last:pair.last)
          inputs["video_latent"]=prepared.latent;conditioning=prepared.condition
        }
        if let union,let unionGuide {
          let prepared=try union.prepare(generated:state["video"]!,reference:unionGuide)
          inputs["video_latent"]=prepared.latent;conditioning=prepared.condition
        }
        if let ic {
          let prepared=try ic.prepare(generated:state["video"]!,references:icGuides)
          inputs["video_latent"]=prepared.latent;conditioning=prepared.condition
        }
        if let guideLayout,let extensionGuides {
          let sourceVideo=stage == 1 ? extensionGuides.stageOneVideo : extensionGuides.stageTwoVideo
          let targetAudio=frozenAudio ?? state["audio"]!
          let targetAudioCondition=try frozenAudio.map {
            try MLXAudioDenoiseCondition(clean:$0,
              mask:Array(repeating:Float(0),count:g.audioFrames))
          }
          let prepared=try guideLayout.prepare(targetVideo:inputs["video_latent"]!,
            targetVideoCondition:conditioning,targetAudio:targetAudio,
            targetAudioCondition:targetAudioCondition,
            sourceVideo:sourceVideo,sourceAudio:extensionGuides.audio,
            guideVideoNoise:guideNoise["video"]!,guideAudioNoise:guideNoise["audio"]!,
            sigma:Float(schedule.sigmas[0]))
          inputs["video_latent"]=prepared.video;inputs["audio_latent"]=prepared.audio
          conditioning=prepared.videoCondition;audioConditioning=prepared.audioCondition
        }
        let videoTokens=guideLayout?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? g.videoTokens
        let audioTokens=guideLayout?.audioTokens ?? g.audioFrames
        inputs["video_positions"]=MLXArray(guideLayout?.videoPositions ?? layout?.positions ?? union?.positions ?? ic?.positions ?? g.videoPositions,[videoTokens,3])
        inputs["audio_positions"]=MLXArray(guideLayout?.audioPositions ?? g.audioPositions,[audioTokens,1])
        let bf16:Set<String> = self.noisePolicy == .releasedMLX
          ? (frozenAudio != nil ? (stage == 1 ? ["video"] : []) : (stage == 1 ? ["video","audio"] : ["audio"])) : []
        let sampled=try sampler.evaluate(inputs,schedule:schedule,videoConditioning:conditioning,
          audioConditioning:audioConditioning,
          frozenAudio:frozenAudio != nil && guideLayout == nil,bfloat16State:bf16,
          fixedWeights:source.readFixed,blockWeights:source.readBlock,
          fixedAdapters:source.fixedAdapters,blockAdapters:source.blockAdapters,noise:noise,
          stageProgress:{ _,event in
            if event.completedBlocks==48,let detail=event.stack {
              self.stageSeconds["stage\(stage)_block_load",default:0] += detail.loadSeconds
              self.stageSeconds["stage\(stage)_block_compute",default:0] += detail.computeSeconds
            }
            try report("stage\(stage):"+event.stage,event.completedBlocks,48)
          },
          progress:{ event in try report("sampling",(stage == 1 ? 0 : 8)+event.completedSteps,11) })
        return ["video":sampled["video"]![0..<g.videoTokens],"audio":sampled["audio"]![0..<g.audioFrames]]
      }
      self.stageSeconds["stage\(stage)"]=Date().timeIntervalSince(start)
      try report("stage\(stage)_weights_released",stage,2)
      return output
    }
    let upscale:MLXTwoStageTrajectory.Upscale={ video,shape in
      Stream.gpu.synchronize(); Memory.clearCache()
      let start=Date()
      let output=try autoreleasepool {
        let array=try self.upscaler.upscale(video.reshaped(shape),maximumActivationBytes:self.maximumActivationBytes,
          progress:{ try report("upscale:"+$0,0,1) })
        return array.reshaped([self.recipe.high.videoTokens,128])
      }
      self.stageSeconds["latent_upscale_mlx"]=Date().timeIntervalSince(start)
      try report("upscaler_weights_released",1,1)
      return output
    }
    let trajectory=MLXTwoStageTrajectory()
    if let guide=guideLayouts[0] {
      return try trajectory.evaluateWithGuides(recipe:recipe,
        videoGuideFrames:guide.videoGuideLatentFrames,audioGuideTokens:guide.audioGuideTokens,
        stageOneVideoObserver:stageOneVideoObserver,sample:sampleStage,upscale:upscale)
    }
    return try trajectory.evaluate(recipe:recipe,noisePolicy:noisePolicy,frozenAudio:frozenAudio,
      stageOneVideoObserver:stageOneVideoObserver,
      sample:{ stage,g,state,schedule,noise in
        try sampleStage(stage,g,state,[:],schedule,noise)
      },upscale:upscale)
  }
}
