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
  private let guidedSampling:MLXGuidedSampling?
  private let noisePolicy:MLXNoisePolicy
  private let maximumActivationBytes:Int
  private let weights:[MLXDenoiserWeights]
  private let upscaler:MLXLatentUpscaler
  private let gate=NSLock()
  private let layouts:[MLXReferenceLayout?]
  private let ordinaryLayouts:[MLXOrdinaryKeyframeLayout?]
  private let guideLayouts:[MLXExtensionGuideLayout?]
  private let sceneLayouts:[MLXSceneKeyframeLayout?]
  private let unionLayout:MLXUnionControlLayout?
  private let icLayout:MLXICControlLayout?
  public private(set) var stageSeconds:[String:Double]=[:]
  var admittedKeyframeMarkers:[Bool] { weights.map(\.requiresKeyframeMarker) }

  public init(recipe:DistilledTwoStageRecipe,transformerRoot:URL,upscalerCheckpoint:URL,
    statisticsCheckpoint:URL,firstStrength:Float?=nil,lastStrength:Float?=nil,
    firstFrame:Int=0,
    ordinaryAnchors:[MLXOrdinaryKeyframeLayout.Anchor]?=nil,generatedKeyframes:Int=0,
    sceneAnchors:[MLXOrdinaryKeyframeLayout.Anchor]?=nil,
    extensionContextFrames:Int?=nil,
    extensionVideoGuideLatentFrames:Int?=nil,extensionAudioGuideTokens:Int?=nil,
    unionControlGuide:MLXUnionControlGuide?=nil,
    icControl:MLXICControl?=nil,
    stageOneLoras:[LoRAAdapter]=[],stageTwoLoras:[LoRAAdapter]=[],
    noisePolicy:MLXNoisePolicy = .native,guidedSampling:MLXGuidedSampling?=nil,maximumActivationBytes:Int=2*1024*1024*1024) throws {
    self.recipe=recipe;self.noisePolicy=noisePolicy;self.guidedSampling=guidedSampling;self.maximumActivationBytes=maximumActivationBytes
    guard sceneAnchors == nil || (ordinaryAnchors == nil && generatedKeyframes == 0 &&
      firstStrength == nil && lastStrength == nil && unionControlGuide == nil && icControl == nil &&
      guidedSampling == nil && noisePolicy == .releasedMLX) else {
      throw LTXError.invalid("Scene image rows require the explicit scene layout and released history contract.")
    }
    guard ordinaryAnchors == nil || (firstStrength == nil && lastStrength == nil &&
      extensionContextFrames == nil && unionControlGuide == nil && icControl == nil && noisePolicy == .releasedMLX),
      ordinaryAnchors != nil || generatedKeyframes == 0 else {
      throw LTXError.invalid("Ordinary keyframes require their explicit layout and released noise contract.")
    }
    guard guidedSampling == nil || (extensionContextFrames == nil && unionControlGuide == nil && icControl == nil && noisePolicy == .releasedMLX) else {
      throw LTXError.invalid("Dev guidance is admitted for ordinary clips and source audio, not specialized guides or extension.")
    }
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
    ordinaryLayouts=try [recipe.low,recipe.high].enumerated().map { index,g in
      try ordinaryAnchors.map { try MLXOrdinaryKeyframeLayout(geometry:g,anchors:$0,
        generatedCount:index == 0 ? generatedKeyframes : 0) }
    }
    let preparedGuideLayouts=try [recipe.low,recipe.high].map { g in
      try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:g,contextFrames:$0,
        videoGuideLatentFrames:extensionVideoGuideLatentFrames,
        audioGuideTokens:extensionAudioGuideTokens) }
    }
    guideLayouts=preparedGuideLayouts
    sceneLayouts=try [recipe.low,recipe.high].enumerated().map { index,g in
      try sceneAnchors.map { try MLXSceneKeyframeLayout(geometry:g,anchors:$0,extensionGuide:preparedGuideLayouts[index]) }
    }
    // Reject impossible stage-two geometry before even opening stage-one files.
    for (index,g) in [recipe.low,recipe.high].enumerated() {
      let block=try MLXAVBlock(configuration:Self.configuration(g,layout:layouts[index],ordinary:ordinaryLayouts[index],scene:sceneLayouts[index],guide:guideLayouts[index],union:index == 0 ? unionLayout : nil,ic:index == 0 ? icLayout : nil),maximumActivationBytes:maximumActivationBytes)
      if layouts[index] != nil || (index == 0 && (unionLayout != nil || icLayout != nil)) { try block.admitPerTokenVideo() }
      if guideLayouts[index] != nil { try block.admitPerTokenAV() }
      if ordinaryLayouts[index]?.anchors.isEmpty == false || sceneLayouts[index]?.ordinary.anchors.isEmpty == false {
        try block.admitPerTokenVideo()
      }
    }
    _ = try MLXLatentUpscaler.admit(shape:[recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128],maximumActivationBytes:maximumActivationBytes)
    var sources:[MLXDenoiserWeights]=[]
    for (index,pair) in [(recipe.low,stageOneLoras),(recipe.high,stageTwoLoras)].enumerated() {
      let (g,adapters)=pair
      let active = adapters + (index == 1 ? guidedSampling.map {
        [LoRAAdapter(path:$0.distilledAdapterPath,strength:1)]
      } ?? [] : []) + (index == 0 ? unionControlGuide.map {
        [LoRAAdapter(path:$0.adapterPath,strength:$0.adapterStrength)]
      } ?? [] : []) + (index == 0 ? icControl?.adapters.map { LoRAAdapter(path:$0.path,strength:$0.strength) } ?? [] : [])
      let source=try MLXDenoiserWeights(root:transformerRoot,
        configuration:Self.configuration(g,layout:layouts[index],ordinary:ordinaryLayouts[index],scene:sceneLayouts[index],guide:guideLayouts[index],union:index == 0 ? unionLayout : nil,ic:index == 0 ? icLayout : nil),
        adapters:active,unionControlAdapterPath:index == 0 ? unionControlGuide?.adapterPath : nil,
        maximumActivationBytes:maximumActivationBytes,
        requireKeyframeMarker:(ordinaryLayouts[index]?.slotTokens ?? 0)>0,
        icControlFamilies:index == 0 ? Dictionary(uniqueKeysWithValues:icControl?.adapters.map { ($0.path,$0.family) } ?? []) : [:])
      let expected=guidedSampling == nil ? "ltx-2.5-22b-distilled-transformer-bf16.safetensors" : "ltx-2.5-22b-dev-transformer-bf16.safetensors"
      guard source.sourceCheckpoint == expected else {
        throw LTXError.invalid("Two-stage sampling requires the selected distilled or Dev transformer provenance.")
      }
      sources.append(source)
    }
    weights=sources
    upscaler=try MLXLatentUpscaler(checkpoint:upscalerCheckpoint,statisticsCheckpoint:statisticsCheckpoint)
  }
  private static func configuration(_ g:AVGeometry,layout:MLXReferenceLayout?=nil,
    ordinary:MLXOrdinaryKeyframeLayout?=nil,scene:MLXSceneKeyframeLayout?=nil,
    guide:MLXExtensionGuideLayout?=nil,union:MLXUnionControlLayout?=nil,ic:MLXICControlLayout?=nil) throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoTokens:scene?.videoTokens ?? ordinary?.videoTokens ?? guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? g.videoTokens,
      audioTokens:scene?.audioTokens ?? guide?.audioTokens ?? g.audioFrames,textTokens:1024)
  }
  public func evaluate(videoContext:MLXArray,audioContext:MLXArray,
    ordinaryReferences:[[MLXArray]]=[],sceneReferences:[[MLXArray]]=[],
    references:[(first:MLXArray,last:MLXArray?)]=[],frozenAudio:MLXArray?=nil,negativeContexts:[String:MLXArray]?=nil,
    extensionGuides:ExtensionGuides?=nil,unionGuide:MLXArray?=nil,icGuides:[MLXArray]=[],
    progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> [String:MLXArray] {
    try evaluateWithStageOneCapture(videoContext:videoContext,audioContext:audioContext,
      ordinaryReferences:ordinaryReferences,sceneReferences:sceneReferences,
      references:references,frozenAudio:frozenAudio,negativeContexts:negativeContexts,extensionGuides:extensionGuides,unionGuide:unionGuide,icGuides:icGuides,
      stageOneVideoObserver:nil,progress:progress)
  }

  public func evaluateWithStageOneCapture(videoContext:MLXArray,audioContext:MLXArray,
    ordinaryReferences:[[MLXArray]]=[],sceneReferences:[[MLXArray]]=[],
    references:[(first:MLXArray,last:MLXArray?)]=[],frozenAudio:MLXArray?=nil,negativeContexts:[String:MLXArray]?=nil,
    extensionGuides:ExtensionGuides?=nil,unionGuide:MLXArray?=nil,icGuides:[MLXArray]=[],
    stageOneVideoObserver:((MLXArray) throws -> Void)?=nil,
    progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Two-stage MLX sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    try Task.checkCancellation()
    guard references.count == (layouts[0] == nil ? 0 : 2) else { throw LTXError.invalid("Both stage references must be encoded before sampling.") }
    guard ordinaryLayouts[0] == nil ? ordinaryReferences.isEmpty :
      (ordinaryReferences.count == 2 && ordinaryReferences.allSatisfy { $0.count == ordinaryLayouts[0]!.anchors.count }) else {
      throw LTXError.invalid("Ordinary references must retain every image at both resolutions.")
    }
    guard sceneLayouts[0] == nil ? sceneReferences.isEmpty :
      (sceneReferences.count == 2 && sceneReferences.allSatisfy { $0.count == sceneLayouts[0]!.ordinary.anchors.count }) else {
      throw LTXError.invalid("Scene references must retain all routed images at both resolutions.")
    }
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
        let ordinary=self.ordinaryLayouts[stage-1]
        let scene=self.sceneLayouts[stage-1]
        let guideLayout=self.guideLayouts[stage-1]
        let union=stage == 1 ? self.unionLayout : nil
        let ic=stage == 1 ? self.icLayout : nil
        var inputs=text
        inputs["video_latent"]=state["video"]!; inputs["audio_latent"]=state["audio"]!
        var conditioning:MLXVideoDenoiseCondition?
        var audioConditioning:MLXAudioDenoiseCondition?
        if let layout {
          let pair=references[stage-1]
          let prepared=try layout.prepare(generated:state["video"]!,first:pair.first,last:pair.last)
          inputs["video_latent"]=prepared.latent;conditioning=prepared.condition
        }
        if let ordinary {
          // Released legacy scalar-noise flow applies conditionings after the
          // main video draw. Generated slots start at zero; no extra RNG draw.
          let prepared=try ordinary.prepare(generated:state["video"]!,anchors:ordinaryReferences[stage-1])
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
        if let scene {
          let targetAudio=frozenAudio ?? state["audio"]!
          let targetAudioCondition=try scene.sourceAudioConditionForHistory(frozenAudio)
          let guides=extensionGuides.map { source in
            MLXSceneKeyframeLayout.Guides(video:stage == 1 ? source.stageOneVideo:source.stageTwoVideo,
              audio:source.audio,videoNoise:guideNoise["video"]!,audioNoise:guideNoise["audio"]!)
          }
          let prepared=try scene.prepare(generated:state["video"]!,anchors:sceneReferences[stage-1],
            targetAudio:targetAudio,targetAudioCondition:targetAudioCondition,guides:guides,sigma:Float(schedule.sigmas[0]))
          inputs["video_latent"]=prepared.video;inputs["audio_latent"]=prepared.audio
          conditioning=scene.ordinary.anchors.isEmpty && guideLayout == nil ? nil:prepared.videoCondition
          audioConditioning=prepared.audioCondition
        }
        if scene == nil,let guideLayout,let extensionGuides {
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
        let videoTokens=scene?.videoTokens ?? ordinary?.videoTokens ?? guideLayout?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? g.videoTokens
        let audioTokens=scene?.audioTokens ?? guideLayout?.audioTokens ?? g.audioFrames
        inputs["video_positions"]=MLXArray(scene?.videoPositions ?? ordinary?.positions ?? guideLayout?.videoPositions ?? layout?.positions ?? union?.positions ?? ic?.positions ?? g.videoPositions,[videoTokens,3])
        inputs["audio_positions"]=MLXArray(scene?.audioPositions ?? guideLayout?.audioPositions ?? g.audioPositions,[audioTokens,1])
        let bf16:Set<String> = self.noisePolicy == .releasedMLX
          ? (frozenAudio != nil ? (stage == 1 ? ["video"] : []) : (stage == 1 ? ["video","audio"] : ["audio"])) : []
        let sampled:[String:MLXArray]
        let firstUpdates=self.guidedSampling.map { $0.steps+($0.mode == .guidedHQ ? 1 : 0) } ?? 8
        if stage == 1,let guidance=self.guidedSampling {
          guard let negativeContexts else { throw LTXError.invalid("Dev guidance requires encoded negative contexts before transformer loading.") }
          let sampler=try MLXGuidedSamplingRunner(configuration:Self.configuration(g,layout:layout,ordinary:ordinary),
            sampling:guidance,maximumActivationBytes:self.maximumActivationBytes,keyframeMarkerRows:ordinary?.slotTokens ?? 0)
          sampled=try sampler.evaluate(inputs,schedule:schedule,negativeContexts:negativeContexts,
            videoConditioning:conditioning,frozenAudio:frozenAudio != nil,
            fixedWeights:source.readFixed,blockWeights:source.readBlock,
            fixedAdapters:source.fixedAdapters,blockAdapters:source.blockAdapters,
            stageProgress:{ _,event in
              if event.completedBlocks==48,let detail=event.stack {
                self.stageSeconds["stage1_block_load",default:0] += detail.loadSeconds
                self.stageSeconds["stage1_block_compute",default:0] += detail.computeSeconds
              }
              try report("stage1:"+event.stage,event.completedBlocks,48)
            },progress:{ try report("sampling",$0.completedSteps,firstUpdates+3) })
        } else {
          let sampler=try MLXSamplingRunner(configuration:Self.configuration(g,layout:layout,ordinary:ordinary,scene:scene,guide:guideLayout,union:union,ic:ic),maximumActivationBytes:self.maximumActivationBytes,
            keyframeMarkerRows:ordinary?.slotTokens ?? 0)
        sampled=try sampler.evaluate(inputs,schedule:schedule,videoConditioning:conditioning,
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
          progress:{ event in try report("sampling",(stage == 1 ? 0 : firstUpdates)+event.completedSteps,firstUpdates+3) })
        }
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
      firstSchedule:try guidedSampling?.schedule(videoTokens:recipe.low.videoTokens),
      stageOneVideoObserver:stageOneVideoObserver,
      sample:{ stage,g,state,schedule,noise in
        try sampleStage(stage,g,state,[:],schedule,noise)
      },upscale:upscale)
  }
}
