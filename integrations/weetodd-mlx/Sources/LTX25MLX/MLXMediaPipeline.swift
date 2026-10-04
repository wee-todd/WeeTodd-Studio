import Foundation
import Darwin
import MLX
import LTX25Engine
import LTX25Video
import LTX25Audio
import InferenceMedia
import InferenceContracts
import AdapterRuntime

/// Developer end-to-end audiovisual integration, with no Python or NNC dependency.
/// The worker/UI protocol and production-size streaming are separate release gates.
public final class MLXMediaPipeline {
  /// Owns source guide tensors only until sampling finishes. Video and audio
  /// decoders can then load without retaining the encoded context on the GPU.
  public final class ExtensionGuideLease {
    public private(set) var guides:MLXDistilledSamplingRunner.ExtensionGuides?
    public init(_ guides:MLXDistilledSamplingRunner.ExtensionGuides) { self.guides=guides }
    public func release() { guides=nil }
  }
  public static let maximumVideoActivationMiB=32768
  public static let maximumTransformerActivationMiB=32768
  public struct Admission:Sendable {
    public let videoFrames:Int
    public let audioSamples:Int
    public let videoActivationBytes:Int
    public let audioEstimatedBytes:UInt64
  }
  static func extensionAudioRange(contextFrames:Int,additionalFrames:Int,
    fps:Double,decodedSamples:Int) throws -> Range<Int> {
    guard contextFrames>0,additionalFrames>0,fps.isFinite,fps>0 else {
      throw LTXError.invalid("LTX extension audio needs a finite positive causal window.")
    }
    let start=Int((Double(contextFrames)/fps*48000).rounded(.toNearestOrEven))
    let count=Int((Double(additionalFrames)/fps*48000).rounded(.toNearestOrEven))
    guard start>=0,count>0,start<=decodedSamples,count<=decodedSamples-start else {
      throw LTXError.invalid("LTX extension audio decoder did not cover the exact published interval.")
    }
    return start..<(start+count)
  }
  private let request:MLXDistilledRequest
  private let sampler:MLXDistilledSamplingRunner?
  private let singleStageSampler:MLXSingleStageSamplingRunner?
  private let ingredientsSampler:MLXSingleStageRipple?
  private let msrSampler:MLXSingleStageMSR?
  private let dfrSampler:MLXDFRSamplingRunner?
  private let videoActivationBytes:Int
  private let transformerActivationBytes:Int
  private let videoBackend:MLXVideoBackend
  private let audioBackend:MLXAudioBackend
  private let saveLatents:Bool
  private let saveFrames:Bool
  private let extensionContextFrames:Int?
  private let gate=NSLock()

  /// Admission and execution must use the same validated job geometry. Standalone
  /// decoder defaults are conservative probe limits, not Studio duration limits.
  static func videoConfiguration(for geometry:AVGeometry,activationBytes:Int) -> VideoDecodeConfiguration {
    var configuration=VideoDecodeConfiguration()
    configuration.frameRate=geometry.fps
    configuration.maximumLatentFrames=geometry.latentFrames
    configuration.maximumLatentHeight=geometry.latentHeight
    configuration.maximumLatentWidth=geometry.latentWidth
    configuration.maximumActivationBytes=activationBytes
    return configuration
  }

  public static func admit(_ request:MLXDistilledRequest,extensionContextFrames:Int?=nil,
    videoActivationBytes:Int=512*1024*1024,transformerActivationBytes:Int=2*1024*1024*1024,
    videoBackend:MLXVideoBackend = .mps,audioBackend:MLXAudioBackend = .mps) throws -> Admission {
    guard (1...maximumVideoActivationMiB*1024*1024).contains(videoActivationBytes) else {
      throw LTXError.invalid("Developer decoder workspace must be positive and at most \(maximumVideoActivationMiB) MiB.")
    }
    guard (request.guidedSampling == nil && request.singleStageSampling == nil) || extensionContextFrames == nil else {
      throw LTXError.invalid("Dev guided continuation has not been admitted.")
    }
    let recipe=try request.recipe(), g=recipe.high
    let decodeG=try request.dfr.map { dfr in
      try dfr.temporalRounds == 0 ? g : AVGeometry(width:g.width,height:g.height,
        frames:MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:dfr.temporalRounds),
        fps:g.fps*Double(1 << dfr.temporalRounds))
    } ?? g
    let geometries=request.ingredientsSheet == nil && request.msr == nil && request.singleStageSampling == nil ? [recipe.low,recipe.high] : [recipe.high]
    for (index,geometry) in geometries.enumerated() {
      let ordinary=try request.ordinaryKeyframeLayout(geometry:geometry,stage:index)
      let singleStage=try request.singleStageControlLayout()
      let reserve=try request.singleStageSampling?.method == .cfgpp
        ? singleStage!.cfgppReserveBytes(audioTokens:geometry.audioFrames)
        : request.ingredientsSampling == .ancestralCFGPP
        ? MLXSingleStageRipple.cfgppReserveBytes(geometry:geometry) :
        (index == 0 && request.guidedSampling != nil ? MLXGuidedSampling.reserveBytes(
          videoTokens:ordinary?.videoTokens ?? (geometry.videoTokens+(request.referenceImages.count == 2 ? geometry.latentHeight*geometry.latentWidth : 0)),
          audioTokens:geometry.audioFrames) : 0)
      guard reserve < transformerActivationBytes else {
        throw LTXError.invalid("CFG++ exceeds the configured activation budget before model loading.")
      }
      let blockBudget=transformerActivationBytes-reserve
      let layout=try (request.usesOrdinaryKeyframes ? nil : request.referenceImages.first).map { try MLXReferenceLayout(geometry:geometry,firstStrength:$0.strength,lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil) }
      let guide=try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:geometry,contextFrames:$0) }
      let union=try index == 0
        ? request.unionControlGuide.map { try MLXUnionControlLayout(geometry:geometry,strength:$0.referenceStrength) }
        : nil
      let ic=try index == 0 ? request.icControl.map { try MLXICControlLayout(geometry:geometry,control:$0) } : nil
      let ingredients=try request.ingredientsSheet.map {
        try MLXReferenceVideoLayout(geometry:geometry,strength:$0.referenceStrength)
      }
      let msr=try MLXMSRReferencePlan.resolve(request,target:geometry)
      let dfr=try request.dfr.map { _ in try MLXDFRLayout(geometry:geometry,
        slotFrames:MLXDFRCanvas(frames:recipe.high.frames).slotFrames,
        reference:index == 1 ? recipe.low : nil,
        firstStrength:request.referenceImages.first?.strength,
        lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil,
        lastFrame:request.referenceImages.count == 2 ? request.frames-1 : nil) }
      guard layout == nil || guide == nil else { throw LTXError.invalid("LTX extension cannot combine with endpoint references.") }
      let block=try MLXAVBlock(configuration:AVBlockConfiguration(
        videoTokens:singleStage?.videoTokens ?? ordinary?.videoTokens ?? dfr?.videoTokens ?? guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? ingredients?.videoTokens ?? msr?.layout.videoTokens ?? geometry.videoTokens,
        audioTokens:guide?.audioTokens ?? geometry.audioFrames,textTokens:1024),
        maximumActivationBytes:blockBudget)
      if singleStage?.requiresPerTokenVideo == true || ordinary?.anchors.isEmpty == false || layout != nil || union != nil || ic != nil || ingredients != nil || msr != nil || (dfr?.referenceTokens ?? 0) > 0 ||
        (dfr != nil && !request.referenceImages.isEmpty) { try block.admitPerTokenVideo() }
      if guide != nil { try block.admitPerTokenAV() }
      _ = try MLXDenoiser.admitRotary(configuration:block.configuration,maximumActivationBytes:blockBudget)
    }
    if let dfr=request.dfr,dfr.temporalRounds>0 {
      let configurations=try MLXDFRTemporalPlan.admissionConfigurations(
        geometry:g,requestedFrames:request.frames,
        slots:MLXDFRCanvas(frames:g.frames).slotFrames,
        rounds:dfr.temporalRounds,endpointCount:request.referenceImages.count)
      for configuration in configurations {
        let block=try MLXAVBlock(configuration:configuration,
          maximumActivationBytes:transformerActivationBytes)
        try block.admitPerTokenVideo()
        _ = try MLXDenoiser.admitRotary(configuration:configuration,
          maximumActivationBytes:transformerActivationBytes)
      }
      var latentFrames=g.latentFrames
      for _ in 0..<dfr.temporalRounds {
        latentFrames=try MLXTemporalUpscaler.admit(shape:[latentFrames,g.latentHeight,
          g.latentWidth,128],maximumActivationBytes:transformerActivationBytes)[0]
      }
    }
    if !request.referenceImages.isEmpty {
      for geometry in [recipe.low,recipe.high] { _ = try MLXImageEncodePlan(width:geometry.width,height:geometry.height) }
    }
    if request.unionControlGuide != nil {
      _ = try MLXVideoEncodeTilePlan(frames:recipe.low.frames,
        width:recipe.low.width/2,height:recipe.low.height/2,
        maximumOwnedBufferBytes:videoActivationBytes)
    }
    if let control=request.icControl {
      let geometry=try control.guideGeometry(target:recipe.low)
      _ = try MLXVideoEncodeTilePlan(frames:geometry.frames,width:geometry.width,
        height:geometry.height,maximumOwnedBufferBytes:videoActivationBytes)
    }
    if request.ingredientsSheet != nil {
      _ = try MLXVideoEncodeTilePlan(frames:1,
        width:recipe.high.width,height:recipe.high.height,
        maximumOwnedBufferBytes:videoActivationBytes)
    }
    if let msr=try MLXMSRReferencePlan.resolve(request,target:g) {
      for plan in msr.plans {
        _ = try MLXVideoEncodeTilePlan(frames:plan.geometry.frames,
          width:plan.geometry.width,height:plan.geometry.height,
          maximumOwnedBufferBytes:videoActivationBytes)
      }
    }
    let c=videoConfiguration(for:decodeG,activationBytes:videoActivationBytes)
    let video=try MLXNativeVideoDecoder.admit(checkpoint:URL(fileURLWithPath:request.videoCheckpoint),
      settings:request.diffusionVAE,shape:decodeG.videoShape,configuration:c,backend:videoBackend)
    let videoShape=video.shape,videoBytes=video.bytes
    let audio=try audioBackend == .mlx ? MLXAudioDecoder.estimatedPeakBytes(latentFrames:g.audioFrames) : AudioDecoder.estimatedPeakBytes(latentFrames:g.audioFrames)
    guard videoShape == [decodeG.frames,decodeG.height,decodeG.width,3], g.audioFrames <= 1501, audio <= 2*1024*1024*1024 else {
      throw LTXError.invalid("Media geometry exceeds the native decoder admission.")
    }
    return Admission(videoFrames:decodeG.frames,audioSamples:try AudioDecoder.sampleCount(latentFrames:g.audioFrames),
      videoActivationBytes:videoBytes,audioEstimatedBytes:audio)
  }
  public init(request:MLXDistilledRequest,extensionContextFrames:Int?=nil,
    videoActivationBytes:Int=512*1024*1024,transformerActivationBytes:Int=2*1024*1024*1024,
    videoBackend:MLXVideoBackend = .mps,audioBackend:MLXAudioBackend = .mps,saveLatents:Bool=false,saveFrames:Bool=false) throws {
    _ = try Self.admit(request,extensionContextFrames:extensionContextFrames,
      videoActivationBytes:videoActivationBytes,transformerActivationBytes:transformerActivationBytes,
      videoBackend:videoBackend,audioBackend:audioBackend)
    guard request.dfr?.temporalRounds == 0 || !saveLatents else {
      throw LTXError.invalid("Temporal DFR latent capture needs a separate video/audio geometry contract.")
    }
    let output=URL(fileURLWithPath:request.outputDirectory)
    guard !FileManager.default.fileExists(atPath:output.path) else { throw LTXError.invalid("Output directory already exists.") }
    self.request=request
    self.videoBackend=videoBackend;self.audioBackend=audioBackend;self.saveLatents=saveLatents;self.saveFrames=saveFrames
    self.extensionContextFrames=extensionContextFrames
    self.videoActivationBytes=videoActivationBytes;self.transformerActivationBytes=transformerActivationBytes
    if let control=request.icControl {
      let geometry=try control.guideGeometry(target:request.recipe().low)
      let expected=Int64(geometry.frames)*Int64(geometry.width)*Int64(geometry.height)*3
      for guide in control.guides {
        try NativeMediaSource(path:guide.path,sha256:guide.sourceSHA256).verify()
        let attributes=try FileManager.default.attributesOfItem(atPath:guide.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
          (attributes[.size] as? NSNumber)?.int64Value == expected else {
          throw LTXError.invalid("IC control RGB24 guide differs from its admitted geometry.")
        }
      }
      if let audio=control.publicationAudio {
        try NativeMediaSource(path:audio.path,sha256:audio.sourceSHA256).verify()
        _ = try MLXSourceAudioInterval(source:URL(fileURLWithPath:audio.path),
          sourceStartSeconds:audio.sourceStartSeconds,sourceDurationSeconds:audio.sourceDurationSeconds,
          durationSeconds:Double(request.frames)/request.fps)
      }
      _ = try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
    }
    if let policy=request.singleStageSampling {
      sampler=nil;ingredientsSampler=nil;msrSampler=nil;dfrSampler=nil
      singleStageSampler=try MLXSingleStageSamplingRunner(geometry:request.recipe().high,policy:policy,
        anchors:zip(request.referenceFrames,request.referenceImages).map { .init(frame:$0.0,strength:$0.1.strength) },
        generatedCount:request.generatedKeyframes ?? 0,transformerRoot:URL(fileURLWithPath:request.transformerRoot),
        adapters:request.stageOneLoras,icControl:request.icControl,unionControlGuide:request.unionControlGuide,
        maximumActivationBytes:transformerActivationBytes)
    } else if let sheet=request.ingredientsSheet {
      singleStageSampler=nil
      sampler=nil
      msrSampler=nil
      dfrSampler=nil
      ingredientsSampler=try MLXSingleStageRipple(geometry:request.recipe().high,
        referenceStrength:sheet.referenceStrength,
        transformerRoot:URL(fileURLWithPath:request.transformerRoot),
        adapters:[LoRAAdapter(path:sheet.adapterPath,strength:sheet.adapterStrength)],
        task:.ingredients,ingredientsSampling:request.ingredientsSampling,
        maximumActivationBytes:transformerActivationBytes)
    } else if let msr=request.msr {
      singleStageSampler=nil
      sampler=nil;ingredientsSampler=nil;dfrSampler=nil
      guard let layout=try MLXMSRReferencePlan.resolve(request,target:request.recipe().high)?.layout else {
        throw LTXError.invalid("MSR reference layout is missing.")
      }
      msrSampler=try MLXSingleStageMSR(layout:layout,
        transformerRoot:URL(fileURLWithPath:request.transformerRoot),
        adapter:LoRAAdapter(path:msr.adapterPath,strength:msr.adapterStrength),
        maximumActivationBytes:transformerActivationBytes)
    } else if let dfr=request.dfr {
      singleStageSampler=nil
      sampler=nil;ingredientsSampler=nil;msrSampler=nil
      let recipe=try request.recipe()
      dfrSampler=try MLXDFRSamplingRunner(recipe:recipe,
        transformerRoot:URL(fileURLWithPath:request.transformerRoot),
        upscalerCheckpoint:URL(fileURLWithPath:request.spatialUpscalerCheckpoint),
        statisticsCheckpoint:URL(fileURLWithPath:request.videoCheckpoint),
        slotFrames:try MLXDFRCanvas(frames:recipe.high.frames).slotFrames,
        detailingAdapter:LoRAAdapter(path:dfr.adapterPath,strength:dfr.adapterStrength),
        firstStrength:request.referenceImages.first?.strength,
        lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil,
        temporalUpscalerCheckpoint:dfr.temporalUpscalerPath.map(URL.init(fileURLWithPath:)),
        temporalRounds:dfr.temporalRounds,requestedFrames:request.frames,
        maximumActivationBytes:transformerActivationBytes)
    } else {
      singleStageSampler=nil
      ingredientsSampler=nil;msrSampler=nil;dfrSampler=nil
      sampler=try MLXDistilledSamplingRunner(recipe:request.recipe(),transformerRoot:URL(fileURLWithPath:request.transformerRoot),
        upscalerCheckpoint:URL(fileURLWithPath:request.spatialUpscalerCheckpoint),statisticsCheckpoint:URL(fileURLWithPath:request.videoCheckpoint),
        firstStrength:request.usesOrdinaryKeyframes ? nil : request.referenceImages.first?.strength,
        lastStrength:!request.usesOrdinaryKeyframes && request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil,
        ordinaryAnchors:request.usesOrdinaryKeyframes ? zip(request.referenceFrames,request.referenceImages).map { .init(frame:$0.0,strength:$0.1.strength) } : nil,
        generatedKeyframes:request.generatedKeyframes ?? 0,
        extensionContextFrames:extensionContextFrames,
        unionControlGuide:request.unionControlGuide,icControl:request.icControl,
        stageOneLoras:request.stageOneLoras,stageTwoLoras:request.stageTwoLoras,noisePolicy:request.noisePolicy,guidedSampling:request.guidedSampling,maximumActivationBytes:transformerActivationBytes)
    }
    for reference in request.referenceImages { try MLXReferenceImage.inspect(URL(fileURLWithPath:reference.path)) }
    if !request.referenceImages.isEmpty { _ = try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint)) }
    if let union=request.unionControlGuide {
      try NativeMediaSource(path:union.path,sha256:union.sourceSHA256).verify()
      let recipe=try request.recipe()
      let attributes=try FileManager.default.attributesOfItem(atPath:union.path)
      let expected=Int64(recipe.low.frames)*Int64(recipe.low.height/2)*Int64(recipe.low.width/2)*3
      guard attributes[.type] as? FileAttributeType == .typeRegular,
        (attributes[.size] as? NSNumber)?.int64Value == expected else {
        throw LTXError.invalid("Union Control RGB24 guide must match the half-resolution stage-one timeline.")
      }
      _ = try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
    }
    if let sheet=request.ingredientsSheet {
      try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
      try MLXReferenceImage.inspect(URL(fileURLWithPath:sheet.path))
      _ = try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
    }
    if let msr=request.msr {
      for reference in msr.references {
        try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256).verify()
        try MLXReferenceImage.inspect(URL(fileURLWithPath:reference.path))
      }
      _ = try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
    }
    if let source=request.audioReference {
      _ = try MLXSourceAudioInterval(source:URL(fileURLWithPath:source.path),
        sourceStartSeconds:source.sourceStartSeconds,sourceDurationSeconds:source.sourceDurationSeconds,
        durationSeconds:Double(request.frames)/request.fps)
      _ = try MLXAudioEncoder(checkpoint:URL(fileURLWithPath:request.audioCheckpoint))
    }
    // Every component header/tokenizer is validated before the first weighted stage.
    try autoreleasepool {
      let text=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:request.gemmaRoot),connectorURL:URL(fileURLWithPath:request.connectorCheckpoint))
      _ = try MLXTextEncodingPlan(promptTokens:text.tokenize(request.prompt).count)
      _ = try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:request.videoCheckpoint),settings:request.diffusionVAE)
      try AudioDecoder.validateCheckpoint(URL(fileURLWithPath:request.audioCheckpoint))
    }
  }
  public func run(ffmpeg:URL,progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> URL {
    try run(ffmpeg:ffmpeg,decodedPreview:nil,progress:progress)
  }
  public func run(ffmpeg:URL,preparedAudio:MLXSourceAudioInterval.Prepared?=nil,
    preparedPublicationAudio:MLXSourceAudioInterval.Prepared?=nil,
    extensionGuideLease:ExtensionGuideLease?=nil,
    preparedTextLease:MLXPreparedTextLease?=nil,textPreparationBinding:MLXTextPreparationBinding?=nil,
    textPreparationSeconds:Double=0,
    decodedPreview:((Int,Data) throws -> Void)?,beforePublish:((URL,[String:Any]) throws -> Void)? = nil,progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> URL {
    defer { preparedTextLease?.release() }
    guard (request.automaticDuration != nil) == (preparedTextLease != nil),
      (preparedTextLease != nil) == (textPreparationBinding != nil),
      request.automaticDuration.map({ $0.headHeaderSHA256 != nil }) ?? true,
      textPreparationSeconds.isFinite,textPreparationSeconds>=0 else {
      throw LTXError.invalid("Automatic geometry requires its single-use prepared text lease and frozen binding.")
    }
    let automaticResolution=preparedTextLease?.resolution
    guard gate.try() else { throw LTXError.invalid("Media pipeline is already running.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    defer { extensionGuideLease?.release() }
    guard ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath:ffmpeg.path) else {
      throw LTXError.invalid("An explicit executable FFmpeg is required for this developer probe.")
    }
    let fm=FileManager.default, output=URL(fileURLWithPath:request.outputDirectory)
    guard !fm.fileExists(atPath:output.path) else { throw LTXError.invalid("Output already exists.") }
    let parent=output.deletingLastPathComponent()
    try fm.createDirectory(at:parent,withIntermediateDirectories:true)
    let temporary=parent.appendingPathComponent(".weetodd-partial-"+UUID().uuidString)
    try fm.createDirectory(at:temporary,withIntermediateDirectories:false)
    var published=false
    defer { if !published { try? fm.removeItem(at:temporary) } }
    let frames=temporary.appendingPathComponent("frames")
    let rawVideo=videoBackend == .mlx && !saveFrames
    if !rawVideo { try fm.createDirectory(at:frames,withIntermediateDirectories:false) }
    try JSONEncoder().encode(request).write(to:temporary.appendingPathComponent("request.json"),options:.withoutOverwriting)
    let admission=try Self.admit(request,extensionContextFrames:extensionContextFrames,
      videoActivationBytes:videoActivationBytes,transformerActivationBytes:transformerActivationBytes,
      videoBackend:videoBackend,audioBackend:audioBackend), g=try request.recipe().high, started=Date()
    let outputG=try request.dfr.map { dfr in
      try dfr.temporalRounds == 0 ? g : AVGeometry(width:g.width,height:g.height,
        frames:MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:dfr.temporalRounds),
        fps:g.fps*Double(1 << dfr.temporalRounds))
    } ?? g
    guard (request.audioReference == nil) == (preparedAudio == nil),
      preparedAudio == nil || (preparedAudio!.publicationSamples == Int((Double(g.frames)/g.fps*48000).rounded(.toNearestOrEven)) &&
        preparedAudio!.conditioningSamples == Int((Double(g.frames)/g.fps*16000).rounded(.toNearestOrEven))) else {
      throw LTXError.invalid("A2V needs its exact prepared source interval before weighted execution.")
    }
    guard (request.icControl?.publicationAudio != nil) == (preparedPublicationAudio != nil),
      preparedPublicationAudio == nil || (preparedAudio == nil &&
        preparedPublicationAudio!.publicationSamples == Int((Double(g.frames)/g.fps*48000).rounded(.toNearestOrEven))) else {
      throw LTXError.invalid("CrossView requires its exact publication-only source waveform before weighted execution.")
    }
    let publicationSource=preparedAudio ?? preparedPublicationAudio
    guard (extensionGuideLease?.guides != nil) == (extensionContextFrames != nil),
      extensionGuideLease == nil || preparedAudio == nil else {
      throw LTXError.invalid("LTX extension needs its staged audiovisual source guides without an A2V driver.")
    }
    var timings:[String:Double]=[:], ids:[Int]=[]
    func report(_ stage:String,_ completed:Int=0,_ total:Int=1) throws {
      try Task.checkCancellation(); try progress(stage,completed,total); try Task.checkCancellation()
    }
    if preparedTextLease == nil { Memory.peakMemory=0 }
    let imageDirectory=temporary.appendingPathComponent("prepared-references")
    try fm.createDirectory(at:imageDirectory,withIntermediateDirectories:false)
    let preparationStart=Date(),recipe=try request.recipe()
    let ingredientsGuide=try request.ingredientsSheet.map {
      try MLXIngredientsGuide.prepare($0,geometry:recipe.high,ffmpeg:ffmpeg,directory:imageDirectory)
    }
    let msrResolved=try MLXMSRReferencePlan.resolve(request,target:recipe.high)
    let msrGuides:[URL]=try msrResolved.map { resolved in
      try zip(resolved.references,resolved.plans).enumerated().map { index,pair in
        try report("msr_reference_prepare",index,resolved.references.count)
        return try MLXMSRGuide.prepare(pair.0,plan:pair.1,ffmpeg:ffmpeg,
          directory:imageDirectory,index:index)
      }
    } ?? []
    let preparedImages=try MLXReferenceImage.prepareStages(request.referenceImages,
      sizes:request.singleStageSampling == nil ? [(recipe.low.width,recipe.low.height),(recipe.high.width,recipe.high.height)] : [(recipe.high.width,recipe.high.height)],ffmpeg:ffmpeg,directory:imageDirectory) {
        try report("reference_prepare:stage\($0+1):"+$1)
      }
    if !request.referenceImages.isEmpty {
      timings["reference_prepare"]=Date().timeIntervalSince(preparationStart)
      try report("reference_preparation_complete")
    }
    let latents=try autoreleasepool {
      let frozenAudio:MLXArray?=try autoreleasepool {
        guard let source=preparedAudio else { return nil }
        let start=Date()
        let mel=try MLXAudioMel.encode(wav:source.conditioning)
        let encoder=try MLXAudioEncoder(checkpoint:URL(fileURLWithPath:request.audioCheckpoint))
        let raw=try encoder.encode(mel:mel) { try report("audio_encode",$0,$1) }
        let fitted=raw.shape[0] < g.audioFrames
          ? concatenated([raw,MLXArray.zeros([g.audioFrames-raw.shape[0],128])],axis:0)
          : raw[0..<g.audioFrames]
        eval(fitted);timings["audio_encode"]=Date().timeIntervalSince(start)
        try report("audio_encoder_weights_released")
        return fitted
      }
      let textStarted=Date()
      let (contexts,unconditionalContexts)=try autoreleasepool {
        if let preparedTextLease,let textPreparationBinding {
          let expected=try MLXTextPreparationBinding(originalRecipeSHA256:textPreparationBinding.originalRecipeSHA256,
            prompt:request.prompt,negativePrompt:request.guidedSampling?.negativePrompt,
            gemmaRoot:request.gemmaRoot,connectorCheckpoint:request.connectorCheckpoint)
          let text=try preparedTextLease.consume(expectedBinding:expected,resolvedFrames:request.frames,fps:request.fps)
          return (text.positive,text.negative)
        }
        let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:request.gemmaRoot),connectorURL:URL(fileURLWithPath:request.connectorCheckpoint))
        let positive=try encoder.encode(prompt:request.prompt) { try report("text:"+$0.stage,$0.completed,$0.total) }
        let negative:MLXTextEncoder.Output?
        if request.ingredientsSampling == .ancestralCFGPP || request.guidedSampling != nil || request.singleStageSampling?.method == .cfgpp {
          negative=try encoder.encode(prompt:request.singleStageSampling?.negativePrompt ?? request.guidedSampling?.negativePrompt ?? "") { try report("text:unconditional:"+$0.stage,$0.completed,$0.total) }
        } else { negative=nil }
        return (positive,negative)
      }
      ids=contexts.tokenIDs; timings["text"]=preparedTextLease == nil ? Date().timeIntervalSince(textStarted) : textPreparationSeconds
      try report("text_weights_released")
      let referenceStart=Date()
      var singleStageControlGuides:[MLXArray]=[]
      let preparedGuides:([(first:MLXArray,last:MLXArray?)],MLXArray?,[MLXArray])=try autoreleasepool {
        if request.singleStageSampling != nil && (request.icControl != nil || request.unionControlGuide != nil) {
          // Ordinary tensors and control tensors have independent typed inputs.
          // Release the ordinary-image encoder before streaming any video guide.
          let ordinary:[MLXArray]=try autoreleasepool {
            guard !request.referenceImages.isEmpty else { return [] }
            guard preparedImages.count==1,preparedImages[0].count==request.referenceImages.count else {
              throw LTXError.invalid("Single-stage ordinary references changed their prepared geometry.")
            }
            let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
            var cached:[URL:MLXArray]=[:],encoded:[MLXArray]=[]
            for (index,image) in preparedImages[0].enumerated() {
              if let reused=cached[image.file] { encoded.append(reused.reshaped(reused.shape));continue }
              let latent=try encoder.encode(MLXArray(try image.pixels(),[1,image.height,image.width,3])) {
                try report("reference_encode:stage1:"+request.referenceImages[index].role,$0,42)
              }
              cached[image.file]=latent;encoded.append(latent)
            }
            return encoded
          }
          Stream.gpu.synchronize();Memory.clearCache()
          if let control=request.icControl {
            let geometry=try control.guideGeometry(target:recipe.high)
            let plan=try MLXVideoEncodeTilePlan(frames:geometry.frames,width:geometry.width,height:geometry.height,
              maximumOwnedBufferBytes:videoActivationBytes)
            for (index,guide) in control.guides.enumerated() {
              let latent=try MLXTiledVideoEncoder.encode(guide:URL(fileURLWithPath:guide.path),
                checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:plan) {
                  try report("ic_guide_encode:\(index+1)",$0,$1)
                }
              let tokens=latent.reshaped([geometry.videoTokens,128]).asType(.float32)
              eval(tokens);singleStageControlGuides.append(tokens)
              try NativeMediaSource(path:guide.path,sha256:guide.sourceSHA256).verify()
              try report("ic_guide_ready",index+1,control.guides.count)
            }
          } else if let union=request.unionControlGuide {
            let plan=try MLXVideoEncodeTilePlan(frames:recipe.high.frames,width:recipe.high.width/2,
              height:recipe.high.height/2,maximumOwnedBufferBytes:videoActivationBytes)
            let latent=try MLXTiledVideoEncoder.encode(guide:URL(fileURLWithPath:union.path),
              checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:plan) {
                try report("control_guide_encode",$0,$1)
              }
            let rows=recipe.high.latentFrames*(recipe.high.latentHeight/2)*(recipe.high.latentWidth/2)
            singleStageControlGuides=[latent.reshaped([rows,128]).asType(.float32)]
            eval(singleStageControlGuides[0])
            try NativeMediaSource(path:union.path,sha256:union.sourceSHA256).verify()
          }
          return ([],nil,ordinary)
        }
        if let resolved=msrResolved {
          guard let msr=request.msr else { throw LTXError.invalid("MSR request is missing.") }
          let slots=try MLXMSRSlotEmbedding.load(URL(fileURLWithPath:msr.adapterPath))
          var encoded:[MLXArray]=[]
          for (index,plan) in resolved.plans.enumerated() {
            let tile=try MLXVideoEncodeTilePlan(frames:plan.geometry.frames,
              width:plan.geometry.width,height:plan.geometry.height,
              maximumOwnedBufferBytes:videoActivationBytes)
            let latent=try MLXTiledVideoEncoder.encode(guide:msrGuides[index],
              checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:tile) {
                try report("msr_guide_encode:\(index+1)",$0,$1)
              }
            let slot=try MLXMSRSlotEmbedding.embedding(slotID:index+1,state:slots)
            let tokens=latent.reshaped([plan.geometry.videoTokens,128]).asType(.float32)+slot
            eval(tokens);encoded.append(tokens)
            try NativeMediaSource(path:resolved.references[index].path,
              sha256:resolved.references[index].sourceSHA256).verify()
            try report("msr_guide_ready",index+1,resolved.plans.count)
          }
          return ([],nil,encoded)
        }
        if let ingredientsGuide {
          let encoded=try MLXIngredientsGuide.encodeStatic(guide:ingredientsGuide,
            checkpoint:URL(fileURLWithPath:request.videoCheckpoint),geometry:recipe.high,
            maximumOwnedBufferBytes:videoActivationBytes) {
            try report("ingredients_guide_encode",$0,$1)
          }
          if let sheet=request.ingredientsSheet {
            try NativeMediaSource(path:sheet.path,sha256:sheet.sourceSHA256).verify()
          }
          return ([],encoded.reshaped([recipe.high.videoTokens,128]),[])
        }
        if let control=request.icControl {
          let geometry=try control.guideGeometry(target:recipe.low)
          let plan=try MLXVideoEncodeTilePlan(frames:geometry.frames,width:geometry.width,
            height:geometry.height,maximumOwnedBufferBytes:videoActivationBytes)
          var encoded:[MLXArray]=[]
          for (index,guide) in control.guides.enumerated() {
            let latent=try MLXTiledVideoEncoder.encode(guide:URL(fileURLWithPath:guide.path),
              checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:plan) {
                try report("ic_guide_encode:\(index+1)",$0,$1)
              }
            let tokens=latent.reshaped([geometry.videoTokens,128]).asType(.float32)
            eval(tokens);encoded.append(tokens)
            try NativeMediaSource(path:guide.path,sha256:guide.sourceSHA256).verify()
            try report("ic_guide_ready",index+1,control.guides.count)
          }
          return ([],nil,encoded)
        }
        if let union=request.unionControlGuide {
          let plan=try MLXVideoEncodeTilePlan(frames:recipe.low.frames,
            width:recipe.low.width/2,height:recipe.low.height/2,
            maximumOwnedBufferBytes:videoActivationBytes)
          let encoded=try MLXTiledVideoEncoder.encode(guide:URL(fileURLWithPath:union.path),
            checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:plan) {
            try report("control_guide_encode",$0,$1)
          }
          try NativeMediaSource(path:union.path,sha256:union.sourceSHA256).verify()
          return ([],encoded.reshaped([plan.latentShape[0]*plan.latentShape[1]*plan.latentShape[2],128]),[])
        }
        guard !request.referenceImages.isEmpty else { return ([],nil,[]) }
        let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
        var cached:[URL:MLXArray]=[:],prepared:[(first:MLXArray,last:MLXArray?)]=[],ordinary:[MLXArray]=[]
        for (index,images) in preparedImages.enumerated() {
          var tokens:[MLXArray]=[]
          for (referenceIndex,image) in images.enumerated() {
            if let reused=cached[image.file] { tokens.append(reused.reshaped(reused.shape));continue }
            let encoded=try autoreleasepool {
              let rgb=try image.pixels()
              return try encoder.encode(MLXArray(rgb,[1,image.height,image.width,3])) {
                try report("reference_encode:stage\(index+1):"+request.referenceImages[referenceIndex].role,$0,42)
              }
            }
            cached[image.file]=encoded;tokens.append(encoded)
          }
          if request.usesOrdinaryKeyframes { ordinary.append(contentsOf:tokens) }
          else { prepared.append((tokens[0],tokens.count == 2 ? tokens[1] : nil)) }
        }
        return (prepared,nil,ordinary)
      }
      try fm.removeItem(at:imageDirectory)
      if !preparedGuides.0.isEmpty || preparedGuides.1 != nil || !preparedGuides.2.isEmpty || !singleStageControlGuides.isEmpty {
        timings["reference_encode"]=Date().timeIntervalSince(referenceStart)
        try report("reference_weights_released")
      }
      let sampled:[String:MLXArray]
      if let singleStageSampler {
        let started=Date()
        sampled=try singleStageSampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,
          negativeContexts:unconditionalContexts.map { ["video_text":$0.video,"audio_text":$0.audio] },
          references:preparedGuides.2,guides:singleStageControlGuides,frozenAudio:frozenAudio,seed:request.seed,progress:report)
        timings["sampling"]=Date().timeIntervalSince(started)
      } else if let dfrSampler {
        let started=Date()
        sampled=try dfrSampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,
          references:preparedGuides.0,
          progress:report)
        timings["sampling"]=Date().timeIntervalSince(started)
        timings.merge(dfrSampler.stageSeconds) { _,new in new }
      } else if let msrSampler {
        let started=Date()
        sampled=try msrSampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,
          references:preparedGuides.2,seed:request.seed,progress:report)
        timings["sampling"]=Date().timeIntervalSince(started)
      } else if let ingredientsSampler {
        guard let guide=preparedGuides.1 else {
          throw LTXError.invalid("Ingredients reference sheet was not encoded.")
        }
        let started=Date()
        sampled=try ingredientsSampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,
          unconditionalVideoContext:unconditionalContexts?.video,unconditionalAudioContext:unconditionalContexts?.audio,
          referenceVideo:guide,seed:request.seed,progress:report)
        timings["sampling"]=Date().timeIntervalSince(started)
      } else {
        guard let sampler else { throw LTXError.invalid("LTX sampler is missing.") }
        let count=request.referenceImages.count
        let ordinary:[[MLXArray]]=request.usesOrdinaryKeyframes ?
          [Array(preparedGuides.2.prefix(count)),Array(preparedGuides.2.dropFirst(count))] : []
        sampled=try sampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,
          ordinaryReferences:ordinary,references:preparedGuides.0,
          frozenAudio:frozenAudio,negativeContexts:unconditionalContexts.map { ["video_text":$0.video,"audio_text":$0.audio] },
          extensionGuides:extensionGuideLease?.guides,
          unionGuide:preparedGuides.1,icGuides:request.icControl != nil ? preparedGuides.2 : [],progress:report)
        timings.merge(sampler.stageSeconds) { _,new in new }
      }
      // One final evaluated download for existing native decoders. The contexts
      // and all MLX latent handles leave scope before a VAE owns weighted buffers.
      return AVLatents(video:sampled["video"]!.asArray(Float.self),audio:sampled["audio"]!.asArray(Float.self))
    }
    extensionGuideLease?.release()
    Stream.gpu.synchronize(); Memory.clearCache()
    try report("sampling_weights_released")
    if saveLatents {
      let captureStart=Date()
      try autoreleasepool { try MLXDecodeSnapshot.write(latents,geometry:g,to:temporary.appendingPathComponent("latents.safetensors")) }
      Memory.clearCache()
      timings["latent_capture"]=Date().timeIntervalSince(captureStart)
      try report("latents_saved",1,1)
    }
    let videoConfig=Self.videoConfiguration(for:outputG,activationBytes:videoActivationBytes)
    let contextFrames=extensionContextFrames ?? 0
    let publishedFrames=request.dfr == nil ? g.frames-contextFrames
      : request.dfr!.temporalRounds == 0 ? request.frames : outputG.frames
    let videoStart=Date(); var written=0,videoCacheBytes:Int?
    try autoreleasepool {
      func receive(_ chunk:VideoFrameChunk) throws {
        guard chunk.startFrame == written, chunk.frameCount == 1, chunk.width == outputG.width,
          chunk.height == outputG.height, chunk.frameRate == outputG.fps else { throw LTXError.invalid("Decoded frame contract mismatch.") }
        if written >= contextFrames && written-contextFrames < publishedFrames {
          try MediaOutput.writePNG(chunk.rgb,width:outputG.width,height:outputG.height,
            to:frames.appendingPathComponent(String(format:"%06d.png",written-contextFrames)))
        }
        written += 1; try report("video_decode",written,outputG.frames)
      }
      let unpacked=try outputG.unpackVideo(latents.video),checkpoint=URL(fileURLWithPath:request.videoCheckpoint)
      switch videoBackend {
      case .mps:
        try VideoDecoder(checkpoint:checkpoint).decode(latent:unpacked,shape:outputG.videoShape,configuration:videoConfig,receive:receive)
      case .mlx:
        let decoder=try MLXNativeVideoDecoder(checkpoint:checkpoint,settings:request.diffusionVAE,
          maximumWorkspaceBytes:videoActivationBytes)
        videoCacheBytes=decoder.cacheLimitBytes
        if decoder.isDiffusion && !rawVideo {
          try decoder.decodeRGB8(latent:MLXArray(unpacked,outputG.videoShape),configuration:videoConfig,
            progress:{ try report("video_layers",$0,$1) }) { index,bytes in
              if index >= contextFrames && index-contextFrames < publishedFrames {
                try MediaOutput.writeRGB8PNG(bytes,width:outputG.width,height:outputG.height,
                  to:frames.appendingPathComponent(String(format:"%06d.png",index-contextFrames)))
                try decodedPreview?(index-contextFrames,bytes)
              }
              written += 1;try report("video_decode",written,outputG.frames)
            }
        } else if rawVideo {
          let writer=try RawVideoWriter(ffmpeg:ffmpeg,output:temporary.appendingPathComponent("video.mp4"),
            width:outputG.width,height:outputG.height,frames:publishedFrames,fps:outputG.fps)
          defer { writer.cancel() }
          try decoder.decodeRGB8(latent:MLXArray(unpacked,outputG.videoShape),configuration:videoConfig,
            progress:{ try report("video_layers",$0,$1) }) { index,bytes in
              if index >= contextFrames && index-contextFrames < publishedFrames {
                try writer.append(bytes,frame:index-contextFrames)
                try decodedPreview?(index-contextFrames,bytes)
              }
              written += 1;try report("video_decode",written,outputG.frames)
            }
          try writer.finish()
        } else {
          try decoder.decode(latent:MLXArray(unpacked,outputG.videoShape),configuration:videoConfig,
            progress:{ try report("video_layers",$0,$1) },receive:receive)
        }
      }
    }
    guard written == admission.videoFrames else { throw LTXError.invalid("Incomplete video decode.") }
    timings["video_decode"]=Date().timeIntervalSince(videoStart); try report("video_weights_released")
    let audioStart=Date()
    if let source=publicationSource {
      try Self.publishSourceAudio(source,to:temporary.appendingPathComponent("audio.wav"))
      try report("source_audio_published")
    } else { try autoreleasepool {
      let wave=try Self.decodeAudio(latent:g.unpackAudio(latents.audio),latentFrames:g.audioFrames,
        checkpoint:URL(fileURLWithPath:request.audioCheckpoint),backend:audioBackend) { try report("audio:"+$0) }
      guard wave.frameCount == admission.audioSamples, wave.channels == 2, wave.sampleRate == 48000 else {
        throw LTXError.invalid("Decoded audio timing contract mismatch.")
      }
      let range=request.dfr != nil ? 0..<min(wave.frameCount,Int((Double(publishedFrames)/outputG.fps*48000).rounded(.toNearestOrEven)))
        : contextFrames==0 ? 0..<wave.frameCount
        : try Self.extensionAudioRange(contextFrames:contextFrames,
          additionalFrames:publishedFrames,fps:g.fps,decodedSamples:wave.frameCount)
      let published=Array(wave.samples[(range.lowerBound*wave.channels)..<(range.upperBound*wave.channels)])
      try MediaOutput.writeWAV(samples:published,sampleRate:wave.sampleRate,channels:wave.channels,
        to:temporary.appendingPathComponent("audio.wav"))
    } }
    timings[publicationSource == nil ? "audio_decode" : "source_audio_copy"]=Date().timeIntervalSince(audioStart)
    if publicationSource == nil { try report("audio_weights_released") }
    let muxStart=Date()
    try Self.mux(ffmpeg:ffmpeg,directory:temporary,fps:outputG.fps,rawVideo:rawVideo)
    timings["mux"]=Date().timeIntervalSince(muxStart)
    var info=task_vm_info_data_t(), count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { ptr in ptr.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
      task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
    } }
    guard status == KERN_SUCCESS else { throw LTXError.invalid("Cannot measure process memory.") }
    let publishedAudioSamples:Int
    if let source=publicationSource { publishedAudioSamples=source.publicationSamples }
    else if request.dfr != nil { publishedAudioSamples=min(admission.audioSamples,
      Int((Double(publishedFrames)/outputG.fps*48000).rounded(.toNearestOrEven))) }
    else if contextFrames==0 { publishedAudioSamples=admission.audioSamples }
    else {
      publishedAudioSamples=try Self.extensionAudioRange(contextFrames:contextFrames,
        additionalFrames:publishedFrames,fps:g.fps,decodedSamples:admission.audioSamples).count
    }
    var metadata:[String:Any]=["status":"complete","scope":"developer Swift MLX distilled audiovisual generation",
      "recipe":request.singleStageSampling == nil ? request.dfr.map { $0.temporalRounds > 0 ? "ltx25-dfr-spatiotemporal-distilled-v1" : "ltx25-dfr-spatial-distilled-v1" } ?? (request.msr != nil ? "ltx25-msr-single-stage-v1" : request.ingredientsSheet == nil ? DistilledTwoStageRecipe.identifier : request.ingredientsSampling == .ancestralCFGPP ? "ltx25-ingredients-ancestral-cfgpp-v1" : "ltx25-ingredients-single-stage-v1") : "ltx25-ordinary-single-stage-v1",
      "noise_algorithm":request.noisePolicy.algorithm,"seed":request.seed,
      "text_token_ids":ids,"width":outputG.width,"height":outputG.height,"frames":publishedFrames,"fps":outputG.fps,
      "video_seconds":Double(publishedFrames)/outputG.fps,
      "audio_samples":publishedAudioSamples,
      "audio_sample_rate":48000,
      "audio_seconds":Double(publishedAudioSamples)/48000,
      "audio_timing":request.dfr != nil ? "crop padded DFR canvas audio to requested video duration"
        : contextFrames>0 ? "crop decoded causal audio to exact generated duration"
        : publicationSource == nil ? "retain actual causal decoder samples; no stretch or shortest trim"
        : "original source waveform, trimmed and padded to output duration",
      "stage_seconds":timings,"seconds":Date().timeIntervalSince(started),"peak_mlx_bytes":Memory.peakMemory,
      "peak_process_footprint_bytes":info.ledger_phys_footprint_peak,"current_process_footprint_bytes":info.phys_footprint,
      "process_memory_scope":"Swift process; external FFmpeg process excluded",
      "video_decoder":videoBackend.rawValue,"decoder_latents_saved":saveLatents,
      "video_publication":rawVideo ? "gpu-rgb24-pipe" : "png-sequence",
      "video_precision":videoBackend == .mlx ? "bfloat16" : "float32",
      "block_loading":"mlx-native-batched-evaluation","compiled_blocks":true,
      "spatial_upscaler":request.ingredientsSheet == nil && request.msr == nil && request.singleStageSampling == nil ? "mlx" : "none",
      "text_weight_loading":"mlx-native-layer-batched-with-bounded-row-projections",
      "fixed_weight_loading":"mlx-native-evaluated-parameters",
      "audio_decoder":audioBackend.rawValue,"audio_estimated_workspace_bytes":admission.audioEstimatedBytes,
      "transformer_activation_budget_bytes":transformerActivationBytes,"video_activation_budget_bytes":videoActivationBytes,"video_admitted_activation_bytes":admission.videoActivationBytes,
      "python_inference":false,"production_qualified":false,"task":request.task,
      "reference_count":request.referenceImages.count,"reference_preparation":MLXReferenceImage.policy,
      "reference_conditioning":"first latent replacement and appended last-frame tokens at both resolutions; no output-frame paste"]
    let decoderSelection=try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:request.videoCheckpoint),settings:request.diffusionVAE)
    metadata["video_decoder_architecture"]=decoderSelection.isDiffusion ? "ltx25-one-step-diffusion-vae" : "ltx25-convolutional-vae"
    if decoderSelection.isDiffusion {
      metadata["video_precision"]="bfloat16"
      metadata["video_tiling_policy"]="diffusion-internal-query-context-width-tiling"
      metadata["video_input_latent_dtype"]="float32"
      metadata["video_stage_latent_dtype"]="bfloat16"
      metadata["diffvae_noise_seed"]=0
      metadata["diffvae_noise_layout"]="B3FHW before c,w,h patching"
      metadata["diffvae_optimization"]=request.diffusionVAE?.optimization.rawValue ?? "combined"
      metadata["diffvae_query_chunk_size"]=request.diffusionVAE?.queryChunkSize ?? 512
      metadata["diffvae_context_width_chunks"]=request.diffusionVAE?.contextWidthChunks ?? 4
      metadata["diffvae_stage4_tile_width"]=request.diffusionVAE?.stage4TileWidth ?? 0
    }
    if let automaticResolution,let policy=request.automaticDuration {
      metadata["duration_mode"]="automatic"
      metadata["automatic_duration"]=["predicted_duration_seconds":automaticResolution.predictedDurationSeconds,
        "resolved_frames":automaticResolution.resolvedFrames,"fps":automaticResolution.fps,
        "minimum_seconds":policy.minimumSeconds,"maximum_seconds":policy.maximumSeconds,
        "head_checkpoint_path":policy.headCheckpointPath,"head_header_sha256":policy.headHeaderSHA256!]
      metadata["automatic_duration_prepare_seconds"]=textPreparationSeconds
      metadata["peak_memory_scope"]="automatic text and duration preparation through completed media"
    }
    if let guided=request.guidedSampling {
      let branches=1+(guided.videoCFG != 1 || guided.audioCFG != 1 ? 1 : 0)+(guided.stg != 0 ? 1 : 0)+(guided.modality != 1 ? 1 : 0)
      let evaluations=(guided.mode == .guided ? guided.steps : guided.steps*2+1)*branches
      metadata["scope"]="developer Swift MLX Dev guided audiovisual generation"
      metadata["recipe"]="ltx25-dev-"+guided.mode.rawValue+"-two-stage-v1"
      metadata["guided_sampling"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(guided))
      metadata["stage1_model_evaluations"]=evaluations
      metadata["stage1_updates"]=guided.steps+(guided.mode == .guided ? 0 : 1)
      metadata["stage2_model_evaluations"]=3
    }
    if let source=publicationSource {
      metadata["audio_samples"]=source.publicationSamples
      metadata["audio_decoder"]="none-source-publication"
      metadata["audio_estimated_workspace_bytes"]=0
      metadata["audio_conditioning"]=preparedAudio != nil ? "frozen source latents in both distilled stages" : "none; original source waveform preserved at publication"
      metadata["audio_source_start_seconds"]=source.sourceStartSeconds
      metadata["reference_conditioning"]="none"
    }
    if extensionContextFrames != nil {
      metadata["task"]="extension"
      metadata["extension_context_frames"]=contextFrames
      metadata["extension_generated_frames"]=publishedFrames
      metadata["reference_conditioning"]="appended synchronized source video/audio latent guides at half strength"
    }
    if let control=request.icControl {
      metadata["reference_count"]=control.guides.count
      metadata["reference_preparation"]="ordered frozen RGB24 guides encoded sequentially with bounded VAE tiles"
      metadata["reference_conditioning"]="trained IC guide layout appended in stage one; clean stage two"
      metadata["ic_family"]=control.family
      metadata["ic_adapter_scope"]="stage_one_only"
      metadata["ic_guide_sha256"]=control.guides.map(\.sourceSHA256)
      metadata["ic_guide_roles"]=control.guides.map(\.role)
    } else if let union=request.unionControlGuide {
      metadata["reference_count"]=1
      metadata["reference_preparation"]="frozen preprocessed RGB24 guide, half the stage-one canvas"
      metadata["reference_conditioning"]="VAE-encoded half-resolution Union guide appended in stage one; clean stage two"
      metadata["union_guide_sha256"]=union.sourceSHA256
      metadata["union_adapter_scope"]="stage_one_only"
    } else if let sheet=request.ingredientsSheet {
      metadata["reference_count"]=1
      metadata["reference_preparation"]="one frozen RGB24 still encoded once, with normalized latent repeated over the guide timeline"
      metadata["reference_conditioning"]="full-resolution VAE-encoded static sheet guide in one distilled stage"
      metadata["ingredients_sheet_sha256"]=sheet.sourceSHA256
      metadata["ingredients_guide_encoding"]="one_static_rgb_frame_then_latent_repeat_v1"
      metadata["ingredients_adapter_scope"]="single_full_resolution_stage"
      metadata["ingredients_sampling"]=request.ingredientsSampling.rawValue
      metadata["transformer_evaluations"]=request.ingredientsSampling.transformerEvaluations
      if request.ingredientsSampling == .ancestralCFGPP {
        metadata["sampler_state_precision"]="float32_sampler_bf16_model_v1"
        metadata["ancestral_noise_policy"]="mlx_threefry_bf16_step_modality_seed_plus_10000_v1"
        metadata["unconditional_prompt"]=""
      }
    } else if let msr=request.msr {
      metadata["reference_count"]=msr.references.count
      metadata["reference_preparation"]="bounded per-image RGB24 stills repeated for 25/33 causal frames"
      metadata["reference_conditioning"]="ordered VAE latent groups with learned slot embeddings, negative reference times and grouped attention"
      metadata["msr_adapter_scope"]="single_full_resolution_stage"
      metadata["msr_reference_sha256"]=msr.references.map(\.sourceSHA256)
    } else if let dfr=request.dfr {
      metadata["reference_conditioning"]="stage-one generated keyframe slots; stage-two clean half-resolution reference plus upscaled seeded slots"
      metadata["dfr_adapter_scope"]="stage_two_only"
      metadata["dfr_detailing_adapter"]=(dfr.adapterPath as NSString).lastPathComponent
      metadata["dfr_generated_slots"]=try MLXDFRCanvas(frames:g.frames).slotFrames
      metadata["dfr_canvas_frames"]=g.frames
      metadata["dfr_requested_frames"]=request.frames
      metadata["dfr_temporal_rounds"]=dfr.temporalRounds
      if dfr.temporalRounds > 0 {
        metadata["dfr_temporal_upscaler"]=(dfr.temporalUpscalerPath! as NSString).lastPathComponent
        metadata["dfr_temporal_audio_policy"]="frozen stage-one audio; no temporal audio resampling at publication"
      }
    } else if request.usesOrdinaryKeyframes {
      metadata["reference_conditioning"]="frame-zero replacement when supplied; ordered exact pixel-time anchors at both resolutions; globally attending generated slots in stage one only"
      metadata["reference_frames"]=request.referenceFrames
      metadata["generated_keyframe_count"]=request.generatedKeyframes ?? 0
      metadata["generated_keyframe_frames"]=try request.ordinaryKeyframeLayout(geometry:recipe.low,stage:0)?.generatedFrames ?? []
      metadata["generated_keyframe_stage_scope"]="stage_one_only"
    } else if request.referenceImages.isEmpty && extensionContextFrames == nil {
      metadata["reference_conditioning"]="none"
    }
    if let policy=request.singleStageSampling {
      metadata["single_stage_sampling"]=policy.method.rawValue
      metadata["cfg_pp_schedule"]=policy.negativeSchedule.rawValue
      metadata["transformer_evaluations"]=policy.transformerEvaluations
      metadata["negative_prompt"]=policy.negativePrompt
      metadata["cfg_pp_execution"]="serial_bounded_residency"
      metadata["reference_conditioning"]="ordered full-resolution keyframes; generated slots in the only sampling stage"
      metadata["generated_keyframe_stage_scope"]="single_stage_only"
      metadata["reference_frames"]=request.referenceFrames
      metadata["generated_keyframe_count"]=request.generatedKeyframes ?? 0
      metadata["generated_keyframe_frames"]=try request.singleStageControlLayout()?.generatedFrames ?? []
      let controlCount=request.icControl?.guides.count ?? (request.unionControlGuide == nil ? 0 : 1)
      metadata["ordinary_reference_count"]=request.referenceImages.count
      metadata["control_reference_count"]=controlCount
      metadata["reference_count"]=request.referenceImages.count+controlCount
      if request.icControl != nil { metadata["ic_adapter_scope"]="single_full_resolution_stage" }
      if request.unionControlGuide != nil { metadata["union_adapter_scope"]="single_full_resolution_stage" }
      if controlCount>0 {
        metadata["reference_preparation"]="ordinary still images and separately encoded ordered frozen control RGB24 guides"
        metadata["reference_conditioning"]="full-resolution ordinary anchors, trained IC/Union guide grid, trailing generated slots in one stage"
      }
      if let layout=try request.singleStageControlLayout() {
        metadata["single_stage_video_tokens"]=layout.videoTokens
        metadata["single_stage_control_tokens"]=layout.guideTokens
        metadata["single_stage_conditioning_order"]="main, ordered nonzero ordinary images, ordered control guides, generated slots"
      }
    }
    if let videoCacheBytes { metadata["video_allocator_cache_limit_bytes"]=videoCacheBytes }
    try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys])
      .write(to:temporary.appendingPathComponent("report.json"),options:.withoutOverwriting)
    try Self.publish(staging:temporary,output:output) {
      try beforePublish?(temporary,metadata)
      try report("ready_to_publish",1,1)
    }
    published=true
    return output.appendingPathComponent("render.mp4")
  }
  static func publish(staging:URL,output:URL,prepare:() throws -> Void) throws {
    try Task.checkCancellation()
    try prepare()
    try Task.checkCancellation()
    try FileManager.default.moveItem(at:staging,to:output)
  }
  static func publishSourceAudio(_ source:MLXSourceAudioInterval.Prepared,to output:URL) throws {
    guard source.publication.isFileURL,output.isFileURL,
      !FileManager.default.fileExists(atPath:output.path) else {
      throw LTXError.invalid("Source audio publication needs a new local WAV output.")
    }
    try FileManager.default.copyItem(at:source.publication,to:output)
  }
  static func decodeAudio(latent:[Float],latentFrames:Int,checkpoint:URL,backend:MLXAudioBackend,
    progress:@escaping (String) throws -> Void) throws -> AudioWaveform {
    switch backend {
    case .mps: return try AudioDecoder(checkpoint:checkpoint,maximumLatentFrames:latentFrames).decode(latent:latent,latentFrames:latentFrames,progress:progress)
    case .mlx: return try MLXAudioDecoder(checkpoint:checkpoint,maximumLatentFrames:latentFrames).decode(latent:latent,latentFrames:latentFrames,progress:progress)
    }
  }
  static func mux(ffmpeg:URL,directory:URL,fps:Double,rawVideo:Bool) throws {
    let log=directory.appendingPathComponent("mux.log")
    guard FileManager.default.createFile(atPath:log.path,contents:nil) else { throw LTXError.invalid("Cannot create mux log.") }
    let handle=try FileHandle(forWritingTo:log); defer { try? handle.close() }
    let process=Process(); process.executableURL=ffmpeg
    let videoInput=rawVideo ? ["-i",directory.appendingPathComponent("video.mp4").path] : ["-framerate",String(fps),"-i",directory.appendingPathComponent("frames/%06d.png").path]
    let videoCodec=rawVideo ? ["-c:v","copy"] : ["-c:v","libx264","-crf","18","-pix_fmt","yuv420p"]
    process.arguments=["-v","error","-nostdin","-n"] + videoInput +
      ["-i",directory.appendingPathComponent("audio.wav").path,"-map","0:v:0","-map","1:a:0"] + videoCodec +
      ["-c:a","aac","-b:a","192k","-movflags","+faststart",directory.appendingPathComponent("render.mp4").path]
    process.standardOutput=handle; process.standardError=handle
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning { process.terminate(); usleep(100000); if process.isRunning { kill(process.processIdentifier,SIGKILL) } }
      process.waitUntilExit()
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10000) }
    guard process.terminationStatus == 0 else { throw LTXError.invalid("FFmpeg could not publish synchronized media (exit \(process.terminationStatus)).") }
    try Task.checkCancellation()
  }
}
