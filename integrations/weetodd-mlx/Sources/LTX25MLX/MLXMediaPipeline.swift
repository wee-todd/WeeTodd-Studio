import Foundation
import Darwin
import MLX
import LTX25Engine
import LTX25Video
import LTX25Audio
import InferenceMedia
import InferenceContracts

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
  private let sampler:MLXDistilledSamplingRunner
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
    let recipe=try request.recipe(), g=recipe.high
    for (index,geometry) in [recipe.low,recipe.high].enumerated() {
      let layout=try request.referenceImages.first.map { try MLXReferenceLayout(geometry:geometry,firstStrength:$0.strength,lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil) }
      let guide=try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:geometry,contextFrames:$0) }
      let union=try index == 0
        ? request.unionControlGuide.map { try MLXUnionControlLayout(geometry:geometry,strength:$0.referenceStrength) }
        : nil
      guard layout == nil || guide == nil else { throw LTXError.invalid("LTX extension cannot combine with endpoint references.") }
      let block=try MLXAVBlock(configuration:AVBlockConfiguration(
        videoTokens:guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? geometry.videoTokens,
        audioTokens:guide?.audioTokens ?? geometry.audioFrames,textTokens:1024),
        maximumActivationBytes:transformerActivationBytes)
      if layout != nil || union != nil { try block.admitPerTokenVideo() }
      if guide != nil { try block.admitPerTokenAV() }
      _ = try MLXDenoiser.admitRotary(configuration:block.configuration,maximumActivationBytes:transformerActivationBytes)
    }
    if !request.referenceImages.isEmpty {
      for geometry in [recipe.low,recipe.high] { _ = try MLXImageEncodePlan(width:geometry.width,height:geometry.height) }
    }
    if request.unionControlGuide != nil {
      _ = try MLXVideoEncodeTilePlan(frames:recipe.low.frames,
        width:recipe.low.width/2,height:recipe.low.height/2,
        maximumOwnedBufferBytes:videoActivationBytes)
    }
    let c=videoConfiguration(for:g,activationBytes:videoActivationBytes)
    let videoShape:[Int],videoBytes:Int
    switch videoBackend {
    case .mlx:
      let plan=try MLXVideoDecodePlan(shape:g.videoShape,configuration:c)
      videoShape=plan.outputShape;videoBytes=plan.admittedActivationBytes
    case .mps:
      let plan=try VideoDecodePlan(shape:g.videoShape,configuration:c)
      videoShape=plan.outputShape;videoBytes=plan.admittedActivationBytes
    }
    let audio=try audioBackend == .mlx ? MLXAudioDecoder.estimatedPeakBytes(latentFrames:g.audioFrames) : AudioDecoder.estimatedPeakBytes(latentFrames:g.audioFrames)
    guard videoShape == [g.frames,g.height,g.width,3], g.audioFrames <= 1501, audio <= 2*1024*1024*1024 else {
      throw LTXError.invalid("Media geometry exceeds the native decoder admission.")
    }
    return Admission(videoFrames:g.frames,audioSamples:try AudioDecoder.sampleCount(latentFrames:g.audioFrames),
      videoActivationBytes:videoBytes,audioEstimatedBytes:audio)
  }
  public init(request:MLXDistilledRequest,extensionContextFrames:Int?=nil,
    videoActivationBytes:Int=512*1024*1024,transformerActivationBytes:Int=2*1024*1024*1024,
    videoBackend:MLXVideoBackend = .mps,audioBackend:MLXAudioBackend = .mps,saveLatents:Bool=false,saveFrames:Bool=false) throws {
    _ = try Self.admit(request,extensionContextFrames:extensionContextFrames,
      videoActivationBytes:videoActivationBytes,transformerActivationBytes:transformerActivationBytes,
      videoBackend:videoBackend,audioBackend:audioBackend)
    let output=URL(fileURLWithPath:request.outputDirectory)
    guard !FileManager.default.fileExists(atPath:output.path) else { throw LTXError.invalid("Output directory already exists.") }
    self.request=request
    self.videoBackend=videoBackend;self.audioBackend=audioBackend;self.saveLatents=saveLatents;self.saveFrames=saveFrames
    self.extensionContextFrames=extensionContextFrames
    self.videoActivationBytes=videoActivationBytes;self.transformerActivationBytes=transformerActivationBytes
    sampler=try MLXDistilledSamplingRunner(recipe:request.recipe(),transformerRoot:URL(fileURLWithPath:request.transformerRoot),
      upscalerCheckpoint:URL(fileURLWithPath:request.spatialUpscalerCheckpoint),statisticsCheckpoint:URL(fileURLWithPath:request.videoCheckpoint),
      firstStrength:request.referenceImages.first?.strength,lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil,
      extensionContextFrames:extensionContextFrames,
      unionControlGuide:request.unionControlGuide,
      stageOneLoras:request.stageOneLoras,stageTwoLoras:request.stageTwoLoras,noisePolicy:request.noisePolicy,maximumActivationBytes:transformerActivationBytes)
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
      _ = try VideoDecoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
      try AudioDecoder.validateCheckpoint(URL(fileURLWithPath:request.audioCheckpoint))
    }
  }
  public func run(ffmpeg:URL,progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> URL {
    try run(ffmpeg:ffmpeg,decodedPreview:nil,progress:progress)
  }
  public func run(ffmpeg:URL,preparedAudio:MLXSourceAudioInterval.Prepared?=nil,
    extensionGuideLease:ExtensionGuideLease?=nil,
    decodedPreview:((Int,Data) throws -> Void)?,beforePublish:((URL,[String:Any]) throws -> Void)? = nil,progress:@escaping (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> URL {
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
    guard (request.audioReference == nil) == (preparedAudio == nil),
      preparedAudio == nil || (preparedAudio!.publicationSamples == Int((Double(g.frames)/g.fps*48000).rounded(.toNearestOrEven)) &&
        preparedAudio!.conditioningSamples == Int((Double(g.frames)/g.fps*16000).rounded(.toNearestOrEven))) else {
      throw LTXError.invalid("A2V needs its exact prepared source interval before weighted execution.")
    }
    guard (extensionGuideLease?.guides != nil) == (extensionContextFrames != nil),
      extensionGuideLease == nil || preparedAudio == nil else {
      throw LTXError.invalid("LTX extension needs its staged audiovisual source guides without an A2V driver.")
    }
    var timings:[String:Double]=[:], ids:[Int]=[]
    func report(_ stage:String,_ completed:Int=0,_ total:Int=1) throws {
      try Task.checkCancellation(); try progress(stage,completed,total); try Task.checkCancellation()
    }
    Memory.peakMemory=0
    let imageDirectory=temporary.appendingPathComponent("prepared-references")
    try fm.createDirectory(at:imageDirectory,withIntermediateDirectories:false)
    let preparationStart=Date(),recipe=try request.recipe()
    let preparedImages=try MLXReferenceImage.prepareStages(request.referenceImages,
      sizes:[(recipe.low.width,recipe.low.height),(recipe.high.width,recipe.high.height)],ffmpeg:ffmpeg,directory:imageDirectory) {
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
      let contexts=try autoreleasepool {
        let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:request.gemmaRoot),connectorURL:URL(fileURLWithPath:request.connectorCheckpoint))
        return try encoder.encode(prompt:request.prompt) { try report("text:"+$0.stage,$0.completed,$0.total) }
      }
      ids=contexts.tokenIDs; timings["text"]=Date().timeIntervalSince(textStarted)
      try report("text_weights_released")
      let referenceStart=Date()
      let preparedGuides:([(first:MLXArray,last:MLXArray?)],MLXArray?)=try autoreleasepool {
        if let union=request.unionControlGuide {
          let plan=try MLXVideoEncodeTilePlan(frames:recipe.low.frames,
            width:recipe.low.width/2,height:recipe.low.height/2,
            maximumOwnedBufferBytes:videoActivationBytes)
          let encoded=try MLXTiledVideoEncoder.encode(guide:URL(fileURLWithPath:union.path),
            checkpoint:URL(fileURLWithPath:request.videoCheckpoint),plan:plan) {
            try report("control_guide_encode",$0,$1)
          }
          try NativeMediaSource(path:union.path,sha256:union.sourceSHA256).verify()
          return ([],encoded.reshaped([plan.latentShape[0]*plan.latentShape[1]*plan.latentShape[2],128]))
        }
        guard !request.referenceImages.isEmpty else { return ([],nil) }
        let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
        var cached:[URL:MLXArray]=[:],prepared:[(first:MLXArray,last:MLXArray?)]=[]
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
          prepared.append((tokens[0],tokens.count == 2 ? tokens[1] : nil))
        }
        return (prepared,nil)
      }
      try fm.removeItem(at:imageDirectory)
      if !preparedGuides.0.isEmpty || preparedGuides.1 != nil {
        timings["reference_encode"]=Date().timeIntervalSince(referenceStart)
        try report("reference_weights_released")
      }
      let sampled=try sampler.evaluate(videoContext:contexts.video,audioContext:contexts.audio,references:preparedGuides.0,
        frozenAudio:frozenAudio,extensionGuides:extensionGuideLease?.guides,
        unionGuide:preparedGuides.1,progress:report)
      timings.merge(sampler.stageSeconds) { _,new in new }
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
    let videoConfig=Self.videoConfiguration(for:g,activationBytes:videoActivationBytes)
    let contextFrames=extensionContextFrames ?? 0
    let publishedFrames=g.frames-contextFrames
    let videoStart=Date(); var written=0,videoCacheBytes:Int?
    try autoreleasepool {
      func receive(_ chunk:VideoFrameChunk) throws {
        guard chunk.startFrame == written, chunk.frameCount == 1, chunk.width == g.width,
          chunk.height == g.height, chunk.frameRate == g.fps else { throw LTXError.invalid("Decoded frame contract mismatch.") }
        if written >= contextFrames {
          try MediaOutput.writePNG(chunk.rgb,width:g.width,height:g.height,
            to:frames.appendingPathComponent(String(format:"%06d.png",written-contextFrames)))
        }
        written += 1; try report("video_decode",written,g.frames)
      }
      let unpacked=try g.unpackVideo(latents.video),checkpoint=URL(fileURLWithPath:request.videoCheckpoint)
      switch videoBackend {
      case .mps:
        try VideoDecoder(checkpoint:checkpoint).decode(latent:unpacked,shape:g.videoShape,configuration:videoConfig,receive:receive)
      case .mlx:
        let decoder=try MLXVideoDecoder(checkpoint:checkpoint)
        videoCacheBytes=decoder.cacheLimitBytes
        if rawVideo {
          let writer=try RawVideoWriter(ffmpeg:ffmpeg,output:temporary.appendingPathComponent("video.mp4"),
            width:g.width,height:g.height,frames:publishedFrames,fps:g.fps)
          defer { writer.cancel() }
          try decoder.decodeRGB8(latent:MLXArray(unpacked,g.videoShape),configuration:videoConfig,
            progress:{ try report("video_layers",$0,$1) }) { index,bytes in
              if index >= contextFrames {
                try writer.append(bytes,frame:index-contextFrames)
                try decodedPreview?(index-contextFrames,bytes)
              }
              written += 1;try report("video_decode",written,g.frames)
            }
          try writer.finish()
        } else {
          try decoder.decode(latent:MLXArray(unpacked,g.videoShape),configuration:videoConfig,
            progress:{ try report("video_layers",$0,$1) },receive:receive)
        }
      }
    }
    guard written == admission.videoFrames else { throw LTXError.invalid("Incomplete video decode.") }
    timings["video_decode"]=Date().timeIntervalSince(videoStart); try report("video_weights_released")
    let audioStart=Date()
    if let source=preparedAudio {
      try Self.publishSourceAudio(source,to:temporary.appendingPathComponent("audio.wav"))
      try report("source_audio_published")
    } else { try autoreleasepool {
      let wave=try Self.decodeAudio(latent:g.unpackAudio(latents.audio),latentFrames:g.audioFrames,
        checkpoint:URL(fileURLWithPath:request.audioCheckpoint),backend:audioBackend) { try report("audio:"+$0) }
      guard wave.frameCount == admission.audioSamples, wave.channels == 2, wave.sampleRate == 48000 else {
        throw LTXError.invalid("Decoded audio timing contract mismatch.")
      }
      let range=contextFrames==0 ? 0..<wave.frameCount
        : try Self.extensionAudioRange(contextFrames:contextFrames,
          additionalFrames:publishedFrames,fps:g.fps,decodedSamples:wave.frameCount)
      let published=Array(wave.samples[(range.lowerBound*wave.channels)..<(range.upperBound*wave.channels)])
      try MediaOutput.writeWAV(samples:published,sampleRate:wave.sampleRate,channels:wave.channels,
        to:temporary.appendingPathComponent("audio.wav"))
    } }
    timings[preparedAudio == nil ? "audio_decode" : "source_audio_copy"]=Date().timeIntervalSince(audioStart)
    if preparedAudio == nil { try report("audio_weights_released") }
    let muxStart=Date()
    try Self.mux(ffmpeg:ffmpeg,directory:temporary,fps:g.fps,rawVideo:rawVideo)
    timings["mux"]=Date().timeIntervalSince(muxStart)
    var info=task_vm_info_data_t(), count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { ptr in ptr.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
      task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
    } }
    guard status == KERN_SUCCESS else { throw LTXError.invalid("Cannot measure process memory.") }
    let publishedAudioSamples:Int
    if let source=preparedAudio { publishedAudioSamples=source.publicationSamples }
    else if contextFrames==0 { publishedAudioSamples=admission.audioSamples }
    else {
      publishedAudioSamples=try Self.extensionAudioRange(contextFrames:contextFrames,
        additionalFrames:publishedFrames,fps:g.fps,decodedSamples:admission.audioSamples).count
    }
    var metadata:[String:Any]=["status":"complete","scope":"developer Swift MLX distilled audiovisual generation",
      "recipe":DistilledTwoStageRecipe.identifier,"noise_algorithm":request.noisePolicy.algorithm,"seed":request.seed,
      "text_token_ids":ids,"width":g.width,"height":g.height,"frames":publishedFrames,"fps":g.fps,
      "video_seconds":Double(publishedFrames)/g.fps,
      "audio_samples":publishedAudioSamples,
      "audio_sample_rate":48000,
      "audio_seconds":Double(publishedAudioSamples)/48000,
      "audio_timing":contextFrames>0 ? "crop decoded causal audio to exact generated duration"
        : preparedAudio == nil ? "retain actual causal decoder samples; no stretch or shortest trim"
        : "original source waveform, trimmed and padded to output duration",
      "stage_seconds":timings,"seconds":Date().timeIntervalSince(started),"peak_mlx_bytes":Memory.peakMemory,
      "peak_process_footprint_bytes":info.ledger_phys_footprint_peak,"current_process_footprint_bytes":info.phys_footprint,
      "process_memory_scope":"Swift process; external FFmpeg process excluded",
      "video_decoder":videoBackend.rawValue,"decoder_latents_saved":saveLatents,
      "video_publication":rawVideo ? "gpu-rgb24-pipe" : "png-sequence",
      "video_precision":videoBackend == .mlx ? "bfloat16" : "float32",
      "block_loading":"mlx-native-batched-evaluation","compiled_blocks":true,"spatial_upscaler":"mlx",
      "text_weight_loading":"mlx-native-layer-batched-with-bounded-row-projections",
      "fixed_weight_loading":"mlx-native-evaluated-parameters",
      "audio_decoder":audioBackend.rawValue,"audio_estimated_workspace_bytes":admission.audioEstimatedBytes,
      "transformer_activation_budget_bytes":transformerActivationBytes,"video_activation_budget_bytes":videoActivationBytes,"video_admitted_activation_bytes":admission.videoActivationBytes,
      "python_inference":false,"production_qualified":false,"task":request.task,
      "reference_count":request.referenceImages.count,"reference_preparation":MLXReferenceImage.policy,
      "reference_conditioning":"first latent replacement and appended last-frame tokens at both resolutions; no output-frame paste"]
    if let source=preparedAudio {
      metadata["audio_samples"]=source.publicationSamples
      metadata["audio_decoder"]="none-source-publication"
      metadata["audio_estimated_workspace_bytes"]=0
      metadata["audio_conditioning"]="frozen source latents in both distilled stages"
      metadata["audio_source_start_seconds"]=source.sourceStartSeconds
      metadata["reference_conditioning"]="none"
    }
    if extensionContextFrames != nil {
      metadata["task"]="extension"
      metadata["extension_context_frames"]=contextFrames
      metadata["extension_generated_frames"]=publishedFrames
      metadata["reference_conditioning"]="appended synchronized source video/audio latent guides at half strength"
    }
    if let union=request.unionControlGuide {
      metadata["reference_count"]=1
      metadata["reference_preparation"]="frozen preprocessed RGB24 guide, half the stage-one canvas"
      metadata["reference_conditioning"]="VAE-encoded half-resolution Union guide appended in stage one; clean stage two"
      metadata["union_guide_sha256"]=union.sourceSHA256
      metadata["union_adapter_scope"]="stage_one_only"
    } else if request.referenceImages.isEmpty && extensionContextFrames == nil {
      metadata["reference_conditioning"]="none"
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
