import AVFoundation
import Darwin
import Foundation
import MLX
import InferenceContracts
import InferenceMedia
import LTX25Engine
import LTX25Video

/// End-to-end source-movie inference through the existing trained native VAEs,
/// learned upscaler and shared streamed denoiser. Chunking executes actual
/// weighted windows and publishes original source PCM once after concatenation.
public final class MLXMovieUpscalePipeline: @unchecked Sendable {
  public struct Admission:Sendable {
    public let chunks:[MLXMovieUpscalePlan.Chunk],componentIdentitySHA256:String
    public let maximumVideoActivationBytes:Int,maximumTransformerActivationBytes:Int,maximumTextOwnedBufferBytes:Int
  }
  private final class TextLease: @unchecked Sendable {
    let video:MLXArray,audio:MLXArray
    init(_ text:MLXTextEncoder.Output) { video=text.video.reshaped(text.video.shape);audio=text.audio.reshaped(text.audio.shape) }
  }
  private final class Timings: @unchecked Sendable {
    private let lock=NSLock();private var values:[String:Double]=[:]
    func add(_ name:String,_ seconds:Double) { lock.withLock { values[name,default:0]+=seconds } }
    var snapshot:[String:Double] { lock.withLock { values } }
  }
  private let request:MLXMovieUpscaleRequest
  private let gate=NSLock();private var active=false
  private var text:TextLease?
  public init(request:MLXMovieUpscaleRequest) { self.request=request }
  private func enter()->Bool { gate.withLock { if active { return false };active=true;return true } }
  private func leave() { gate.withLock { active=false } }
  public func preflight(maximumVideoActivationBytes:Int=4*1024*1024*1024,
    maximumTransformerActivationBytes:Int=2*1024*1024*1024,
    maximumTextOwnedBufferBytes:Int=3*1024*1024*1024) async throws -> Admission {
    guard maximumVideoActivationBytes>0,maximumTransformerActivationBytes>0,maximumTextOwnedBufferBytes>0 else {
      throw LTXError.invalid("Movie stages need positive owned-buffer admission.")
    }
    try request.validateMediaSources()
    for reference in request.referenceImages {
      try Task.checkCancellation()
      try MLXReferenceImage.inspect(URL(fileURLWithPath:reference.path))
    }
    let source=try MLXMovieSourceVideo(source:URL(fileURLWithPath:request.source.path),sha256:request.source.sha256,
      plan:request.plan,startSeconds:request.source.startSeconds)
    _ = try await source.preflight()
    let cuts=try request.chunking ? MLXMovieSourceVideo.sceneCuts(rgb24:URL(fileURLWithPath:request.source.rgbPath),plan:request.plan) : []
    let chunks=try request.chunkPlans(cutFrames:cuts)
    let componentIdentity=try MLXMovieCheckpointIdentity.capture(request.components)
    if request.plan.mode != .latentOnly { _ = try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!)) }
    _ = try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!))
    _ = try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!),settings:request.diffusionVAE)
    if request.plan.mode != .latentOnly {
      let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:request.components["gemma_root"]!),
        connectorURL:URL(fileURLWithPath:request.components["connector_checkpoint"]!))
      _ = try MLXTextEncodingPlan(promptTokens:encoder.tokenize(request.prompt).count,maximumOwnedBufferBytes:maximumTextOwnedBufferBytes)
      _ = try MLXAudioEncoder(checkpoint:URL(fileURLWithPath:request.components["audio_checkpoint"]!),maximumMelFrames:MLXAudioMelPlan.maximumMelFrames)
    }
    // Only one header provider is needed at a time; never retain one set of
    // fifty headers per planned chunk or allocate a pixel/latent tensor here.
    for chunk in chunks {
      let local=try MLXMovieUpscalePlan(mode:request.plan.mode,width:request.plan.size.width,height:request.plan.size.height,
        frames:chunk.frames,fps:request.plan.fps,sizePolicy:.strict,refinementStrength:request.plan.refinementStrength)
      _ = try MLXVideoEncodeTilePlan(frames:chunk.paddedFrames,width:local.size.width,height:local.size.height,
        maximumOwnedBufferBytes:maximumVideoActivationBytes)
      let g=try AVGeometry(width:local.size.outputWidth,height:local.size.outputHeight,frames:chunk.paddedFrames,fps:local.fps)
      _ = try MLXNativeVideoDecoder.admit(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!),settings:request.diffusionVAE,shape:g.videoShape,configuration:Self.videoConfiguration(g,maximumVideoActivationBytes),backend:.mlx)
      if request.plan.mode != .latentOnly {
        let samples=Int(ceil(Double(chunk.frames)/request.plan.fps*16000))
        let mel=try MLXAudioMelPlan(samples:samples)
        _ = try MLXAudioEncodePlan(melFrames:mel.melFrames,maximumMelFrames:MLXAudioMelPlan.maximumMelFrames)
      }
    }
    // Admit the greatest token geometry once, with both possible endpoints.
    // Weight/header shapes do not depend on which source interval it covers.
    let largest=chunks.max { $0.paddedFrames<$1.paddedFrames }!
    let usesFirst=request.anchors != .none || request.referenceImages.contains { $0.role=="first" }
    let usesLast=request.anchors == .firstLast || request.referenceImages.contains { $0.role=="last" }
    _ = try MLXMovieUpscaleRunner(request:request,chunk:largest,firstStrength:usesFirst ? request.anchorStrength : nil,
      lastStrength:usesLast ? request.anchorStrength : nil,maximumActivationBytes:maximumTransformerActivationBytes)
    return Admission(chunks:chunks,componentIdentitySHA256:componentIdentity,maximumVideoActivationBytes:maximumVideoActivationBytes,
      maximumTransformerActivationBytes:maximumTransformerActivationBytes,maximumTextOwnedBufferBytes:maximumTextOwnedBufferBytes)
  }
  private static func videoConfiguration(_ g:AVGeometry,_ bytes:Int)->LTX25Video.VideoDecodeConfiguration {
    var c=LTX25Video.VideoDecodeConfiguration();c.frameRate=g.fps;c.maximumLatentFrames=g.latentFrames
    c.maximumLatentHeight=g.latentHeight;c.maximumLatentWidth=g.latentWidth;c.maximumActivationBytes=bytes;return c
  }
  private func endpointStrengths(chunk:Int,chunks:Int)->(first:Float?,last:Float?) {
    guard request.plan.mode != .latentOnly else { return (nil,nil) }
    let first=request.anchors != .none || (chunk==0 && request.referenceImages.contains { $0.role=="first" })
    let last=request.anchors == .firstLast || (chunk==chunks-1 && request.referenceImages.contains { $0.role=="last" })
    return (first ? request.anchorStrength : nil,last ? request.anchorStrength : nil)
  }
  public func run(ffmpeg:URL,workerSHA256:String,admission:Admission,
    progress:@escaping @Sendable(String,Int,Int)throws->Void={ _,_,_ in },
    decodedPreview:(@Sendable(Int,Data)throws->Void)?=nil,
    latentPreview:((MLXArray,AVGeometry,Int,Int)throws->Void)?=nil,
    beforePublish:((URL,[String:Any])throws->Void)?=nil) async throws -> URL {
    guard enter() else { throw LTXError.invalid("Movie pipeline is already running.") }
    defer { text=nil;Stream.gpu.synchronize();Memory.clearCache();leave() }
    guard MLXMovieUpscaleRequest.validSHA(workerSHA256),FileManager.default.isExecutableFile(atPath:ffmpeg.path),
      try MLXMovieCheckpointIdentity.capture(request.components)==admission.componentIdentitySHA256 else {
      throw LTXError.invalid("Movie renderer or checkpoint identity changed after preflight.")
    }
    try request.validateMediaSources();try Task.checkCancellation()
    let output=URL(fileURLWithPath:request.outputDirectory),fm=FileManager.default
    guard !fm.fileExists(atPath:output.path) else { throw LTXError.invalid("Movie final output already exists; use the host's verified completed-job resume.") }
    // The durable sibling survives cancellation. Final publication stays atomic.
    let work=output.deletingLastPathComponent().appendingPathComponent(output.lastPathComponent+".movie-work")
    if fm.fileExists(atPath:work.path),!request.resume { throw LTXError.invalid("Movie work already exists; explicitly select resume before reusing or replacing anything.") }
    try fm.createDirectory(at:work,withIntermediateDirectories:true)
    let pipelineLock=work.appendingPathComponent(".pipeline.lock")
    let lockFD=Darwin.open(pipelineLock.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0o600)
    guard lockFD>=0 else { throw LTXError.invalid("Another movie pipeline owns this work directory, or its interrupted lock requires inspection.") }
    Darwin.close(lockFD);defer { try? fm.removeItem(at:pipelineLock) }
    let audioDirectory=work.appendingPathComponent("source-audio")
    if fm.fileExists(atPath:audioDirectory.path) { try fm.removeItem(at:audioDirectory) }
    let clock=try await MLXMovieFiles.videoClock(URL(fileURLWithPath:request.source.path))
    let completeSource=request.source.startSeconds==clock.times[0] && clock.times.count==request.plan.frames
    let audioURL:URL?,audioSourceSHA:String?,audioStart:Double,audioDuration:Double?
    switch request.audioPolicy {
    case .source:
      audioURL=URL(fileURLWithPath:request.source.path);audioSourceSHA=request.source.sha256
      audioStart=request.source.startSeconds;audioDuration=completeSource ? nil : request.source.durationSeconds
    case .sidecar:
      let selected=request.audioSource!
      audioURL=URL(fileURLWithPath:selected.path);audioSourceSHA=selected.sha256
      audioStart=selected.startSeconds;audioDuration=selected.durationSeconds
      guard !(try await AVURLAsset(url:audioURL!).loadTracks(withMediaType:.audio)).isEmpty else {
        throw LTXError.invalid("Selected movie sidecar contains no audio track; use explicit silence instead.")
      }
    case .silence:
      audioURL=nil;audioSourceSHA=nil;audioStart=0;audioDuration=nil
    }
    let audioSource=try MLXMovieSourceAudio(source:audioURL,sha256:audioSourceSHA,
      sourceStartSeconds:audioStart,sourceDurationSeconds:audioDuration,
      frames:request.plan.frames,fps:request.plan.fps,maximumDriftSeconds:request.maximumAudioDriftSeconds)
    let audio=try await audioSource.prepare(ffmpeg:ffmpeg,directory:audioDirectory)
    let audioSHA=try MLXMovieFiles.digest(audio.publication)
    let binding=try MLXMovieChunkCoordinator.Binding(request:request,audioSHA256:audioSHA,
      componentIdentitySHA256:admission.componentIdentitySHA256,workerSHA256:workerSHA256)
    let times=Timings()
    let start=Date()
    let completed=try await MLXMovieChunkCoordinator.execute(request:request,chunks:admission.chunks,binding:binding,
      directory:work.appendingPathComponent("chunks"),progress:progress) { chunk,index,temp in
      try Task.checkCancellation()
      guard try MLXMovieCheckpointIdentity.capture(self.request.components)==admission.componentIdentitySHA256,
        try MLXMovieFiles.digest(audio.publication)==audioSHA else { throw LTXError.invalid("Movie source audio or model identity changed before a weighted chunk.") }
      let rgb=temp.appendingPathComponent("source-padded.rgb24")
      try MLXMovieFiles.copyRGBRange(source:URL(fileURLWithPath:self.request.source.rgbPath),to:rgb,
        frameBytes:self.request.plan.size.width*self.request.plan.size.height*3,
        visibleRange:chunk.startFrame..<chunk.endFrame,paddedFrames:chunk.paddedFrames)
      let strengths=self.endpointStrengths(chunk:index,chunks:admission.chunks.count)
      let runner=try MLXMovieUpscaleRunner(request:self.request,chunk:chunk,firstStrength:strengths.first,lastStrength:strengths.last,
        maximumActivationBytes:admission.maximumTransformerActivationBytes)
      let sourceStart=Date()
      let source=try autoreleasepool { () throws -> MLXArray in
        let plan=try MLXVideoEncodeTilePlan(frames:chunk.paddedFrames,width:self.request.plan.size.width,height:self.request.plan.size.height,
          maximumOwnedBufferBytes:admission.maximumVideoActivationBytes)
        return try MLXTiledVideoEncoder.encode(guide:rgb,checkpoint:URL(fileURLWithPath:self.request.components["video_checkpoint"]!),
          plan:plan,progress:{ try progress("movie_source_video_encode",$0,$1) })
      }
      times.add("source_video_encode",Date().timeIntervalSince(sourceStart))
      try progress("movie_source_video_weights_released",index+1,admission.chunks.count)
      var first:MLXArray?,last:MLXArray?,frozenAudio:MLXArray?
      if self.request.plan.mode != .latentOnly {
        let g=runner.layout!.geometry
        func anchor(_ role:String,_ frame:Int) throws -> MLXArray {
          let globalOuter=(role=="first" && index==0) || (role=="last" && index==admission.chunks.count-1)
          let external=globalOuter ? self.request.referenceImages.first(where:{ $0.role==role }) : nil
          let image:URL,crf:Int
          if let external { image=URL(fileURLWithPath:external.path);crf=external.crf }
          else {
            image=temp.appendingPathComponent("source-"+role+".png");crf=33
            try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-f","rawvideo","-pixel_format","rgb24",
              "-video_size","\(self.request.plan.size.width)x\(self.request.plan.size.height)","-framerate",String(self.request.plan.fps),"-i",rgb.path,
              "-vf","select=eq(n\\,\(frame))","-frames:v","1","-update","1",image.path],log:temp.appendingPathComponent("source-"+role+".log"))
          }
          let pixels=try MLXReferenceImage.prepare(image,width:g.width,height:g.height,crf:crf,ffmpeg:ffmpeg,temporaryParent:temp)
          return try autoreleasepool {
            let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:self.request.components["video_checkpoint"]!))
            return try encoder.encode(MLXArray(pixels,[1,g.height,g.width,3]),maximumOwnedBufferBytes:admission.maximumVideoActivationBytes)
          }
        }
        if strengths.first != nil { first=try anchor("first",0) }
        if strengths.last != nil { last=try anchor("last",chunk.frames-1) }
        try progress("movie_endpoint_weights_released",1,1)
        if self.text == nil {
          let textStart=Date()
          self.text=try autoreleasepool {
            let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:self.request.components["gemma_root"]!),connectorURL:URL(fileURLWithPath:self.request.components["connector_checkpoint"]!))
            return TextLease(try encoder.encode(prompt:self.request.prompt,maximumOwnedBufferBytes:admission.maximumTextOwnedBufferBytes,
              progress:{ try progress("movie_text:"+$0.stage,$0.completed,$0.total) }))
          }
          times.add("text_encode",Date().timeIntervalSince(textStart))
          try progress("movie_text_weights_released",1,1)
        }
        let sampleRange=try audio.contract.publicationSampleBounds(startFrame:chunk.startFrame,endFrame:chunk.endFrame,fps:self.request.plan.fps)
        let wav=temp.appendingPathComponent("context-16k.wav")
        try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-i",audio.publication.path,"-af",
          "atrim=start_sample=\(sampleRange.lowerBound):end_sample=\(sampleRange.upperBound),asetpts=PTS-STARTPTS,aresample=16000",
          "-ac","2","-c:a","pcm_f32le",wav.path],log:temp.appendingPathComponent("audio-context.log"))
        let audioStart=Date()
        frozenAudio=try autoreleasepool {
          let mel=try MLXAudioMel.encode(wav:wav)
          let encoder=try MLXAudioEncoder(checkpoint:URL(fileURLWithPath:self.request.components["audio_checkpoint"]!),maximumMelFrames:MLXAudioMelPlan.maximumMelFrames)
          return try encoder.encode(mel:mel,progress:{ try progress("movie_audio_context_encode",$0,$1) })
        }
        times.add("source_audio_encode",Date().timeIntervalSince(audioStart))
        try progress("movie_audio_context_weights_released",1,1)
        _ = g
      }
      let result=try runner.run(source:source,audio:frozenAudio,videoContext:self.text?.video,audioContext:self.text?.audio,
        first:first,last:last,progress:progress,preview:latentPreview.map { callback in
          { state,event in try callback(state["video"]![0..<runner.layout!.geometry.videoTokens],runner.layout!.geometry,event.completedSteps,event.totalSteps) }
        })
      for (name,value) in result.stageSeconds { times.add(name,value) }
      frozenAudio=nil;first=nil;last=nil
      let writer=try RawVideoWriter(ffmpeg:ffmpeg,output:temp.appendingPathComponent("video.mp4"),
        width:result.geometry.width,height:result.geometry.height,frames:chunk.frames,fps:result.geometry.fps)
      defer { writer.cancel() }
      let decodeStart=Date()
      try autoreleasepool {
        let decoder=try MLXNativeVideoDecoder(checkpoint:URL(fileURLWithPath:self.request.components["video_checkpoint"]!),settings:self.request.diffusionVAE,maximumWorkspaceBytes:admission.maximumVideoActivationBytes)
        let latent=result.video.reshaped([result.geometry.latentFrames,result.geometry.latentHeight,result.geometry.latentWidth,128]).transposed(3,0,1,2).expandedDimensions(axis:0)
        try decoder.decodeRGB8(latent:latent,configuration:Self.videoConfiguration(result.geometry,admission.maximumVideoActivationBytes),progress:{ try progress("movie_video_layers",$0,$1) }) { frame,bytes in
          if frame<chunk.frames { try writer.append(bytes,frame:frame);try decodedPreview?(chunk.startFrame+frame,bytes) }
        }
        try writer.finish()
      }
      times.add("video_decode_and_encode",Date().timeIntervalSince(decodeStart))
      try progress("movie_video_weights_released",index+1,admission.chunks.count)
      try fm.removeItem(at:rgb)
    }
    text=nil;Stream.gpu.synchronize();Memory.clearCache()
    let staging=output.deletingLastPathComponent().appendingPathComponent(".movie-publication-"+UUID().uuidString)
    try fm.createDirectory(at:staging,withIntermediateDirectories:false);defer { try? fm.removeItem(at:staging) }
    // Independently verified silent chunks share the exact encoder/timebase.
    // Copy video packets; never add interpolation or a second lossy encode.
    let list=staging.appendingPathComponent("chunks.ffconcat")
    var parts:[URL]=[]
    for (index,chunk) in completed.enumerated() {
      let part=staging.appendingPathComponent(String(format:"part-%06d.mp4",index))
      try fm.linkItem(at:chunk.video,to:part);parts.append(part)
    }
    let lines=parts.map { "file '"+$0.lastPathComponent+"'" }.joined(separator:"\n")+"\n"
    try Data(lines.utf8).write(to:list,options:.withoutOverwriting)
    try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-f","concat","-safe","1","-i",list.path,
      "-an","-c:v","copy","-frames:v",String(request.plan.frames),staging.appendingPathComponent("video.mp4").path],log:staging.appendingPathComponent("concat.log"))
    for part in parts { try fm.removeItem(at:part) }
    try fm.copyItem(at:audio.publication,to:staging.appendingPathComponent("audio.wav"))
    try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-i",staging.appendingPathComponent("video.mp4").path,"-i",staging.appendingPathComponent("audio.wav").path,
      "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-frames:v",String(request.plan.frames),"-movflags","+faststart",staging.appendingPathComponent("render.mp4").path],log:staging.appendingPathComponent("mux.log"))
    let rendered=try await MLXMovieFiles.videoClock(staging.appendingPathComponent("render.mp4"),maximumFrames:request.plan.frames)
    guard rendered.times.count==request.plan.frames,rendered.width==request.plan.size.outputWidth,rendered.height==request.plan.size.outputHeight,abs(rendered.fps-request.plan.fps)<0.001,
      try MLXMovieFiles.digest(staging.appendingPathComponent("audio.wav"))==audioSHA,
      try MLXMovieCheckpointIdentity.capture(request.components)==admission.componentIdentitySHA256 else { throw LTXError.invalid("Movie final publication changed its source/audio/canvas contract.") }
    try request.validateMediaSources();try Task.checkCancellation()
    var metadata:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale","nativeRuntime":"swift-mlx","pythonModelInference":false,
      "source_movie_sha256":request.source.sha256,"source_rgb_sha256":request.source.rgbSHA256,
      "source_start_seconds":request.source.startSeconds,"source_duration_seconds":request.source.durationSeconds,
      "reference_image_sha256":request.referenceImageSHA256,"seed":request.seed,
      "refinement_sigmas":request.plan.mode == .latentOnly ? [] : request.plan.sigmas,
      "transformer_evaluations_per_chunk":request.plan.mode == .latentOnly ? 0 : 3,
      "mode":request.plan.mode.rawValue,"frames":request.plan.frames,"padded_chunk_frames":admission.chunks.map(\.paddedFrames),"width":rendered.width,"height":rendered.height,"fps":request.plan.fps,"video_seconds":request.plan.seconds,
      "audio_policy":request.audioPolicy.rawValue,"source_audio_sample_rate":audio.contract.publicationSampleRate,"source_audio_samples":audio.contract.publicationSamples,"source_audio_sha256":audioSHA,
      "worker_sha256":workerSHA256,"component_identity_sha256":admission.componentIdentitySHA256,"contract_sha256":request.contractSHA256,
      "executed_chunks":completed.filter { !$0.reused }.count,"reused_chunks":completed.filter(\.reused).count,"stage_seconds":times.snapshot,"pipeline_seconds":Date().timeIntervalSince(start)]
    let selected=try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!),settings:request.diffusionVAE)
    metadata.merge(MLXNativeVideoDecoder.publicationMetadata(isDiffusion:selected.isDiffusion,settings:request.diffusionVAE)) { _,new in new }
    try JSONSerialization.data(withJSONObject:metadata,options:[.sortedKeys,.prettyPrinted]).write(to:staging.appendingPathComponent("metadata.json"),options:.withoutOverwriting)
    try beforePublish?(staging,metadata)
    try request.validateMediaSources();try Task.checkCancellation()
    try fm.moveItem(at:staging,to:output)
    if !request.keepChunks { try? fm.removeItem(at:work) }
    return output
  }
}
