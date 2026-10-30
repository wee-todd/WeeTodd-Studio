import Foundation
import InferenceContracts

public struct H3RegenerateService {
  public init() {}
  static func workspace() throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-h3-restore-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false);return url
  }
  public func preflightResources(settings:H3RegenerateSettings,runtime:H3RegenerateRuntime) async throws -> H3RegenerateResources {
    let cancellation=NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{
      try await Task.detached { try H3RegeneratePreparation.resources(settings:settings,runtime:runtime,cancellation:cancellation) }.value
    },onCancel:{cancellation.cancel()})
  }
  public func preflight(request:H3RegenerateRequest) async throws -> H3RegenerateResolvedPlan {
    let cancellation=NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{ try await Task.detached {
      guard request.format=="weetodd-h3-regenerate-v1",request.sourcePath.hasPrefix("/"),!request.sourcePath.utf8.contains(0) else { throw StudioError.invalid("Choose a local H3 restoration source.") }
      try request.settings.validate()
      let root=try Self.workspace();defer{try? FileManager.default.removeItem(at:root)}
      let runtime=try request.runtime.resolved()
      // Inspect every source timestamp before hashing any model asset.
      let media=try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:request.sourcePath),runtime:runtime.mediaRuntime,directory:root,cancellation:cancellation)
      let grid=try H3RegeneratePreparation.geometry(media:media,settings:request.settings,firstFrame:request.firstFrame,frameCount:request.frameCount,latent:request.nativeLatents)
      let source=try FlashVSRFile.capture(URL(fileURLWithPath:request.sourcePath))
      guard request.sourceSHA256==nil || request.sourceSHA256==source.sha256 else { throw StudioError.invalid("The selected H3 source changed.") }
      try H3RegeneratePreparation.validateLatentSelection(request.nativeLatents,settings:request.settings,sourceSHA256:source.sha256,firstFrame:request.firstFrame,frames:grid.frames)
      if let latent=request.nativeLatents {try latent.file.verify()}
      var assetSettings=request.settings
      if request.nativeLatents != nil {assetSettings.encoderProfile = .nativePosteriorMeanF16;assetSettings.posteriorSeed=nil}
      let resources=try H3RegeneratePreparation.resources(settings:assetSettings,runtime:runtime,cancellation:cancellation)
      var frozen=request;frozen.runtime=runtime
      let policy=request.nativeLatents != nil ? "explicit-normalized-rows-v1"
        : request.settings.fit == .fit ? "source-ceil32-center-pad-v1":"source-ceil32-lanczos-fill-center-crop-v1"
      let plan=H3RegenerateResolvedPlan(request:frozen,source:source,media:media,resources:resources,frames:grid.frames,paddedFrames:grid.paddedFrames,sourceWidth:grid.sourceWidth,sourceHeight:grid.sourceHeight,temporaryBytes:grid.temporaryBytes,preparationPolicy:policy+";target-ceil32-center-crop;repeat-last-to-5plus17-v1")
      let prepared=try Self.prepare(plan:plan,directory:root,cancellation:cancellation)
      let job=try Self.job(plan:plan,payload:prepared.payload,audio:prepared.audio)
      let response=try Self.call(job:job,runtime:runtime,directory:root,output:root.appendingPathComponent("admission"),mode:"preflight",cancellation:cancellation,emit:{_ in})
      try Self.validateResult(response,job:job,render:false)
      try source.verify();return plan
    }.value },onCancel:{cancellation.cancel()})
  }
  static func ffmpeg(_ args:[String],runtime:H3RegenerateRuntime,directory:URL,name:String,cancellation:NativeHeadlessCancellation) throws {
    let result=try ExternalEngineProcess.run(executable:URL(fileURLWithPath:runtime.ffmpegPath),arguments:["-v","error","-nostdin","-n"]+args,directory:directory,log:directory.appendingPathComponent(name+".log"),cancellation:cancellation)
    guard result.exitCode==0 else { throw StudioError.invalid("H3 media preparation failed. See " + name + ".log.") }
  }
  struct Prepared {let payload:FlashVSRFile,audio:FlashVSRFile?}
  static func prepare(plan:H3RegenerateResolvedPlan,directory:URL,cancellation:NativeHeadlessCancellation) throws -> Prepared {
    let request=plan.request,runtime=request.runtime
    try plan.source.verify();try cancellation.check()
    let payload:FlashVSRFile
    if let latent=request.nativeLatents {
      try H3RegeneratePreparation.validateLatentSelection(latent,settings:request.settings,sourceSHA256:plan.source.sha256,firstFrame:request.firstFrame,frames:plan.frames)
      try latent.file.verify();payload=latent.file
    } else {
      let rgb=directory.appendingPathComponent("source.rgb24")
      let canvas=request.settings.fit == .fit
        ? "pad=\(plan.sourceWidth):\(plan.sourceHeight):(ow-iw)/2:(oh-ih)/2:color=black"
        : "scale=\(plan.sourceWidth):\(plan.sourceHeight):force_original_aspect_ratio=increase:flags=lanczos,crop=\(plan.sourceWidth):\(plan.sourceHeight)"
      let filter="trim=start_frame=\(request.firstFrame):end_frame=\(request.firstFrame+plan.frames),setpts=PTS-STARTPTS,\(canvas),setsar=1"
      try ffmpeg(["-i",plan.source.path,"-map","0:v:0","-vf",filter,"-fps_mode","passthrough","-an","-pix_fmt","rgb24","-f","rawvideo",rgb.path],runtime:runtime,directory:directory,name:"prepare-source",cancellation:cancellation)
      payload=try .capture(rgb)
      guard payload.bytes==Int64(plan.frames*plan.sourceWidth*plan.sourceHeight*3) else {throw StudioError.invalid("Prepared H3 source frame count or RGB byte layout changed.")}
    }
    var audio:FlashVSRFile?
    if plan.media.hasAudio {
      guard let rate=plan.media.audioRate else{throw StudioError.invalid("Source audio has no valid sample clock.")}
      let url=directory.appendingPathComponent("source-audio.wav")
      try FlashVSRAudioPreparation.prepare(source:URL(fileURLWithPath:plan.source.path),output:url,ffmpeg:URL(fileURLWithPath:runtime.ffmpegPath),directory:directory,
        startTimestamp:plan.media.firstTimestamp+Double(request.firstFrame)/24,frames:plan.frames,fps:plan.media.fps,sampleRate:rate,cancellation:cancellation)
      audio=try .capture(url)
    }
    try plan.source.verify();return Prepared(payload:payload,audio:audio)
  }
  static func job(plan:H3RegenerateResolvedPlan,payload:FlashVSRFile,audio:FlashVSRFile?) throws -> H3RegenerateJob {
    let r=plan.request,s=r.settings,native=r.nativeLatents != nil
    return try .init(source:.init(kind:native ? .normalizedVideoRows:.rgb24,payload:.init(path:payload.path,sha256:payload.sha256),originalSHA256:plan.source.sha256,
      audioSHA256:audio?.sha256 ?? H3RegeneratePreparation.hash(Data()),width:plan.sourceWidth,height:plan.sourceHeight,visibleFrames:plan.frames,paddedFrames:plan.paddedFrames,fps:24,
      encoderProfile:native ? .nativeNormalizedLatents:s.encoderProfile),prompt:s.prompt,promptSHA256:H3RegeneratePreparation.hash(Data(s.prompt.utf8)),qwen:plan.resources.qwen,
      components:plan.resources.components,schedule:s.schedule,seed:s.seed,posteriorSeed:native ? nil:s.posteriorSeed,target:.init(width:s.targetWidth,height:s.targetHeight),
      latentUpscaleMode:s.latentUpscaleMode,decoder:s.decoder,budget:.init(maximumStageBytes:min(128*1024*1024*1024,ProcessInfo.processInfo.physicalMemory*3/4),availableTemporaryBytes:UInt64(plan.temporaryBytes)))
  }
  static func progressEvent(_ value:[String:Any],mode:String) -> FlashVSREvent? {
    guard value["status"] == nil else { return nil }
    let stage = value["stage"] as? String ?? mode
    let completed = (value["completed"] as? NSNumber)?.doubleValue, total = (value["total"] as? NSNumber)?.doubleValue
    var message = "H3 Regenerate · "
    if let step = (value["step"] as? NSNumber)?.intValue { message += "step \(step) · " }
    message += stage.replacingOccurrences(of: "_", with: " ")
    var fraction:Double?
    if let completed, let total, completed.isFinite, total.isFinite, total > 0 {
      if let count=Int(exactly:completed),let limit=Int(exactly:total) { message += " \(count)/\(limit)" }
      fraction = max(0,min(1,completed/total))
    }
    return .init(stage:stage,message:message,fraction:fraction)
  }
  static func call(job:H3RegenerateJob,runtime:H3RegenerateRuntime,directory:URL,output:URL,mode:String,cancellation:NativeHeadlessCancellation,emit:@escaping @Sendable(FlashVSREvent)->Void) throws -> [String:Any] {
    let recipe=directory.appendingPathComponent(UUID().uuidString+"-h3-job.json")
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    let bytes=try encoder.encode(job);try bytes.write(to:recipe,options:.withoutOverwriting)
    let result=try NativeHeadlessExecutor.worker(runtime.workerPath,recipe:recipe,output:output,mode:mode,cancellation:cancellation,emit:{data in
      guard let value=try? JSONSerialization.jsonObject(with:data) as? [String:Any],let event=progressEvent(value,mode:mode) else{return}
      emit(event)
    })
    guard result["fingerprint"] as? String==H3RegeneratePreparation.hash(bytes) else{throw StudioError.invalid("H3 worker response belongs to a different request.")}
    return result
  }
  static func validateResult(_ result:[String:Any],job:H3RegenerateJob,render:Bool) throws {
    func int(_ key:String,_ expected:Int)->Bool {
      guard let n=result[key] as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID() else{return false}
      return n.doubleValue==Double(expected)
    }
    guard result["nativeRuntime"] as? String=="swift-mlx",result["engine"] as? String=="h3",result["task"] as? String=="h3_regenerate",
      result["inferenceExecuted"] as? Bool==render,result["sourceSHA256"] as? String==job.source.originalSHA256,
      result["sourcePayloadSHA256"] as? String==job.source.payload.sha256,result["sourceAudioSHA256"] as? String==job.source.audioSHA256,
      result["decoder"] as? String==job.decoder.rawValue,result["encoderProfile"] as? String==job.source.encoderProfile.rawValue,
      int("fps",24),int("visibleFrames",job.source.visibleFrames),int("paddedFrames",job.source.paddedFrames),int("width",job.target.width),int("height",job.target.height),
      !render || result["audioDecoded"] as? Bool==false && result["sourceAudioPreservationRequired"] as? Bool==true else {
      throw StudioError.invalid("H3 worker result changed the frozen source, timing, encoder or selected decoder.")
    }
  }
  public func run(plan:H3RegenerateResolvedPlan,directory:URL,intermediateOnly:Bool=false,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> H3RegenerateReceipt {
    let cancellation=NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{try await Task.detached {
      try Self.execute(plan:plan,directory:directory,cancellation:cancellation,intermediateOnly:intermediateOnly,emit:emit)
    }.value},onCancel:{cancellation.cancel()})
  }
  static func execute(plan:H3RegenerateResolvedPlan,directory:URL,cancellation:NativeHeadlessCancellation,intermediateOnly:Bool=false,emit:@escaping @Sendable(FlashVSREvent)->Void,
    resourceResolver:((H3RegenerateSettings,H3RegenerateRuntime,NativeHeadlessCancellation)throws->H3RegenerateResources)?=nil,
    worker:((H3RegenerateJob,URL)throws->[String:Any])?=nil) throws -> H3RegenerateReceipt {
    let runtime=plan.request.runtime,s=plan.request.settings,started=Date()
    let grid=try H3RegeneratePreparation.geometry(media:plan.media,settings:s,firstFrame:plan.request.firstFrame,frameCount:plan.frames,latent:plan.request.nativeLatents)
    guard plan.frames==grid.frames,plan.paddedFrames==grid.paddedFrames,plan.sourceWidth==grid.sourceWidth,plan.sourceHeight==grid.sourceHeight,plan.temporaryBytes>=grid.temporaryBytes,
      plan.source.path==URL(fileURLWithPath:plan.request.sourcePath).resolvingSymlinksInPath().path,
      directory.isFileURL,!FileManager.default.fileExists(atPath:directory.path) else{throw StudioError.invalid("The H3 saved source plan changed or output already exists.")}
    try plan.source.verify();try cancellation.check()
    let inspection=try workspace();defer{try? FileManager.default.removeItem(at:inspection)}
    let actual=try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:plan.source.path),runtime:runtime.mediaRuntime,directory:inspection,cancellation:cancellation)
    guard actual==plan.media else{throw StudioError.invalid("The saved H3 source clock changed before execution.")}
    var assetSettings=s
    if plan.request.nativeLatents != nil{assetSettings.encoderProfile = .nativePosteriorMeanF16;assetSettings.posteriorSeed=nil}
    let current=try resourceResolver?(assetSettings,runtime,cancellation)
      ?? H3RegeneratePreparation.resources(settings:assetSettings,runtime:runtime,cancellation:cancellation)
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    guard try encoder.encode(current)==encoder.encode(plan.resources) else{throw StudioError.invalid("H3 model or worker identities changed after preflight.")}
    try plan.source.verify();try cancellation.check()
    let parent=directory.deletingLastPathComponent();try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
    let available=(try FileManager.default.attributesOfFileSystem(forPath:parent.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    guard available>plan.temporaryBytes else{throw StudioError.invalid("The output volume lacks H3 restoration temporary space.")}
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    let output=directory.appendingPathComponent(intermediateOnly ? "restored.mkv":"output.mov");var accepted=false
    defer{if !accepted{try? FileManager.default.removeItem(at:output)}}
    emit(.init(stage:"prepare",message:"Preparing the frozen 24 FPS source interval",fraction:0))
    let prepared=try prepare(plan:plan,directory:directory,cancellation:cancellation),job=try job(plan:plan,payload:prepared.payload,audio:prepared.audio)
    let workerOutput=directory.appendingPathComponent("native")
    let result=try worker?(job,workerOutput)
      ?? call(job:job,runtime:runtime,directory:directory,output:workerOutput,mode:"render",cancellation:cancellation,emit:emit)
    try validateResult(result,job:job,render:true)
    let savedResult=try H3RegeneratePreparation.boundedJSON(workerOutput.appendingPathComponent("result.json"))
    guard try JSONSerialization.data(withJSONObject:result,options:.sortedKeys)
      == JSONSerialization.data(withJSONObject:savedResult,options:.sortedKeys),
      try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:H3RegeneratePreparation.boundedJSON(workerOutput.appendingPathComponent("worker-job.json")))) == job else {
      throw StudioError.invalid("H3 worker saved metadata differs from its accepted immutable request or completion.")
    }
    let rgb=try FlashVSRFile.capture(workerOutput.appendingPathComponent("video.rgb24"))
    guard result["outputRGBPath"] as? String==rgb.path,result["outputRGBSHA256"] as? String==rgb.sha256,
      (result["outputRGBBytes"] as? NSNumber)?.int64Value==rgb.bytes,rgb.bytes==Int64(plan.frames*s.targetWidth*s.targetHeight*3) else{throw StudioError.invalid("H3 worker RGB output differs from its frozen receipt.")}
    let restored=directory.appendingPathComponent("restored.mkv")
    try ffmpeg(["-f","rawvideo","-pix_fmt","rgb24","-video_size","\(s.targetWidth)x\(s.targetHeight)","-framerate","24","-i",rgb.path,"-an","-c:v","ffv1","-pix_fmt","bgr0","-fps_mode","passthrough",restored.path],runtime:runtime,directory:directory,name:"restore-lossless",cancellation:cancellation)
    try verifyMovie(restored,plan:plan,directory:directory,cancellation:cancellation,audio:false)
    let decodedRGB=try ExternalEngineProcess.run(executable:URL(fileURLWithPath:runtime.ffmpegPath),
      arguments:["-v","error","-nostdin","-i",restored.path,"-map","0:v:0","-an","-pix_fmt","rgb24","-c:v","rawvideo","-f","hash","-hash","sha256","-"],
      directory:directory,log:directory.appendingPathComponent("verify-lossless-rgb.log"),cancellation:cancellation,maximumOutput:65536)
    guard decodedRGB.exitCode==0,String(decoding:decodedRGB.output,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)=="SHA256="+rgb.sha256 else {
      throw StudioError.invalid("The H3 restored intermediate changed decoded RGB bytes.")
    }
    if intermediateOnly {
      guard prepared.audio==nil else{throw StudioError.invalid("H3 intermediate-only execution requires a silent lossless sequence source.")}
    } else {
    var final=["-i",restored.path]
    if let audio=prepared.audio{final += ["-i",audio.path]}
    final += ["-map","0:v:0"]
    final += prepared.audio == nil ? ["-an"]:["-map","1:a:0","-c:a","pcm_f32le"]
    final += ["-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709","-color_range","tv","-movflags","+faststart","-fps_mode","passthrough",output.path]
    try ffmpeg(final,runtime:runtime,directory:directory,name:"deliver",cancellation:cancellation)
    try verifyMovie(output,plan:plan,directory:directory,cancellation:cancellation,audio:prepared.audio != nil)
    if let audio=prepared.audio {
      guard try pcmHash(URL(fileURLWithPath:audio.path),runtime:runtime,directory:directory,cancellation:cancellation)==pcmHash(output,runtime:runtime,directory:directory,cancellation:cancellation) else{throw StudioError.invalid("H3 delivery changed original source audio samples.")}
    }
    }
    let latent=try FlashVSRFile.capture(workerOutput.appendingPathComponent("refined-video.f32"))
    let rows=(2+5*((plan.paddedFrames-5)/17))*((s.targetWidth+31)/32)*((s.targetHeight+31)/32)
    guard result["refinedLatentsPath"] as? String==latent.path,result["refinedLatentsSHA256"] as? String==latent.sha256,
      result["refinedLatentsShape"] as? [Int]==[1,rows,96],result["refinedLatentsDType"] as? String=="F32LE",latent.bytes==Int64(rows*96*4) else{throw StudioError.invalid("H3 refined latent artifact differs from its declared normalized row layout.")}
    try plan.source.verify();for file in plan.resources.engineFiles{try file.verify()};try cancellation.check()
    let receipt=H3RegenerateReceipt(requestFingerprint:try plan.fingerprint,plan:plan,source:plan.source,video:try .capture(output),restored:try .capture(restored),audio:prepared.audio,
      workerResult:try .capture(workerOutput.appendingPathComponent("result.json")),workerJob:try .capture(workerOutput.appendingPathComponent("worker-job.json")),refinedLatents:latent,refinedLatentsShape:[1,rows,96],elapsedSeconds:Date().timeIntervalSince(started))
    try encoder.encode(receipt).write(to:directory.appendingPathComponent("receipt.json"),options:.withoutOverwriting)
    accepted=true;emit(.init(stage:"completed",message:"H3 restoration verified",fraction:1));return receipt
  }
  static func verifyMovie(_ url:URL,plan:H3RegenerateResolvedPlan,directory:URL,cancellation:NativeHeadlessCancellation,audio:Bool) throws {
    let info=try FlashVSRMediaInfo.inspect(url,runtime:plan.request.runtime.mediaRuntime,directory:directory,cancellation:cancellation,expectedFrameRate:try .init(numerator:24,denominator:1))
    guard info.frames==plan.frames,info.width==plan.request.settings.targetWidth,info.height==plan.request.settings.targetHeight,info.fps.value==24,info.hasAudio==audio,
      !audio || info.audioRate==plan.media.audioRate && info.audioChannels==plan.media.audioChannels else{throw StudioError.invalid("H3 delivered video/audio clock or geometry changed.")}
  }
  static func pcmHash(_ url:URL,runtime:H3RegenerateRuntime,directory:URL,cancellation:NativeHeadlessCancellation) throws -> String {
    let result=try ExternalEngineProcess.run(executable:URL(fileURLWithPath:runtime.ffmpegPath),arguments:["-v","error","-nostdin","-i",url.path,"-map","0:a:0","-vn","-c:a","pcm_f32le","-f","hash","-hash","sha256","-"],directory:directory,log:directory.appendingPathComponent(UUID().uuidString+"-audio.log"),cancellation:cancellation,maximumOutput:65536)
    let hash=String(decoding:result.output,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
    guard result.exitCode==0,hash.hasPrefix("SHA256="),hash.count==71 else{throw StudioError.invalid("Cannot verify original H3 source audio PCM.")};return hash
  }
}
