import Darwin
import Foundation

public struct VideoProcessingStageExecution:Sendable {
  public let payload:VideoProcessingStagePayload,restored:FlashVSRFile,receiptFile:FlashVSRFile
  public init(payload:VideoProcessingStagePayload,restored:FlashVSRFile,receiptFile:FlashVSRFile){self.payload=payload;self.restored=restored;self.receiptFile=receiptFile}
}
public enum VideoProcessingSequenceExecutor {
  public typealias Worker = @Sendable(VideoProcessingStep,FlashVSRFile,FlashVSRMediaInfo,URL) async throws -> VideoProcessingStageExecution
  private struct PreparedSource:Codable {let file:FlashVSRFile,media:FlashVSRMediaInfo,rgb:String,sourceSHA256:String}
  private struct Checkpoint:Codable {let key:String,attempt:UUID,receiptSHA256:String}
  static func encode<T:Encodable>(_ value:T) throws -> Data {let e=JSONEncoder();e.outputFormatting=[.sortedKeys];return try e.encode(value)}
  static func settingsFingerprint(_ step:VideoProcessingStep) throws -> String {
    var normalized=step;normalized.id="restore";normalized.input="source"
    return NativeHeadlessJob.hash(try encode(normalized))
  }
  static func resourceBytes(plan:NativeVideoProcessingPlan,step:VideoProcessingStep) throws -> Data {
    if step.isH3Regenerate,let resource=plan.h3Resources?[step.id]{return try encode(resource)}
    if step.isNative,let resource=plan.pixelResources?[step.id]{return try encode(resource)}
    if step.isFlashVSR,let resource=plan.resources[step.id]{return try encode(resource)}
    throw StudioError.invalid("Every sequence stage must have its own admitted resources: "+step.id)
  }
  static func stageKey(plan:NativeVideoProcessingPlan,step:VideoProcessingStep,inputRGBSHA256:String,media:FlashVSRMediaInfo) throws -> String {
    let clock="\(media.width)x\(media.height):\(media.frames):\(media.fps.numerator)/\(media.fps.denominator)"
    return NativeHeadlessJob.hash(try Data(("sequence-stage-v2|"+inputRGBSHA256+"|"+clock+"|"+settingsFingerprint(step)).utf8)+resourceBytes(plan:plan,step:step))
  }
  static func mediaRuntime(_ plan:NativeVideoProcessingPlan) throws -> FlashVSRRuntime {
    guard let first=plan.recipe.steps.first else{throw StudioError.invalid("Empty processing sequence.")}
    if first.isH3Regenerate,let runtime=plan.h3Resources?[first.id]?.runtime{return runtime.mediaRuntime}
    if first.isNative,let runtime=plan.pixelResources?[first.id]?.runtime{return runtime.mediaRuntime}
    return plan.runtime
  }
  static func ffmpeg(_ args:[String],runtime:FlashVSRRuntime,directory:URL,name:String,cancellation:NativeHeadlessCancellation) throws -> Data {
    let result=try ExternalEngineProcess.run(executable:URL(fileURLWithPath:runtime.ffmpegPath),arguments:["-v","error","-nostdin","-n"]+args,directory:directory,log:directory.appendingPathComponent(name+".log"),cancellation:cancellation,maximumOutput:65536)
    guard result.exitCode==0 else{throw StudioError.invalid("Sequence media verification failed; see "+name+".log.")};return result.output
  }
  static func hashMedia(_ file:FlashVSRFile,video:Bool,runtime:FlashVSRRuntime,directory:URL,cancellation:NativeHeadlessCancellation) throws -> String {
    let args=video ? ["-map","0:v:0","-an","-pix_fmt","rgb24","-c:v","rawvideo"]:["-map","0:a:0","-vn","-c:a","pcm_f32le"]
    let bytes=try ffmpeg(["-i",file.path]+args+["-f","hash","-hash","sha256","-"],runtime:runtime,directory:directory,name:UUID().uuidString+"-content",cancellation:cancellation)
    let text=String(decoding:bytes,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
    guard text.hasPrefix("SHA256="),text.count==71 else{throw StudioError.invalid("Missing sequence media content digest.")};return String(text.dropFirst(7))
  }
  static func sameClock(_ a:FlashVSRMediaInfo,_ b:FlashVSRMediaInfo)->Bool {a.frames==b.frames && a.fps.numerator*b.fps.denominator==b.fps.numerator*a.fps.denominator}
  static func canonical(_ source:FlashVSRFile,media:FlashVSRMediaInfo,output:URL,runtime:FlashVSRRuntime,directory:URL,cancellation:NativeHeadlessCancellation) throws -> (FlashVSRFile,FlashVSRMediaInfo,String) {
    try source.verify()
    let before=try hashMedia(source,video:true,runtime:runtime,directory:directory,cancellation:cancellation)
    _=try ffmpeg(["-i",source.path,"-map","0:v:0","-an","-vf","setpts=PTS-STARTPTS,setsar=1","-fps_mode","passthrough","-c:v","ffv1","-pix_fmt","bgr0","-fflags","+bitexact","-flags:v","+bitexact",output.path],runtime:runtime,directory:directory,name:UUID().uuidString+"-lossless",cancellation:cancellation)
    let file=try FlashVSRFile.capture(output),actual=try FlashVSRMediaInfo.inspect(output,runtime:runtime,directory:directory,cancellation:cancellation,expectedFrameRate:media.fps)
    let after=try hashMedia(file,video:true,runtime:runtime,directory:directory,cancellation:cancellation)
    guard before==after,sameClock(media,actual),actual.width==media.width,actual.height==media.height,!actual.hasAudio else{throw StudioError.invalid("Lossless sequence handoff changed RGB, frame count or source cadence.")}
    return (file,actual,after)
  }
  static func contained(_ file:FlashVSRFile,in folder:URL)->Bool {URL(fileURLWithPath:file.path).resolvingSymlinksInPath().path.hasPrefix(folder.resolvingSymlinksInPath().path+"/")}
  static func bounded<T:Decodable>(_ type:T.Type,_ url:URL) throws -> T {
    let info=try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
    guard info.isRegularFile==true,let bytes=info.fileSize,(1...8*1024*1024).contains(bytes) else{throw StudioError.invalid("Invalid bounded sequence cache record.")}
    return try JSONDecoder().decode(type,from:Data(contentsOf:url))
  }
  static func verifyPayload(_ payload:VideoProcessingStagePayload,file:FlashVSRFile) throws {
    let expected=try encode(payload)
    let saved:VideoProcessingStagePayload
    switch payload {
    case .flashVSR:saved = .flashVSR(try bounded(FlashVSRReceipt.self,URL(fileURLWithPath:file.path)))
    case .h3Regenerate:saved = .h3Regenerate(try bounded(H3RegenerateReceipt.self,URL(fileURLWithPath:file.path)))
    case .pixelCQ:saved = .pixelCQ(try bounded(NativePixelCQReceipt.self,URL(fileURLWithPath:file.path)))
    }
    guard try encode(saved)==expected else{throw StudioError.invalid("Stage completion differs from its immutable engine receipt.")}
  }
  static func verifyResources(plan:NativeVideoProcessingPlan,step:VideoProcessingStep,nativeWorker:NativePixelCQService.Worker?,skipPixelAdmission:Bool,cancellation:NativeHeadlessCancellation) async throws {
    let files:[FlashVSRFile]
    if step.isH3Regenerate {files=plan.h3Resources?[step.id]?.files ?? []}
    else if step.isNative {files=plan.pixelResources?[step.id]?.workerFiles ?? []}
    else if let resource=plan.resources[step.id] {files=resource.components.files+resource.engineFiles}
    else {files=[]}
    _=try resourceBytes(plan:plan,step:step)
    try await Task.detached {for file in files{try Task.checkCancellation();try file.verify()}}.value
    if step.isNative,!skipPixelAdmission,let resources=plan.pixelResources?[step.id],let settings=step.pixelCQ {
      let current=try await NativePixelCQService.preflight(recipe:.init(pixelCQ:settings),runtime:resources.runtime,worker:nativeWorker,cancellation:cancellation)
      guard try encode(current)==encode(resources) else{throw StudioError.invalid("Pixel stage components changed after admission.")}
    }
  }
  static func pixelBridge(_ input:FlashVSRFile,media:FlashVSRMediaInfo,runtime:FlashVSRRuntime,directory:URL,cancellation:NativeHeadlessCancellation) async throws -> FlashVSRFile {
    let output=directory.appendingPathComponent("pixel-source.mov")
    let file=try await Task.detached {
      let before=try hashMedia(input,video:true,runtime:runtime,directory:directory,cancellation:cancellation)
      _=try ffmpeg(["-i",input.path,"-map","0:v:0","-an","-c:v","qtrle","-pix_fmt","rgb24","-fps_mode","passthrough",output.path],runtime:runtime,directory:directory,name:"pixel-lossless-bridge",cancellation:cancellation)
      let file=try FlashVSRFile.capture(output),actual=try FlashVSRMediaInfo.inspect(output,runtime:runtime,directory:directory,cancellation:cancellation)
      guard sameClock(actual,media),actual.width==media.width,actual.height==media.height,!actual.hasAudio,
        try hashMedia(file,video:true,runtime:runtime,directory:directory,cancellation:cancellation)==before else{throw StudioError.invalid("Pixel source bridge changed lossless RGB or the frozen clock.")}
      return file
    }.value
    let clock=try await NativeMovieFrozenMedia.videoClock(output)
    guard clock.width==media.width,clock.height==media.height,clock.times.count==media.frames,abs(clock.fps-media.fps.value)<0.00001 else{throw StudioError.invalid("The installed Pixel media reader cannot preserve this source clock.")}
    return file
  }
  static func executeStage(plan:NativeVideoProcessingPlan,step:VideoProcessingStep,input:FlashVSRFile,media:FlashVSRMediaInfo,directory:URL,
    backend:(any FlashVSRBackend)?,nativeWorker:NativePixelCQService.Worker?,cancellation:NativeHeadlessCancellation,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> VideoProcessingStageExecution {
    if step.isH3Regenerate {
      guard let resource=plan.h3Resources?[step.id],let settings=step.h3Regenerate else{throw StudioError.invalid("H3 stage resources missing.")}
      var request=H3RegenerateRequest(sourcePath:input.path,settings:settings,runtime:resource.runtime);request.sourceSHA256=input.sha256
      let service=H3RegenerateService(),resolved=try await service.preflight(request:request)
      guard resolved.source==input,try encode(resolved.resources)==encode(resource) else{throw StudioError.invalid("H3 stage resources changed after admission.")}
      let receipt=try await service.run(plan:resolved,directory:directory,intermediateOnly:true,emit:emit)
      return .init(payload:.h3Regenerate(receipt),restored:receipt.restored,receiptFile:try .capture(directory.appendingPathComponent("receipt.json")))
    }
    if step.isNative {
      guard let resources=plan.pixelResources?[step.id],let settings=step.pixelCQ else{throw StudioError.invalid("Pixel stage resources missing.")}
      try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
      let bridge=try await pixelBridge(input,media:media,runtime:resources.runtime.mediaRuntime,directory:directory,cancellation:cancellation)
      let result=try await NativePixelCQService.execute(resources:resources,recipe:.init(pixelCQ:settings),source:bridge,directory:directory,cancellation:cancellation,worker:nativeWorker,emit:emit,intermediateOnly:true)
      guard let receipt=result.receipt.native,let restored=receipt.restored else{throw StudioError.invalid("This Pixel worker does not expose a verified lossless intermediate. Update the complete native helper before running a sequence.")}
      return .init(payload:.pixelCQ(receipt),restored:restored,receiptFile:try .capture(URL(fileURLWithPath:result.receiptPath)))
    }
    let selected=backend ?? FlashVSRService.backend(step.settings.backend)
    guard let lossless=selected as? any FlashVSRIntermediateBackend else{throw StudioError.invalid("The selected FlashVSR backend does not provide lossless sequence handoff. Select native Swift/MLX.")}
    var request=VideoUpscaleRequest(sourcePath:input.path,settings:step.settings,runtime:plan.runtime);request.sourceSHA256=input.sha256
    let resolved=try await selected.preflight(request:request)
    guard resolved.source==input,plan.resources[step.id]==FlashVSRResources(components:resolved.components,engineFiles:resolved.engineFiles) else{throw StudioError.invalid("FlashVSR stage resources changed after admission.")}
    let receipt=try await lossless.runIntermediate(plan:resolved,directory:directory,emit:emit)
    return .init(payload:.flashVSR(receipt),restored:receipt.restored,receiptFile:try .capture(directory.appendingPathComponent("receipt.json")))
  }
  /// Propagate known output canvases before any stage hashes or loads models.
  /// Every family preserves this source clock; no inferred conversion is allowed.
  static func validateGeometry(recipe:VideoProcessingRecipe,sourceMedia:FlashVSRMediaInfo) throws {
    var current=sourceMedia
    for step in recipe.steps {
      let width:Int,height:Int
      if let settings=step.h3Regenerate {
        _ = try H3RegeneratePreparation.geometry(media:current,settings:settings,firstFrame:0,frameCount:current.frames)
        width=settings.targetWidth;height=settings.targetHeight
      } else if step.isNative {
        guard (1...60).contains(current.fps.value),current.width%32==0,current.height%32==0,current.width<=2048,current.height<=2048 else{
          throw StudioError.invalid("Sequence stage "+step.id+" requires Pixel's 1–60 FPS source on a 32-pixel grid, at most 2048 pixels per axis. The upstream stages would not produce a compatible source; no model was run.")
        }
        width=current.width*2;height=current.height*2
      } else {width=step.settings.width;height=step.settings.height}
      current = .init(width:width,height:height,frames:current.frames,fps:current.fps,firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    }
  }
  static func run(plan:NativeVideoProcessingPlan,source:FlashVSRFile,root:URL,backend:(any FlashVSRBackend)?,nativeWorker:NativePixelCQService.Worker?,worker:Worker?,
    cancellation:NativeHeadlessCancellation,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> NativeVideoProcessingResult {
    try plan.recipe.validate();guard plan.recipe.isSequence else{throw StudioError.invalid("Linear execution requires a version 2 recipe.")}
    for step in plan.recipe.steps{_ = try resourceBytes(plan:plan,step:step)}
    let runtime=try mediaRuntime(plan)
    let preparation=root.appendingPathComponent("sequence-source-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:preparation,withIntermediateDirectories:false)
    let sourceMedia=try await Task.detached {try source.verify();return try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:source.path),runtime:runtime,directory:preparation,cancellation:cancellation)}.value
    try validateGeometry(recipe:plan.recipe,sourceMedia:sourceMedia)
    for step in plan.recipe.steps{try await verifyResources(plan:plan,step:step,nativeWorker:nativeWorker,skipPixelAdmission:worker != nil,cancellation:cancellation)}
    let preparedSource=try await Task.detached {try prepareSource(source,media:sourceMedia,runtime:runtime,root:root,cancellation:cancellation)}.value
    var input=preparedSource.file,media=preparedSource.media,inputRGB=preparedSource.rgb
    var entries=[VideoProcessingSequenceReceipt.Stage](),reused=0
    for step in plan.recipe.steps {
      try cancellation.check();try Task.checkCancellation()
      let key=try stageKey(plan:plan,step:step,inputRGBSHA256:inputRGB,media:media),folder=root.appendingPathComponent("stage-"+key)
      try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
      guard folder.resolvingSymlinksInPath().path.hasPrefix(root.path+"/") else{throw StudioError.invalid("Sequence cache escaped its root.")}
      let checkpointURL=folder.appendingPathComponent("completed.json")
      var accepted:VideoProcessingStageReceipt?,receiptFile:FlashVSRFile?
      if FileManager.default.fileExists(atPath:checkpointURL.path) {
        do {
          let checkpoint=try bounded(Checkpoint.self,checkpointURL),attempt=folder.appendingPathComponent(checkpoint.attempt.uuidString)
          let receiptURL=attempt.appendingPathComponent("stage-receipt.json"),file=try FlashVSRFile.capture(receiptURL),saved=try bounded(VideoProcessingStageReceipt.self,receiptURL)
          guard checkpoint.key==key,file.sha256==checkpoint.receiptSHA256,saved.key==key,saved.format=="weetodd-video-processing-stage-v2",saved.inputRGBSHA256==inputRGB,
            saved.settingsFingerprint==(try settingsFingerprint(step)),saved.payload.kind==step.kind,sameClock(saved.media,media),contained(saved.restored,in:attempt),contained(saved.engineReceipt,in:attempt) else{throw StudioError.invalid("Sequence checkpoint does not match its input or settings.")}
          try await Task.detached {for artifact in [saved.restored,saved.engineReceipt]+saved.payload.files {guard contained(artifact,in:attempt) else{throw StudioError.invalid("Stage artifact escaped its attempt.")};try artifact.verify()};try verifyPayload(saved.payload,file:saved.engineReceipt)}.value
          let actualRGB=try await Task.detached {try hashMedia(saved.restored,video:true,runtime:runtime,directory:preparation,cancellation:cancellation)}.value
          guard actualRGB==saved.outputRGBSHA256 else{throw StudioError.invalid("Cached stage RGB changed.")}
          accepted=saved;receiptFile=file;reused+=1;emit(.init(stage:"reuse",message:"Reusing verified "+step.label))
        } catch {try cancellation.check();try Task.checkCancellation();emit(.init(stage:"retry",message:"Retaining an incomplete or changed stage and starting a fresh attempt."))}
      }
      if accepted==nil {
        let id=UUID(),attempt=folder.appendingPathComponent(id.uuidString)
        try FileManager.default.createDirectory(at:attempt,withIntermediateDirectories:false)
        let engineDirectory=attempt.appendingPathComponent("engine")
        let output:VideoProcessingStageExecution
        if let worker {output=try await worker(step,input,media,engineDirectory)}
        else {output=try await executeStage(plan:plan,step:step,input:input,media:media,directory:engineDirectory,backend:backend,nativeWorker:nativeWorker,cancellation:cancellation,emit:emit)}
        try cancellation.check();try Task.checkCancellation()
        guard output.payload.kind==step.kind,contained(output.restored,in:attempt),contained(output.receiptFile,in:attempt) else{throw StudioError.invalid("The stage returned another engine or an external output.")}
        let oldMedia=media,inputFile=input,inputDigest=inputRGB
        let converted=try await Task.detached {() throws -> (FlashVSRFile,FlashVSRMediaInfo,String) in
          for file in [output.restored,output.receiptFile]+output.payload.files{guard contained(file,in:attempt) else{throw StudioError.invalid("Stage output escaped its attempt.")};try file.verify()}
          try verifyPayload(output.payload,file:output.receiptFile)
          let actual=try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:output.restored.path),runtime:runtime,directory:attempt,cancellation:cancellation,expectedFrameRate:oldMedia.fps)
          let expectedWidth=step.isNative ? oldMedia.width*2:step.isH3Regenerate ? step.h3Regenerate!.targetWidth:step.settings.width
          let expectedHeight=step.isNative ? oldMedia.height*2:step.isH3Regenerate ? step.h3Regenerate!.targetHeight:step.settings.height
          guard sameClock(actual,oldMedia),!actual.hasAudio,actual.width==expectedWidth,actual.height==expectedHeight else{throw StudioError.invalid("A sequence stage changed the frozen clock or unexpectedly attached audio.")}
          return try canonical(output.restored,media:actual,output:attempt.appendingPathComponent("restored.mkv"),runtime:runtime,directory:attempt,cancellation:cancellation)
        }.value
        let saved=VideoProcessingStageReceipt(key:key,settingsFingerprint:try settingsFingerprint(step),inputRGBSHA256:inputDigest,outputRGBSHA256:converted.2,input:inputFile,restored:converted.0,media:converted.1,engineReceipt:output.receiptFile,payload:output.payload)
        let receiptURL=attempt.appendingPathComponent("stage-receipt.json");try encode(saved).write(to:receiptURL,options:.withoutOverwriting)
        try await Task.detached {try source.verify();try inputFile.verify()}.value;try cancellation.check();try Task.checkCancellation()
        let file=try FlashVSRFile.capture(receiptURL)
        try encode(Checkpoint(key:key,attempt:id,receiptSHA256:file.sha256)).write(to:checkpointURL,options:.atomic)
        accepted=saved;receiptFile=file
      }
      let saved=accepted!;entries.append(.init(id:step.id,receiptFile:receiptFile!,receipt:saved))
      input=saved.restored;media=saved.media;inputRGB=saved.outputRGBSHA256
    }
    let stages=entries,restored=input,finalMedia=media
    return try await Task.detached {try finish(plan:plan,source:source,sourceMedia:sourceMedia,stages:stages,restored:restored,media:finalMedia,runtime:runtime,root:root,reused:reused,cancellation:cancellation)}.value
  }
  private static func prepareSource(_ source:FlashVSRFile,media:FlashVSRMediaInfo,runtime:FlashVSRRuntime,root:URL,cancellation:NativeHeadlessCancellation) throws -> PreparedSource {
    let utilities=try [runtime.ffmpegPath,runtime.ffprobePath].map{try FlashVSRFile.capture(URL(fileURLWithPath:$0))}
    let key=NativeHeadlessJob.hash(try encode(source)+encode(utilities)),folder=root.appendingPathComponent("source-"+key)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    guard folder.resolvingSymlinksInPath().path.hasPrefix(root.path+"/") else{throw StudioError.invalid("Source preparation escaped the cache.")}
    let receipt=folder.appendingPathComponent("source.json")
    if let saved=try? bounded(PreparedSource.self,receipt),saved.sourceSHA256==source.sha256,contained(saved.file,in:folder),sameClock(saved.media,media),saved.media.width==media.width,saved.media.height==media.height {
      do {try saved.file.verify();guard try hashMedia(saved.file,video:true,runtime:runtime,directory:folder,cancellation:cancellation)==saved.rgb else{throw StudioError.invalid("Prepared source RGB identity changed.")};return saved}catch{try cancellation.check()}
    }
    let free=(try FileManager.default.attributesOfFileSystem(forPath:root.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    guard free>Int64(media.frames)*Int64(media.width)*Int64(media.height)*4 else{throw StudioError.invalid("The output volume lacks space for lossless source preparation.")}
    let output=folder.appendingPathComponent(UUID().uuidString+".mkv")
    let (file,checked,rgb)=try canonical(source,media:media,output:output,runtime:runtime,directory:folder,cancellation:cancellation)
    let saved=PreparedSource(file:file,media:checked,rgb:rgb,sourceSHA256:source.sha256)
    try encode(saved).write(to:receipt,options:.atomic);return saved
  }
  static func finish(plan:NativeVideoProcessingPlan,source:FlashVSRFile,sourceMedia:FlashVSRMediaInfo,stages:[VideoProcessingSequenceReceipt.Stage],restored:FlashVSRFile,media:FlashVSRMediaInfo,runtime:FlashVSRRuntime,root:URL,reused:Int,cancellation:NativeHeadlessCancellation) throws -> NativeVideoProcessingResult {
    let key=NativeHeadlessJob.hash(try Data((plan.recipe.fingerprint+"|"+source.sha256+"|sequence-delivery-h264-pcm-v1").utf8)+encode(stages.map(\.receiptFile)))
    let folder=root.appendingPathComponent("sequence-"+key);try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    guard folder.resolvingSymlinksInPath().path.hasPrefix(root.path+"/") else{throw StudioError.invalid("Sequence delivery escaped its cache.")}
    let checkpointURL=folder.appendingPathComponent("completed.json")
    if FileManager.default.fileExists(atPath:checkpointURL.path) {
      do {
        let checkpoint=try bounded(Checkpoint.self,checkpointURL),attempt=folder.appendingPathComponent(checkpoint.attempt.uuidString),url=attempt.appendingPathComponent("receipt.json")
        let receipt=try bounded(VideoProcessingSequenceReceipt.self,url)
        guard checkpoint.key==key,try NativeHeadlessJob.fileHash(url)==checkpoint.receiptSHA256,receipt.requestFingerprint==key,receipt.recipe==plan.recipe,receipt.source==source,receipt.stages.map(\.receiptFile)==stages.map(\.receiptFile),contained(receipt.video,in:attempt) else{throw StudioError.invalid("Sequence delivery identity changed.")}
        for file in [receipt.video,receipt.restored]+[receipt.audio].compactMap({$0}){try file.verify()};try source.verify();try cancellation.check()
        return .init(source:source,recipe:plan.recipe,sequenceReceipt:receipt,receiptPath:url.path,reusedSteps:reused)
      } catch{try cancellation.check();try Task.checkCancellation()}
    }
    let id=UUID(),attempt=folder.appendingPathComponent(id.uuidString);try FileManager.default.createDirectory(at:attempt,withIntermediateDirectories:false)
    var audio:FlashVSRFile?,sourcePCM:String?,outputPCM:String?
    if sourceMedia.hasAudio {
      guard let rate=sourceMedia.audioRate else{throw StudioError.invalid("Source soundtrack sample clock missing.")}
      let url=attempt.appendingPathComponent("original-audio.wav")
      try FlashVSRAudioPreparation.prepare(source:URL(fileURLWithPath:source.path),output:url,ffmpeg:URL(fileURLWithPath:runtime.ffmpegPath),directory:attempt,startTimestamp:sourceMedia.firstTimestamp,frames:sourceMedia.frames,fps:sourceMedia.fps,sampleRate:rate,cancellation:cancellation)
      audio=try .capture(url);sourcePCM=try hashMedia(audio!,video:false,runtime:runtime,directory:attempt,cancellation:cancellation)
    }
    let videoURL=attempt.appendingPathComponent("processed.mov")
    var args=["-i",restored.path];if let audio{args += ["-i",audio.path]};args += ["-map","0:v:0"]
    args += audio == nil ? ["-an"]:["-map","1:a:0","-c:a","pcm_f32le"]
    args += ["-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709","-color_range","tv","-fps_mode","passthrough","-movflags","+faststart",videoURL.path]
    _=try ffmpeg(args,runtime:runtime,directory:attempt,name:"delivery",cancellation:cancellation)
    let video=try FlashVSRFile.capture(videoURL),actual=try FlashVSRMediaInfo.inspect(videoURL,runtime:runtime,directory:attempt,cancellation:cancellation)
    guard sameClock(actual,sourceMedia),actual.width==media.width,actual.height==media.height,actual.hasAudio==sourceMedia.hasAudio,
      !actual.hasAudio || actual.audioRate==sourceMedia.audioRate&&actual.audioChannels==sourceMedia.audioChannels else{throw StudioError.invalid("Sequence delivery changed source clock, soundtrack or target dimensions.")}
    if audio != nil{outputPCM=try hashMedia(video,video:false,runtime:runtime,directory:attempt,cancellation:cancellation)}
    guard sourcePCM==outputPCM else{throw StudioError.invalid("Sequence delivery changed original PCM samples.")}
    try source.verify();try restored.verify();for stage in stages{try stage.receiptFile.verify()};try cancellation.check()
    let receipt=VideoProcessingSequenceReceipt(requestFingerprint:key,recipe:plan.recipe,source:source,sourceMedia:sourceMedia,stages:stages,video:video,restored:restored,audio:audio,sourcePCMHash:sourcePCM,outputPCMHash:outputPCM,media:actual)
    let url=attempt.appendingPathComponent("receipt.json");try encode(receipt).write(to:url,options:.withoutOverwriting)
    try encode(Checkpoint(key:key,attempt:id,receiptSHA256:try NativeHeadlessJob.fileHash(url))).write(to:checkpointURL,options:.atomic)
    return .init(source:source,recipe:plan.recipe,sequenceReceipt:receipt,receiptPath:url.path,reusedSteps:reused)
  }
}
