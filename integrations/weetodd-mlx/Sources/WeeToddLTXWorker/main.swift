import Foundation
import Darwin
import LTX25MLX
import LTX25Engine
import InferenceContracts
import InferenceMedia
import MLX
import CryptoKit

private final class MovieWorkerEvents:@unchecked Sendable {
  private let lock=NSLock(),preview:URL,width:Int,height:Int,frames:Int
  private var progress:MLXMovieStudioProgress,lastPreview=Date.distantPast,revision=0
  init(preview:URL,width:Int,height:Int,frames:Int,chunks:Int) {
    self.preview=preview;self.width=width;self.height=height;self.frames=frames
    progress=MLXMovieStudioProgress(chunks:chunks)
  }
  func report(_ stage:String,_ completed:Int,_ total:Int) throws {
    try lock.withLock { try Task.checkCancellation();try LTXWorker.emit(progress.event(stage:stage,completed:completed,total:total)) }
  }
  func decoded(_ index:Int,_ bytes:Data) throws {
    try lock.withLock {
      guard MLXStudioPreview.shouldEmit(index:index,total:frames,secondsSinceLast:Date().timeIntervalSince(lastPreview)) else { return }
      try Task.checkCancellation()
      try autoreleasepool { try MLXStudioPreview.write(rgb:bytes,width:width,height:height,to:preview) }
      revision += 1;lastPreview=Date()
      var event=progress.event(stage:"movie_decoded_preview",completed:index+1,total:frames)
      event["previewPath"]=preview.path;event["previewRevision"]=revision
      try LTXWorker.emit(event)
    }
  }
}

@main struct LTXWorker {
  static func emit(_ object:[String:Any]) throws {
    try FileHandle.standardOutput.write(contentsOf:JSONSerialization.data(withJSONObject:object,options:.sortedKeys)+Data([10]))
  }
  static func main() async {
    signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN)
    let job=Task.detached { try await execute() }
    let signals=[SIGINT,SIGTERM].map { number in
      let source=DispatchSource.makeSignalSource(signal:number,queue:.global())
      source.setEventHandler { @Sendable in job.cancel() };source.resume();return source
    }
    do { try await job.value }
    catch {
      try? emit(["status":error is CancellationError ? "cancelled":"error","error":String(describing:error)])
      for source in signals { source.cancel() }
      exit(error is CancellationError ? 130:1)
    }
    for source in signals { source.cancel() }
  }
  static func invalid(_ text:String) -> NSError { NSError(domain:"WeeToddLTXWorker",code:1,userInfo:[NSLocalizedDescriptionKey:text]) }
  static func read(_ path:String) throws -> Data {
    guard path.hasPrefix("/"),!path.utf8.contains(0) else { throw invalid("An absolute local request path is required.") }
    let fd=Darwin.open(path,O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw invalid("Cannot open request: \(path)") }
    let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? file.close() }
    var status=stat()
    guard fstat(fd,&status) == 0,status.st_mode & S_IFMT == S_IFREG,status.st_size <= 1024*1024 else { throw invalid("Request must be a regular file of at most 1 MiB.") }
    let data=try file.read(upToCount:1024*1024+1) ?? Data()
    guard data.count <= 1024*1024 else { throw invalid("Request exceeds 1 MiB.") };return data
  }
  static func execute() async throws {
    let start=Date(),args=Array(CommandLine.arguments.dropFirst())
    let executable=URL(fileURLWithPath:CommandLine.arguments[0]).resolvingSymlinksInPath()
    if executable.deletingLastPathComponent().lastPathComponent == "MacOS" {
      let library=executable.deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Resources/LTXNative/mlx.metallib")
      guard FileManager.default.fileExists(atPath:library.path) else { throw invalid("The packaged MLX Metal library is missing. Reinstall the complete app bundle.") }
      GPU.metallib=library
    }
    guard args.count == 5,["preflight","render","preflight-transformer-conversion","convert-transformer"].contains(args[0]),args[1] == "--request",args[3] == "--output" else {
      throw invalid("Usage: WeeToddLTXWorker preflight|render|preflight-transformer-conversion|convert-transformer --request REQUEST --output DIRECTORY")
    }
    try Task.checkCancellation()
    if args[0].contains("transformer") {
      try executeTransformerConversion(data:read(args[2]),outputDirectory:args[4],
        preflight:args[0] == "preflight-transformer-conversion",started:start)
      return
    }
    let envelope=try NativeVideoJobEnvelope.decode(read(args[2]),expectedEngine:.ltx25,
      expectedOutputDirectory:args[4])
    let recipeData=try read(envelope.recipePath)
    try envelope.validateRecipe(recipeData)
    if let root=try JSONSerialization.jsonObject(with:recipeData) as? [String:Any] {
      if root["task"] as? String == "video_upscale" || root["movie_upscale"] != nil {
        try await executeMovieUpscale(recipeData:recipeData,envelope:envelope,
          outputDirectory:args[4],preflight:args[0] == "preflight",started:start,executable:executable)
        return
      }
      if root["scene"] != nil {
        try await executeScene(recipeData:recipeData,envelope:envelope,
          outputDirectory:args[4],preflight:args[0] == "preflight",started:start)
        return
      }
      if root["task"] as? String == "ripple" {
        try await executeRipple(recipeData:recipeData,envelope:envelope,ffmpegOverride:envelope.ffmpegPath,
          outputDirectory:args[4],preflight:args[0] == "preflight",started:start)
        return
      }
      if (root["conditioning"] as? [String:Any])?["task"] as? String == "extension" {
        try await executeExtension(recipeData:recipeData,envelope:envelope,
          outputDirectory:args[4],preflight:args[0] == "preflight",started:start)
        return
      }
    }
    var request:MLXDistilledRequest
    if let direct=try JSONSerialization.jsonObject(with:recipeData) as? [String:Any],
      ((direct["version"] as? Int == 5 && direct["task"] as? String == "union_control") ||
        (([6,11,17].contains(direct["version"] as? Int ?? -1)) && direct["task"] as? String == "ingredients") ||
        ((direct["version"] as? Int == 7 || direct["version"] as? Int == 16) && direct["task"] as? String == "msr") ||
        ((direct["version"] as? Int == 8 || direct["version"] as? Int == 9) && direct["task"] as? String == "dfr") ||
        (direct["version"] as? Int == 10 && direct["task"] as? String == "ic_control") ||
        direct["version"] as? Int == 12 || direct["version"] as? Int == 13 || direct["version"] as? Int == 14 || direct["version"] as? Int == 15) {
      request=try JSONDecoder().decode(MLXDistilledRequest.self,from:recipeData)
      guard URL(fileURLWithPath:request.outputDirectory).standardizedFileURL.path ==
        URL(fileURLWithPath:args[4]).standardizedFileURL.path else {
        throw invalid("Direct LTX task output must match the authenticated job envelope.")
      }
    } else {
      request=try MLXStudioRecipe.compile(data:recipeData,outputDirectory:args[4])
    }
    let recipe=try JSONSerialization.jsonObject(with:recipeData) as! [String:Any]
    let configured=envelope.ffmpegPath ?? ""
    let ffmpeg=configured.isEmpty ? recipe["ffmpeg"] as? String ?? "" : configured
    guard ffmpeg.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:ffmpeg) else { throw invalid("Select an executable FFmpeg in Runtime Settings.") }
    let automaticHead=try request.automaticDuration.map { try MLXDurationHead(checkpoint:URL(fileURLWithPath:$0.headCheckpointPath)) }
    if let policy=request.automaticDuration,let automaticHead {
      guard policy.headHeaderSHA256 == nil || policy.headHeaderSHA256 == automaticHead.headerSHA256 else {
        throw invalid("The automatic duration head changed after Studio preparation.")
      }
      request=try request.replacingFrames(policy.maximumFrames(fps:request.fps),automaticHeadHeaderSHA256:automaticHead.headerSHA256)
    }
    var memory=try MLXStudioMemoryPlan(request:request,physicalMemory:ProcessInfo.processInfo.physicalMemory,
      recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    var pipeline=try MLXMediaPipeline(request:request,videoActivationBytes:memory.videoActivationBytes,
      transformerActivationBytes:memory.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx)
    try Task.checkCancellation()
    if args[0] == "preflight" {
      let publishedFrames=try request.dfr.map {
        try MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:$0.temporalRounds)
      } ?? request.frames
      let publishedFPS=request.fps*Double(1 << (request.dfr?.temporalRounds ?? 0))
      var admission:[String:Any]=["nativeRuntime":"swift-mlx","jobID":envelope.jobID.uuidString,
        "frames":publishedFrames,"fps":publishedFPS,"task":request.task,
        "transformerActivationBytes":memory.transformerActivationBytes,"videoActivationBytes":memory.videoActivationBytes,
        "activationCeilingBytes":memory.activationCeilingBytes]
      if request.automaticDuration != nil {
        admission["frame_resolution"]="admitted_maximum_before_prediction"
        admission["duration_mode"]="automatic"
      }
      try emit(["status":"success","result":admission])
      return
    }
    let inferenceLease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }
    var preparedTextLease:MLXPreparedTextLease?,textBinding:MLXTextPreparationBinding?
    var textPreparationSeconds:Double=0
    defer { preparedTextLease?.release() }
    if let policy=request.automaticDuration,let automaticHead {
      Memory.peakMemory=0
      let binding=try MLXTextPreparationBinding(originalRecipeSHA256:envelope.recipeSHA256,
        prompt:request.prompt,negativePrompt:request.guidedSampling?.negativePrompt,
        gemmaRoot:request.gemmaRoot,connectorCheckpoint:request.connectorCheckpoint)
      textBinding=binding
      let preparationStarted=Date()
      preparedTextLease=try MLXAutomaticTextPreparation.prepare(binding:binding,policy:policy,fps:request.fps,
        expectedHeadHeaderSHA256:automaticHead.headerSHA256,admitResolvedFrames:{ frames in
          let resolved=try request.replacingFrames(frames)
          let resolvedMemory=try MLXStudioMemoryPlan(request:resolved,physicalMemory:ProcessInfo.processInfo.physicalMemory,
            recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
          let resolvedPipeline=try MLXMediaPipeline(request:resolved,videoActivationBytes:resolvedMemory.videoActivationBytes,
            transformerActivationBytes:resolvedMemory.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx)
          request=resolved;memory=resolvedMemory;pipeline=resolvedPipeline
        },progress:{ progress in
          var event:[String:Any]=["event":"progress","stage":progress.stage,
            "message":progress.stage == "duration_resolved" ? "Resolved automatic clip duration" : "Preparing prompt and automatic duration"]
          if let text=progress.text { event["completed"]=text.completed;event["total"]=text.total }
          if let resolution=progress.resolution {
            event["predicted_duration_seconds"]=resolution.predictedDurationSeconds
            event["resolved_frames"]=resolution.resolvedFrames;event["fps"]=resolution.fps
            event["effective_duration_seconds"]=resolution.effectiveDurationSeconds
          }
          try emit(event)
        })
      textPreparationSeconds=Date().timeIntervalSince(preparationStarted)
    }
    let output=URL(fileURLWithPath:args[4]),preview=output.appendingPathExtension("preview.png")
    let sourceDirectory=output.appendingPathExtension("source-audio-"+UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:sourceDirectory) }
    let preparedAudio:MLXSourceAudioInterval.Prepared?
    if let reference=request.audioReference {
      let interval=try MLXSourceAudioInterval(source:URL(fileURLWithPath:reference.path),
        sourceStartSeconds:reference.sourceStartSeconds,sourceDurationSeconds:reference.sourceDurationSeconds,
        durationSeconds:Double(request.frames)/request.fps)
      preparedAudio=try await interval.extract(ffmpeg:URL(fileURLWithPath:ffmpeg),directory:sourceDirectory)
    } else { preparedAudio=nil }
    let publicationDirectory=output.appendingPathExtension("publication-audio-"+UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:publicationDirectory) }
    let preparedPublicationAudio:MLXSourceAudioInterval.Prepared?
    if let reference=request.icControl?.publicationAudio {
      try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256).verify()
      let interval=try MLXSourceAudioInterval(source:URL(fileURLWithPath:reference.path),
        sourceStartSeconds:reference.sourceStartSeconds,sourceDurationSeconds:reference.sourceDurationSeconds,
        durationSeconds:Double(request.frames)/request.fps)
      preparedPublicationAudio=try await interval.extract(ffmpeg:URL(fileURLWithPath:ffmpeg),directory:publicationDirectory)
      try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256).verify()
    } else { preparedPublicationAudio=nil }
    let voiceDirectory=output.appendingPathExtension("msr-voices-"+UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:voiceDirectory) }
    var preparedMSRAudio:[MLXSourceAudioInterval.Prepared]=[]
    for reference in request.msr?.audioReferences ?? [] {
      try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256).verify()
      let interval=try MLXSourceAudioInterval(source:URL(fileURLWithPath:reference.path),
        sourceStartSeconds:reference.sourceStartSeconds,sourceDurationSeconds:reference.sourceDurationSeconds,
        durationSeconds:reference.effectiveDurationSeconds)
      preparedMSRAudio.append(try await interval.extract(ffmpeg:URL(fileURLWithPath:ffmpeg),
        directory:voiceDirectory.appendingPathComponent("slot-\(reference.imageSlot)")))
      try NativeMediaSource(path:reference.path,sha256:reference.sourceSHA256).verify()
    }
    let previewFrames=try request.dfr.map {
      try MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:$0.temporalRounds)
    } ?? request.frames
    let firstUpdates=request.guidedSampling.map { $0.steps+($0.mode == .guidedHQ ? 1 : 0) } ?? 8
    let noRefinement=request.singleStageSampling != nil || request.ingredientsSheet != nil || request.msr != nil
    var progress=MLXStudioProgress(temporalRounds:request.dfr?.temporalRounds ?? 0,
      samplingSteps:firstUpdates+(noRefinement ? 0 : 3))
    var lastPreview=Date.distantPast,revision=0
    var result:[String:Any]=[:]
    defer { try? FileManager.default.removeItem(at:preview) }
    _ = try pipeline.run(ffmpeg:URL(fileURLWithPath:ffmpeg),preparedAudio:preparedAudio,preparedPublicationAudio:preparedPublicationAudio,preparedMSRAudio:preparedMSRAudio,
      preparedTextLease:preparedTextLease,textPreparationBinding:textBinding,textPreparationSeconds:textPreparationSeconds,decodedPreview:{ index,bytes in
      guard MLXStudioPreview.shouldEmit(index:index,total:previewFrames,
        secondsSinceLast:Date().timeIntervalSince(lastPreview)) else { return }
      try Task.checkCancellation()
      try autoreleasepool { try MLXStudioPreview.write(rgb:bytes,width:request.width,height:request.height,to:preview) }
      revision += 1;lastPreview=Date()
      var event=progress.event(stage:"video_decode",completed:index+1,total:previewFrames)
      event["previewPath"]=preview.path;event["previewRevision"]=revision
      try emit(event)
    },beforePublish:{ staging,report in
      let stages=report["stage_seconds"] as? [String:Double] ?? [:]
      result=["video":output.appendingPathComponent("render.mp4").path,"jobID":envelope.jobID.uuidString,
        "seconds":Date().timeIntervalSince(start),
        "sampling_seconds":stages["sampling"] ?? (stages["stage1"] ?? 0)+(stages["stage2"] ?? 0),
        "metadata":report,"nativeRuntime":"swift-mlx",
        "elapsed_scope":"worker initialization through completed media, before atomic publication"]
      if request.automaticDuration != nil {
        result["use_complete_duration"]=true
        result["usable_source_in"]=0.0
        result["usable_duration"]=Double(request.frames)/request.fps
      }
      try recipeData.write(to:staging.appendingPathComponent("studio-recipe.json"),options:.withoutOverwriting)
      try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
    },progress:{ stage,completed,total in try emit(progress.event(stage:stage,completed:completed,total:total)) })
    try Task.checkCancellation()
    try emit(["status":"success","result":result])
  }

  static func executeMovieUpscale(recipeData:Data,envelope:NativeVideoJobEnvelope,
    outputDirectory:String,preflight:Bool,started:Date,executable:URL) async throws {
    let output=URL(fileURLWithPath:outputDirectory)
    let compiled=try MLXMovieStudioRecipe.compile(recipeData,outputDirectory:output)
    let request=compiled.request
    guard let configured=envelope.ffmpegPath,configured.hasPrefix("/"),
      FileManager.default.isExecutableFile(atPath:configured) else {
      throw invalid("Movie upscale requires an executable FFmpeg in its frozen native job.")
    }
    let cuts=try request.chunking ? MLXMovieSourceVideo.sceneCuts(
      rgb24:URL(fileURLWithPath:request.source.rgbPath),plan:request.plan) : []
    let memory=try MLXMovieMemoryPlan(request:request,cutFrames:cuts,
      physicalMemory:ProcessInfo.processInfo.physicalMemory,
      recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    let pipeline=MLXMovieUpscalePipeline(request:request)
    let admission=try await pipeline.preflight(maximumVideoActivationBytes:memory.videoActivationBytes,
      maximumTransformerActivationBytes:memory.transformerActivationBytes,
      maximumTextOwnedBufferBytes:memory.textOwnedBufferBytes)
    guard admission.chunks==memory.chunks else {
      throw invalid("Movie chunk geometry changed after memory admission.")
    }
    if preflight {
      let chunks=admission.chunks.map { ["startFrame":$0.startFrame,"endFrame":$0.endFrame,
        "paddedFrames":$0.paddedFrames,"reason":$0.reason] as [String:Any] }
      try emit(["status":"success","result":["nativeRuntime":"swift-mlx","jobID":envelope.jobID.uuidString,
        "task":"video_upscale","frames":request.plan.frames,"fps":request.plan.fps,
        "width":request.plan.size.outputWidth,"height":request.plan.size.outputHeight,
        "componentIdentitySHA256":admission.componentIdentitySHA256,"chunks":chunks,
        "videoActivationBytes":admission.maximumVideoActivationBytes,
        "transformerActivationBytes":admission.maximumTransformerActivationBytes,
        "textOwnedBufferBytes":admission.maximumTextOwnedBufferBytes,
        "activationCeilingBytes":memory.activationCeilingBytes,
        "memoryAdmissionScope":"Estimated per-stage owned buffers; system reserve retained; not measured peak memory"]])
      return
    }
    let lease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,"message":"Waiting for another local inference job"])
    }
    defer { lease.release() }
    let handle=try FileHandle(forReadingFrom:executable);defer { try? handle.close() }
    var hash=SHA256()
    while let data=try handle.read(upToCount:1024*1024),!data.isEmpty { try Task.checkCancellation();hash.update(data:data) }
    let workerSHA=hash.finalize().map { String(format:"%02x",$0) }.joined()
    try NativeMediaSource(path:executable.path,sha256:workerSHA).verify()
    let preview=output.appendingPathExtension("preview.png")
    defer { try? FileManager.default.removeItem(at:preview) }
    let events=MovieWorkerEvents(preview:preview,width:request.plan.size.outputWidth,height:request.plan.size.outputHeight,
      frames:request.plan.frames,chunks:admission.chunks.count)
    Memory.peakMemory=0
    _ = try await pipeline.run(ffmpeg:URL(fileURLWithPath:configured),workerSHA256:workerSHA,admission:admission,
      progress:events.report,decodedPreview:events.decoded,beforePublish:{ staging,report in
        try events.report("ready_to_publish",1,1)
        var info=task_vm_info_data_t(),count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
        let measured=withUnsafeMutablePointer(to:&info) { pointer in pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
          task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
        } }
        guard measured == KERN_SUCCESS else { throw invalid("Cannot measure movie worker memory.") }
        var metadata=report
        metadata["peak_mlx_bytes"]=Memory.peakMemory;metadata["peak_process_footprint_bytes"]=info.ledger_phys_footprint_peak
        metadata["current_process_footprint_bytes"]=info.phys_footprint
        metadata["process_memory_scope"]="Swift process; external FFmpeg process excluded"
        let result:[String:Any]=["nativeRuntime":"swift-mlx","jobID":envelope.jobID.uuidString,
          "video":output.appendingPathComponent("render.mp4").path,"metadata":metadata,
          "use_complete_duration":true,"usable_source_in":0.0,"usable_duration":request.plan.seconds,
          "seconds":Date().timeIntervalSince(started),
          "elapsed_scope":"worker initialization through completed media, before atomic publication"]
        try recipeData.write(to:staging.appendingPathComponent("studio-recipe.json"),options:.withoutOverwriting)
        try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys])
          .write(to:staging.appendingPathComponent("metadata.json"),options:.atomic)
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
          .write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
      })
    try Task.checkCancellation()
    let result=try JSONSerialization.jsonObject(with:read(output.appendingPathComponent("result.json").path))
    try emit(["status":"success","result":result])
  }

  static func executeTransformerConversion(data:Data,outputDirectory:String,
    preflight:Bool,started:Date) throws {
    let request=try MLXTransformerConversionRequest(data:data,
      outputDirectory:URL(fileURLWithPath:outputDirectory),requiresSourceIdentity:!preflight)
    let plan=try request.preflight()
    let identity=try JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.sourceIdentity))
    var result:[String:Any]=["nativeRuntime":"swift-mlx","task":"transformer-page-conversion",
      "source_path":plan.source.path,"sourceIdentity":identity,
      "output_directory":request.outputDirectory.path,"source_tensor_bytes":plan.sourceTensorBytes,
      "output_tensor_bytes":plan.outputTensorBytes,"maximum_working_bytes":plan.maximumWorkingBytes,
      "largest_quantization_input_bytes":plan.largestQuantizationInputBytes,"block_count":plan.blockCount]
    if preflight {
      result["status"]="admitted"
      try emit(["status":"success","result":result])
      return
    }
    let lease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { lease.release() }
    Memory.peakMemory=0
    let output=try MLXTransformerPageConverter.convert(plan,destination:request.outputDirectory) { progress in
      let fraction=Double(progress.completedTensors)/Double(max(1,progress.totalTensors))
      try emit(["event":"progress","stage":"transformer_conversion","phase":progress.stage,
        "fraction":fraction,"message":"Preparing native transformer pages: "+progress.stage,
        "completed_tensors":progress.completedTensors,"total_tensors":progress.totalTensors,
        "completed_pages":progress.completedPages,"total_pages":progress.totalPages])
    }
    try Task.checkCancellation()
    result["status"]="complete"
    result["manifestPath"]=output.appendingPathComponent("paged_manifest.json").path
    result["outputDirectory"]=output.path
    result["seconds"]=Date().timeIntervalSince(started)
    result["peakMLXBytes"]=Memory.peakMemory
    result["afterReleaseActiveBytes"]=Memory.activeMemory
    result["afterReleaseCacheBytes"]=Memory.cacheMemory
    try emit(["status":"success","result":result])
  }

  static func executeScene(recipeData:Data,envelope:NativeVideoJobEnvelope,
    outputDirectory:String,preflight:Bool,started:Date) async throws {
    let compiled=try MLXStudioSceneRecipe.compile(data:recipeData,
      outputDirectory:outputDirectory)
    let recipe=try JSONSerialization.jsonObject(with:recipeData) as! [String:Any]
    let configured=envelope.ffmpegPath ?? ""
    let ffmpeg=configured.isEmpty ? recipe["ffmpeg"] as? String ?? "" : configured
    guard ffmpeg.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:ffmpeg) else {
      throw invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    let physical=ProcessInfo.processInfo.physicalMemory
    let workingSet=UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize)
    let plans=try compiled.requests.enumerated().map { index,request in
      try MLXStudioMemoryPlan(request:request,
        extensionContextFrames:index == 0 ? nil : compiled.plan.overlapFrames,
        physicalMemory:physical,recommendedWorkingSet:workingSet)
    }
    let ceiling=plans.map(\.activationCeilingBytes).min()!
    let transformer=max(plans.map(\.transformerActivationBytes).max()!,
      try MLXSceneSampler.estimatedTransformerActivationBytes(compiled))
    guard transformer<=ceiling else { throw invalid("Scene images and audiovisual history exceed this Mac’s admitted transformer memory allowance.") }
    let first=compiled.requests[0]
    let geometry=try AVGeometry(width:first.width,height:first.height,
      frames:compiled.plan.totalFrames,fps:compiled.plan.fps)
    let sceneDecoder=try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:first.videoCheckpoint),settings:first.diffusionVAE)
    let scenePublicationMode=sceneDecoder.isDiffusion ? "diffusion_internal_tiling_native_latent_chain":compiled.decodeMode.publicationMode
    let decodePlan=try MLXSceneDecodeWindowPlan(geometry:geometry,checkpoint:URL(fileURLWithPath:first.videoCheckpoint),settings:first.diffusionVAE,
      plan:compiled.plan,strictBoundaries:compiled.strictBoundaries,maximumActivationBytes:ceiling,
      maximumWindowFrames:compiled.decodeMode.maximumWindowFrames)
    if case .single=compiled.decodeMode,decodePlan.latentRanges.count != 1,
      compiled.strictBoundaries.isEmpty {
      throw invalid("The complete LTX scene exceeds this Mac's admitted video decoder memory allowance.")
    }
    _=try MLXSceneSampler.preflight(compiled,
      maximumActivationBytes:transformer,decodePlan:decodePlan)
    if preflight {
      try emit(["status":"success","result":["nativeRuntime":"swift-mlx",
        "task":"scene","jobID":envelope.jobID.uuidString,
        "frames":geometry.frames,"fps":geometry.fps,
        "windowFrames":compiled.plan.windowFrames,
        "transformerActivationBytes":transformer,
        "videoActivationBytes":decodePlan.admittedActivationBytes,
        "videoDecodeWindows":decodePlan.latentRanges.count,
        "videoDecodeMode":scenePublicationMode,
        "activationCeilingBytes":ceiling]])
      return
    }
    let lease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { lease.release() }
    Memory.peakMemory=0
    let audioRoot=URL(fileURLWithPath:outputDirectory)
      .appendingPathExtension("source-audio-"+UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:audioRoot) }
    var preparedAudio:[MLXSourceAudioInterval.Prepared?]=[]
    var sourcePublication:MLXSourceAudioInterval.Prepared?
    if let firstSource=first.audioReference {
      for (index,request) in compiled.requests.enumerated() {
        guard let source=request.audioReference else {
          throw invalid("Every audio-driven scene window needs the continuous source interval.")
        }
        let interval=try MLXSourceAudioInterval(source:URL(fileURLWithPath:source.path),
          sourceStartSeconds:source.sourceStartSeconds,
          sourceDurationSeconds:source.sourceDurationSeconds,
          durationSeconds:Double(request.frames)/request.fps)
        preparedAudio.append(try await interval.extract(ffmpeg:URL(fileURLWithPath:ffmpeg),
          directory:audioRoot.appendingPathComponent("window-\(index)")))
      }
      let full=try MLXSourceAudioInterval(source:URL(fileURLWithPath:firstSource.path),
        sourceStartSeconds:firstSource.sourceStartSeconds,
        sourceDurationSeconds:firstSource.sourceDurationSeconds,
        durationSeconds:Double(MLXSceneMediaPublisher.deliveredFrames(plan:compiled.plan))/compiled.plan.fps)
      sourcePublication=try await full.extract(ffmpeg:URL(fileURLWithPath:ffmpeg),
        directory:audioRoot.appendingPathComponent("publication"))
    }
    var progress=MLXStudioProgress(sceneWindowCount:compiled.requests.count)
    let sampled=try MLXSceneSampler.sample(compiled,
      ffmpeg:URL(fileURLWithPath:ffmpeg),
      preparedAudio:preparedAudio,
      maximumActivationBytes:transformer) { stage,completed,total in
      try emit(progress.event(stage:stage,completed:completed,total:total))
    }
    let output=URL(fileURLWithPath:outputDirectory)
    let preview=output.appendingPathExtension("preview.png")
    defer { try? FileManager.default.removeItem(at:preview) }
    var lastPreview=Date.distantPast,revision=0
    var result:[String:Any]=[:]
    _=try MLXSceneMediaPublisher.publish(sampled,plan:compiled.plan,
      videoCheckpoint:URL(fileURLWithPath:first.videoCheckpoint),
      audioCheckpoint:URL(fileURLWithPath:first.audioCheckpoint),
      ffmpeg:URL(fileURLWithPath:ffmpeg),output:output,
      decodePlan:decodePlan,decodeMode:compiled.decodeMode,
      diffusionVAE:first.diffusionVAE,sourceAudio:sourcePublication,
      preview:{ index,bytes in
        guard index == 0 || index == geometry.frames-2 ||
          Date().timeIntervalSince(lastPreview) >= 1 else { return }
        try Task.checkCancellation()
        try autoreleasepool { try MLXStudioPreview.write(rgb:bytes,width:geometry.width,
          height:geometry.height,to:preview) }
        revision += 1;lastPreview=Date()
        var event=progress.event(stage:"video_decode",completed:index+1,total:geometry.frames)
        event["previewPath"]=preview.path;event["previewRevision"]=revision
        try emit(event)
      },beforePublish:{ staging,report in
        let members:[[String:Any]]=zip(compiled.clipIDs,zip(compiled.plan.segmentStarts,compiled.plan.segmentFrames)).map { id,range in
          ["clip_id":id,"source_in":Double(range.0)/geometry.fps,
            "duration":Double(range.1)/geometry.fps]
        }
        let scene:[String:Any]=["version":compiled.version,"members":members,
          "frame_rate":geometry.fps,
          "publication_mode":scenePublicationMode]
        result=["video":output.appendingPathComponent("render.mp4").path,
          "jobID":envelope.jobID.uuidString,"nativeRuntime":"swift-mlx",
          "seconds":Date().timeIntervalSince(started),"metadata":report,
          "scene":scene,
          "elapsed_scope":"worker initialization through completed media, before atomic publication"]
        try recipeData.write(to:staging.appendingPathComponent("studio-recipe.json"),
          options:.withoutOverwriting)
        try JSONSerialization.data(withJSONObject:result,
          options:[.prettyPrinted,.sortedKeys]).write(
            to:staging.appendingPathComponent("result.json"),
            options:.withoutOverwriting)
      },progress:{ stage,completed,total in
        try emit(progress.event(stage:stage,completed:completed,total:total))
      })
    try emit(["status":"success","result":result])
  }

  static func executeExtension(recipeData:Data,envelope:NativeVideoJobEnvelope,
    outputDirectory:String,preflight:Bool,started:Date) async throws {
    let compiled=try MLXStudioExtensionRecipe.compile(data:recipeData,
      outputDirectory:outputDirectory)
    let request=compiled.request,window=compiled.window
    let recipe=try JSONSerialization.jsonObject(with:recipeData) as! [String:Any]
    let configured=envelope.ffmpegPath ?? ""
    let ffmpeg=configured.isEmpty ? recipe["ffmpeg"] as? String ?? "" : configured
    guard ffmpeg.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:ffmpeg) else {
      throw invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    let memory=try MLXStudioMemoryPlan(request:request,
      extensionContextFrames:window.contextFrames,
      physicalMemory:ProcessInfo.processInfo.physicalMemory,
      recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    let interval=try MLXSourceMovieInterval(source:compiled.source,
      sha256:compiled.sourceSHA256,window:window)
    let encoder=try MLXExtensionGuideEncoder(window:window,
      videoCheckpoint:URL(fileURLWithPath:request.videoCheckpoint),
      audioCheckpoint:URL(fileURLWithPath:request.audioCheckpoint),
      maximumOwnedBufferBytes:memory.activationCeilingBytes)
    let pipeline=try MLXMediaPipeline(request:request,
      extensionContextFrames:window.contextFrames,
      videoActivationBytes:memory.videoActivationBytes,
      transformerActivationBytes:memory.transformerActivationBytes,
      videoBackend:.mlx,audioBackend:.mlx)
    let sourceStatus=try await interval.preflight()
    try Task.checkCancellation()
    if preflight {
      try emit(["status":"success","result":["nativeRuntime":"swift-mlx",
        "jobID":envelope.jobID.uuidString,"task":"extension",
        "frames":window.additionalFrames,"totalFrames":window.totalFrames,
        "contextFrames":window.contextFrames,"sourceFrames":sourceStatus.sourceFrames,
        "fps":request.fps,"transformerActivationBytes":memory.transformerActivationBytes,
        "videoActivationBytes":memory.videoActivationBytes,
        "activationCeilingBytes":memory.activationCeilingBytes]])
      return
    }
    let inferenceLease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }
    let output=URL(fileURLWithPath:outputDirectory)
    let preview=output.appendingPathExtension("preview.png")
    let sourceDirectory=output.appendingPathExtension("source-extension-"+UUID().uuidString)
    defer {
      try? FileManager.default.removeItem(at:sourceDirectory)
      try? FileManager.default.removeItem(at:preview)
    }
    var progress=MLXStudioProgress(),lastPreview=Date.distantPast,revision=0
    let prepared=try await interval.prepare(ffmpeg:URL(fileURLWithPath:ffmpeg),
      directory:sourceDirectory)
    let guideLease=MLXMediaPipeline.ExtensionGuideLease(try encoder.encode(prepared) {
      stage,completed,total in
      try emit(progress.event(stage:stage,completed:completed,total:total))
    })
    var result:[String:Any]=[:]
    _ = try pipeline.run(ffmpeg:URL(fileURLWithPath:ffmpeg),extensionGuideLease:guideLease,
      decodedPreview:{ index,bytes in
        guard index == 0 || index == window.additionalFrames-1 ||
          Date().timeIntervalSince(lastPreview) >= 1 else { return }
        try Task.checkCancellation()
        try autoreleasepool {
          try MLXStudioPreview.write(rgb:bytes,width:request.width,height:request.height,to:preview)
        }
        revision += 1;lastPreview=Date()
        var event=progress.event(stage:"video_decode",completed:index+1,total:window.additionalFrames)
        event["previewPath"]=preview.path;event["previewRevision"]=revision
        try emit(event)
      },beforePublish:{ staging,report in
        let stages=report["stage_seconds"] as? [String:Double] ?? [:]
        result=["video":output.appendingPathComponent("render.mp4").path,
          "jobID":envelope.jobID.uuidString,"task":"extension",
          "usable_source_in":0.0,"usable_duration":window.additionalDuration,
          "seconds":Date().timeIntervalSince(started),
          "sampling_seconds":(stages["stage1"] ?? 0)+(stages["stage2"] ?? 0),
          "metadata":report,"nativeRuntime":"swift-mlx",
          "elapsed_scope":"worker initialization through completed media, before atomic publication"]
        try recipeData.write(to:staging.appendingPathComponent("studio-recipe.json"),
          options:.withoutOverwriting)
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
          .write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
      },progress:{ stage,completed,total in
        try emit(progress.event(stage:stage,completed:completed,total:total))
      })
    try Task.checkCancellation()
    try emit(["status":"success","result":result])
  }

  static func executeRipple(recipeData:Data,envelope:NativeVideoJobEnvelope,
    ffmpegOverride:String?,outputDirectory:String,preflight:Bool,started:Date) async throws {
    let request=try JSONDecoder().decode(MLXRippleRequest.self,from:recipeData)
    guard URL(fileURLWithPath:request.outputDirectory).standardizedFileURL.path ==
      URL(fileURLWithPath:outputDirectory).standardizedFileURL.path else {
      throw invalid("Ripple output directory differs from the signed job envelope.")
    }
    let configured=ffmpegOverride.flatMap { $0.isEmpty ? nil : $0 } ?? request.ffmpegPath
    guard configured.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:configured) else {
      throw invalid("Select an executable FFmpeg in Runtime Settings for Ripple.")
    }
    let pipeline=try MLXRipplePipeline(request:request,
      physicalMemory:ProcessInfo.processInfo.physicalMemory,
      recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    if preflight {
      try pipeline.preflight()
      try emit(["status":"success","result":["nativeRuntime":"swift-mlx",
        "jobID":envelope.jobID.uuidString,"task":"ripple",
        "frames":request.frames,"editorialFrames":request.editorialFrames,
        "fps":request.fps,"transformerActivationBytes":pipeline.memory.transformerActivationBytes,
        "videoActivationBytes":pipeline.memory.videoActivationBytes,
        "activationCeilingBytes":pipeline.memory.activationCeilingBytes]])
      return
    }
    let inferenceLease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }
    let output=URL(fileURLWithPath:outputDirectory)
    let preview=output.appendingPathExtension("preview.png")
    var progress=MLXStudioProgress(),lastPreview=Date.distantPast,revision=0
    defer { try? FileManager.default.removeItem(at:preview) }
    let result=try await pipeline.run(ffmpeg:URL(fileURLWithPath:configured),decodedPreview:{ index,bytes in
      guard index == 0 || index == request.editorialFrames-1 ||
        Date().timeIntervalSince(lastPreview) >= 1 else { return }
      try Task.checkCancellation()
      try autoreleasepool { try MLXStudioPreview.write(rgb:bytes,width:request.width,
        height:request.height,to:preview) }
      revision += 1;lastPreview=Date()
      var event=progress.event(stage:"video_decode",completed:index+1,
        total:request.editorialFrames)
      event["previewPath"]=preview.path;event["previewRevision"]=revision
      try emit(event)
    },beforePublish:{ staging,result in
      var saved=result
      saved["jobID"]=envelope.jobID.uuidString
      saved["worker_seconds"]=Date().timeIntervalSince(started)
      try recipeData.write(to:staging.appendingPathComponent("ripple-request.json"),
        options:.withoutOverwriting)
      try JSONSerialization.data(withJSONObject:saved,options:[.prettyPrinted,.sortedKeys])
        .write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
    },progress:{ stage,completed,total in
      try emit(progress.event(stage:stage,completed:completed,total:total))
    })
    var final=result
    final["jobID"]=envelope.jobID.uuidString
    final["worker_seconds"]=Date().timeIntervalSince(started)
    try emit(["status":"success","result":final])
  }
}
