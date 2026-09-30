import Foundation
import Darwin
import LTX25MLX
import InferenceContracts
import MLX

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
    guard args.count == 5,["preflight","render"].contains(args[0]),args[1] == "--request",args[3] == "--output" else {
      throw invalid("Usage: WeeToddLTXWorker preflight|render --request ENVELOPE --output DIRECTORY")
    }
    try Task.checkCancellation()
    let envelope=try NativeVideoJobEnvelope.decode(read(args[2]),expectedEngine:.ltx25,
      expectedOutputDirectory:args[4])
    let recipeData=try read(envelope.recipePath)
    try envelope.validateRecipe(recipeData)
    if let root=try JSONSerialization.jsonObject(with:recipeData) as? [String:Any] {
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
    let request=try MLXStudioRecipe.compile(data:recipeData,outputDirectory:args[4])
    let recipe=try JSONSerialization.jsonObject(with:recipeData) as! [String:Any]
    let configured=envelope.ffmpegPath ?? ""
    let ffmpeg=configured.isEmpty ? recipe["ffmpeg"] as? String ?? "" : configured
    guard ffmpeg.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:ffmpeg) else { throw invalid("Select an executable FFmpeg in Runtime Settings.") }
    let memory=try MLXStudioMemoryPlan(request:request,physicalMemory:ProcessInfo.processInfo.physicalMemory,
      recommendedWorkingSet:UInt64(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    let pipeline=try MLXMediaPipeline(request:request,videoActivationBytes:memory.videoActivationBytes,
      transformerActivationBytes:memory.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx)
    try Task.checkCancellation()
    if args[0] == "preflight" {
      try emit(["status":"success","result":["nativeRuntime":"swift-mlx","jobID":envelope.jobID.uuidString,
        "frames":request.frames,"fps":request.fps,"task":request.task,
        "transformerActivationBytes":memory.transformerActivationBytes,"videoActivationBytes":memory.videoActivationBytes,
        "activationCeilingBytes":memory.activationCeilingBytes]])
      return
    }
    let inferenceLease=try NativeInferenceLease.acquire {
      try? emit(["event":"progress","stage":"waiting","fraction":0,
        "message":"Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }
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
    var progress=MLXStudioProgress(),lastPreview=Date.distantPast,revision=0
    var result:[String:Any]=[:]
    defer { try? FileManager.default.removeItem(at:preview) }
    _ = try pipeline.run(ffmpeg:URL(fileURLWithPath:ffmpeg),preparedAudio:preparedAudio,decodedPreview:{ index,bytes in
      guard index == 0 || index == request.frames-1 || Date().timeIntervalSince(lastPreview) >= 1 else { return }
      try Task.checkCancellation()
      try autoreleasepool { try MLXStudioPreview.write(rgb:bytes,width:request.width,height:request.height,to:preview) }
      revision += 1;lastPreview=Date()
      var event=progress.event(stage:"video_decode",completed:index+1,total:request.frames)
      event["previewPath"]=preview.path;event["previewRevision"]=revision
      try emit(event)
    },beforePublish:{ staging,report in
      let stages=report["stage_seconds"] as? [String:Double] ?? [:]
      result=["video":output.appendingPathComponent("render.mp4").path,"jobID":envelope.jobID.uuidString,
        "seconds":Date().timeIntervalSince(start),
        "sampling_seconds":(stages["stage1"] ?? 0)+(stages["stage2"] ?? 0),"metadata":report,"nativeRuntime":"swift-mlx",
        "elapsed_scope":"worker initialization through completed media, before atomic publication"]
      try recipeData.write(to:staging.appendingPathComponent("studio-recipe.json"),options:.withoutOverwriting)
      try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
    },progress:{ stage,completed,total in try emit(progress.event(stage:stage,completed:completed,total:total)) })
    try Task.checkCancellation()
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
