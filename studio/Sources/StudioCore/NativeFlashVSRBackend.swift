import Darwin
import Foundation
import InferenceContracts

/// Studio owns source media and delivery; the dedicated native helper owns all
/// model execution. Saved vpipe plans continue through their explicit adapter.
public struct NativeFlashVSRBackend: FlashVSRIntermediateBackend {
  public init() {}
  static func workspace() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-native-flash-" + UUID().uuidString)
    try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false); return url
  }
  static func workerPath(_ runtime: FlashVSRRuntime) throws -> String {
    let candidate = runtime.swiftWorkerPath ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddFlashVSRWorker").path ?? ""
    guard candidate.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:candidate) else {
      throw StudioError.invalid("The native FlashVSR+ helper is missing. Use a complete current Studio app or connect its Swift worker.")
    }
    return candidate
  }
  static func recipe(settings:FlashVSRSettings,runtime:FlashVSRRuntime,sourceID:String,input:URL,inputSHA:String,
    frames:Int,fps:FlashVSRFrameRate,output:URL) throws -> NativeFlashVSRRecipe {
    try .init(sourceID:sourceID,inputRGBPath:input.path,inputSHA256:inputSHA,
      outputRGBPath:output.appendingPathComponent("restored.rgb").path,
      scratchDirectory:output.appendingPathComponent("scratch").path,modelRoot:runtime.modelRoot,
      width:settings.width,height:settings.height,frames:frames,fpsNumerator:fps.numerator,fpsDenominator:fps.denominator,
      temporalCompression:settings.temporalCompression,variant:settings.variant == .turbo ? .turbo:.base,
      decoder:settings.decoder == .tiny ? .tiny:.wan,seed:settings.seed,
      colorFix:settings.colorFix == .wavelet ? .wavelet:settings.colorFix == .adain ? .adain:.off,
      waveletLevels:settings.waveletLevels,temporalSigma:settings.temporalSigma,
      maximumActiveBytes:min(128*1024*1024*1024,ProcessInfo.processInfo.physicalMemory*3/4),
      leasePath:runtime.leasePath.isEmpty ? nil:runtime.leasePath,ownerPID:getpid())
  }
  static func call(_ recipe:NativeFlashVSRRecipe,runtime:FlashVSRRuntime,mode:String,directory:URL,output:URL,
    cancellation:NativeHeadlessCancellation,emit:@escaping @Sendable(FlashVSREvent)->Void) throws -> [String:Any] {
    let path = directory.appendingPathComponent(UUID().uuidString + ".flash-recipe.json")
    try JSONEncoder().encode(recipe).write(to:path,options:.withoutOverwriting)
    return try NativeHeadlessExecutor.worker(try workerPath(runtime),recipe:path,output:output,mode:mode,cancellation:cancellation,emit:{ data in
      guard let value = try? JSONSerialization.jsonObject(with:data) as? [String:Any], value["status"] == nil else { return }
      let stage = value["stage"] as? String ?? mode
      emit(.init(stage:stage,message:value["message"] as? String ?? "FlashVSR+ · " + stage.replacingOccurrences(of:"_",with:" "),fraction:value["fraction"] as? Double))
    })
  }
  static func inspectResources(settings:FlashVSRSettings,runtime:FlashVSRRuntime,directory:URL,
    cancellation:NativeHeadlessCancellation) throws -> FlashVSRResources {
    try settings.validateExecution()
    guard settings.backend == .swiftMLX else { throw StudioError.invalid("The native FlashVSR adapter requires an explicit Swift/MLX selection.") }
    guard runtime.modelRoot.hasPrefix("/"), !runtime.modelRoot.utf8.contains(0) else { throw StudioError.invalid("Connect an absolute local FlashVSR+ model folder.") }
    for (name,file) in [("FFmpeg",runtime.ffmpegPath),("FFprobe",runtime.ffprobePath)] {
      guard file.hasPrefix("/"),FileManager.default.isExecutableFile(atPath:file) else { throw StudioError.invalid("Connect the existing " + name + " media utility before restoration.") }
    }
    let worker = try workerPath(runtime)
    let components = try FlashVSRModelManifest.resolve(root:URL(fileURLWithPath:runtime.modelRoot),settings:settings)
    try FlashVSRModelManifest.verifyPublished(components,settings:settings)
    var engine = try [worker,runtime.ffmpegPath,runtime.ffprobePath].map { try FlashVSRFile.capture(URL(fileURLWithPath:$0)) }
    engine.append(try .capture(NativeMLXMetalLibrary.resolve(executable: URL(fileURLWithPath: worker))))
    let canvas = try settings.canvas(), input = directory.appendingPathComponent("probe.rgb")
    try Data(count:canvas.width*canvas.height*3).write(to:input,options:.withoutOverwriting)
    let hash = try FlashVSRFile.capture(input).sha256, output = directory.appendingPathComponent("admission")
    let probe = try recipe(settings:settings,runtime:runtime,sourceID:hash,input:input,inputSHA:hash,
      frames:1,fps:.init(numerator:24,denominator:1),output:output)
    let result = try call(probe,runtime:runtime,mode:"preflight",directory:directory,output:output,cancellation:cancellation,emit:{ _ in })
    guard result["preflight"] as? Bool == true,result["admission"] is [String:Any] else { throw StudioError.invalid("The native helper did not admit the selected FlashVSR+ model and decoder.") }
    for file in engine { try file.verify() }; try cancellation.check()
    return .init(components:components,engineFiles:engine)
  }
  public func preflightResources(settings:FlashVSRSettings,runtime:FlashVSRRuntime) async throws -> FlashVSRResources {
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{
      try await Task.detached {
        let root = try Self.workspace(); defer { try? FileManager.default.removeItem(at:root) }
        return try Self.inspectResources(settings:settings,runtime:runtime,directory:root,cancellation:cancellation)
      }.value
    },onCancel:{ cancellation.cancel() })
  }
  public func preflight(request:VideoUpscaleRequest) async throws -> FlashVSRResolvedPlan {
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{
      try await Task.detached {
        try request.settings.validateExecution()
        guard request.settings.backend == .swiftMLX,request.format == "weetodd-video-upscale-v1",request.firstFrame >= 0,
          request.sourcePath.hasPrefix("/"), !request.sourcePath.utf8.contains(0),
          request.frameCount == nil || request.frameCount! > 0 else { throw StudioError.invalid("Native FlashVSR+ requires an absolute local source and a valid selected interval.") }
        let root = try Self.workspace(); defer { try? FileManager.default.removeItem(at:root) }
        let source = try FlashVSRFile.capture(URL(fileURLWithPath:request.sourcePath))
        guard request.sourceSHA256 == nil || request.sourceSHA256 == source.sha256 else { throw StudioError.invalid("The source movie changed before restoration.") }
        let media = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:source.path),runtime:request.runtime,directory:root,cancellation:cancellation)
        let frames = request.frameCount ?? media.frames-request.firstFrame
        guard frames > 0,frames <= media.frames,request.firstFrame <= media.frames-frames else { throw StudioError.invalid("The selected restoration interval exceeds the source movie.") }
        let resources = try Self.inspectResources(settings:request.settings,runtime:request.runtime,directory:root,cancellation:cancellation)
        let output = root.appendingPathComponent("planned"), raw = root.appendingPathComponent("unprepared.rgb")
        let planRecipe = try Self.recipe(settings:request.settings,runtime:request.runtime,sourceID:source.sha256,input:raw,inputSHA:source.sha256,frames:frames,fps:media.fps,output:output)
        let response = try Self.call(planRecipe,runtime:request.runtime,mode:"plan",directory:root,output:output,cancellation:cancellation,emit:{ _ in })
        guard response["planOnly"] as? Bool == true,let memory = response["memory"] as? [String:Any],
          let nativeBytes = memory["temporaryBytes"] as? NSNumber else { throw StudioError.invalid("The native helper returned no bounded disk plan.") }
        let mediaBytes = try FlashVSRResourcePlan.temporaryBytes(media:media,settings:request.settings,frames:frames)
        let addition = mediaBytes.addingReportingOverflow(nativeBytes.int64Value)
        guard !addition.overflow,nativeBytes.int64Value > 0 else { throw StudioError.invalid("Native restoration disk estimate overflow.") }
        return .init(request:request,source:source,media:media,components:resources.components,engineFiles:resources.engineFiles,frames:frames,temporaryBytes:addition.partialValue)
      }.value
    },onCancel:{ cancellation.cancel() })
  }
  public func run(plan:FlashVSRResolvedPlan,directory:URL,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> FlashVSRReceipt {
    try await execute(plan: plan, directory: directory, intermediateOnly: false, emit: emit)
  }
  public func runIntermediate(plan:FlashVSRResolvedPlan,directory:URL,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> FlashVSRReceipt {
    try await execute(plan: plan, directory: directory, intermediateOnly: true, emit: emit)
  }
  private func execute(plan:FlashVSRResolvedPlan,directory:URL,intermediateOnly:Bool,emit:@escaping @Sendable(FlashVSREvent)->Void) async throws -> FlashVSRReceipt {
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation:{
      try await Task.detached {
        guard plan.request.settings.backend == .swiftMLX else { throw StudioError.invalid("Native restoration cannot execute another backend's saved plan.") }
        try FlashVSRModelManifest.verifyLookup(plan.components,settings:plan.request.settings)
        return try FlashVSRMediaPipeline.execute(plan:plan,directory:directory,cancellation:cancellation,emit:emit,intermediateOnly:intermediateOnly,restore:Self.restore)
      }.value
    },onCancel:{ cancellation.cancel() })
  }
  static func restore(plan:FlashVSRResolvedPlan,selected:URL,restored:URL,directory:URL,
    cancellation:NativeHeadlessCancellation,emit:@escaping @Sendable(FlashVSREvent)->Void) throws {
    let settings = plan.request.settings,runtime = plan.request.runtime,canvas = try settings.canvas()
    func ffmpeg(_ args:[String],_ name:String) throws {
      let result = try ExternalEngineProcess.run(executable:URL(fileURLWithPath:runtime.ffmpegPath),arguments:["-v","error","-nostdin","-n"]+args,directory:directory,log:directory.appendingPathComponent(name+".log"),cancellation:cancellation)
      guard result.exitCode == 0 else { throw StudioError.invalid("Native FlashVSR media transport failed: " + name) }
    }
    let input = directory.appendingPathComponent("input.rgb"),output = directory.appendingPathComponent("native-worker")
    let fit = settings.fit == .fit
      ? "scale=\(settings.width):\(settings.height):force_original_aspect_ratio=decrease:force_divisible_by=2,pad=\(settings.width):\(settings.height):(ow-iw)/2:(oh-ih)/2"
      : "scale=\(settings.width):\(settings.height):force_original_aspect_ratio=increase,crop=\(settings.width):\(settings.height)"
    let right = canvas.width-settings.width,bottom = canvas.height-settings.height
    var filter = fit+",setsar=1"
    if right > 0 || bottom > 0 { filter += ",pad=\(canvas.width):\(canvas.height):0:0,fillborders=left=0:right=\(right):top=0:bottom=\(bottom):mode=smear" }
    try ffmpeg(["-i",selected.path,"-map","0:v:0","-vf",filter,"-fps_mode","passthrough","-an","-pix_fmt","rgb24","-f","rawvideo",input.path],"native-prepare")
    let frozen = try FlashVSRFile.capture(input)
    let recipe = try Self.recipe(settings:settings,runtime:runtime,sourceID:plan.source.sha256,input:input,inputSHA:frozen.sha256,frames:plan.frames,fps:plan.media.fps,output:output)
    let result = try call(recipe,runtime:runtime,mode:"render",directory:directory,output:output,cancellation:cancellation,emit:emit)
    guard result["pixelFormat"] as? String == "rgb24",result["video"] as? String == recipe.outputRGBPath,
      let metrics = result["metrics"] as? [String:Any],metrics["inputSHA256"] as? String == frozen.sha256,
      metrics["visibleFrames"] as? Int == plan.frames,metrics["fpsNumerator"] as? Int == plan.media.fps.numerator,
      metrics["fpsDenominator"] as? Int == plan.media.fps.denominator,
      let expected = metrics["outputSHA256"] as? String else { throw StudioError.invalid("Native restoration returned a different source, clock or RGB transport.") }
    let raw = try FlashVSRFile.capture(URL(fileURLWithPath:recipe.outputRGBPath))
    guard raw.sha256 == expected,raw.bytes == Int64(recipe.outputBytes) else { throw StudioError.invalid("Native RGB output size or digest differs from its completed receipt.") }
    try frozen.verify();try cancellation.check()
    try ffmpeg(["-f","rawvideo","-pixel_format","rgb24","-video_size","\(settings.width)x\(settings.height)","-framerate",plan.media.fps.argument,"-i",raw.path,"-an","-c:v","rawvideo","-f","nut",restored.path],"native-lossless")
    try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("native-result.json"),options:.withoutOverwriting)
  }
}
