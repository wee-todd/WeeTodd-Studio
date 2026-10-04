import CryptoKit
import Foundation
import ImageIO

/// Dedicated source-movie producer. It never uses the extension tail extractor,
/// regenerates source audio, changes source FPS or invokes an inference model.
public enum NativeLTXMoviePreparation {
  /// Timing-only inspection also admits silent movies and rotated video tracks.
  /// No source pixels, audio samples or model weights are decoded here.
  public static func inspectSource(_ source:URL) async throws -> [String:Any] {
    let clock=try await NativeMovieFrozenMedia.videoClock(source.standardizedFileURL.resolvingSymlinksInPath())
    return ["width":clock.width,"height":clock.height,"fps":clock.fps,
      "frames":clock.times.count,"duration":Double(clock.times.count)/clock.fps,
      "startSeconds":clock.times[0]]
  }
  public struct Endpoint:Sendable {
    public let path:String,role:String,strength:Double,crf:Int
    public init(path:String,role:String,strength:Double,crf:Int=33) {
      self.path=path;self.role=role;self.strength=strength;self.crf=crf
    }
  }
  public struct Sidecar:Sendable {
    public let path:String,startSeconds:Double,durationSeconds:Double?
    public init(path:String,startSeconds:Double=0,durationSeconds:Double?=nil) {
      self.path=path;self.startSeconds=startSeconds;self.durationSeconds=durationSeconds
    }
  }
  /// Resolve existing native model-setup fields without changing task provenance.
  /// Component compatibility is subsequently admitted by the shared worker.
  public static func components(from profile:[String:Any],mode:LTX25MovieUpscaleSettings.Mode,
    pixelSpatialAdapterPath:String?=nil) throws -> [String:String] {
    let mapping=["video_checkpoint":"video_vae_path","spatial_upscaler_checkpoint":"spatial_upscaler_path",
      "gemma_root":"text_encoder_path","transformer_root":"transformer_path","audio_checkpoint":"audio_vae_path"]
    var result:[String:String]=[:]
    for (target,key) in mapping where mode != .latentOnly || ["video_checkpoint","spatial_upscaler_checkpoint"].contains(target) {
      guard let path=profile[key] as? String,path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0) else {
        throw StudioError.invalid("Movie upscaling is missing native model component: \(key)")
      }
      result[target]=URL(fileURLWithPath:path).resolvingSymlinksInPath().standardizedFileURL.path
    }
    if mode != .latentOnly {
      let transformer=URL(fileURLWithPath:result["transformer_root"]!)
      var directory:ObjCBool=false
      guard FileManager.default.fileExists(atPath:transformer.path,isDirectory:&directory),directory.boolValue else {
        throw StudioError.invalid("Movie refinement requires completed native distilled transformer pages, not an unconverted raw checkpoint.")
      }
      result["connector_checkpoint"]=(profile["connector_checkpoint"] as? String).flatMap { $0.isEmpty ? nil:$0 }
        ?? transformer.appendingPathComponent("pages/fixed.safetensors").path
    }
    if mode == .pixelSpatial {
      guard let adapter=pixelSpatialAdapterPath ?? profile["pixel_spatial_adapter"] as? String ?? profile["dfr_detailing_lora_path"] as? String,
        adapter.hasPrefix("/"),adapter.utf8.count<=4096,!adapter.utf8.contains(0) else {
        throw StudioError.invalid("Pixel-Spatial movie refinement needs the compatible official Pixel-Spatial adapter.")
      }
      result["pixel_spatial_adapter"]=adapter
    }
    return result
  }
  public static func validateEditor(clip:Clip,settings:LTX25MovieUpscaleSettings,isContinuousScene:Bool=false) throws {
    try settings.validate()
    let selection=clip.generationSelection
    guard clip.engine == .ltx25,clip.continuityMode == "independent",!isContinuousScene,
      clip.extensionDirection.isEmpty,clip.extensionSource.isEmpty,clip.rippleDraft == nil,
      clip.negativePrompt.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
      selection?.ltx25Guidance == nil,selection?.ltx25AutomaticDuration == nil,
      selection?.ltx25Keyframes == nil,selection?.ltx25SingleStage == nil,
      selection?.cfg == nil,selection?.shift == nil,selection?.h3SamplingMethod == nil,
      selection?.h3Reference == nil,selection?.h3Joint == nil,selection?.h3MotionFidelity == nil,
      clip.attachments.allSatisfy({$0.h3LoRA == nil && $0.h3ReferencePlacement == nil}),
      selection?.projectionBackend == nil,selection?.transformerBackend == nil,
      selection?.memoryPolicy == nil || selection?.memoryPolicy == "recipe",
      clip.audioDriverSelection == nil,clip.musicSource == nil,
      selection?.steps == nil || selection?.steps == (settings.mode == .latentOnly ? 0:3),
      selection?.refinementSteps == nil || selection?.refinementSteps == 3 else {
      throw StudioError.invalid("Movie upscaling cannot ignore alternate sampling, guidance, automatic duration, slots or continuity settings.")
    }
    guard clip.attachments.allSatisfy({ [MediaRole.reference,.first,.last,.audioDriver].contains($0.role) }),
      !clip.attachments.contains(where:{ $0.role == .lora && $0.isEnabled }) else {
      throw StudioError.invalid("Movie upscaling uses its movie source and explicit endpoints; unrelated controls and LoRA stacks are unsupported.")
    }
  }
  static func grid(width:Int,height:Int,policy:LTX25MovieUpscaleSettings.SizePolicy) throws -> (width:Int,height:Int,left:Int,top:Int,resize:Bool) {
    guard (32...8192).contains(width),(32...8192).contains(height) else { throw StudioError.invalid("Movie source dimensions are outside the native movie contract.") }
    if width%32==0,height%32==0 { return (width,height,0,0,false) }
    guard policy != .strict else { throw StudioError.invalid("Strict movie upscaling needs source dimensions on the 32-pixel grid.") }
    if policy == .fitNearest {
      let low=max(32,Int(floor(Double(height)*0.65/32))*32),high=max(32,Int(ceil(Double(height)*1.35/32))*32)
      let aspect=Double(width)/Double(height)
      var best:(distance:Double,down:Int,error:Double,w:Int,h:Int)?
      for h in stride(from:low,through:high,by:32) {
        let w=max(32,Int((Double(h)*aspect/32).rounded(.toNearestOrEven))*32)
        let error=abs((Double(w)/Double(h))/aspect-1)
        if error>0.005 { continue }
        let scale=Double(h)/Double(height),candidate=(abs(log(scale)),scale>=1 ? 0:1,error,w,h)
        if let old=best {
          if candidate < (old.distance,old.down,old.error,old.w,old.h) { best=candidate }
        } else { best=candidate }
      }
      if let best { return (best.w,best.h,0,0,true) }
    }
    let w=width/32*32,h=height/32*32
    return (w,h,(width-w)/2,(height-h)/2,false)
  }
  public static func prepare(source:URL,startSeconds:Double=0,visibleFrames:Int?=nil,
    settings:LTX25MovieUpscaleSettings,components:[String:String],prompt:String,seed:UInt64,
    endpoints:[Endpoint]=[],sidecar:Sidecar?=nil,ffmpeg:URL,destination:URL,
    outputDirectory:URL,editorRequest:[String:Any]?=nil,diffusionVAE:LTX25DiffusionVAESettings?=nil) async throws -> [String:Any] {
    try settings.validate()
    guard source.isFileURL,source.path.utf8.count<=4096,!source.path.utf8.contains(0),
      startSeconds.isFinite,startSeconds>=0,endpoints.count<=2,Set(endpoints.map(\.role)).count==endpoints.count,
      endpoints.allSatisfy({ ["first","last"].contains($0.role) && $0.strength==settings.anchorStrength && (0...51).contains($0.crf)
        && $0.path.hasPrefix("/") && $0.path.utf8.count<=4096 && !$0.path.utf8.contains(0) }),
      (settings.audioPolicy == .sidecar)==(sidecar != nil),settings.mode != .latentOnly || endpoints.isEmpty,
      destination.isFileURL,outputDirectory.isFileURL,!FileManager.default.fileExists(atPath:destination.path) else {
      throw StudioError.invalid("Movie upscaling needs one frozen source interval, exact endpoint strengths and explicit audio selection.")
    }
    let required:Set<String>=["video_checkpoint","spatial_upscaler_checkpoint"]
    let refine:Set<String>=["gemma_root","connector_checkpoint","transformer_root","audio_checkpoint"]
    let keys=required.union(settings.mode == .latentOnly ? []:refine).union(settings.mode == .pixelSpatial ? ["pixel_spatial_adapter"]:[])
    guard Set(components.keys)==keys,components.values.allSatisfy({ $0.hasPrefix("/") && $0.utf8.count<=4096 && !$0.utf8.contains(0) }),
      prompt.utf8.count<=65536,!prompt.utf8.contains(0),settings.mode == .latentOnly ? prompt.isEmpty : !prompt.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
      throw StudioError.invalid("Movie mode must use exactly its required native components and an applicable prompt.")
    }
    let diffusion=try diffusionVAE.map { try NativeLTXDiffusionVAE.wire($0,checkpoint:URL(fileURLWithPath:components["video_checkpoint"]!)) }
    let original=source.resolvingSymlinksInPath().standardizedFileURL
    let mediaPaths=[original.path]+endpoints.map { URL(fileURLWithPath:$0.path).standardizedFileURL.resolvingSymlinksInPath().path }
      + (sidecar.map { [URL(fileURLWithPath:$0.path).standardizedFileURL.resolvingSymlinksInPath().path] } ?? [])
    guard Set(mediaPaths).count==mediaPaths.count else {
      throw StudioError.invalid("Movie source, endpoint and sidecar roles must use distinct canonical media paths.")
    }
    let sourceSHA=try NativeMovieFrozenMedia.digest(original),signature=try NativeMovieFrozenMedia.Identity(original)
    let clock=try await NativeMovieFrozenMedia.videoClock(original)
    let position=((startSeconds-clock.times[0])*clock.fps).rounded(.toNearestOrEven)
    guard position.isFinite,position>=0,position<Double(clock.times.count) else { throw StudioError.invalid("Movie source start is outside its actual visible interval.") }
    let start=Int(position),count=visibleFrames ?? (clock.times.count-start)
    guard count>0,count<=clock.times.count-start,abs(clock.times[start]-startSeconds)*clock.fps<0.05 else {
      throw StudioError.invalid("Movie source selection must preserve its exact CFR frames and FPS.")
    }
    let size=try grid(width:clock.width,height:clock.height,policy:settings.sizePolicy)
    guard size.width<=2048,size.height<=2048 else { throw StudioError.invalid("Movie upscaling needs a processed source at most2048 per axis.") }
    let fm=FileManager.default,parent=destination.deletingLastPathComponent()
    try fm.createDirectory(at:parent,withIntermediateDirectories:true)
    let staging=parent.appendingPathComponent(".movie-prepare-"+UUID().uuidString)
    try fm.createDirectory(at:staging,withIntermediateDirectories:false);defer { try? fm.removeItem(at:staging) }
    let raw=staging.appendingPathComponent("source.rgb24")
    let operation=size.resize ? "scale=\(size.width):\(size.height):flags=lanczos" : "crop=\(size.width):\(size.height):\(size.left):\(size.top)"
    try NativeMovieFrozenMedia.run(ffmpeg,["-v","error","-nostdin","-n","-i",original.path,"-map","0:v:0","-vf",
      "trim=start_frame=\(start):end_frame=\(start+count),setpts=PTS-STARTPTS,format=rgb24,"+operation,
      "-vsync","0","-frames:v",String(count),"-an","-f","rawvideo","-pix_fmt","rgb24",raw.path],log:staging.appendingPathComponent("source.log"))
    guard (try NativeMovieFrozenMedia.Identity(raw)).bytes==Int64(count)*Int64(size.width)*Int64(size.height)*3 else { throw StudioError.invalid("Prepared movie bytes differ from its visible frame/grid contract.") }
    let rawSHA=try NativeMovieFrozenMedia.digest(raw)
    var inputs:[[String:Any]]=[["kind":"video","path":original.path,"sha256":sourceSHA],["kind":"rgb24","path":destination.appendingPathComponent("source.rgb24").path,"sha256":rawSHA]]
    var references:[[String:Any]]=[],referenceSHA:[String:String]=[:],frozenInputs:[NativeMovieFrozenMedia.Identity]=[signature]
    var usedEndpoints=Set<String>()
    for endpoint in endpoints {
      let url=URL(fileURLWithPath:endpoint.path).resolvingSymlinksInPath().standardizedFileURL
      let identity=try NativeMovieFrozenMedia.Identity(url)
      guard identity.bytes<=64*1024*1024,usedEndpoints.insert(url.path).inserted,
        let image=CGImageSourceCreateWithURL(url as CFURL,[kCGImageSourceShouldCache:false] as CFDictionary),CGImageSourceGetCount(image)==1 else {
        throw StudioError.invalid("Movie endpoints need distinct bounded single-frame image files.")
      }
      frozenInputs.append(identity)
      let sha=try NativeMovieFrozenMedia.digest(url)
      references.append(["path":url.path,"role":endpoint.role,"strength":endpoint.strength,"crf":endpoint.crf])
      referenceSHA[url.path]=sha;inputs.append(["kind":"image","path":url.path,"sha256":sha])
    }
    var audio:Any=NSNull()
    if let sidecar {
      guard sidecar.path.hasPrefix("/"),sidecar.path.utf8.count<=4096,!sidecar.path.utf8.contains(0),sidecar.startSeconds.isFinite,sidecar.startSeconds>=0,
        sidecar.durationSeconds.map({ $0.isFinite && $0>0 }) ?? true else { throw StudioError.invalid("Movie audio sidecar interval is invalid.") }
      let url=URL(fileURLWithPath:sidecar.path).resolvingSymlinksInPath().standardizedFileURL,sha=try NativeMovieFrozenMedia.digest(URL(fileURLWithPath:sidecar.path).resolvingSymlinksInPath())
      frozenInputs.append(try NativeMovieFrozenMedia.Identity(url))
      audio=["path":url.path,"sha256":sha,"start_seconds":sidecar.startSeconds,"duration_seconds":sidecar.durationSeconds.map { $0 as Any } ?? NSNull()]
      inputs.append(["kind":"audio","path":url.path,"sha256":sha])
    }
    var body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale","source":["path":original.path,"sha256":sourceSHA,
      "rgb_path":destination.appendingPathComponent("source.rgb24").path,"rgb_sha256":rawSHA,"width":clock.width,"height":clock.height,"frames":count,"fps":clock.fps,"start_seconds":startSeconds,"duration_seconds":Double(count)/clock.fps],
      "components":components,"mode":settings.mode.rawValue,"size_policy":settings.sizePolicy.rawValue,"output_directory":outputDirectory.standardizedFileURL.resolvingSymlinksInPath().path,
      "prompt":prompt,"seed":seed,"refinement_strength":settings.refinementStrength,"anchors":settings.anchors.rawValue,"anchor_strength":settings.anchorStrength,"pixel_strength":settings.pixelStrength,
      "reference_images":references,"reference_image_sha256":referenceSHA,"audio_policy":settings.audioPolicy.rawValue,"audio_source":audio,"maximum_audio_drift_seconds":settings.maximumAudioDriftSeconds,
      "chunking":settings.chunking,"chunk_frame_megapixel_budget":settings.chunkFrameMegapixelBudget,"resume":settings.resume,"keep_chunks":settings.keepChunks]
    if let diffusion { body["diffusion_vae"]=diffusion }
    guard Set(inputs.map { $0["path"] as! String }).count==inputs.count else {
      throw StudioError.invalid("Movie publication cannot alias frozen source roles.")
    }
    inputs.sort { ($0["path"] as! String)<($1["path"] as! String) }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":prompt,"config":[:],"components":components,
      "conditioning":["task":"video_upscale","inputs":inputs],"movie_upscale":body]
    let bytes=try JSONSerialization.data(withJSONObject:recipe,options:[.sortedKeys,.withoutEscapingSlashes])
    let fingerprint="swift-json-v1:"+SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
    let report:[String:Any]=["task":"video_upscale","resolvedFingerprint":fingerprint,"pythonModelInference":false,
      "use_complete_duration":true,"usable_duration":Double(count)/clock.fps,"sourceFrames":count,"fps":clock.fps,
      "width":size.width*2,"height":size.height*2,"sourceFrameStart":start,"sourceMovieSHA256":sourceSHA,"sourceRGBSHA256":rawSHA,
      "nativeFPS":clock.fps,"preserveEditorialDuration":false,"nativePreparation":"swift","productionQualified":false,
      "generation":["supportedTasks":["video_upscale"],"pipelineMode":settings.mode.rawValue,
        "controls":["evaluations":settings.mode == .latentOnly ? 0:3,"stepsEditable":false,"refinementStepsEditable":false,"cfgEditable":false,"shiftEditable":false,
          "stepsExplanation":"Learned 2× optionally followed by three full-resolution source-video refinement evaluations.","cfgExplanation":"Uses released distilled refinement without CFG.","shiftExplanation":"Uses the released three-step refinement schedule."],"presets":[]]]
    try bytes.write(to:staging.appendingPathComponent("recipe.json"),options:.withoutOverwriting)
    try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys]).write(to:staging.appendingPathComponent("report.json"),options:.withoutOverwriting)
    if let editorRequest {
      try JSONSerialization.data(withJSONObject:editorRequest,options:[.sortedKeys,.withoutEscapingSlashes])
        .write(to:staging.appendingPathComponent("editor-request.json"),options:.withoutOverwriting)
    }
    guard try frozenInputs.allSatisfy({ $0 == (try NativeMovieFrozenMedia.Identity(URL(fileURLWithPath:$0.path))) }) else {
      throw StudioError.invalid("Movie source, sidecar or endpoint changed during preparation.")
    }
    try Task.checkCancellation();try fm.moveItem(at:staging,to:destination)
    return ["recipePath":destination.appendingPathComponent("recipe.json").path,"prompt":prompt,"report":report]
  }
}
