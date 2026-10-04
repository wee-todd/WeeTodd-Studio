import Foundation

public enum NativeH3MotionFidelityPreparation {
  public static func apply(_ settings:H3MotionFidelitySettings?,to ordinary:[String:Any],clip:Clip) throws -> [String:Any] {
    guard let settings else { return ordinary }
    try settings.validate(duration:clip.duration,seed:clip.seed,width:clip.generationWidth,height:clip.generationHeight)
    guard clip.continuityMode=="independent",clip.extensionDirection.isEmpty,clip.audioDriverSelection==nil,
      clip.musicSource==nil,clip.generationSelection?.h3Reference==nil,clip.generationSelection?.h3Joint==nil,
      ["t2v","t2va"].contains(clip.inferredTask),clip.attachments.allSatisfy({$0.role == .lora}),
      ordinary["continuation"]==nil,ordinary["joint_latents"]==nil,ordinary["refinement"]==nil,
      let config=ordinary["config"] as? [String:Any],let points=config["steps"] as? Int,points>=16,
      (config["projection_backend"] as? String ?? "mlx")=="mlx",(config["transformer_backend"] as? String ?? "mlx")=="mlx",
      let components=ordinary["components"] as? [String:Any],components["task"] as? String=="t2va",components["fun_controlnet"]==nil,
      let conditioning=ordinary["conditioning"] as? [String:Any],conditioning["task"] as? String=="t2v",
      (conditioning["inputs"] as? [Any])?.isEmpty==true else {
      throw StudioError.invalid("H3 Motion Fidelity needs independent plain T2VA with at least 15 evaluations, explicit MLX and immediate standard LoRAs.")
    }
    if let descriptors=(ordinary["loras"] as? [String:Any])?["adapters"] as? [[String:Any]] {
      for item in descriptors {
        guard (item["start_after_evaluations"] as? Int ?? 0)==0,item["profile"] as? String != "turbo",let path=item["path"] as? String else { throw StudioError.invalid("H3 motion repair cannot use deferred or Turbo adapters.") }
        _=try NativeLoRAInspection.validateH3Sampling(path:path,selectedProfile:"standard",schedulePoints:points)
      }
    }
    for pair in components["loras"] as? [[Any]] ?? [] {
      guard let path=pair.first as? String else { throw StudioError.invalid("Invalid H3 motion adapter pair.") }
      _=try NativeLoRAInspection.validateH3Sampling(path:path,selectedProfile:"standard",schedulePoints:points)
    }
    let source=URL(fileURLWithPath:try H3JointLatentArtifact.localPath(settings.sourceVideo)).standardizedFileURL.resolvingSymlinksInPath()
    guard let ffmpeg=ordinary["ffmpeg"] as? String else { throw StudioError.invalid("H3 Motion Fidelity requires native FFmpeg and FFprobe.") }
    let probe=URL(fileURLWithPath:ffmpeg).deletingLastPathComponent().appendingPathComponent("ffprobe")
    guard FileManager.default.isExecutableFile(atPath:probe.path) else { throw StudioError.invalid("Select an FFmpeg installation containing executable FFprobe for H3 motion inspection.") }
    let digest=try NativeMovieFrozenMedia.digest(source)
    var fields:[String:Any]=["version":1,"source_video":source.path,"source_sha256":digest,"source_in":settings.sourceIn,
      "duration_seconds":clip.duration,"ffprobe":probe.path,"mode":settings.mode.rawValue,"strength":settings.strength,
      "max_hold":settings.maxHold,"sensitivity":settings.sensitivity,"seed":clip.seed,"max_frames":settings.maxFrames]
    if let evaluations=settings.evaluations { fields["evaluations"]=evaluations }
    var recipe=ordinary;recipe["motion_fidelity"]=fields;return recipe
  }
  public static func inspect(request:[String:Any]) async throws {
    guard let raw=request["project"] as? [String:Any],let clipID=request["clipID"] as? String,
      let clips=raw["clips"] as? [[String:Any]],let rawClip=clips.first(where:{($0["id"] as? String)?.caseInsensitiveCompare(clipID) == .orderedSame}),
      (rawClip["generationSelection"] as? [String:Any])?["h3MotionFidelity"] != nil else { return }
    let project=try JSONDecoder().decode(StudioProject.self,from:JSONSerialization.data(withJSONObject:raw))
    guard let clip=project.clips.first(where:{$0.id.uuidString.caseInsensitiveCompare(clipID) == .orderedSame}),
      clip.engine == .h3,let settings=clip.generationSelection?.h3MotionFidelity else { throw StudioError.invalid("Select an H3 clip for Motion Fidelity.") }
    try await inspect(settings,clip:clip)
  }
  /// Compressed sample timing only: no pixels, model tensors or audio conversion.
  public static func inspect(_ settings:H3MotionFidelitySettings,clip:Clip) async throws {
    try settings.validate(duration:clip.duration,seed:clip.seed,width:clip.generationWidth,height:clip.generationHeight)
    let source=URL(fileURLWithPath:try H3JointLatentArtifact.localPath(settings.sourceVideo)).standardizedFileURL.resolvingSymlinksInPath()
    let clock=try await NativeMovieFrozenMedia.videoClock(source)
    let visible=clock.times.filter {$0>=settings.sourceIn-0.0001 && $0<settings.sourceIn+clip.duration-0.0001}
    let frames=Int((clip.duration*24).rounded(.toNearestOrEven))
    guard abs(clock.fps-24)<0.0001,clock.width==clip.generationWidth,clock.height==clip.generationHeight,
      visible.count==frames,let first=visible.first,abs(first-settings.sourceIn)<=0.0001,
      zip(visible,visible.dropFirst()).allSatisfy({abs($0.1-$0.0-1/24)<=0.0001}) else {
      throw StudioError.invalid("H3 Motion Fidelity source must cover the exact edited 24 fps interval at the original generation canvas.")
    }
  }
}
