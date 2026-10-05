import CryptoKit
import Foundation

/// Editor adapter for the dedicated shared source-movie renderer. No ordinary
/// generated-video recipe, temporal tail extractor or Python inference is used.
public enum NativeLTXMovieEditorPreparation {
  private struct Context {
    let project:StudioProject,clip:Clip,settings:LTX25MovieUpscaleSettings
    let source:MediaAsset,sourceAttachment:Attachment
    let endpoints:[NativeLTXMoviePreparation.Endpoint],sidecar:NativeLTXMoviePreparation.Sidecar?
    let components:[String:String],profileID:String,selectionFingerprint:String,ffmpeg:URL
  }
  public static func matches(request:[String:Any]) throws -> Bool {
    guard let raw=request["project"],let id=request["clipID"] as? String else { return false }
    let project=try JSONDecoder().decode(StudioProject.self,from:JSONSerialization.data(withJSONObject:raw))
    guard let clip=project.clips.first(where:{ $0.id.uuidString.caseInsensitiveCompare(id) == .orderedSame }) else { return false }
    return clip.generationSelection?.ltx25MovieUpscale != nil || clip.generationSelection?.task == "video_upscale"
  }
  private static func context(_ request:[String:Any]) throws -> Context {
    guard let raw=request["project"],let id=request["clipID"] as? String else { throw StudioError.invalid("Missing movie editor request.") }
    let project=try JSONDecoder().decode(StudioProject.self,from:JSONSerialization.data(withJSONObject:raw))
    guard let clip=project.clips.first(where:{ $0.id.uuidString.caseInsensitiveCompare(id) == .orderedSame }),
      let settings=clip.generationSelection?.ltx25MovieUpscale,clip.generationSelection?.task == "video_upscale" else {
      throw StudioError.invalid("Select source movie upscaling with explicit experimental movie settings.")
    }
    try NativeLTXMoviePreparation.validateEditor(clip:clip,settings:settings,isContinuousScene:project.isContinuousSceneMember(clip))
    guard clip.seed>=0 else { throw StudioError.invalid("Movie seed must be nonnegative.") }
    let global=try JSONDecoder().decode([MediaAsset].self,from:JSONSerialization.data(withJSONObject:request["globalAssets"] ?? []))
    let assets=project.assets+global
    var sources:[(MediaAsset,Attachment)]=[],endpoints:[NativeLTXMoviePreparation.Endpoint]=[]
    var sidecars:[NativeLTXMoviePreparation.Sidecar]=[],used=Set<String>()
    for attachment in clip.attachments {
      let candidates=assets.filter { $0.id == attachment.assetID }
      guard candidates.count==1,let asset=candidates.first,asset.path.hasPrefix("/"),
        asset.path.utf8.count<=4096,!asset.path.utf8.contains(0),
        used.insert(URL(fileURLWithPath:asset.path).standardizedFileURL.resolvingSymlinksInPath().path).inserted,
        attachment.isEnabled,attachment.time==0,attachment.attentionStrength==nil,
        attachment.referenceRole==nil,attachment.referencePriority==nil,attachment.referenceFrames==nil,
        attachment.referenceSizePolicy==nil,attachment.msrAudioReferenceID==nil else {
        throw StudioError.invalid("Movie attachments require distinct local media and cannot ignore reference timing, attention or disabled controls.")
      }
      switch attachment.role {
      case .reference:
        guard asset.kind == .video,attachment.strength==1,
          attachment.audioSourceStart==nil,attachment.audioSourceDuration==nil else { throw StudioError.invalid("Choose exactly one movie as the unscaled source reference.") }
        sources.append((asset,attachment))
      case .first,.last:
        guard asset.kind == .image,settings.mode != .latentOnly,settings.anchors != .none,
          attachment.role != .last || settings.anchors == .firstLast,
          attachment.strength==settings.anchorStrength,
          attachment.sourceStartSeconds==nil,attachment.sourceDurationSeconds==nil,
          attachment.audioSourceStart==nil,attachment.audioSourceDuration==nil else {
          throw StudioError.invalid("Movie endpoints must use the selected common anchor strength and enabled anchor roles.")
        }
        endpoints.append(.init(path:asset.path,role:attachment.role == .first ? "first":"last",strength:attachment.strength))
      case .audioDriver:
        guard asset.kind == .audio,settings.audioPolicy == .sidecar,attachment.strength==1,
          attachment.sourceStartSeconds==nil,attachment.sourceDurationSeconds==nil else { throw StudioError.invalid("An audio sidecar must be an explicit source-audio attachment.") }
        sidecars.append(.init(path:asset.path,startSeconds:attachment.audioSourceStart ?? 0,durationSeconds:attachment.audioSourceDuration))
      default: throw StudioError.invalid("This attachment cannot execute in source movie upscaling.")
      }
    }
    guard sources.count==1,sidecars.count == (settings.audioPolicy == .sidecar ? 1:0),
      Set(endpoints.map(\.role)).count==endpoints.count else { throw StudioError.invalid("Select exactly one movie, at most one endpoint per role, and the audio policy's exact sidecar.") }
    let source=sources[0]
    guard source.1.sourceStartSeconds.map({ $0.isFinite && $0>=0 }) ?? true,
      source.1.sourceDurationSeconds.map({ $0.isFinite && $0>0 }) ?? true else { throw StudioError.invalid("Movie source interval must be finite and positive.") }
    // Actual attachment admission precedes the component-only profile probe.
    let binding=try NativeLTXPreparation.componentsForMovie(request:request,settings:settings)
    return Context(project:project,clip:clip,settings:settings,source:source.0,sourceAttachment:source.1,
      endpoints:endpoints,sidecar:sidecars.first,components:binding.components,profileID:binding.profileID,
      selectionFingerprint:binding.selectionFingerprint,ffmpeg:binding.ffmpeg)
  }
  private static func descriptor(_ settings:LTX25MovieUpscaleSettings) -> [String:Any] {
    ["supportedTasks":["video_upscale"],"pipelineMode":settings.mode.rawValue,
      "controls":["evaluations":settings.mode == .latentOnly ? 0:3,"stepsEditable":false,"refinementStepsEditable":false,
        "cfgEditable":false,"shiftEditable":false,"stepsExplanation":"Source-video learned 2× with optional three-step refinement.",
        "cfgExplanation":"Distilled refinement does not use CFG.","shiftExplanation":"Uses the released refinement schedule."],"presets":[]]
  }
  public static func describe(request:[String:Any]) throws -> [String:Any] {
    let value=try context(request)
    var paths=[value.profileID,value.source.path]+Array(value.components.values)+value.endpoints.map(\.path)
    if let sidecar=value.sidecar { paths.append(sidecar.path) }
    for component in value.components.values {
      for name in ["paged_manifest.json","model_identity.json","conversion_provenance.json"] {
        let path=URL(fileURLWithPath:component).appendingPathComponent(name).path
        if FileManager.default.fileExists(atPath:path) { paths.append(path) }
      }
    }
    let bytes=try JSONSerialization.data(withJSONObject:request,options:[.sortedKeys,.withoutEscapingSlashes])
    let fingerprint="swift-json-v1:"+SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
    return ["profileID":value.profileID,"generation":descriptor(value.settings),"fingerprint":fingerprint,
      "selectionFingerprint":value.selectionFingerprint,"sourcePaths":Array(Set(paths)).sorted(),
      "warnings":["Native movie upscaling is experimental until real-model qualification. It preserves the selected source FPS and original audio interval."],"readinessErrors":[]]
  }
  public static func prepare(request:[String:Any],destination:URL) async throws -> [String:Any] {
    let value=try context(request)
    let source=URL(fileURLWithPath:value.source.path).standardizedFileURL.resolvingSymlinksInPath()
    let clock=try await NativeMovieFrozenMedia.videoClock(source)
    let start=value.sourceAttachment.sourceStartSeconds ?? 0
    var frames:Int?
    if let duration=value.sourceAttachment.sourceDurationSeconds {
      let exact=duration*clock.fps,rounded=exact.rounded(.toNearestOrEven)
      guard exact.isFinite,rounded>0,rounded<=Double(clock.times.count),abs(exact-rounded)<0.05 else {
        throw StudioError.invalid("Movie source duration must select an exact visible CFR interval, without an 8n+1 conversion.")
      }
      frames=Int(rounded)
    }
    let output=destination.appendingPathComponent("native-render")
    return try await NativeLTXMoviePreparation.prepare(source:source,startSeconds:start,visibleFrames:frames,
      settings:value.settings,components:value.components,prompt:value.clip.prompt.trimmingCharacters(in:.whitespacesAndNewlines),
      seed:UInt64(value.clip.seed),endpoints:value.endpoints,sidecar:value.sidecar,ffmpeg:value.ffmpeg,
      destination:destination,outputDirectory:output,editorRequest:request,diffusionVAE:value.clip.generationSelection?.ltx25DiffusionVAE)
  }
}
