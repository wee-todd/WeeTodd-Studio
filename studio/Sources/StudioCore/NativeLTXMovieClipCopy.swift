import Foundation

/// Creates a separate source-movie edit without replacing the accepted take.
public enum NativeLTXMovieClipCopy {
  public static func create(source:Clip,media:MediaAsset,
    method: MovieUpscaleMethod = .ltx25)throws->(clip:Clip,asset:MediaAsset) {
    guard media.kind == .video,media.path==source.sourcePath,media.path.hasPrefix("/"),
      source.sourceIn.isFinite,source.sourceIn>=0,source.duration.isFinite,source.duration>0,
      media.duration.isFinite,media.duration>0,media.fps.isFinite,media.fps > 0,
      method == .relayVSR ? source.sourceIn < media.duration : (1...60).contains(media.fps),
      source.sourceIn+source.duration<=media.duration+0.5/media.fps else {
      throw StudioError.invalid("Select a valid rendered or imported movie interval before making an upscale copy.")
    }
    var copy=Clip(name:source.name+" · "+method.label,engine:.ltx25)
    copy.duration = method == .relayVSR ? min(source.duration, media.duration - source.sourceIn) : source.duration
    copy.seed = source.seed
    copy.generationSelection=GenerationSelection(task:"video_upscale",preset:.custom)
    if method == .relayVSR {
      copy.generationSelection?.movieUpscaleMethod = method
    } else {
      copy.prompt=source.prompt.isEmpty ? "Preserve the source subject, clothing, motion, camera, lighting and scene while refining detail." : source.prompt
      copy.generationSelection?.ltx25MovieUpscale=LTX25MovieUpscaleSettings()
    }
    var asset=media;asset.id=UUID();asset.scope = .clip;asset.owner=copy.id
    var reference=Attachment(assetID:asset.id,role:.reference)
    reference.sourceStartSeconds=source.sourceIn;reference.sourceDurationSeconds=copy.duration
    copy.attachments=[reference]
    return (copy,asset)
  }
}
