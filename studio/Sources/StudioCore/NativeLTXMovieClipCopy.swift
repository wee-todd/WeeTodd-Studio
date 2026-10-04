import Foundation

/// Creates a separate source-movie edit without replacing the accepted take.
public enum NativeLTXMovieClipCopy {
  public static func create(source:Clip,media:MediaAsset)throws->(clip:Clip,asset:MediaAsset) {
    guard media.kind == .video,media.path==source.sourcePath,media.path.hasPrefix("/"),
      source.sourceIn.isFinite,source.sourceIn>=0,source.duration.isFinite,source.duration>0,
      media.duration.isFinite,media.duration>0,media.fps.isFinite,(1...60).contains(media.fps),
      source.sourceIn+source.duration<=media.duration+0.5/media.fps else {
      throw StudioError.invalid("Select a valid rendered or imported movie interval before making an LTX upscale copy.")
    }
    var copy=Clip(name:source.name+" · LTX 2×",engine:.ltx25)
    copy.duration=source.duration;copy.seed=source.seed
    copy.prompt=source.prompt.isEmpty ? "Preserve the source subject, clothing, motion, camera, lighting and scene while refining detail." : source.prompt
    copy.generationSelection=GenerationSelection(task:"video_upscale",preset:.custom)
    copy.generationSelection?.ltx25MovieUpscale=LTX25MovieUpscaleSettings()
    var asset=media;asset.id=UUID();asset.scope = .clip;asset.owner=copy.id
    var reference=Attachment(assetID:asset.id,role:.reference)
    reference.sourceStartSeconds=source.sourceIn;reference.sourceDurationSeconds=source.duration
    copy.attachments=[reference]
    return (copy,asset)
  }
}
