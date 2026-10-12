import Foundation

extension StudioProject {
  /// Video scheduling remains independent of free audio regions.
  public var playbackDuration: Double { timelineContentDuration }

  @discardableResult public mutating func placeTimelineAudio(_ asset: MediaAsset, role: AudioTrackRole,
                                                            at start: Double) throws -> AudioRegion {
    guard [.voice, .music].contains(role), asset.kind == .audio, !asset.path.isEmpty,
      start.isFinite, start >= 0, asset.duration.isFinite, asset.duration > 0,
      start + asset.duration <= 3600 else {
      throw StudioError.invalid("Choose valid audio and a timeline position within one hour.")
    }
    let track: AudioTrack
    if let found = audioTracks.first(where: { $0.role == role && !$0.replacesSource }) { track = found }
    else { track = AudioTrack(name: role == .voice ? "Voice" : "Music", role: role); audioTracks.append(track) }
    if !assets.contains(where: { $0.id == asset.id }) { assets.append(asset) }
    var region = AudioRegion(assetID: asset.id, path: asset.path)
    region.start = start; region.duration = asset.duration; region.trackID = track.id
    region.volume = 1; region.fade = 0.01
    audio.append(region)
    return region
  }
}
