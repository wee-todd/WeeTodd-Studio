import Foundation

extension StudioProject {
  /// Editing needs space for the full song and titles while video coverage is still being built.
  /// Export and playback duration continue to follow the actual video timeline.
  public var timelineContentDuration: Double {
    let ends = [duration] + audio.map { $0.start + $0.duration } + titles.map { $0.start + $0.duration }
    return max(0, ends.filter { $0.isFinite }.max() ?? 0)
  }
}

/// Lightweight editing playback uses cuts at incoming clip starts. Rendered movie preview
/// remains authoritative for transitions and finishing. No media is copied into this plan.
public struct TimelinePlaybackSpan: Equatable {
  public let clipID: UUID
  public let start: Double
  public let duration: Double
  public let path: String
  public let sourceIn: Double
  public let volume: Double
  public var end: Double { start + duration }
  public var isStill: Bool {
    ["png", "jpg", "jpeg", "webp", "tif", "tiff", "heic"].contains(
      URL(fileURLWithPath: path).pathExtension.lowercased())
  }
}

public struct TimelinePlaybackPlan: Equatable {
  public let spans: [TimelinePlaybackSpan]
  public let duration: Double
  public let fps: Double
  public let width: Int
  public let height: Int
  public let fit: String
  public let audio: [AudioRegion]
  public let audioTracks: [AudioTrack]

  public init(project: StudioProject) {
    var starts: [Double] = []
    var cursor = 0.0
    for (index, clip) in project.clips.enumerated() {
      let length = clip.duration.isFinite ? max(0, clip.duration) : 0
      let overlap = project.overlap(before: index)
      let start = max(0, cursor - (overlap.isFinite ? overlap : 0))
      starts.append(start)
      cursor = start + length
    }
    duration = cursor
    spans = project.clips.enumerated().map { index, clip in
      let end = index + 1 < starts.count ? starts[index + 1] : cursor
      return TimelinePlaybackSpan(clipID: clip.id, start: starts[index],
        duration: max(0, end - starts[index]), path: clip.playbackPath,
        sourceIn: clip.playbackIn, volume: clip.volume)
    }
    fps = project.settings.fps.isFinite ? max(1, project.settings.fps) : 24
    // Keep live preview bounded by display needs; export retains the chosen dimensions.
    let scale = min(1, 1280 / Double(max(64, project.settings.width, project.settings.height)))
    width = max(2, Int(Double(project.settings.width) * scale) / 2 * 2)
    height = max(2, Int(Double(project.settings.height) * scale) / 2 * 2)
    fit = project.settings.fit
    audio = project.audio
    audioTracks = project.audioTracks
  }

  public func span(at seconds: Double) -> TimelinePlaybackSpan? {
    guard seconds.isFinite, seconds >= 0, seconds <= duration else { return nil }
    if seconds == duration { return spans.last(where: { $0.duration > 0 }) }
    return spans.last(where: { seconds >= $0.start && seconds < $0.end })
  }

  /// Ruler clicks reject the blank area. Handle drags stay at the nearest timeline edge.
  public func seekPosition(_ seconds: Double, clamped: Bool) -> Double? {
    guard seconds.isFinite, duration > 0 else { return nil }
    let target = clamped ? min(duration, max(0, seconds)) : seconds
    return span(at: target) == nil ? nil : target
  }
}
