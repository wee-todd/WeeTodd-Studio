import Darwin
import AVFoundation
import Combine
import StudioCore

/// High-frequency transport updates are observed only by the viewport and playhead.
/// Publishing them on StudioStore would invalidate the entire editor each frame.
@MainActor final class TimelinePlaybackPosition: ObservableObject {
  @Published var seconds: Double = 0
}

struct TimelinePlaybackMedia {
  let composition: AVMutableComposition
  let videoComposition: AVMutableVideoComposition?
  let audioMix: AVMutableAudioMix
  let unavailableClips: [UUID: String]
  let warnings: [String]
  let hasMedia: Bool
}

/// Loads file metadata asynchronously and references source ranges in place. A single video
/// track decodes only the current shot, including when several shots share one generated movie.
struct TimelinePlaybackBuilder {
  static func time(_ seconds: Double) -> CMTime {
    guard seconds.isFinite, abs(seconds) < Double(Int64.max) / 60000 else { return .invalid }
    return CMTime(value: Int64((seconds * 60000).rounded()), timescale: 60000)
  }

  static func build(_ plan: TimelinePlaybackPlan, canonicalAudio: String? = nil) async throws -> TimelinePlaybackMedia {
    let composition = AVMutableComposition()
    let video = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)!
    let sourceAudio = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
    let sourceMix = AVMutableAudioMixInputParameters(track: sourceAudio)
    var instructions: [AVMutableVideoCompositionInstruction] = []
    var unavailable: [UUID: String] = [:]
    var warnings: [String] = []
    var hasMedia = false
    var hasVideo = false
    var assets: [String: AVURLAsset] = [:]
    let solo = plan.audioTracks.contains(where: \.solo)
    let activeAudio = (canonicalAudio == nil ? plan.audio : []).filter { region in
      let track = plan.audioTracks.first(where: { $0.id == region.trackID })
      return track?.muted != true && (!solo || track?.solo == true)
    }
    let replacements = activeAudio.filter { region in
      plan.audioTracks.first(where: { $0.id == region.trackID })?.replacesSource == true
    }
    func asset(_ path: String) -> AVURLAsset {
      if let value = assets[path] { return value }
      let value = AVURLAsset(url: URL(fileURLWithPath: path))
      assets[path] = value
      return value
    }
    func instruction(start: Double, duration: Double,
                     layer: AVMutableVideoCompositionLayerInstruction? = nil) {
      guard duration > 0 else { return }
      let startTime = time(start)
      // Quantize shared boundaries, not durations independently: separately rounded
      // starts and lengths can leave a one-tick hole that invalidates the whole compositor.
      let range = CMTimeRange(start: startTime, end: time(start + duration))
      guard range.duration > .zero else { return }
      let value = AVMutableVideoCompositionInstruction()
      value.timeRange = range
      value.backgroundColor = CGColor(gray: 0, alpha: 1)
      value.layerInstructions = layer.map { [$0] } ?? []
      instructions.append(value)
    }
    for span in plan.spans where span.duration > 0 {
      try Task.checkCancellation()
      if span.isStill || span.path.isEmpty {
        if span.isStill && !FileManager.default.isReadableFile(atPath: span.path) {
          unavailable[span.clipID] = "The source image is missing or unreadable. Relink it to preview this shot."
        }
        instruction(start: span.start, duration: span.duration)
        continue
      }
      do {
        guard span.sourceIn.isFinite, span.sourceIn >= 0 else {
          throw StudioError.invalid("Invalid source trim")
        }
        let media = asset(span.path)
        guard let track = try await media.loadTracks(withMediaType: .video).first else {
          throw StudioError.invalid("No playable video track")
        }
        let sourceRange = try await track.load(.timeRange)
        let available = CMTimeRangeGetEnd(sourceRange).seconds - span.sourceIn
        let length = min(span.duration, available)
        guard length.isFinite, length > 0 else {
          throw StudioError.invalid("The trim starts beyond the source movie")
        }
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        try Task.checkCancellation()
        let range = CMTimeRange(start: time(span.sourceIn), duration: time(length))
        try video.insertTimeRange(range, of: track, at: time(span.start))
        hasMedia = true; hasVideo = true
        let layer = AVMutableVideoCompositionLayerInstruction(assetTrack: video)
        let bounds = CGRect(origin: .zero, size: naturalSize).applying(transform)
        let xScale = Double(plan.width) / max(1, bounds.width)
        let yScale = Double(plan.height) / max(1, bounds.height)
        let scale = plan.fit == "fill" ? max(xScale, yScale) : min(xScale, yScale)
        let fitted = transform
          .concatenating(CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY))
          .concatenating(CGAffineTransform(scaleX: scale, y: scale))
          .concatenating(CGAffineTransform(translationX: (Double(plan.width) - bounds.width * scale) / 2,
                                         y: (Double(plan.height) - bounds.height * scale) / 2))
        layer.setTransform(fitted, at: time(span.start))
        instruction(start: span.start, duration: length, layer: layer)
        instruction(start: span.start + length, duration: span.duration - length)
        if length < span.duration - 1 / plan.fps {
          warnings.append("A source movie is shorter than its timeline range; its missing tail plays black.")
        }
        do {
          if canonicalAudio == nil, let audio = try await media.loadTracks(withMediaType: .audio).first {
            let audioRange = try await audio.load(.timeRange)
            let audioStart = max(span.sourceIn, audioRange.start.seconds)
            let audioEnd = min(span.sourceIn + length, CMTimeRangeGetEnd(audioRange).seconds)
            if audioEnd > audioStart {
              try sourceAudio.insertTimeRange(CMTimeRange(start: time(audioStart), duration: time(audioEnd - audioStart)),
                of: audio, at: time(span.start + audioStart - span.sourceIn))
            }
          }
        } catch is CancellationError { throw CancellationError() }
        catch { warnings.append("A shot's audio could not be played: \(error.localizedDescription)") }
      } catch is CancellationError { throw CancellationError() }
      catch {
        // A missing shot keeps its time, so later clips and titles cannot shift forward.
        unavailable[span.clipID] = error.localizedDescription
        if instructions.last.map({ CMTimeRangeGetEnd($0.timeRange).seconds <= span.start }) ?? true {
          instruction(start: span.start, duration: span.duration)
        }
      }
      let boundaries = Set([span.start, span.end] + replacements.flatMap {
        [max(span.start, min(span.end, $0.start)), max(span.start, min(span.end, $0.start + $0.duration))]
      }).sorted()
      for index in 0..<max(0, boundaries.count - 1) {
        let middle = (boundaries[index] + boundaries[index + 1]) / 2
        let muted = solo || replacements.contains { middle >= $0.start && middle < $0.start + $0.duration }
        sourceMix.setVolume(muted ? 0 : Float(max(0, min(2, span.volume))), at: time(boundaries[index]))
      }
    }
    var parameters = [sourceMix]
    if let canonicalAudio {
      let mixed = AVURLAsset(url: URL(fileURLWithPath: canonicalAudio))
      guard let audio = try await mixed.loadTracks(withMediaType: .audio).first else {
        throw StudioError.invalid("The prepared audio mix is unavailable.")
      }
      let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
      try track.insertTimeRange(CMTimeRange(start: .zero, duration: time(plan.duration)), of: audio, at: .zero)
      let parameter = AVMutableAudioMixInputParameters(track: track); parameter.setVolume(1, at: .zero)
      parameters.append(parameter); hasMedia = true
    }
    for region in activeAudio {
      try Task.checkCancellation()
      guard region.start.isFinite, region.sourceIn.isFinite, region.duration.isFinite,
        region.start >= 0, region.sourceIn >= 0, region.duration > 0, region.start < plan.duration else { continue }
      do {
        guard let audio = try await asset(region.path).loadTracks(withMediaType: .audio).first else { continue }
        let sourceRange = try await audio.load(.timeRange)
        let length = min(region.duration, plan.duration - region.start,
                         CMTimeRangeGetEnd(sourceRange).seconds - region.sourceIn)
        guard length > 0 else { continue }
        let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
        try track.insertTimeRange(CMTimeRange(start: time(region.sourceIn), duration: time(length)),
          of: audio, at: time(region.start))
        let mix = AVMutableAudioMixInputParameters(track: track)
        let volume = Float(max(0, min(2, region.volume)))
        let fade = min(max(0, region.fade), length / 2)
        mix.setVolume(volume, at: time(region.start))
        if fade > 0 {
          mix.setVolumeRamp(fromStartVolume: 0, toEndVolume: volume,
            timeRange: CMTimeRange(start: time(region.start), duration: time(fade)))
          mix.setVolumeRamp(fromStartVolume: volume, toEndVolume: 0,
            timeRange: CMTimeRange(start: time(region.start + length - fade), duration: time(fade)))
        }
        parameters.append(mix); hasMedia = true
      } catch is CancellationError { throw CancellationError() }
      catch { warnings.append("Could not play an audio region: \(error.localizedDescription)") }
    }
    for track in composition.tracks {
      let end = CMTimeRangeGetEnd(track.timeRange).seconds
      if end < plan.duration {
        track.insertEmptyTimeRange(CMTimeRange(start: time(end), duration: time(plan.duration - end)))
      }
    }
    let videoComposition: AVMutableVideoComposition?
    if hasVideo {
      let result = AVMutableVideoComposition()
      result.renderSize = CGSize(width: plan.width, height: plan.height)
      result.frameDuration = time(1 / plan.fps)
      result.instructions = instructions
      videoComposition = result
    } else { videoComposition = nil }
    let mix = AVMutableAudioMix(); mix.inputParameters = parameters
    try Task.checkCancellation()
    return TimelinePlaybackMedia(composition: composition, videoComposition: videoComposition,
      audioMix: mix, unavailableClips: unavailable, warnings: warnings, hasMedia: hasMedia)
  }
}

/// Keep observer ownership paired with its AVPlayer, including short-lived document/test stores.
final class TimelinePlayerObservation {
  private let player: AVPlayer
  private var timeObserver: Any?
  private var endObserver: NSObjectProtocol?
  init(player: AVPlayer, tick: @escaping (CMTime, AVPlayerItem?) -> Void,
       ended: @escaping (AVPlayerItem) -> Void) {
    self.player = player
    timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 1 / 30, preferredTimescale: 600), queue: .main) {
      [weak player] time in tick(time, player?.currentItem)
    }
    endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main) {
      notification in
      if let item = notification.object as? AVPlayerItem { ended(item) }
    }
  }
  deinit {
    if let timeObserver { player.removeTimeObserver(timeObserver) }
    if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
  }
}

final class TimelineFallbackClock {
  private var timer: Timer?
  init(start: Double, tick: @escaping (Double) -> Void) {
    let began = ProcessInfo.processInfo.systemUptime
    timer = Timer(timeInterval: 1 / 30, repeats: true) { _ in
      tick(start + ProcessInfo.processInfo.systemUptime - began)
    }
    RunLoop.main.add(timer!, forMode: .common)
  }
  deinit { timer?.invalidate() }
}

@MainActor extension StudioStore {
  var effectivePreviewDuration: Double { max(0, project.duration) }
  var previewClip: Clip? {
    guard let span = (timelinePlaybackPlan ?? TimelinePlaybackPlan(project: project)).span(at: playhead) else { return nil }
    return project.clips.first(where: { $0.id == span.clipID })
  }
  var selectedClipPlayhead: Double? {
    guard let clip = selectedClip,
      let index = project.clips.firstIndex(where: { $0.id == clip.id }) else { return nil }
    let local = playhead - project.start(of: index)
    return local >= 0 && local < clip.duration ? local : nil
  }

  func prepareTimelinePlayback(force: Bool = false) {
    let plan = TimelinePlaybackPlan(project: project)
    guard force || previewMode == "Movie" || timelinePlaybackPlan != plan else { return }
    timelineBuildTask?.cancel(); audioMixBridge.cancel()
    timelineBuildID = UUID()
    let snapshot = project.audioPlaybackProject
    let request = timelineBuildID
    let session = documentSessionID
    previewMode = "Timeline"
    timelinePlaybackPlan = plan
    timelinePlaybackIssues = [:]
    timelinePlaybackWarning = nil
    timelineItemStatus = nil
    playhead = min(max(0, playhead), plan.duration)
    preparingTimelinePlayback = plan.duration > 0
    guard plan.duration > 0 else {
      pausePlayback(); player.replaceCurrentItem(with: nil); timelineAudioLease = nil
      return
    }
    timelineBuildTask = Task { [weak self] in
      do {
        // The builder runs outside the main actor; file metadata loading never blocks gestures.
        try await Task.sleep(nanoseconds: 150_000_000)
        guard let self else { return }
        while self.audioMixBridge.busy {
          try Task.checkCancellation()
          try await Task.sleep(nanoseconds: 50_000_000)
        }
        let mixed = try await self.audioMixBridge.invoke("audio-mix", runtime: self.runtime,
          payload: ["project": try snapshot.object(), "purpose": "preview", "disposablePreview": true],
          output: self.dataDirectory.appendingPathComponent("AudioPreview"))
        try Task.checkCancellation()
        guard let path = mixed["path"] as? String else { throw StudioError.invalid("No audio mix was prepared.") }
        let lease = try AudioMixLease(path: path)
        let media = try await TimelinePlaybackBuilder.build(plan, canonicalAudio: path)
        guard !Task.isCancelled, self.timelineBuildID == request,
          self.documentSessionID == session, self.previewMode == "Timeline" else { return }
        self.preparingTimelinePlayback = false
        self.timelinePlaybackIssues = media.unavailableClips
        self.timelinePlaybackWarning = media.warnings.first
        if media.hasMedia {
          let item = AVPlayerItem(asset: media.composition)
          item.videoComposition = media.videoComposition
          item.audioMix = media.audioMix
          item.forwardPlaybackEndTime = TimelinePlaybackBuilder.time(plan.duration)
          self.player.replaceCurrentItem(with: item)
          self.timelineAudioLease = lease
          self.observeTimelineItem(item)
        }
        self.seek(self.playhead)
        if self.isPlaying { self.startPlaybackClock() }
      } catch is CancellationError { }
      catch {
        guard let self, self.timelineBuildID == request else { return }
        self.preparingTimelinePlayback = false
        self.pausePlayback()
        self.timelinePlaybackWarning = "Timeline playback could not load: \(error.localizedDescription)"
      }
    }
  }

  func observeTimelineItem(_ item: AVPlayerItem) {
    timelineItemStatus = item.observe(\.status, options: [.initial, .new]) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item, self.player.currentItem === item, item.status == .failed else { return }
        self.pausePlayback()
        self.timelinePlaybackWarning = item.error?.localizedDescription ?? "This movie could not be played."
      }
    }
  }

  func refreshPreview() { prepareTimelinePlayback(force: true) }
  func pausePlayback() {
    player.pause()
    timelineClock = nil
    isPlaying = false
  }
  func seekToEnd() { seek(effectivePreviewDuration) }
  func seek(_ seconds: Double) {
    guard seconds.isFinite else { return }
    playhead = min(max(0, seconds), effectivePreviewDuration)
    timelineClock = nil
    timelineSeekID = UUID()
    let request = timelineSeekID
    timelineSeekPending = false
    guard let item = player.currentItem else {
      if isPlaying && !preparingTimelinePlayback { startPlaybackClock() }
      return
    }
    // Coalesce drag seeks: stale completions and clock callbacks cannot pull the thumb backward.
    timelineSeekPending = true
    item.cancelPendingSeeks()
    player.seek(to: TimelinePlaybackBuilder.time(playhead), toleranceBefore: .zero, toleranceAfter: .zero) {
      [weak self, weak item] finished in
      Task { @MainActor in
        guard let self, self.timelineSeekID == request, self.player.currentItem === item else { return }
        self.timelineSeekPending = false
        if finished && self.isPlaying { self.player.play() }
      }
    }
  }
  func togglePlayback() {
    if isPlaying { pausePlayback(); return }
    guard effectivePreviewDuration > 0 else { return }
    if previewMode != "Movie" { prepareTimelinePlayback() }
    if playhead >= effectivePreviewDuration - 0.0001 { seek(0) }
    isPlaying = true
    if !preparingTimelinePlayback { startPlaybackClock() }
  }
  func startPlaybackClock() {
    guard isPlaying else { return }
    if player.currentItem != nil {
      if !timelineSeekPending { player.play() }
    } else {
      // Still-only and unfinished timelines retain their timing without producing proxy movies.
      let request = timelineSeekID
      timelineClock = TimelineFallbackClock(start: playhead) { [weak self] seconds in
        Task { @MainActor in
          guard let self, self.timelineSeekID == request, self.isPlaying, self.player.currentItem == nil else { return }
          self.updatePlaybackPosition(seconds)
        }
      }
    }
  }
  func updatePlaybackPosition(_ seconds: Double) {
    guard seconds.isFinite else { return }
    playhead = min(max(0, seconds), effectivePreviewDuration)
    if playhead >= effectivePreviewDuration { pausePlayback() }
  }
  func playbackTick(_ seconds: Double, item: AVPlayerItem?) {
    guard isPlaying, !timelineSeekPending, let item, player.currentItem === item else { return }
    updatePlaybackPosition(seconds)
  }
  func playbackEnded(item: AVPlayerItem) {
    guard player.currentItem === item, !timelineSeekPending, isPlaying else { return }
    updatePlaybackPosition(effectivePreviewDuration)
  }
  func scrubTimeline(to seconds: Double, clamped: Bool = false) {
    let plan = timelinePlaybackPlan ?? TimelinePlaybackPlan(project: project)
    guard let target = plan.seekPosition(seconds, clamped: clamped) else { return }
    if scrubWasPlaying == nil {
      scrubWasPlaying = isPlaying
      pausePlayback()
    }
    seek(target)
  }
  func endTimelineScrub() {
    guard let resume = scrubWasPlaying else { return }
    scrubWasPlaying = nil
    if resume && playhead < effectivePreviewDuration {
      isPlaying = true
      if !preparingTimelinePlayback { startPlaybackClock() }
    }
  }
  func invalidateTimelinePlayback() {
    pausePlayback()
    timelineBuildTask?.cancel(); audioMixBridge.cancel()
    timelineBuildID = UUID()
    timelineSeekID = UUID()
    timelineSeekPending = false
    timelineItemStatus = nil
    scrubWasPlaying = nil
    preparingTimelinePlayback = false
  }
}

/// A shared filesystem lease keeps the currently playing PCM out of cache eviction.
final class AudioMixLease {
  let descriptor: Int32
  init(path: String) throws {
    let lock = URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("lease.lock")
    descriptor = open(lock.path, O_CREAT | O_RDWR, 0o600)
    guard descriptor >= 0 else { throw StudioError.invalid("Cannot retain prepared audio.") }
    guard flock(descriptor, LOCK_SH | LOCK_NB) == 0 else {
      close(descriptor); throw StudioError.invalid("Prepared audio is being refreshed; retry playback.")
    }
  }
  deinit { flock(descriptor, LOCK_UN); close(descriptor) }
}
