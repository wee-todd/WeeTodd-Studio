import Foundation

public enum AudioTrackRole: String, Codable, CaseIterable { case voice, music, effects, other }
public enum AudioDriverMode: String, Codable, CaseIterable { case voice, music, voiceAndMusic }
public struct ClipAudioAnchor: Codable, Equatable {
  public var clipID: UUID
  public var offsetSeconds: Double
  public init(clipID: UUID, offsetSeconds: Double) { self.clipID = clipID; self.offsetSeconds = offsetSeconds }
}
public struct AudioDriverSelection: Codable, Equatable {
  public var mode: AudioDriverMode
  public var voiceTrackIDs: [UUID]
  public var musicTrackIDs: [UUID]
  public init(mode: AudioDriverMode = .voiceAndMusic, voiceTrackIDs: [UUID] = [], musicTrackIDs: [UUID] = []) {
    self.mode = mode; self.voiceTrackIDs = voiceTrackIDs; self.musicTrackIDs = musicTrackIDs
  }
}
public struct AudioDucking: Codable, Equatable {
  public var amountDb: Double = 12
  public var thresholdDb: Double = -36
  public var attack: Double = 0.02
  public var release: Double = 0.25
  public init() {}
}
public func resolvedAudioStart(_ region: AudioRegion, in project: StudioProject) throws -> Double {
  guard let anchor = region.anchor else { return region.start }
  guard let index = project.clips.firstIndex(where: { $0.id == anchor.clipID }) else {
    throw StudioError.invalid("The audio region's clip is missing. Relink or remove this region.")
  }
  return project.start(of: index) + anchor.offsetSeconds
}
extension StudioProject {
  public var resolvedAudio: [AudioRegion] {
    audio.map { region in
      var resolved = region
      resolved.start = (try? resolvedAudioStart(region, in: self)) ?? region.start
      return resolved
    }
  }
  public func validateAudio() throws {
    guard Set(audio.map(\.id)).count == audio.count,
      Set(audioTracks.map(\.id)).count == audioTracks.count else {
      throw StudioError.invalid("Audio region and track IDs must be unique.")
    }
    for track in audioTracks {
      guard track.gainDb.isFinite, (-60...12).contains(track.gainDb),
        track.pan.isFinite, (-1...1).contains(track.pan) else {
        throw StudioError.invalid("Audio gain must be −60 to +12 dB and pan must be −1 to +1.")
      }
      try track.reverb?.validate()
      if let d = track.ducking {
        guard [d.amountDb, d.thresholdDb, d.attack, d.release].allSatisfy(\.isFinite),
          (0...36).contains(d.amountDb), (-80...0).contains(d.thresholdDb),
          (0.001...5).contains(d.attack), (0.001...10).contains(d.release) else {
          throw StudioError.invalid("Check music ducking amount, threshold, attack and release.")
        }
      }
    }
    for region in audio {
      let start = try resolvedAudioStart(region, in: self)
      guard [start, region.sourceIn, region.duration, region.volume, region.effectiveFadeIn,
        region.effectiveFadeOut].allSatisfy(\.isFinite), start >= 0, region.sourceIn >= 0,
        region.duration > 0, (0...4).contains(region.volume),
        region.effectiveFadeIn >= 0, region.effectiveFadeOut >= 0 else {
        throw StudioError.invalid("Check audio region timing, volume and fades.")
      }
      if let window = region.envelope {
        guard [window.offset, window.duration, window.fadeIn, window.fadeOut].allSatisfy(\.isFinite),
          window.offset >= 0, window.duration > 0, window.fadeIn >= 0, window.fadeOut >= 0,
          ["linear", "equalPower"].contains(window.curve) else {
          throw StudioError.invalid("Check the split audio envelope.")
        }
      }
      // Legacy unassigned/orphan tracks remain audible; new placement validates destination IDs.
      if let anchor = region.anchor, !anchor.offsetSeconds.isFinite || anchor.offsetSeconds < 0 {
        throw StudioError.invalid("Audio clip offset must be finite and nonnegative.")
      }
    }
    for clip in clips {
      guard (clip.sourcePan ?? 0).isFinite, (-1...1).contains(clip.sourcePan ?? 0) else {
        throw StudioError.invalid("Source audio pan must be −1 to +1.")
      }
    }
  }
  public mutating func splitAnchoredAudio(clipID: UUID, secondID: UUID, at seconds: Double) {
    var additions: [AudioRegion] = []
    for index in audio.indices where audio[index].anchor?.clipID == clipID {
      let offset = audio[index].anchor!.offsetSeconds
      if offset >= seconds {
        audio[index].anchor = ClipAudioAnchor(clipID: secondID, offsetSeconds: offset - seconds)
      } else if offset + audio[index].duration > seconds {
        let firstLength = seconds - offset
        let region = audio[index]
        let legacy = audioMixPolicy == nil || audioMixPolicy == "legacy-v1"
        let fadeIn = min(region.effectiveFadeIn, legacy && region.fadeIn == nil ? region.duration / 2 : region.duration)
        let fadeOut = min(region.effectiveFadeOut, legacy && region.fadeOut == nil ? region.duration / 2 : region.duration)
        let window = region.envelope ?? AudioEnvelopeSlice(duration: region.duration,
          fadeIn: fadeIn, fadeOut: fadeOut, curve: region.fadeCurve ?? "linear")
        audio[index].envelope = window
        var tail = audio[index]; tail.id = UUID()
        tail.envelope?.offset += firstLength
        tail.anchor = ClipAudioAnchor(clipID: secondID, offsetSeconds: 0)
        tail.sourceIn += firstLength; tail.duration -= firstLength; tail.fadeIn = 0
        audio[index].duration = firstLength; audio[index].fadeOut = 0
        additions.append(tail)
      }
    }
    audio.append(contentsOf: additions)
  }
}
extension StudioProject {
  public func audioDriverRevision(for clip: Clip) -> String {
    guard let selection = clip.audioDriverSelection,
      let index = clips.firstIndex(where: { $0.id == clip.id }) else { return "" }
    let end = start(of: index) + clip.duration + 1
    let tracks = audioTracks.filter { track in
      let roleSelected = selection.mode == .voiceAndMusic || selection.mode.rawValue == track.role.rawValue
      let ids = track.role == .voice ? selection.voiceTrackIDs : selection.musicTrackIDs
      return [.voice, .music].contains(track.role) && roleSelected && (ids.isEmpty || ids.contains(track.id))
    }
    let ids = Set(tracks.map(\.id))
    let regions = resolvedAudio.filter { $0.start < end && $0.trackID.map(ids.contains) == true }
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    var normalized = tracks
    for i in normalized.indices { normalized[i].solo = false; normalized[i].name = "" }
    return "\(start(of: index))|\(clip.duration)|" + ((try? encoder.encode(selection).base64EncodedString()) ?? "")
      + ((try? encoder.encode(normalized).base64EncodedString()) ?? "")
      + ((try? encoder.encode(regions).base64EncodedString()) ?? "")
  }
}

/// An immutable envelope window lets split regions reproduce their unsplit gain curve.
public struct AudioEnvelopeSlice: Codable, Equatable {
  public var offset: Double = 0
  public var duration: Double
  public var fadeIn: Double
  public var fadeOut: Double
  public var curve: String
}
