import Foundation

extension StudioProject {
  /// Realign edited shot lengths to their original continuous song interval.
  @discardableResult
  public mutating func realignPlanningMusicIntervals() throws -> Int {
    guard var plan = planning, !plan.shots.isEmpty, (1...120).contains(plan.frameRate),
      abs(settings.fps - Double(plan.frameRate)) < 0.000001,
      let first = plan.shots.first?.musicSource else {
      throw StudioError.invalid("Use a music shot plan with the movie's frame rate.")
    }
    let fps = Double(plan.frameRate)
    var sourceEnd = first.start
    var totalFrames: Int64 = 0
    for shot in plan.shots {
      guard shot.combinedShots == nil, (1...100_000_000).contains(shot.frameCount),
        let source = shot.musicSource, !source.path.isEmpty,
        source.path == first.path, source.sha256 == first.sha256,
        source.start.isFinite, source.start >= 0, source.duration.isFinite, source.duration > 0,
        abs(source.start - sourceEnd) < 0.000001 else {
        throw StudioError.invalid("Realignment requires one continuous song interval and uncombined shots. Restore combined shots or replan separate songs first.")
      }
      sourceEnd = source.start + source.duration
      totalFrames += Int64(shot.frameCount)
    }
    let tail = Double(totalFrames) / fps - (sourceEnd - first.start)
    guard sourceEnd.isFinite, totalFrames <= 100_000_000, tail >= -0.000001, tail < 1 / fps else {
      throw StudioError.invalid("Keep the total shot length equal to the original song interval, allowing only the final partial video frame. Move frames between shots before realigning.")
    }
    var updated = self
    var cursor: Int64 = 0
    var changed = 0
    for i in plan.shots.indices {
      let original = plan.shots[i].musicSource!
      let start = first.start + Double(cursor) / fps
      let length = Double(plan.shots[i].frameCount) / fps
      let duration = i == plan.shots.count - 1 ? sourceEnd - start : length
      guard duration > 0 else { throw StudioError.invalid("The final shot must retain audible song content.") }
      defer { cursor += Int64(plan.shots[i].frameCount) }
      guard abs(original.start - start) >= 0.000001 || abs(original.duration - duration) >= 0.000001 else { continue }
      var source = original; source.start = start; source.duration = duration
      if let id = plan.shots[i].linkedClipID {
        guard let c = updated.clips.firstIndex(where: { $0.id == id }),
          updated.clips[c].musicSource == original,
          abs(updated.start(of: c) - Double(cursor) / fps) < 0.000001,
          abs(updated.clips[c].duration - length) < 0.000001 else {
          throw StudioError.invalid("Trim existing timeline clips to match the revised shot starts and lengths before realigning. Existing footage is never trimmed automatically.")
        }
        guard original.task != "a2v", !updated.clips[c].attachments.contains(where: { $0.role == .audioDriver }) else {
          throw StudioError.invalid("An applied audio-driven shot cannot be assigned a different song interval. Replan and generate a new take for that interval.")
        }
        updated.clips[c].musicSource = source
        updated.clips[c].reviewedTakeFingerprint = nil
      }
      plan.shots[i].musicSource = source
      plan.shots[i].approvedRevision = nil
      changed += 1
    }
    updated.planning = plan
    self = updated
    return changed
  }
}
