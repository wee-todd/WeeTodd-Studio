import Foundation

/// Original source coordinates; editorial trims never rewrite or stretch the song.
public struct MusicShotSource: Codable, Equatable {
  public var path: String
  public var sha256: String
  public var start: Double
  public var duration: Double
  public var task: String
  public var sceneEligible: Bool?
  public var maximumGenerationSeconds: Double?
  public init(path: String, sha256: String, start: Double, duration: Double, task: String) {
    self.path = path; self.sha256 = sha256; self.start = start; self.duration = duration; self.task = task
  }
}

public struct MusicVideoTiming: Codable, Equatable {
  public struct SourceAudio: Codable, Equatable {
    public var path: String
    public var sha256: String
    public var sourceStartSeconds: Double
    public var sourceEndSeconds: Double
  }
  public struct Shot: Codable, Equatable {
    public var clipID: String
    public var startFrame: Int
    public var frameCount: Int
    public var sourceStartSeconds: Double
    public var sourceDurationSeconds: Double
    public var sceneEligible: Bool?
  }
  public struct Bounds: Codable, Equatable {
    public var modelMaximumSeconds: Double?
    public var maximumSeconds: Double
  }
  public var bounds: Bounds?
  public var sourceAudio: SourceAudio
  public var fps: Int
  public var totalFrames: Int
  public var clips: [Shot]
  public var suppliedLyrics: String
  public var lyricStatus: String
  public var engine: String
  public var task: String

  public func validate() throws {
    guard (1...120).contains(fps), totalFrames > 0, totalFrames <= fps * 3600,
      !sourceAudio.path.isEmpty, sourceAudio.sha256.count == 64,
      sourceAudio.sha256.allSatisfy({ $0.isHexDigit }),
      sourceAudio.sourceStartSeconds.isFinite, sourceAudio.sourceStartSeconds >= 0,
      sourceAudio.sourceEndSeconds.isFinite, sourceAudio.sourceEndSeconds > sourceAudio.sourceStartSeconds,
      abs(sourceAudio.sourceEndSeconds - sourceAudio.sourceStartSeconds - Double(totalFrames) / Double(fps)) <= 1 / Double(fps),
      ["a2v", "t2v"].contains(task), Engine(rawValue: engine) != nil, engine != "movie",
      !clips.isEmpty, Set(clips.map(\.clipID)).count == clips.count else {
      throw StudioError.invalid("The music-video plan has invalid source audio, model or timing.")
    }
    var cursor = 0
    for (index, clip) in clips.enumerated() {
      let editorialDuration = Double(clip.frameCount) / Double(fps)
      let tail = editorialDuration - clip.sourceDurationSeconds
      guard clip.sourceDurationSeconds > 0, tail >= -0.000001,
        (abs(tail) < 0.000001 || index == clips.count - 1 && tail < 1 / Double(fps)),
        clip.startFrame == cursor, clip.frameCount > 0, clip.frameCount <= totalFrames,
        clip.sourceStartSeconds.isFinite, clip.sourceDurationSeconds.isFinite,
        abs(clip.sourceStartSeconds - sourceAudio.sourceStartSeconds - Double(cursor) / Double(fps)) < 0.000001,
        clip.sourceStartSeconds + clip.sourceDurationSeconds <= sourceAudio.sourceEndSeconds + 0.000001 else {
        throw StudioError.invalid("Music-video shots must cover the source interval exactly without gaps or overlaps.")
      }
      cursor += clip.frameCount
    }
    guard cursor == totalFrames,
      abs((clips.last!.sourceStartSeconds + clips.last!.sourceDurationSeconds) - sourceAudio.sourceEndSeconds) < 0.000001 else { throw StudioError.invalid("Music-video timing does not cover the selected song interval.") }
  }
}

extension ProjectPlanning {
  static func validGenerationSize(width: Int, height: Int, engine: Engine) -> Bool {
    let grid = engine == .drawThings ? 64 : 32
    return engine != .movie && (128...4096).contains(width) && (128...4096).contains(height)
      && width % grid == 0 && height % grid == 0
  }

  public mutating func setGenerationSettings(_ ids: Set<UUID>, engine: Engine, width: Int, height: Int) throws {
    let selected = shots.indices.filter { ids.contains(shots[$0].id) }
    guard !ids.isEmpty, selected.count == ids.count, selected.allSatisfy({ shots[$0].linkedClipID == nil }),
      Self.validGenerationSize(width: width, height: height, engine: engine) else {
      throw StudioError.invalid("Select unapplied shots and valid render dimensions: multiples of 64 for Draw Things or 32 for native models, between 128 and 4096.")
    }
    guard engine != .drawThings || selected.allSatisfy({ shots[$0].musicSource?.task != "a2v" }) else {
      throw StudioError.invalid("Audio-driven song shots require a supported native model. Replan as soundtrack visuals before selecting Draw Things.")
    }
    for i in selected {
      if shots[i].engine != engine { shots[i].modelID = "" }
      shots[i].engine = engine; shots[i].generationWidth = width; shots[i].generationHeight = height
      shots[i].approvedRevision = nil
    }
  }

  /// Split on the editorial frame grid; keep the original audio endpoint, including a partial final frame.
  @discardableResult
  public mutating func splitShot(_ id: UUID, afterFrames count: Int, boundary: String) throws -> UUID {
    guard let i = shots.firstIndex(where: { $0.id == id }), shots[i].linkedClipID == nil,
      shots[i].combinedShots == nil, (1...120).contains(frameRate),
      count > 0, count < shots[i].frameCount, !boundary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw StudioError.invalid("Select one unapplied shot, an interior frame and a boundary description. Restore a combined shot before splitting it.")
    }
    let original = shots[i]
    let seconds = Double(count) / Double(frameRate)
    let duration = Double(original.frameCount) / Double(frameRate)
    if let source = original.musicSource {
      guard source.start.isFinite, source.start >= 0, source.duration.isFinite,
        source.duration > seconds, source.duration <= duration + 0.000001,
        duration - source.duration < 1 / Double(frameRate) else {
        throw StudioError.invalid("The shot's song interval no longer matches its frames. Restore its timing before splitting.")
      }
    }
    var first = original, second = original
    second.id = UUID(); second.sourceKey = original.sourceKey + ":split:" + second.id.uuidString
    first.name += " · A"; second.name += " · B"
    first.frameCount = count; second.frameCount -= count
    first.lastFrame = boundary; second.firstFrame = boundary; second.continuity = "cut"
    first.lastAssetID = nil; second.firstAssetID = nil
    first.approvedRevision = nil; second.approvedRevision = nil
    if let source = original.musicSource {
      first.musicSource?.duration = seconds
      second.musicSource?.start = source.start + seconds
      second.musicSource?.duration = source.duration - seconds
    }
    shots.replaceSubrange(i...i, with: [first, second])
    return second.id
  }

  @discardableResult
  public mutating func combineShots(_ ids: Set<UUID>, maximumSeconds: Double = 15) throws -> UUID {
    let indices = shots.indices.filter { ids.contains(shots[$0].id) }
    guard ids.count >= 2, indices.count == ids.count, let first = indices.first, let last = indices.last,
      last - first + 1 == indices.count else { throw StudioError.invalid("Select two or more adjacent shots to combine.") }
    let originals = Array(shots[first...last])
    guard originals.allSatisfy({ $0.linkedClipID == nil }) else {
      throw StudioError.invalid("These shots already have timeline clips. Combine an unapplied plan to preserve existing takes.")
    }
    guard Set(originals.map(\.engine)).count == 1, Set(originals.map(\.task)).count == 1,
      Set(originals.map(\.modelID)).count == 1,
      Set(originals.map(\.generationWidth)).count == 1, Set(originals.map(\.generationHeight)).count == 1 else {
      throw StudioError.invalid("Combined shots must use the same generation model, task and render size.")
    }
    guard originals.allSatisfy({ (1...100_000_000).contains($0.frameCount) }) else {
      throw StudioError.invalid("Combined shots need valid positive frame counts.")
    }
    let count = originals.reduce(Int64(0)) { $0 + Int64($1.frameCount) }
    let engineMax: Double
    switch originals[0].engine {
    case .ltx23, .ltx25: engineMax = 30
    case .h3: engineMax = 15
    case .drawThings:
      guard let maximum = originals.compactMap({ $0.musicSource?.maximumGenerationSeconds }).min(), maximum.isFinite, maximum > 0 else {
        throw StudioError.invalid("Plan these shots with the selected Draw Things model's duration bounds before combining them.")
      }
      engineMax = maximum
    case .movie: throw StudioError.invalid("Combine generated shot plans; edit imported movies on the timeline.")
    }
    guard (1...120).contains(frameRate), originals.allSatisfy({ $0.frameCount > 0 }),
      count <= 100_000_000, maximumSeconds.isFinite, maximumSeconds > 0,
      Double(count) / Double(frameRate) <= min(maximumSeconds, engineMax) + 0.000001 else {
      throw StudioError.invalid("The combined shot exceeds the clip maximum. Adjust the maximum within the model's supported range or select fewer shots.")
    }
    var music = originals[0].musicSource
    var cursor = music?.start ?? 0
    for (index, shot) in originals.enumerated() {
      if let source = shot.musicSource, let firstSource = music {
        guard source.path == firstSource.path, source.sha256 == firstSource.sha256, source.task == firstSource.task,
          abs(source.start - cursor) < 0.000001,
          source.duration > 0,
          source.duration <= Double(shot.frameCount) / Double(frameRate) + 0.000001,
          (abs(source.duration - Double(shot.frameCount) / Double(frameRate)) < 0.000001 ||
            index == originals.count - 1 && Double(shot.frameCount) / Double(frameRate) - source.duration < 1 / Double(frameRate)) else {
          throw StudioError.invalid("Combined music shots must use consecutive intervals of the same song.")
        }
        cursor += source.duration
      } else if shot.musicSource != nil || music != nil {
        throw StudioError.invalid("Combined shots must share a music source.")
      }
    }
    if let firstSource = music {
      music?.duration = cursor - firstSource.start
      music?.sceneEligible = originals.allSatisfy { $0.musicSource?.sceneEligible == true }
    }
    var combined = originals[0]
    combined.id = UUID(); combined.sourceKey = "combined:" + combined.id.uuidString
    combined.name = originals.map(\.name).joined(separator: " + ")
    combined.frameCount = Int(count); combined.lastFrame = originals.last!.lastFrame
    combined.lastAssetID = originals.last!.lastAssetID
    combined.combinedShots = originals; combined.approvedRevision = nil; combined.musicSource = music
    var elapsed = 0
    combined.direction = "One continuous take with the following ordered actions.\n\n" + originals.map { shot in
      defer { elapsed += shot.frameCount }
      let interval = String(format: "%.3f–%.3f s", Double(elapsed) / Double(frameRate), Double(elapsed + shot.frameCount) / Double(frameRate))
      return "[\(interval) · \(shot.name)]\n" + [shot.action, shot.direction, shot.camera, shot.dialogue, shot.sound].filter { !$0.isEmpty }.joined(separator: "\n")
    }.joined(separator: "\n\n")
    combined.action = originals.map(\.action).filter { !$0.isEmpty }.joined(separator: "; then ")
    combined.camera = originals.map(\.camera).filter { !$0.isEmpty }.joined(separator: "\n")
    combined.dialogue = originals.map(\.dialogue).filter { !$0.isEmpty }.joined(separator: "\n")
    combined.sound = originals.map(\.sound).filter { !$0.isEmpty }.joined(separator: "\n")
    var subjects = Set<UUID>(), references = Set<UUID>()
    combined.subjectIDs = originals.flatMap(\.subjectIDs).filter { subjects.insert($0).inserted }
    combined.referenceAssetIDs = originals.flatMap(\.referenceAssetIDs).filter { references.insert($0).inserted }
    // Conflicting appearance states need explicit resolution instead of quietly retaining one.
    let states = originals.flatMap { $0.appearanceOverrides ?? [] }
    guard Dictionary(grouping: states, by: \.subjectID).values.allSatisfy({ Set($0.map(\.state)).count <= 1 }) else {
      throw StudioError.invalid("Resolve conflicting subject appearance overrides before combining shots.")
    }
    var seen = Set<UUID>(); combined.appearanceOverrides = states.filter { seen.insert($0.subjectID).inserted }
    if combined.appearanceOverrides?.isEmpty == true { combined.appearanceOverrides = nil }
    shots.replaceSubrange(first...last, with: [combined])
    return combined.id
  }

  public mutating func uncombineShot(_ id: UUID) throws {
    guard let i = shots.firstIndex(where: { $0.id == id }), let originals = shots[i].combinedShots, !originals.isEmpty,
      shots[i].linkedClipID == nil else { throw StudioError.invalid("Select an unapplied combined shot to restore its original shots.") }
    let otherIDs = Set(shots.filter { $0.id != id }.map(\.id))
    guard Set(originals.map(\.id)).isDisjoint(with: otherIDs) else {
      throw StudioError.invalid("An original shot is already in this plan; restore the shot order before uncombining.")
    }
    shots.replaceSubrange(i...i, with: originals)
  }
}

extension StudioProject {
  /// Explicitly link an existing take without replacing its footage, prompt or versions.
  public mutating func reuseTimelineClip(_ clipID: UUID, forPlanningShot shotID: UUID) throws {
    guard var plan = planning, let s = plan.shots.firstIndex(where: { $0.id == shotID }),
      let c = clips.firstIndex(where: { $0.id == clipID }),
      plan.shots[s].linkedClipID == nil || plan.shots[s].linkedClipID == clipID,
      !plan.shots.contains(where: { $0.id != shotID && $0.linkedClipID == clipID }), plan.isShotApproved(shotID, assets: assets),
      abs(settings.fps - Double(plan.frameRate)) < 0.000001 else {
      throw StudioError.invalid("Approve an unapplied shot and select an existing, unlinked timeline take to reuse.")
    }
    let shot = plan.shots[s], clip = clips[c]
    let length = Double(shot.frameCount) / Double(plan.frameRate)
    let offset = Double(plan.startFrame(of: shotID)) / Double(plan.frameRate)
    guard !clip.sourcePath.isEmpty, clip.engine == shot.engine, clip.transition == "cut",
      c + 1 == clips.count || clips[c + 1].transition == "cut",
      abs(start(of: c) - offset) < 0.000001, clip.duration.isFinite, length <= clip.duration + 0.000001,
      clip.duration - length < 1 / Double(plan.frameRate),
      c == clips.count - 1 || abs(clip.duration - length) < 0.000001 else {
      throw StudioError.invalid("The existing take must match this shot's model, start and duration. Only a sub-frame trim of the last clip is allowed; existing footage and later edits are preserved.")
    }
    if let existing = clip.musicSource, let source = shot.musicSource {
      guard existing.sha256 == source.sha256, abs(existing.start - source.start) < 0.000001 else {
        throw StudioError.invalid("This take belongs to a different song interval.")
      }
    }
    if let source = shot.musicSource {
      guard source.start.isFinite, source.start >= 0, source.duration.isFinite, source.duration > 0,
        source.duration <= length + 0.000001, length - source.duration < 1 / Double(plan.frameRate) else {
        throw StudioError.invalid("The shot length changed. Restore its song interval before reusing a take.")
      }
    }
    plan.shots[s].linkedClipID = clipID
    // Linkage is bookkeeping for already reviewed content, not a creative revision.
    try plan.approveShot(shotID, assets: assets)
    clips[c].duration = length; clips[c].musicSource = shot.musicSource
    if shot.musicSource != nil { clips[c].volume = 0 }
    clips[c].reviewedTakeFingerprint = clips[c].reuseFingerprint
    planning = plan
  }

  /// Applies an approved plan as an additive timeline edit. Rendering uses the ordinary clip path.
  public mutating func applyPlanningShots(_ ids: Set<UUID>, continuity: Bool = true) throws {
    guard var plan = planning, !ids.isEmpty else { throw StudioError.invalid("Select approved shots to add to the timeline.") }
    let selected = plan.shots.filter { ids.contains($0.id) }
    guard selected.count == ids.count, selected.allSatisfy({ $0.linkedClipID == nil && plan.isShotApproved($0.id, assets: assets) }) else {
      throw StudioError.invalid("Approve each selected shot and its subjects first. Already applied shots keep their existing timeline clips.")
    }
    guard abs(settings.fps - Double(plan.frameRate)) < 0.000001 else {
      throw StudioError.invalid("Match the movie frame rate to the reviewed shot plan before applying it.")
    }
    var updated = self
    let firstNew = updated.clips.count
    var timelineStart = updated.duration
    var lastRegion: Int?
    for shot in selected {
      var clip = Clip(name: shot.name, engine: shot.engine)
      clip.duration = Double(shot.frameCount) / Double(plan.frameRate)
      if let width = shot.generationWidth { clip.generationWidth = width }
      if let height = shot.generationHeight { clip.generationHeight = height }
      clip.musicSource = shot.musicSource
      clip.settingsOverride = settings
      let objects = try plan.resolvedObjects(shot.subjectIDs)
      let descriptions = objects.map { object in
        let state = shot.appearanceOverrides?.first(where: { $0.subjectID == object.id })?.state
        return "\(object.name): \(object.details)" + (state.map { "\nIn this shot: " + $0 } ?? "")
      }
      clip.prompt = ([shot.direction.isEmpty ? shot.action : shot.direction] + descriptions).filter { !$0.isEmpty }.joined(separator: "\n\n")
      clip.soundscape = shot.sound; clip.selectGenerationTask(shot.task)
      if !shot.modelID.isEmpty { clip.profileID = shot.modelID }
      if let source = shot.musicSource {
        guard source.start.isFinite, source.start >= 0, source.duration.isFinite,
          source.duration > 0, source.duration <= clip.duration + 0.000001,
          clip.duration - source.duration < 1 / Double(plan.frameRate) else { throw StudioError.invalid("The shot length changed. Replan its music interval before adding it to the timeline.") }
        var asset = updated.assets.first { $0.kind == .audio && $0.path == source.path }
          ?? MediaAsset(name: URL(fileURLWithPath: source.path).deletingPathExtension().lastPathComponent, kind: .audio, path: source.path)
        if !updated.assets.contains(where: { $0.id == asset.id }) {
          asset.duration = max(source.start + source.duration, plan.musicTimings?.values.filter { $0.sourceAudio.path == source.path }.map { $0.sourceAudio.sourceEndSeconds }.max() ?? 0)
          updated.assets.append(asset)
        }
        if source.task == "a2v" {
          guard [.ltx25, .ltx23, .h3].contains(shot.engine) else { throw StudioError.invalid("Select a supported native audio-driven model or plan visuals with a music soundtrack.") }
          var driver = Attachment(assetID: asset.id, role: .audioDriver)
          driver.audioSourceStart = source.start; driver.audioSourceDuration = source.duration
          clip.attachments.append(driver)
        }
        clip.volume = 0
        let alignedRegions = updated.audio.filter {
          $0.path == source.path && abs(($0.sourceIn - $0.start) - (source.start - timelineStart)) < 0.000001
        }
        let alreadyCovered = alignedRegions.contains {
          $0.start <= timelineStart + 0.000001 && $0.start + $0.duration >= timelineStart + source.duration - 0.000001
        }
        if !alreadyCovered, alignedRegions.contains(where: {
          $0.start < timelineStart + source.duration - 0.000001 && $0.start + $0.duration > timelineStart + 0.000001
        }) {
          throw StudioError.invalid("An existing song region overlaps part of this shot. Review the music track before adding the plan to avoid a doubled mix.")
        }
        if alreadyCovered {
          // Preserve an existing full song instead of mixing it with a second copy.
          lastRegion = nil
        } else if let index = lastRegion, updated.audio[index].path == source.path,
          abs(updated.audio[index].sourceIn + updated.audio[index].duration - source.start) < 0.000001,
          abs(updated.audio[index].start + updated.audio[index].duration - timelineStart) < 0.000001 {
          updated.audio[index].duration += source.duration
        } else {
          _ = try updated.placeMusic(asset, at: timelineStart, trackID: nil, sourceIn: source.start, duration: source.duration)
          lastRegion = updated.audio.count - 1
        }
      } else { lastRegion = nil }
      func image(_ id: UUID?, role: MediaRole, at time: Double = 0) throws -> Attachment? {
        guard let id else { return nil }
        guard updated.assets.contains(where: { $0.id == id && $0.kind == .image && !$0.path.isEmpty }) else {
          throw StudioError.invalid("A shot frame image is unavailable. Relink it before applying the plan.")
        }
        return Attachment(assetID: id, role: role, time: time)
      }
      if let input = try image(shot.firstAssetID, role: .first) { clip.attachments.append(input) }
      if let input = try image(shot.lastAssetID, role: .last) { clip.attachments.append(input) }
      func appendInteriorFrames(_ children: [PlanningShot], startingAt start: Int) throws {
        var frame = start
        func append(_ id: UUID?, at frame: Int) throws {
          let time = Double(frame) / Double(plan.frameRate)
          guard !clip.attachments.contains(where: { $0.assetID == id && $0.role == .keyframe && abs($0.time - time) < 0.000001 }) else { return }
          if let input = try image(id, role: .keyframe, at: time) { clip.attachments.append(input) }
        }
        for child in children {
          if frame > 0 { try append(child.firstAssetID, at: frame) }
          if let nested = child.combinedShots { try appendInteriorFrames(nested, startingAt: frame) }
          frame += child.frameCount
          if frame < shot.frameCount { try append(child.lastAssetID, at: frame - 1) }
        }
      }
      if let children = shot.combinedShots { try appendInteriorFrames(children, startingAt: 0) }
      if shot.musicSource?.task != "a2v", clip.attachments.contains(where: { [.first,.last,.keyframe].contains($0.role) }) {
        clip.selectGenerationTask(clip.attachments.contains { $0.role == .last || $0.role == .keyframe } ? "fflf" : "i2v")
      }
      if !shot.referenceAssetIDs.isEmpty {
        guard shot.engine == .h3 else { throw StudioError.invalid("Assign these shot references as frame images, or configure a compatible reference adapter in the clip inspector.") }
        for id in shot.referenceAssetIDs { if let input = try image(id, role: .reference) { clip.attachments.append(input) } }
      }
      if continuity, shot.continuity == "continue", updated.clips.count > firstNew,
        let previous = updated.clips.last, clip.engine == .ltx25, previous.engine == .ltx25 {
        let scenePreference = shot.musicSource?.sceneEligible != false && previous.musicSource?.sceneEligible != false
        let currentFrames = shot.frameCount
        let previousFrames = Int((previous.duration * Double(plan.frameRate)).rounded())
        if scenePreference, currentFrames % 8 == 0, previousFrames % 8 == 0, previousFrames >= 32 {
          var candidate = updated; var joined = clip
          joined.continuity = ClipContinuity(mode: "scene", sourceClipID: previous.id)
          candidate.clips.append(joined)
          if candidate.continuousSceneIssues(for: joined).isEmpty { clip = joined }
        }
      }
      updated.clips.append(clip)
      if let i = plan.shots.firstIndex(where: { $0.id == shot.id }) {
        plan.shots[i].linkedClipID = clip.id
        try plan.approveShot(shot.id, assets: updated.assets)
      }
      timelineStart += clip.duration
    }
    updated.planning = plan
    self = updated
  }
}

extension StudioProject {
  /// Relink canonical song coordinates alongside ordinary assets, including unapplied plans.
  public mutating func mapMusicSourcePaths(_ transform: (String) throws -> String) rethrows {
    func mapped(_ original: PlanningShot) throws -> PlanningShot {
      var shot = original
      if let source = shot.musicSource { shot.musicSource?.path = try transform(source.path) }
      if let children = shot.combinedShots { shot.combinedShots = try children.map(mapped) }
      return shot
    }
    for i in clips.indices {
      if let source = clips[i].musicSource { clips[i].musicSource?.path = try transform(source.path) }
    }
    if var plan = planning {
      plan.shots = try plan.shots.map(mapped)
      for key in Array(plan.musicTimings?.keys ?? Dictionary<String, MusicVideoTiming>().keys) {
        if let path = plan.musicTimings?[key]?.sourceAudio.path {
          plan.musicTimings?[key]?.sourceAudio.path = try transform(path)
        }
      }
      // Analysis retains the original provenance hash; only its source location is relocated.
      for key in Array(plan.musicAnalysis?.keys ?? Dictionary<String, JSONValue>().keys) {
        if case .object(var value) = plan.musicAnalysis?[key], case .object(var source) = value["sourceAudio"],
          case .string(let path) = source["path"] {
          source["path"] = .string(try transform(path)); value["sourceAudio"] = .object(source)
          plan.musicAnalysis?[key] = .object(value)
        }
      }
      planning = plan
    }
  }
}

extension StudioProject {
  /// Keep style and shot references even when the agent did not assign them to an object.
  public mutating func retainPlanningImages(_ bindings: [String: String]) {
    for path in Set(bindings.values).sorted() where !path.isEmpty {
      guard !assets.contains(where: { $0.path == path && $0.kind == .image && $0.scope == .project }) else { continue }
      assets.append(MediaAsset(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
        kind: .image, path: path))
    }
  }
}
