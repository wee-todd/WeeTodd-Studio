import CryptoKit
import Foundation

public enum PlanningSubjectKind: String, Codable, CaseIterable, Identifiable {
  case character, prop, location, environment, set, clothing, outfit
  public var id: String { rawValue }
  public var label: String { rawValue.capitalized }
  public static let reviewOrder: [Self] = [.character, .environment, .set, .location, .prop, .clothing, .outfit]
  public var groupTitle: String {
    switch self { case .character: return "Characters"; case .location: return "Locations (unclassified)"; case .prop: return "Props"; case .environment: return "Environments"; case .set: return "Sets"; case .clothing: return "Clothing"; case .outfit: return "Outfits" }
  }
}

public struct PlanningSubject: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var sourceKey = ""
  public var sourceKeyAliases: [String] = []
  public var name: String
  public var kind: PlanningSubjectKind
  public var aliases = ""
  public var details = ""
  public var evidence = ""
  public var suggestions = ""
  public var referenceAssetIDs: [UUID] = []
  public var approvedRevision: String?
  public var approvedReferenceRevision: String?
  public var characterAppearance: CharacterAppearance?
  public var originalAppearanceDescription: String?
  public var descriptionReview: SubjectDescriptionReview?
  public var relationships: [ObjectRelationship]?
  public var relationshipReview: ObjectRelationshipReview?
  public var descriptionMentions: [DescriptionMention]?
  public var mentionSourceDescription: String?
  public var coverageReview: ObjectCoverageReview?
  public var tags: [String]?
  public var environmentID: UUID?
  public var libraryOrigin: LibraryOrigin?
  public init(name: String, kind: PlanningSubjectKind) { self.name = name; self.kind = kind }
  public var revision: String {
    let legacy = planningDigest([name, kind.rawValue, aliases, details, evidence, suggestions])
    let relational: String
    if !(relationships ?? []).isEmpty || environmentID != nil || !(descriptionMentions ?? []).isEmpty {
      var parts = [legacy, planningDigest(relationships ?? []), environmentID?.uuidString ?? ""]
      if !(descriptionMentions ?? []).isEmpty { parts += [planningDigest(descriptionMentions), mentionSourceDescription ?? ""] }
      relational = planningDigest(parts)
    } else { relational = legacy }
    guard let characterAppearance else { return relational }
    return planningDigest([relational, planningDigest(characterAppearance)])
  }

  public static func descriptionProjection(for appearance: CharacterAppearance) -> String {
    // Reuse the compiler's labelled appearance clauses, including explicit absence and named
    // material targets, while excluding generation/style/camera/layout instructions.
    let definition = CharacterSheetDefinition(appearance: appearance,
      settings: CharacterSheetSettings(stylePresetID: "photograph"))
    return CharacterSheetCompiler.compile(definition).sections
      .filter { (2...8).contains($0.index) && !$0.text.isEmpty }
      .map(\.text).joined(separator: ". ")
  }

  public mutating func applyAcceptedAppearance(_ appearance: CharacterAppearance, sourceDescription: String?) {
    characterAppearance = appearance
    originalAppearanceDescription = sourceDescription
    details = Self.descriptionProjection(for: appearance)
  }
}

public struct PlanningShot: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var sourceKey = ""
  public var name: String
  public var frameCount: Int
  public var action = ""
  public var direction = ""
  public var camera = ""
  public var dialogue = ""
  public var sound = ""
  public var firstFrame = ""
  public var lastFrame = ""
  public var continuity = "cut"
  public var subjectIDs: [UUID] = []
  public var referenceAssetIDs: [UUID] = []
  public var firstAssetID: UUID?
  public var lastAssetID: UUID?
  public var engine: Engine = .ltx25
  public var task = "fflf"
  public var modelID = ""
  public var generationWidth: Int?
  public var generationHeight: Int?
  public var linkedClipID: UUID?
  /// Retained source shots make combining reversible without losing references or text.
  public var combinedShots: [PlanningShot]?
  public var musicSource: MusicShotSource?
  public var appearanceOverrides: [ObjectStateOverride]?
  public var approvedRevision: String?
  public init(name: String, frameCount: Int) { self.name = name; self.frameCount = frameCount }
}

/// Optional project-owned planning document. Workflow runs remain independent proposals.
public struct ProjectPlanning: Codable, Equatable {
  public var format = "weetodd-project-planning-v1"
  public var version = 1
  public var sourceText = ""
  public var sourceScripts: [String: String] = [:]
  public var frameRate = 24
  public var subjects: [PlanningSubject] = []
  public var shots: [PlanningShot] = []
  public var musicTimings: [String: MusicVideoTiming]?
  public var musicAnalysis: [String: JSONValue]?
  public var importedSources: [String] = []
  public var reviewNotes: [String] = []
  public init() {}

  public func isSubjectApproved(_ id: UUID) -> Bool {
    guard let s = subjects.first(where: { $0.id == id }), let revision = subjectApprovalRevision(id) else { return false }
    return s.approvedRevision == revision && !s.name.trimmed.isEmpty && !s.details.trimmed.isEmpty
  }
  public mutating func approveSubject(_ id: UUID) throws {
    guard let i = subjects.firstIndex(where: { $0.id == id }),
          !subjects[i].name.trimmed.isEmpty, !subjects[i].details.trimmed.isEmpty else {
      throw StudioError.invalid("Give this subject a name and a description before approval.")
    }
    guard let revision = subjectApprovalRevision(id) else { throw StudioError.invalid("Resolve this object’s missing or invalid links before approving it.") }
    subjects[i].approvedRevision = revision
  }
  private func referenceRevision(_ id: UUID, assets: [MediaAsset]) -> String? {
    guard let s = subjects.first(where: { $0.id == id }), isSubjectApproved(id), !s.referenceAssetIDs.isEmpty else { return nil }
    var keys = [subjectApprovalRevision(id) ?? s.revision]
    for aid in s.referenceAssetIDs {
      guard let a = assets.first(where: { $0.id == aid }), a.kind == .image, !a.path.isEmpty,
            let attributes = try? FileManager.default.attributesOfItem(atPath: a.path),
            let bytes = attributes[.size] as? NSNumber, bytes.intValue > 0,
            let modified = attributes[.modificationDate] as? Date,
            attributes[.type] as? FileAttributeType == .typeRegular else { return nil }
      keys += [aid.uuidString, a.path, a.generation?.requestFingerprint ?? "", bytes.stringValue, String(modified.timeIntervalSince1970)]
    }
    return planningDigest(keys)
  }
  public func areReferencesApproved(_ id: UUID, assets: [MediaAsset]) -> Bool {
    guard let revision = referenceRevision(id, assets: assets) else { return false }
    return subjects.first(where: { $0.id == id })?.approvedReferenceRevision == revision
  }
  public mutating func approveReferences(_ id: UUID, assets: [MediaAsset]) throws {
    guard let i = subjects.firstIndex(where: { $0.id == id }), let revision = referenceRevision(id, assets: assets) else {
      throw StudioError.invalid("Approve the description and link at least one available image before approving references.")
    }
    subjects[i].approvedReferenceRevision = revision
  }
  public func startFrame(of id: UUID) -> Int {
    var total = 0
    for shot in shots {
      if shot.id == id { break }
      let (next, overflow) = total.addingReportingOverflow(shot.frameCount)
      if overflow { return 0 }
      total = next
    }
    return total
  }
  public func issues(for id: UUID, assets: [MediaAsset] = []) -> [String] {
    guard let i = shots.firstIndex(where: { $0.id == id }) else { return ["Shot no longer exists."] }
    let s = shots[i]
    var issues: [String] = []
    if !(1...120).contains(frameRate) || !(1...100_000_000).contains(s.frameCount) { issues.append("Use 1–120 FPS and a positive frame count up to 100,000,000.") }
    if s.generationWidth != nil || s.generationHeight != nil {
      if !Self.validGenerationSize(width: s.generationWidth ?? 0, height: s.generationHeight ?? 0, engine: s.engine) {
        issues.append("Set both render dimensions to supported multiples: 64 pixels for Draw Things, 32 for native models, between 128 and 4096.")
      }
    }
    if s.name.trimmed.isEmpty || s.action.trimmed.isEmpty || s.firstFrame.trimmed.isEmpty || s.lastFrame.trimmed.isEmpty {
      issues.append("Add a shot name, action, first-frame description and last-frame description.")
    }
    if Set(s.subjectIDs).count != s.subjectIDs.count { issues.append("A subject is linked more than once.") }
    let resolved: [PlanningSubject]
    do { resolved = try resolvedObjects(s.subjectIDs) }
    catch { issues.append(error.localizedDescription); resolved = [] }
    let resolvedIDs = Array(Set(s.subjectIDs + resolved.map(\.id)))
    let states = s.appearanceOverrides ?? []
    if Set(states.map(\.subjectID)).count != states.count || states.contains(where: { !resolvedIDs.contains($0.subjectID) || $0.state.count > 2000 }) {
      issues.append("Appearance overrides must refer to this shot’s objects, once each, with at most 2,000 characters.")
    }
    for subjectID in resolvedIDs where !isSubjectApproved(subjectID) {
      issues.append("Approve \(subjects.first(where: { $0.id == subjectID })?.name ?? "the missing subject") before approving this shot.")
    }
    for subjectID in resolvedIDs {
      if let subject = subjects.first(where: { $0.id == subjectID }), !subject.referenceAssetIDs.isEmpty,
         !areReferencesApproved(subjectID, assets: assets) {
        issues.append("Review the reference images for " + subject.name + " before approving this shot.")
      }
    }
    if !["cut", "continue"].contains(s.continuity) { issues.append("Choose Cut or Continue.") }
    if s.continuity == "continue" {
      if i == 0 { issues.append("The first shot must start with a cut.") }
      else {
        if s.firstFrame != shots[i-1].lastFrame { issues.append("A continuous shot must start with the previous ending description.") }
        let locations = Set(subjects.filter { [.location, .set, .environment].contains($0.kind) }.map(\.id))
        if Set(s.subjectIDs).intersection(locations) != Set(shots[i-1].subjectIDs).intersection(locations) {
          issues.append("A location change requires a cut.")
        }
      }
    }
    return issues
  }
  private func shotRevision(_ id: UUID, assets: [MediaAsset]) -> String? {
    guard let i = shots.firstIndex(where: { $0.id == id }) else { return nil }
    var shot = shots[i]; shot.approvedRevision = nil
    var keys = [planningDigest(shot), String(frameRate), String(startFrame(of: id))]
    keys += ((try? resolvedObjects(shot.subjectIDs).map(\.id)) ?? shot.subjectIDs).map { sid in
      guard let s = subjects.first(where: { $0.id == sid }) else { return "missing:\(sid)" }
      return s.id.uuidString + s.revision + planningDigest(s.referenceAssetIDs) + (referenceRevision(sid, assets: assets) ?? "unavailable")
    }
    if shot.continuity == "continue", i > 0 {
      keys += [shots[i-1].id.uuidString, shots[i-1].lastFrame]
    }
    return planningDigest(keys)
  }
  public func isShotApproved(_ id: UUID, assets: [MediaAsset] = []) -> Bool {
    guard issues(for: id, assets: assets).isEmpty, let revision = shotRevision(id, assets: assets) else { return false }
    return shots.first(where: { $0.id == id })?.approvedRevision == revision
  }
  public mutating func approveShot(_ id: UUID, assets: [MediaAsset] = []) throws {
    if let issue = issues(for: id, assets: assets).first { throw StudioError.invalid(issue) }
    guard let i = shots.firstIndex(where: { $0.id == id }), let revision = shotRevision(id, assets: assets) else { return }
    shots[i].approvedRevision = revision
  }

  public mutating func mergeSubject(_ sourceID: UUID, into targetID: UUID) throws {
    var candidate = self
    try candidate.mergeSubjectUnchecked(sourceID, into: targetID)
    _ = try candidate.resolvedObjects([targetID])
    self = candidate
  }
  private mutating func mergeSubjectUnchecked(_ sourceID: UUID, into targetID: UUID) throws {
    guard sourceID != targetID,
          let source = subjects.first(where: { $0.id == sourceID }),
          let target = subjects.first(where: { $0.id == targetID }), source.kind == target.kind else {
      throw StudioError.invalid("Select two different subjects of the same kind.")
    }
    guard !isSubjectApproved(sourceID), !isSubjectApproved(targetID) else {
      throw StudioError.invalid("Unlock both descriptions before merging subjects.")
    }
    let i = subjects.firstIndex(where: { $0.id == targetID })!
    subjects[i].aliases = ([target.aliases, source.name, source.aliases].filter { !$0.isEmpty }).joined(separator: ", ")
    subjects[i].evidence = ([target.evidence, source.evidence].filter { !$0.isEmpty }).joined(separator: "\n")
    subjects[i].suggestions = ([target.suggestions, source.suggestions, "Merged description from " + source.name + ": " + source.details].filter { !$0.isEmpty }).joined(separator: "\n")
    subjects[i].sourceKeyAliases = Array(Set(target.sourceKeyAliases + source.sourceKeyAliases + [source.sourceKey]).filter { !$0.isEmpty }).sorted()
    subjects[i].referenceAssetIDs += source.referenceAssetIDs.filter { !target.referenceAssetIDs.contains($0) }
    subjects[i].approvedRevision = nil; subjects[i].approvedReferenceRevision = nil
    for index in shots.indices {
      var seen = Set<UUID>()
      shots[index].subjectIDs = shots[index].subjectIDs.map { $0 == sourceID ? targetID : $0 }.filter { seen.insert($0).inserted }
      var states = shots[index].appearanceOverrides ?? []
      if let sourceState = states.first(where: { $0.subjectID == sourceID }) {
        if states.contains(where: { $0.subjectID == targetID }) { throw StudioError.invalid("Resolve the two shot appearance overrides before merging these objects.") }
        states.removeAll { $0.subjectID == sourceID }
        states.append(ObjectStateOverride(subjectID: targetID, state: sourceState.state))
        shots[index].appearanceOverrides = states
      }
    }
    subjects[i].relationships = (target.relationships ?? []) + (source.relationships ?? [])
    subjects[i].tags = Array(Set((target.tags ?? []) + (source.tags ?? []))).sorted()
    if subjects[i].environmentID == nil { subjects[i].environmentID = source.environmentID }
    for index in subjects.indices {
      if subjects[index].environmentID == sourceID { subjects[index].environmentID = targetID }
      subjects[index].descriptionMentions = subjects[index].descriptionMentions?.compactMap { mention in
        var value = mention; if value.targetID == sourceID.uuidString { value.targetID = targetID.uuidString }
        return value.targetID == subjects[index].id.uuidString ? nil : value
      }
      subjects[index].relationships = subjects[index].relationships?.compactMap { link in
        var linked = link
        if linked.targetID == sourceID { linked.targetID = targetID }
        return linked.targetID == subjects[index].id ? nil : linked
      }
    }
    subjects.removeAll { $0.id == sourceID }
  }

  /// Import is additive and transactional. Existing records, source text and approvals are never replaced.
  public mutating func importRun(_ run: WorkflowRunSummary, sourceID: String, sourceText: String) throws {
    var candidate = self
    try candidate.addRun(run, sourceID: sourceID, sourceText: sourceText)
    self = candidate
  }
  private mutating func addRun(_ run: WorkflowRunSummary, sourceID: String, sourceText: String) throws {
    guard !sourceID.isEmpty else { throw StudioError.invalid("Workflow source identity is missing.") }
    var outputs: [String: JSONValue] = [:]
    for key in run.steps.keys.sorted() where run.steps[key]?.status == "completed" {
      for (name, value) in run.steps[key]?.outputs ?? [:] { outputs[name] = value }
    }
    func decode<T: Decodable>(_ key: String, _ type: T.Type) throws -> T? {
      guard let value = outputs[key] else { return nil }
      return try JSONDecoder().decode(type, from: JSONEncoder().encode(value))
    }
    if let finalSubjects = run.outputs["subjects"] { outputs["subjects"] = finalSubjects }
    else if let id = run.preferredSubjectStepID, let finalSubjects = run.steps[id]?.outputs?["subjects"] { outputs["subjects"] = finalSubjects }
    let inventory = try decode("subjects", [WorkflowSubjectProposal].self)
    if let inventory {
      guard inventory.count <= 2000, Set(inventory.map(\.id)).count == inventory.count else { throw StudioError.invalid("Workflow inventory has too many subjects or duplicate IDs.") }
      let known = Set(inventory.map(\.id))
      for item in inventory {
        let links = item.relationships ?? []
        guard links.count <= 32, Set(links.map(\.id)).count == links.count,
              links.allSatisfy({ $0.targetID != item.id && known.contains($0.targetID) && $0.placement.count <= 300 }) else {
          throw StudioError.invalid("Workflow inventory has invalid object relationships.")
        }
      }
    }
    let inventoryReview = run.steps[run.preferredSubjectStepID ?? ""]
    let story = try decode("story", WorkflowStoryOutline.self)
    let clips = try decode("clips", WorkflowClipPlan.self)
    if let preview = run.outputs["h3_prompts"] { outputs["h3_prompts"] = preview }
    if let preview = run.outputs["prompt_plan"] { outputs["prompt_plan"] = preview }
    let promptPreview = try decode("prompt_plan", H3PromptPreview.self) ?? decode("h3_prompts", H3PromptPreview.self)
    if let timing = run.outputs["music_timing"] { outputs["music_timing"] = timing }
    let musicTiming = try decode("music_timing", MusicVideoTiming.self)
    try musicTiming?.validate()
    if let musicTiming {
      guard let clips, musicTiming.fps == clips.fps, musicTiming.totalFrames == clips.totalFrames,
        musicTiming.clips.map(\.clipID) == clips.clips.map(\.id),
        zip(musicTiming.clips, clips.clips).allSatisfy({ $0.startFrame == $1.startFrame && $0.frameCount == $1.frameCount }) else {
        throw StudioError.invalid("Music analysis timing differs from the reviewed shot plan. Rebuild timing before importing.")
      }
    }
    var promptSubjects: [String: [String]] = [:]
    if let promptPreview {
      let known = Set((inventory ?? []).map(\.id))
      guard let clips, Set(promptPreview.prompts.map(\.clipID)) == Set(clips.clips.map(\.id)),
            Set(promptPreview.prompts.map(\.clipID)).count == promptPreview.prompts.count,
            promptPreview.prompts.allSatisfy({ Set($0.subjectIDs).isSubset(of: known) }) else {
        throw StudioError.invalid("Prompt drafts have missing shots or unknown subject IDs. Rebuild the preview before importing.")
      }
      promptSubjects = Dictionary(uniqueKeysWithValues: promptPreview.prompts.map { ($0.clipID, $0.subjectIDs) })
    }
    guard inventory != nil || story != nil || clips != nil else { throw StudioError.invalid("Complete a subject, story or clip-planning step first.") }
    for step in run.steps.values where step.status == "completed" {
      for note in step.warnings ?? [] where !reviewNotes.contains(note) { reviewNotes.append(note) }
    }
    if self.sourceText.isEmpty { self.sourceText = sourceText }
    if sourceScripts[sourceID] == nil { sourceScripts[sourceID] = sourceText }
    let prefix = sourceID + ":"
    func sourceBodies(_ text: String) -> [String] {
      guard let regex = try? NSRegularExpression(pattern: "(?im)^\\s*\\[Shot\\s+\\d+\\][^\\n]*") else { return [] }
      let ns = text as NSString
      let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
      return matches.enumerated().map { i, match in
        let end = i + 1 < matches.count ? matches[i+1].range.location : ns.length
        return ns.substring(with: NSRange(location: match.range.location, length: end - match.range.location))
      }
    }
    var newInventoryIDs = Set<UUID>()
    var inventoryIDs: [String: UUID] = [:]
    for item in inventory ?? [] {
      // Classification is editable; workflow identity is the source plus item ID.
      // Match old kind-bearing keys, including aliases retained by object reuse.
      let key = prefix + "subject:" + item.id
      let sourceKeys = Set([key] + PlanningSubjectKind.allCases.map {
        prefix + "subject:" + $0.rawValue + ":" + item.id
      })
      let matches = subjects.filter {
        sourceKeys.contains($0.sourceKey) || !sourceKeys.isDisjoint(with: $0.sourceKeyAliases)
      }
      guard matches.count <= 1 else {
        throw StudioError.invalid("Multiple project objects match workflow subject \(item.name). Merge the duplicate objects before importing again.")
      }
      if let old = matches.first { inventoryIDs[item.id] = old.id; continue }
      var s = PlanningSubject(name: item.name, kind: item.kind)
      s.sourceKey = key; s.details = item.description; s.aliases = item.aliases.joined(separator: ", ")
      s.evidence = item.evidence.joined(separator: "\n"); s.suggestions = item.suggestions.joined(separator: "\n")
      s.descriptionReview = item.descriptionReview
      if inventoryReview?.approved == true || inventoryReview?.items?[item.id]?.approved == true {
        s.approvedRevision = s.revision
      }
      subjects.append(s); newInventoryIDs.insert(s.id); inventoryIDs[item.id] = s.id
    }
    for item in inventory ?? [] {
      guard let id = inventoryIDs[item.id], newInventoryIDs.contains(id), let index = subjects.firstIndex(where: { $0.id == id }) else { continue }
      subjects[index].relationships = try item.relationships?.map { link in
        guard let target = inventoryIDs[link.targetID], target != id else { throw StudioError.invalid("Workflow relationship has an unknown or self target.") }
        return ObjectRelationship(targetID: target, role: link.role, placement: link.placement)
      }
      if item.kind == .set {
        let parentIDs = Set((subjects[index].relationships ?? []).filter { link in
          [.located_in, .part_of].contains(link.role) && subjects.contains(where: { $0.id == link.targetID && $0.kind == .environment })
        }.map(\.targetID))
        if parentIDs.count == 1 { subjects[index].environmentID = parentIDs.first }
      }
      subjects[index].relationshipReview = item.relationshipReview
      subjects[index].coverageReview = item.coverageReview
      subjects[index].mentionSourceDescription = item.mentionSourceDescription
      subjects[index].descriptionMentions = try item.descriptionMentions?.map { mention in
        guard let target = inventoryIDs[mention.targetID] else { throw StudioError.invalid("A phrase reference points to an unknown inventory object.") }
        var value = mention; value.targetID = target.uuidString; return value
      }
    }
    for item in inventory ?? [] {
      if let id = inventoryIDs[item.id], newInventoryIDs.contains(id), let index = subjects.firstIndex(where: { $0.id == id }), subjects[index].approvedRevision != nil {
        subjects[index].approvedRevision = subjectApprovalRevision(id)
      }
    }
    let cast = clips?.characters ?? story?.characters ?? []
    var mapping: [String: UUID] = [:]
    for member in cast {
      let key = prefix + "cast:" + member.id
      if let id = inventoryIDs[member.id], let index = subjects.firstIndex(where: { $0.id == id }) {
        guard subjects[index].kind == .character else { throw StudioError.invalid("A shot character ID refers to a non-character object.") }
        mapping[member.id] = id
        if !subjects[index].sourceKeyAliases.contains(key) { subjects[index].sourceKeyAliases.append(key) }
        continue
      }
      if let old = subjects.first(where: { ($0.sourceKey == key || $0.sourceKeyAliases.contains(key)) }) { mapping[member.id] = old.id; continue }
      var s = PlanningSubject(name: member.id.replacingOccurrences(of: "_", with: " ").capitalized, kind: .character)
      s.sourceKey = key; s.details = member.description
      // Match only an explicit identical name/alias; never infer that two different people are one.
      let matches = subjects.filter { $0.kind == .character && ([$0.name] + $0.aliases.components(separatedBy: ",")).map { $0.trimmed.lowercased().replacingOccurrences(of: "_", with: " ") }.contains(member.id.lowercased().replacingOccurrences(of: "_", with: " ")) }
      if matches.count == 1 {
        mapping[member.id] = matches[0].id
        if let i = subjects.firstIndex(where: { $0.id == matches[0].id }), !subjects[i].sourceKeyAliases.contains(key) { subjects[i].sourceKeyAliases.append(key) }
      }
      else { subjects.append(s); mapping[member.id] = s.id }
    }
    if let clips {
      guard (1...120).contains(clips.fps), Set(clips.clips.map(\.id)).count == clips.clips.count else { throw StudioError.invalid("The workflow has invalid shot IDs or FPS.") }
      if !shots.isEmpty && frameRate != clips.fps { throw StudioError.invalid("The workflow FPS differs from this shot list. Use a separate project or a matching workflow.") }
      var cursor = 0
      for clip in clips.clips {
        guard clip.startFrame == cursor, (1...100_000_000).contains(clip.frameCount), clip.characters.allSatisfy({ mapping[$0] != nil }) else { throw StudioError.invalid("The workflow has invalid timing or unknown characters.") }
        cursor += clip.frameCount
      }
      guard cursor == clips.totalFrames else { throw StudioError.invalid("Shot timing does not cover the workflow movie.") }
      frameRate = clips.fps
      let bodies = sourceBodies(sourceText)
      for (index, clip) in clips.clips.enumerated() {
        let key = prefix + "shot:" + clip.id
        if shots.contains(where: { $0.sourceKey == key }) { continue }
        var shot = PlanningShot(name: "Shot \(index + 1)", frameCount: clip.frameCount)
        if let timing = musicTiming, let interval = timing.clips.first(where: { $0.clipID == clip.id }) {
          shot.engine = Engine(rawValue: timing.engine) ?? .ltx25; shot.task = timing.task
          shot.musicSource = MusicShotSource(path: timing.sourceAudio.path, sha256: timing.sourceAudio.sha256,
            start: interval.sourceStartSeconds, duration: interval.sourceDurationSeconds, task: timing.task)
          shot.musicSource?.sceneEligible = interval.sceneEligible ?? false
          shot.musicSource?.maximumGenerationSeconds = timing.bounds?.modelMaximumSeconds ?? timing.bounds?.maximumSeconds
        }
        shot.sourceKey = key; shot.action = clip.action; shot.firstFrame = clip.startState; shot.lastFrame = clip.endState
        shot.continuity = clip.continuity; shot.subjectIDs = clip.characters.compactMap { mapping[$0] }
        if let references = promptSubjects[clip.id] {
          for sourceID in references {
            if let id = inventoryIDs[sourceID], !shot.subjectIDs.contains(id) { shot.subjectIDs.append(id) }
          }
        }
        if let preview = promptPreview?.prompts.first(where: { $0.clipID == clip.id }) { shot.direction = preview.prompt }
        else if bodies.count == clips.clips.count { shot.direction = bodies[index] }
        if !clip.location.trimmed.isEmpty {
          let locationKey = prefix + "location:" + clip.location
          let locationKinds: Set<PlanningSubjectKind> = [.location, .environment, .set]
          let resolvedLocation = promptSubjects[clip.id] != nil && subjects.contains { locationKinds.contains($0.kind) && shot.subjectIDs.contains($0.id) }
          let candidates = subjects.filter { locationKinds.contains($0.kind) && ($0.sourceKey == locationKey ||
            ([$0.name] + $0.aliases.components(separatedBy: ",")).contains { $0.trimmed.caseInsensitiveCompare(clip.location.trimmed) == .orderedSame } || inventoryIDs[clip.location] == $0.id) }
          if candidates.count == 1, let loc = candidates.first {
            if !shot.subjectIDs.contains(loc.id) { shot.subjectIDs.append(loc.id) }
          } else if !resolvedLocation {
            var loc = PlanningSubject(name: clip.location, kind: .location)
            loc.sourceKey = locationKey; loc.details = clip.location
            subjects.append(loc); shot.subjectIDs.append(loc.id)
          }
        }
        shots.append(shot)
      }
    }
    if let musicTiming, musicTimings?[sourceID] == nil {
      if musicTimings == nil { musicTimings = [:] }; musicTimings?[sourceID] = musicTiming
      if musicAnalysis == nil { musicAnalysis = [:] }; musicAnalysis?[sourceID] = outputs["music_timing"]
    }
    if !importedSources.contains(sourceID) { importedSources.append(sourceID) }
  }
}

public struct WorkflowSubjectProposal: Codable, Identifiable, Equatable {
  public let id: String
  public var kind: PlanningSubjectKind
  public var name: String
  public let aliases: [String]
  public var description: String
  public let evidence: [String]
  public let suggestions: [String]
  public var descriptionReview: SubjectDescriptionReview?
  public var referenceAssets: [String]?
  public var referenceAssetKeys: [String] { referenceAssets ?? descriptionReview?.referenceAssets ?? [] }
  public var relationships: [WorkflowObjectRelationship]?
  public var relationshipReview: ObjectRelationshipReview?
  public var descriptionMentions: [DescriptionMention]?
  public var mentionSourceDescription: String?
  public var coverageReview: ObjectCoverageReview?
  public var reusedDefinition: ReusedDefinition?
  public var reusedOriginalDescription: String?
  public func editing(name: String, description: String) throws -> Self {
    guard !name.trimmed.isEmpty, !description.trimmed.isEmpty else {
      throw StudioError.invalid("Enter a subject name and description.")
    }
    var value = self; value.name = name; value.description = description
    if self.description != description { value.descriptionMentions = nil; value.mentionSourceDescription = nil }
    return value
  }
}

public struct SubjectDescriptionReview: Codable, Equatable {
  public var version: Int
  public var status: String
  public var reviewedDescription: String
  public var criteria: [String]
  public var proposedDetails: [String]
  public var issues: [String]
  public var referenceAssets: [String]
  public var referenceDetails: [String]?
  public func isReady(for description: String) -> Bool {
    status == "ready" && reviewedDescription == description
  }
}

public struct WorkflowSubjectGroup: Identifiable {
  public let kind: PlanningSubjectKind
  public let subjects: [WorkflowSubjectProposal]
  public var id: PlanningSubjectKind { kind }
  public var title: String { kind.groupTitle }
  public static func group(_ subjects: [WorkflowSubjectProposal], includeEmpty: Bool = true) -> [Self] {
    PlanningSubjectKind.reviewOrder.map { kind in
      Self(kind: kind, subjects: subjects.filter { $0.kind == kind }.sorted {
        let order = $0.name.localizedStandardCompare($1.name)
        return order == .orderedSame ? $0.id < $1.id : order == .orderedAscending
      })
    }.filter { includeEmpty || !$0.subjects.isEmpty }
  }
}

func planningDigest<T: Encodable>(_ value: T) -> String {
  let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
  guard let data = try? encoder.encode(value) else { return "invalid" }
  return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
private extension String { var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) } }

public extension StudioProject {
  /// Create a project link to an existing image; never duplicate the media file.
  mutating func attachPlanningReference(_ asset: MediaAsset, subjectID: UUID) throws -> UUID {
    guard asset.kind == .image, !asset.path.isEmpty,
          let index = planning?.subjects.firstIndex(where: { $0.id == subjectID }) else {
      throw StudioError.invalid("Select an image and an existing project subject.")
    }
    let existing = assets.first { $0.kind == .image && $0.scope == .project && ($0.id == asset.id || $0.path == asset.path) }
    var linked = existing ?? asset
    if existing == nil {
      linked.id = UUID(); linked.scope = .project; linked.owner = nil
      assets.append(linked)
    }
    if planning?.subjects[index].referenceAssetIDs.contains(linked.id) != true {
      planning?.subjects[index].referenceAssetIDs.append(linked.id)
    }
    return linked.id
  }
}
