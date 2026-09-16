import Foundation

public struct DescriptionMention: Codable, Equatable, Identifiable {
  public var id: String
  public var targetID: String
  public var phrase: String
  public var occurrence: Int
  public init(id: String, targetID: String, phrase: String, occurrence: Int) {
    self.id = id; self.targetID = targetID; self.phrase = phrase; self.occurrence = occurrence
  }
}
public struct MissingObjectProposal: Codable, Equatable {
  public var name: String
  public var kind: PlanningSubjectKind
  public var description: String
  public var evidence: [String]
}
public struct LibraryObjectMatch: Codable, Equatable, Identifiable {
  public var objectID: String
  public var packageID: String
  public var version: Int
  public var reason: String
  public var scope: String?
  public var definitionRevision: String?
  public var sourceRevision: String?
  public var id: String { (scope ?? "global") + ":" + packageID + ":" + String(version) + ":" + objectID }
}
public struct ReusedDefinition: Codable, Equatable {
  public var objectID: String
  public var packageID: String
  public var version: Int
  public var scope: String?
  public var definitionRevision: String
  public func matches(_ value: LibraryObjectMatch) -> Bool {
    objectID == value.objectID && packageID == value.packageID && version == value.version &&
      (scope ?? "global") == (value.scope ?? "global") && definitionRevision == value.definitionRevision
  }
  public var libraryMatch: LibraryObjectMatch {
    LibraryObjectMatch(objectID: objectID, packageID: packageID, version: version,
      reason: "Selected definition used for this design", scope: scope, definitionRevision: definitionRevision)
  }
}
public struct ObjectCoverageReview: Codable, Equatable {
  public var version: Int
  public var status: String
  public var reviewedDescription: String
  public var issues: [String]
  public var missingObjects: [MissingObjectProposal]
  public var libraryMatches: [LibraryObjectMatch]
}
public struct LibraryObjectCandidate: Codable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var kind: PlanningSubjectKind
  public var aliases: [String]
  public var tags: [String]
  public var description: String
  public var packageID: String
  public var version: Int
  public var definitionRevision: String
  public var scope: String?
  public init(subject: PlanningSubject, packageID: UUID, version: Int, scope: String) {
    id = subject.id.uuidString; name = String(subject.name.prefix(200)); kind = subject.kind
    aliases = subject.aliases.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.prefix(8).map { String($0.prefix(1000)) }
    tags = (subject.tags ?? []).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.prefix(32).map { String($0.prefix(200)) }; description = String(subject.details.prefix(2000))
    self.packageID = packageID.uuidString; self.version = version
    definitionRevision = subject.revision; self.scope = scope
  }
}
public struct ObjectCoverageProposalReport: Codable {
  public var version: Int
  public var proposals: [WorkflowSubjectProposal]
}

public extension StudioProject {
  /// Explicit library reuse redirects references while retaining source-key import aliases.
  /// Only newly imported draft workflow records may be replaced; old movie edits are protected.
  mutating func reuseLibraryObject(replacing sourceID: UUID, targetID: UUID) throws {
    guard sourceID != targetID, var plan = planning,
          let source = plan.subjects.first(where: { $0.id == sourceID }),
          let targetIndex = plan.subjects.firstIndex(where: { $0.id == targetID }) else { throw StudioError.invalid("Choose two available object definitions.") }
    guard source.kind == plan.subjects[targetIndex].kind else { throw StudioError.invalid("The library object has a different type.") }
    let extraKeys = [source.sourceKey] + source.sourceKeyAliases
    plan.subjects[targetIndex].sourceKeyAliases = Array(Set(plan.subjects[targetIndex].sourceKeyAliases + extraKeys).filter { !$0.isEmpty }).sorted()
    for i in plan.subjects.indices where plan.subjects[i].id != sourceID {
      if plan.subjects[i].environmentID == sourceID { plan.subjects[i].environmentID = targetID }
      plan.subjects[i].relationships = plan.subjects[i].relationships?.compactMap { link in
        var value = link; if value.targetID == sourceID { value.targetID = targetID }
        return value.targetID == plan.subjects[i].id ? nil : value
      }
      plan.subjects[i].descriptionMentions = plan.subjects[i].descriptionMentions?.map { mention in
        var value = mention; if value.targetID == sourceID.uuidString { value.targetID = targetID.uuidString }; return value
      }
    }
    for i in plan.shots.indices {
      var seen = Set<UUID>()
      plan.shots[i].subjectIDs = plan.shots[i].subjectIDs.map { $0 == sourceID ? targetID : $0 }.filter { seen.insert($0).inserted }
      if let override = plan.shots[i].appearanceOverrides?.first(where: { $0.subjectID == sourceID }) {
        guard plan.shots[i].appearanceOverrides?.contains(where: { $0.subjectID == targetID }) != true else { throw StudioError.invalid("Resolve duplicate shot appearance overrides before library reuse.") }
        plan.shots[i].appearanceOverrides?.removeAll { $0.subjectID == sourceID }
        plan.shots[i].appearanceOverrides?.append(ObjectStateOverride(subjectID: targetID, state: override.state))
      }
    }
    plan.subjects.removeAll { $0.id == sourceID }
    _ = try plan.resolvedObjects([targetID])
    planning = plan
  }
}

public extension WorkflowSubjectProposal {
  var reuseSelectionRevision: String {
    var values = [name, kind.rawValue, description, planningDigest(aliases), planningDigest(relationships), planningDigest(descriptionMentions)]
    if let reusedDefinition { values.append(planningDigest(reusedDefinition)) }
    return planningDigest(values)
  }
  func validatesReuse(_ match: LibraryObjectMatch) -> Bool {
    match.sourceRevision == reuseSelectionRevision && (reusedDefinition?.matches(match) == true || coverageReview?.libraryMatches.contains(where: {
      $0.id == match.id && $0.definitionRevision == match.definitionRevision
    }) == true)
  }
}
public extension WorkflowRunSummary {
  var preferredSubjectStepID: String? {
    let order = [awaitingStep].compactMap { $0 } + ["subjects_coverage", "design", "subjects_review", "inventory", "links", "subjects_links", "classify", "subjects"] + steps.keys.sorted().reversed()
    return order.first { steps[$0]?.status == "completed" && steps[$0]?.outputs?["subjects"] != nil }
  }
  func subjectsForImport() throws -> [WorkflowSubjectProposal] {
    let value = outputs["subjects"] ?? steps[preferredSubjectStepID ?? ""]?.outputs?["subjects"]
    guard let value else { return [] }
    return try JSONDecoder().decode([WorkflowSubjectProposal].self, from: JSONEncoder().encode(value))
  }
}

public extension ProjectPlanning {
  func reusableObjectRevision(_ id: UUID, assets: [MediaAsset]) throws -> String {
    let snapshot = try resolvedSnapshot([id], assets: assets, requireApproval: false)
    let files = snapshot.assets.map { asset -> String in
      let attributes = try? FileManager.default.attributesOfItem(atPath: asset.path)
      return asset.path + ":" + ((attributes?[.size] as? NSNumber)?.stringValue ?? "missing") + ":" + String((attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
    }
    return planningDigest([snapshot.revision, planningDigest(files)])
  }
}
