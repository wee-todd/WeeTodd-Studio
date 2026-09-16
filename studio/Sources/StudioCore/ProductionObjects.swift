import Foundation

public enum ObjectRelationshipRole: String, Codable, CaseIterable, Identifiable {
  case contains, wears, holds, uses, located_in, part_of
  public var id: String { rawValue }
  public var label: String { rawValue.replacingOccurrences(of: "_", with: " ").capitalized }
}

/// Each relationship identifies a placement/use; several can target the same definition.
public struct ObjectRelationship: Codable, Equatable, Identifiable {
  public var id: UUID
  public var targetID: UUID
  public var role: ObjectRelationshipRole
  public var placement: String
  public init(id: UUID = UUID(), targetID: UUID, role: ObjectRelationshipRole, placement: String = "") {
    self.id = id; self.targetID = targetID; self.role = role; self.placement = placement
  }
}
public struct LibraryOrigin: Codable, Equatable {
  public var packageID: UUID
  public var version: Int
  public var definitionRevision: String
  public init(packageID: UUID, version: Int, definitionRevision: String) {
    self.packageID = packageID; self.version = version; self.definitionRevision = definitionRevision
  }
}
public struct ObjectRelationshipReview: Codable, Equatable {
  public var version: Int
  public var status: String
  public var reviewedDescription: String
  public var issues: [String]
  public var missingObjects: [String]
}
public struct WorkflowObjectRelationship: Codable, Equatable, Identifiable {
  public var id: String
  public var targetID: String
  public var role: ObjectRelationshipRole
  public var placement: String
  public init(id: String, targetID: String, role: ObjectRelationshipRole, placement: String) {
    self.id = id; self.targetID = targetID; self.role = role; self.placement = placement
  }
}
public struct ObjectStateOverride: Codable, Equatable, Identifiable {
  public var subjectID: UUID
  public var state: String
  public var id: UUID { subjectID }
  public init(subjectID: UUID, state: String) { self.subjectID = subjectID; self.state = state }
}
public struct ResolvedObjectSnapshot: Codable, Equatable {
  public var subjects: [PlanningSubject]
  public var assets: [MediaAsset]
  public var revision: String
  public var appearanceOverrides: [ObjectStateOverride]?
}

public extension ProjectPlanning {
  func subjectApprovalRevision(_ id: UUID) -> String? {
    guard let object = subjects.first(where: { $0.id == id }), let dependencies = try? resolvedObjects([id]) else { return nil }
    if dependencies.count == 1 { return object.revision }
    return planningDigest(dependencies.map { $0.id.uuidString + $0.revision })
  }
  /// Follow explicit links and set parents. Publishing an environment optionally includes its sets.
  func resolvedObjects(_ roots: [UUID], includeSets: Bool = false) throws -> [PlanningSubject] {
    guard Set(subjects.map(\.id)).count == subjects.count else { throw StudioError.invalid("Duplicate object IDs in this movie.") }
    let index = Dictionary(uniqueKeysWithValues: subjects.map { ($0.id, $0) })
    var visited = Set<UUID>(); var result: [PlanningSubject] = []
    func visit(_ id: UUID) throws {
      guard visited.insert(id).inserted else { return }
      guard let object = index[id] else { throw StudioError.invalid("A linked object is missing: \(id).") }
      if object.kind == .set && object.environmentID == nil { throw StudioError.invalid("Choose an environment for the set “\(object.name)”.") }
      if let parent = object.environmentID {
        guard object.kind == .set, index[parent]?.kind == .environment else { throw StudioError.invalid("“\(object.name)” needs a valid environment parent.") }
        try visit(parent)
      }
      let links = object.relationships ?? []
      guard links.count <= 32, Set(links.map(\.id)).count == links.count else { throw StudioError.invalid("Use at most 32 distinct placements per object.") }
      for link in links {
        guard link.targetID != id, link.placement.count <= 300 else { throw StudioError.invalid("An object cannot reference itself; placement notes are limited to 300 characters.") }
        try visit(link.targetID)
      }
      result.append(object)
      if includeSets && roots.contains(id) && object.kind == .environment {
        for set in subjects where set.kind == .set && set.environmentID == id { try visit(set.id) }
      }
    }
    for id in roots { try visit(id) }
    return result.sorted { $0.id.uuidString < $1.id.uuidString }
  }
  func resolvedSnapshot(_ roots: [UUID], assets: [MediaAsset], requireApproval: Bool = true) throws -> ResolvedObjectSnapshot {
    let objects = try resolvedObjects(roots)
    if requireApproval {
      for object in objects {
        guard isSubjectApproved(object.id) else { throw StudioError.invalid("Approve \(object.name) before using it for generation.") }
        if !object.referenceAssetIDs.isEmpty && !areReferencesApproved(object.id, assets: assets) {
          throw StudioError.invalid("Approve the reference images for \(object.name).")
        }
      }
    }
    let ids = Set(objects.flatMap(\.referenceAssetIDs))
    let references = assets.filter { ids.contains($0.id) }.sorted { $0.id.uuidString < $1.id.uuidString }
    guard Set(references.map(\.id)) == ids else { throw StudioError.invalid("Some object reference images are missing from the asset stores.") }
    return ResolvedObjectSnapshot(subjects: objects, assets: references,
      revision: planningDigest(objects.map { $0.id.uuidString + $0.revision } + references.map { planningDigest($0) }))
  }
  func resolvedShotSnapshot(_ shotID: UUID, assets: [MediaAsset], requireApproval: Bool = true) throws -> ResolvedObjectSnapshot {
    guard let shot = shots.first(where: { $0.id == shotID }) else { throw StudioError.invalid("Shot not found.") }
    var snapshot = try resolvedSnapshot(shot.subjectIDs, assets: assets, requireApproval: requireApproval)
    let overrides = shot.appearanceOverrides ?? []
    let ids = Set(snapshot.subjects.map(\.id))
    guard Set(overrides.map(\.subjectID)).count == overrides.count,
          overrides.allSatisfy({ ids.contains($0.subjectID) && $0.state.count <= 2000 }) else {
      throw StudioError.invalid("Remove appearance overrides for unlinked objects or shorten them to 2,000 characters.")
    }
    snapshot.appearanceOverrides = shot.appearanceOverrides
    snapshot.revision = planningDigest([snapshot.revision, planningDigest(shot.appearanceOverrides ?? [])])
    return snapshot
  }
  mutating func removeSubject(_ id: UUID) throws {
    guard !shots.contains(where: { $0.subjectIDs.contains(id) }),
          !subjects.contains(where: { $0.id != id && ($0.environmentID == id || ($0.relationships ?? []).contains(where: { $0.targetID == id })) }) else {
      throw StudioError.invalid("This object is used by a shot or another object. Remove those links first.")
    }
    subjects.removeAll { $0.id == id }
  }
}

public struct PlanningExport: Codable {
  public var format = "weetodd-shot-list-v2"
  public var planning: ProjectPlanning
  public var referenceAssets: [MediaAsset]
  public var resolvedShots: [String: ResolvedObjectSnapshot]
  public var issues: [String]
}
public extension ProjectPlanning {
  func exportDocument(assets: [MediaAsset]) -> PlanningExport {
    let referenceIDs = Set(subjects.flatMap(\.referenceAssetIDs) + shots.flatMap { $0.referenceAssetIDs + [$0.firstAssetID, $0.lastAssetID].compactMap { $0 } })
    let references = assets.filter { referenceIDs.contains($0.id) }
    var snapshots: [String: ResolvedObjectSnapshot] = [:]; var messages: [String] = []
    for shot in shots {
      do { snapshots[shot.id.uuidString] = try resolvedShotSnapshot(shot.id, assets: assets, requireApproval: false) }
      catch { messages.append(shot.name + ": " + error.localizedDescription) }
    }
    if Set(references.map(\.id)) != referenceIDs { messages.append("Some reference assets are missing from the movie’s asset stores.") }
    return PlanningExport(planning: self, referenceAssets: references, resolvedShots: snapshots, issues: messages)
  }
}
