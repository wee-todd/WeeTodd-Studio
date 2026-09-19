import AppKit
import StudioCore
import SwiftUI

extension StudioStore {
  var planning: ProjectPlanning { project.planning ?? ProjectPlanning() }
  func changePlanning(_ body: (inout ProjectPlanning) -> Void) {
    change { project in
      var plan = project.planning ?? ProjectPlanning()
      body(&plan); project.planning = plan
    }
  }
  func importPlanning(_ run: WorkflowRunSummary, sourceID: String, sourceText: String, referenceBindings: [String: String] = [:]) throws {
    let existingIDs = Set(planning.subjects.map(\.id))
    var value = planning
    try value.importRun(run, sourceID: sourceID, sourceText: sourceText)
    var updated = project; updated.planning = value
    updated.retainPlanningImages(referenceBindings)
    for subject in value.subjects where !existingIDs.contains(subject.id) && subject.sourceKey.hasPrefix(sourceID + ":subject:") {
      for key in (try run.subjectsForImport()).first(where: { subject.sourceKey.hasSuffix(":" + $0.id) })?.referenceAssetKeys ?? [] {
        guard let path = referenceBindings[key] else { continue }
        let url = URL(fileURLWithPath: path)
        _ = try updated.attachPlanningReference(MediaAsset(name: url.deletingPathExtension().lastPathComponent,
          kind: .image, path: path), subjectID: subject.id)
      }
    }
    change { $0 = updated }
  }
  func linkPlanningReference(_ asset: MediaAsset, subjectID: UUID) {
    do {
      var updated = project
      _ = try updated.attachPlanningReference(asset, subjectID: subjectID)
      change { $0 = updated }
    } catch { self.error = error.localizedDescription }
  }
  func importPlanningReferences(subjectID: UUID) {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
    guard panel.runModal() == .OK else { return }
    for url in panel.urls {
      if let existing = project.assets.first(where: { $0.kind == .image && $0.path == url.path && $0.scope == .project }) {
        linkPlanningReference(existing, subjectID: subjectID)
      } else {
        linkPlanningReference(MediaAsset(name: url.deletingPathExtension().lastPathComponent, kind: .image, path: url.path), subjectID: subjectID)
      }
    }
  }
}

extension StudioStore {
  func objectCatalog(matching text: String) throws -> [LibraryObjectCandidate] {
    let terms = Set(text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).filter { $0.count >= 3 }.map(String.init))
    let projectCandidates = planning.subjects.filter { object in
      let words = Set(([object.name, object.aliases] + (object.tags ?? [])).joined(separator: " ").lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init))
      return !object.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !object.details.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !words.intersection(terms).isEmpty
    }.compactMap { object -> LibraryObjectCandidate? in
      guard let revision = try? planning.reusableObjectRevision(object.id, assets: allAssets) else { return nil }
      var candidate = LibraryObjectCandidate(subject: object, packageID: project.id, version: 1, scope: "project")
      candidate.definitionRevision = revision; return candidate
    }
    let databaseURL = Self.supportDirectory.appendingPathComponent("ProductionLibrary/catalog.sqlite")
    let global = FileManager.default.fileExists(atPath: databaseURL.path) ? try ProductionLibrary(url: databaseURL).candidates(matching: text) : []
    var seen = Set<String>()
    return Array((projectCandidates + global).filter { seen.insert($0.id).inserted }.prefix(64))
  }
  func importPlanning(_ run: WorkflowRunSummary, sourceID: String, sourceText: String,
                      referenceBindings: [String: String], librarySelections: [String: LibraryObjectMatch]) throws {
    let previous = project
    var staged = project; var plan = planning
    try plan.importRun(run, sourceID: sourceID, sourceText: sourceText); staged.planning = plan
    staged.retainPlanningImages(referenceBindings)
    let sourceSubjects = try run.subjectsForImport()
    for (workflowID, match) in librarySelections {
      guard sourceSubjects.first(where: { $0.id == workflowID })?.validatesReuse(match) == true else {
        throw StudioError.invalid("A reusable match belongs to an earlier object description. Review the match again before import.")
      }
      guard let objectID = UUID(uuidString: match.objectID), let packageID = UUID(uuidString: match.packageID),
            let source = staged.planning?.subjects.first(where: { $0.sourceKey.hasPrefix(sourceID + ":subject:") && $0.sourceKey.hasSuffix(":" + workflowID) }) else {
        throw StudioError.invalid("The selected reusable object no longer matches this workflow inventory.")
      }
      guard previous.planning?.subjects.contains(where: { $0.id == source.id }) != true else {
        throw StudioError.invalid("This workflow object already exists in the movie. Its existing definition was preserved.")
      }
      if match.scope == "project" {
        guard packageID == staged.id, let target = staged.planning?.subjects.first(where: { $0.id == objectID }),
              match.definitionRevision == (try staged.planning?.reusableObjectRevision(target.id, assets: staged.assets + globalAssets)) else { throw StudioError.invalid("The chosen movie object changed. Review its match again.") }
      } else {
        let library = try ProductionLibrary(url: Self.supportDirectory.appendingPathComponent("ProductionLibrary/catalog.sqlite"))
        let package = try library.package(rootID: packageID, version: match.version)
        guard let object = package.subjects.first(where: { $0.id == objectID }), match.definitionRevision == object.revision else { throw StudioError.invalid("The chosen library definition is unavailable or changed.") }
        // Reuse only this object's dependency closure, rather than importing unrelated sibling sets.
        var sourcePlan = ProjectPlanning(); sourcePlan.subjects = package.subjects
        let subjects = try sourcePlan.resolvedObjects([objectID])
        let mediaIDs = Set(subjects.flatMap(\.referenceAssetIDs))
        var subset = package; subset.subjects = subjects; subset.rootID = objectID
        subset.assets = package.assets.filter { mediaIDs.contains($0.id) }
        let existingIDs = Set(staged.planning?.subjects.map(\.id) ?? [])
        try staged.importLibraryPackage(subset)
        for object in subjects where !existingIDs.contains(object.id) {
          if let index = staged.planning?.subjects.firstIndex(where: { $0.id == object.id }) {
            staged.planning?.subjects[index].libraryOrigin = LibraryOrigin(packageID: packageID, version: match.version, definitionRevision: object.revision)
          }
        }
      }
      try staged.reuseLibraryObject(replacing: source.id, targetID: objectID)
      // A newly created reference is a movie-specific addition, including when the
      // appearance came from a pinned library definition. Do not modify the library.
      for key in sourceSubjects.first(where: { $0.id == workflowID })?.referenceAssetKeys ?? [] {
        guard let path = referenceBindings[key] else { continue }
        _ = try staged.attachPlanningReference(MediaAsset(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
          kind: .image, path: path), subjectID: objectID)
      }
    }
    // Attach observed references only to newly imported non-reused descriptions.
    for subject in staged.planning?.subjects ?? [] where previous.planning?.subjects.contains(where: { $0.id == subject.id }) != true && subject.sourceKey.hasPrefix(sourceID + ":subject:") {
      for key in (try run.subjectsForImport()).first(where: { subject.sourceKey.hasSuffix(":" + $0.id) })?.referenceAssetKeys ?? [] {
        guard let path = referenceBindings[key] else { continue }
        _ = try staged.attachPlanningReference(MediaAsset(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent, kind: .image, path: path), subjectID: subject.id)
      }
    }
    change { $0 = staged }
  }
}
