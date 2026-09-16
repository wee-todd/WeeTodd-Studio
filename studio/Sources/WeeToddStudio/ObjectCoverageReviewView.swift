import StudioCore
import SwiftUI

struct ObjectCoverageReviewView: View {
  var report: ObjectCoverageReview?
  var candidates: [LibraryObjectCandidate]
  var selectedMatch: LibraryObjectMatch?
  var locked: Bool
  var onSelect: (LibraryObjectMatch?) -> Void
  var onCreate: (MissingObjectProposal) -> Void
  @State private var created = Set<String>()

  var body: some View {
    if let report {
      VStack(alignment: .leading, spacing: 10) {
        Text("Object coverage review").font(.headline)
        if !report.issues.isEmpty { SubjectSourceText(title: "Needs attention", entries: report.issues) }
        ForEach(Array(report.missingObjects.enumerated()), id: \.offset) { _, object in
          VStack(alignment: .leading, spacing: 4) {
            Text("Suggested \(object.kind.label): \(object.name)").font(.subheadline.bold())
            Text(object.description).font(.caption).textSelection(.enabled)
            SubjectSourceText(title: "Source evidence", entries: object.evidence)
            Button(created.contains(object.name) ? "Draft added to movie" : "Create draft in movie") {
              onCreate(object); created.insert(object.name)
            }.disabled(created.contains(object.name))
          }.padding(8).background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
        }
        ForEach(report.libraryMatches) { match in
          let candidate = candidates.first { $0.id == match.objectID && $0.packageID == match.packageID && $0.version == match.version && $0.definitionRevision == match.definitionRevision && ($0.scope ?? "global") == (match.scope ?? "global") }
          VStack(alignment: .leading, spacing: 5) {
            Text("Reusable match: " + (candidate?.name ?? match.objectID)).font(.subheadline.bold())
            Text(match.reason).font(.caption)
            Text(candidate?.description ?? "The candidate snapshot is unavailable. Run coverage review again.")
              .font(.caption).textSelection(.enabled)
            Text((match.scope == "project" ? "Movie object" : "Global library · version \(match.version)") + " · " + match.objectID)
              .font(.caption2).foregroundStyle(.secondary)
            Button(selectedMatch?.id == match.id ? "Keep extracted definition instead" : "Use this definition") {
              onSelect(selectedMatch?.id == match.id ? nil : match)
            }.disabled(locked || candidate == nil)
            if selectedMatch?.id == match.id {
              Text("Add to project will reuse this ID and its references. Other uses of the extracted object will follow it.")
                .font(.caption).foregroundStyle(.secondary)
            }
          }.padding(8).background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 6))
        }
        Text("Clear links were applied to drafts. These remaining suggestions are not approvals.").font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}
