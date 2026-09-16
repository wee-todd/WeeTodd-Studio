import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct WorkflowSubjectReviewView: View {
  @Binding var draft: DirectorReviewDraft
  var bindReferences: ([String]) throws -> [String]
  let subjects: [WorkflowSubjectProposal]
  let step: WorkflowRunSummary.Step
  let referenceBindings: [String: String]
  let onDescriptionReview: ((String, [String]) async -> Bool)?
  var onReferenceSave: ((String, [String]) async -> Bool)? = nil
  var subjectScope = ""
  var onCoverageReview: (() async -> Bool)? = nil
  var libraryCandidates: [LibraryObjectCandidate] = []
  var librarySelections: [String: LibraryObjectMatch] = [:]
  var onLibrarySelect: (String, LibraryObjectMatch?) -> Void = { _, _ in }
  var onCreateObject: (String, MissingObjectProposal) -> Void = { _, _ in }
  var allowKindEditing = false
  var approvalLabel = "description"
  var individualApprovals = true
  let onSave: (WorkflowSubjectProposal) async -> Bool
  let onApproval: (String, Bool) async -> Bool
  private var selectedSubject: String? { get { draft.selectedSubject } nonmutating set { draft.selectedSubject = newValue } }
  @State private var expandedKinds = Set(PlanningSubjectKind.reviewOrder)
  private var drafts: [String: WorkflowSubjectProposal] { get { draft.subjects } nonmutating set { draft.subjects = newValue } }
  private var referencePaths: [String: [String]] { get { draft.referencePaths } nonmutating set { draft.referencePaths = newValue } }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if let onCoverageReview {
        Button("Find missing object links & reusable matches") { Task { _ = await onCoverageReview() } }
          .disabled((!drafts.isEmpty || referencesChanged) || step.status != "completed")
        if !drafts.isEmpty || referencesChanged { Text("Save your description edits before checking coverage.").font(.caption).foregroundStyle(.secondary) }
      }
    HSplitView {
      List(selection: Binding(get: { selectedSubject }, set: { selectedSubject = $0 })) {
        ForEach(WorkflowSubjectGroup.group(subjects, includeEmpty: individualApprovals)) { group in
          DisclosureGroup(isExpanded: Binding(get: { expandedKinds.contains(group.kind) }, set: { value in
            if value { expandedKinds.insert(group.kind) } else { expandedKinds.remove(group.kind) }
          })) {
            ForEach(group.subjects) { subject in
              HStack {
                Text(subject.name).lineLimit(2)
                Spacer()
                if drafts[subject.id] != nil || referencePaths[subject.id].map({ $0 != subject.referenceAssetKeys.compactMap { referenceBindings[$0] } }) == true { Text("Unsaved").font(.caption2).foregroundStyle(.secondary) }
                if step.approved == true || step.items?[subject.id]?.approved == true {
                  Image(systemName: "checkmark.seal.fill").foregroundStyle(.green).accessibilityLabel("Approved")
                }
              }.tag(subject.id)
            }
          } label: { Text("\(group.title) (\(group.subjects.count))").font(.headline) }
        }
      }.listStyle(.sidebar).frame(minWidth: 230, idealWidth: 270, maxWidth: 320)
      if let subject = subjects.first(where: { $0.id == selectedSubject }) {
        ScrollView {
          WorkflowSubjectCard(subject: subject,
            coverageProposal: step.coverageReviewReport?.proposals.first(where: { $0.id == subject.id }),
            libraryCandidates: libraryCandidates, selectedLibraryMatch: librarySelections[subject.id],
            onLibrarySelect: { onLibrarySelect(subject.id, $0) }, onCreateObject: { onCreateObject(subject.id, $0) },
            inventory: subjects.map { drafts[$0.id] ?? $0 }, onNavigate: { selectedSubject = $0 },
            approved: step.approved == true || step.items?[subject.id]?.approved == true,
            editable: step.status == "completed", allowKindEditing: allowKindEditing, approvalLabel: approvalLabel, individualApprovals: individualApprovals, onDescriptionReview: onDescriptionReview,
            onReferenceSave: onReferenceSave, subjectScope: subjectScope,
            savedReferencePaths: subject.referenceAssetKeys.compactMap { referenceBindings[$0] },
            onSave: { edited in
              var value = edited
              guard let references = try? bindReferences(referencePaths[edited.id] ?? subject.referenceAssetKeys.compactMap { referenceBindings[$0] }) else { return false }
              value.referenceAssets = references
              let saved = await onSave(value)
              if saved { drafts.removeValue(forKey: edited.id); referencePaths.removeValue(forKey: edited.id) }
              return saved
            }, onApproval: onApproval, name: draftBinding(subject, \.name), kind: draftBinding(subject, \.kind),
            description: draftBinding(subject, \.description), relationships: draftBinding(subject, \.relationships), imagePaths: Binding(get: {
              referencePaths[subject.id] ?? subject.referenceAssetKeys.compactMap { referenceBindings[$0] }
            }, set: { referencePaths[subject.id] = $0 }))
            .id(subject.id).padding(.horizontal, 12)
        }.frame(minWidth: 330, maxWidth: .infinity)
      } else {
        Text(subjects.isEmpty ? "No subjects identified." : "Select a subject to review.")
          .foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }
    }.onAppear { if selectedSubject == nil { selectedSubject = WorkflowSubjectGroup.group(subjects).flatMap(\.subjects).first?.id } }
  }
  private var referencesChanged: Bool {
    subjects.contains { subject in referencePaths[subject.id].map { $0 != subject.referenceAssetKeys.compactMap { referenceBindings[$0] } } ?? false }
  }
  private func draftBinding<T>(_ subject: WorkflowSubjectProposal,
                            _ key: WritableKeyPath<WorkflowSubjectProposal, T>) -> Binding<T> {
    Binding(get: { (drafts[subject.id] ?? subject)[keyPath: key] }, set: { value in
      var updated = drafts[subject.id] ?? subject; updated[keyPath: key] = value
      drafts[subject.id] = updated == subject ? nil : updated
    })
  }
}

private struct WorkflowSubjectCard: View {
  let subject: WorkflowSubjectProposal
  var coverageProposal: WorkflowSubjectProposal?
  var libraryCandidates: [LibraryObjectCandidate]
  var selectedLibraryMatch: LibraryObjectMatch?
  var onLibrarySelect: (LibraryObjectMatch?) -> Void
  var onCreateObject: (MissingObjectProposal) -> Void
  let inventory: [WorkflowSubjectProposal]
  let onNavigate: (String) -> Void
  let approved: Bool
  let editable: Bool
  let allowKindEditing: Bool
  let approvalLabel: String
  var individualApprovals = true
  let onDescriptionReview: ((String, [String]) async -> Bool)?
  var onReferenceSave: ((String, [String]) async -> Bool)? = nil
  var subjectScope = ""
  let savedReferencePaths: [String]
  let onSave: (WorkflowSubjectProposal) async -> Bool
  let onApproval: (String, Bool) async -> Bool
  @Binding var name: String
  @Binding var kind: PlanningSubjectKind
  @Binding var description: String
  @Binding var relationships: [WorkflowObjectRelationship]?
  @Binding var imagePaths: [String]
  @State private var error: String?
  @State private var saving = false
  @State private var sheetContext: ReferenceSheetContext?
  private var changed: Bool { name != subject.name || kind != subject.kind || description != subject.description || relationships != subject.relationships || imagePaths != savedReferencePaths }
  private var definitionLocked: Bool { approved || !editable || subject.reusedDefinition != nil }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      TextField("Name", text: $name).font(.headline).textFieldStyle(.roundedBorder)
        .accessibilityLabel("Subject name").disabled(definitionLocked)
      DisclosureGroup("Subject details") {
        Text("ID: \(subject.id)").font(.system(.caption2, design: .monospaced)).textSelection(.enabled)
      }.font(.caption).foregroundStyle(.secondary)
      if allowKindEditing {
        Picker("Object type", selection: $kind) {
          ForEach(PlanningSubjectKind.reviewOrder) { item in Text(item == .location ? "Location · unresolved" : item.label).tag(item) }
        }.disabled(definitionLocked)
        if kind == .set {
          Text("Link this set to its environment using Located in or Part of. Its ID stays the same when you correct its type.")
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      if !subject.aliases.isEmpty {
        Text("Also known as: \(subject.aliases.joined(separator: ", "))").font(.caption)
      }
      if !individualApprovals, attention.hasNotes {
        Label(attention.summary, systemImage: attention.needsAttention ? "exclamationmark.triangle" : "info.circle")
          .font(.caption).foregroundStyle(attention.needsAttention ? Color.orange : Color.secondary)
        if let issue = attention.issues.first {
          Text(issue).font(.caption).foregroundStyle(.secondary).lineLimit(2).help(issue)
        }
        if attention.hasOutdatedNotes || subject.descriptionReview.map({ $0.referenceAssets != subject.referenceAssetKeys || imagePaths != savedReferencePaths }) == true {
          Text("Some assistant notes refer to an earlier description or reference selection.").font(.caption).foregroundStyle(.orange)
        }
      }
      Text(approvalLabel == "classification" && subject.descriptionReview == nil ? "Source summary" : "Description")
        .font(.subheadline.bold())
      if approvalLabel == "classification" && subject.descriptionReview == nil {
        Text("Known details from your story and answers. Review the description below to propose a fuller appearance, or continue to Review visual designs after classification and reuse.")
          .font(.caption).foregroundStyle(.secondary)
      }
      if subject.reusedDefinition != nil {
        Text("Using the selected reusable definition. Its appearance is preserved during enrichment.").font(.caption).foregroundStyle(.secondary)
        Button("Use original extracted definition") { onLibrarySelect(nil) }.disabled(approved || !editable || changed)
      }
      LinkedDescriptionEditor(text: $description, targets: descriptionTargets,
        mentions: subject.descriptionMentions ?? [], sourceDescription: subject.mentionSourceDescription,
        editable: !definitionLocked, accessibilityLabel: "Subject description", onNavigate: onNavigate)
        .frame(height: 150)
      Text("Hover over a linked name to read its description. Command-click while editing to open it.")
        .font(.caption).foregroundStyle(.secondary)
      linkedObjects
      if individualApprovals { coverageNotes }
      Text("Reference images").font(.subheadline.bold())
      if onReferenceSave != nil {
        Button("Create reference…") {
          let linked = Set((relationships ?? []).map(\.targetID))
          sheetContext = ReferenceSheetContext(subjectKey: subjectScope + ":" + subject.id,
            name: name, kind: kind, description: description,
            linkedDefinitions: inventory.filter { linked.contains($0.id) }.map { "\($0.id) · \($0.name): \($0.description)" }.joined(separator: "\n"))
        }.disabled(changed || !editable || imagePaths.count >= 8)
        Text("Using a new reference requires renewed review; your saved description is preserved.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      ScrollView(.horizontal) {
        HStack(alignment: .top, spacing: 10) {
          ForEach(ImagePreviewSelection.references(paths: imagePaths, subject: name)) { item in
            VStack {
              PreviewableImage(path: item.path, title: item.title).frame(width: 100, height: 76).clipped()
              Text(item.title).font(.caption2).lineLimit(2).help(item.path)
              HStack {
                Button { moveReference(item.path, by: -1) } label: { Image(systemName: "arrow.left") }.help("Move reference earlier").accessibilityLabel("Move reference earlier").disabled(approved || imagePaths.first == item.path)
                Button { moveReference(item.path, by: 1) } label: { Image(systemName: "arrow.right") }.help("Move reference later").accessibilityLabel("Move reference later").disabled(approved || imagePaths.last == item.path)
              }
              Button("Remove") { imagePaths.removeAll { ImagePreviewSelection(path: $0).id == item.id } }.disabled(approved)
            }.frame(width: 110)
          }
        }
      }
      Button("Add reference images…") {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        panel.message = "Choose images that show this subject. Qwen uses visible details to supplement the script."
        if panel.runModal() == .OK {
          let paths = imagePaths + panel.urls.map(\.path).filter { !imagePaths.contains($0) }
          if paths.count <= 8 { imagePaths = paths } else { error = "Use at most eight images per subject." }
        }
      }.disabled(approved || !editable)
      Text("Optional · Qwen3.5 4B can inspect these images locally. Script/image conflicts are flagged for your review.")
        .font(.caption).foregroundStyle(.secondary)
      if let onDescriptionReview {
        Button(subject.descriptionReview == nil ? "Review & improve description" : "Review description again") { Task {
          saving = true; error = nil
          if !(await onDescriptionReview(subject.id, imagePaths)) { error = "Review did not finish. See the workflow error below." }
          saving = false
        } }.disabled(changed || definitionLocked)
      }
      if individualApprovals { descriptionNotes }
      else if attention.hasNotes {
        DisclosureGroup("Assistant notes") {
          VStack(alignment: .leading, spacing: 12) {
            descriptionNotes
            relationshipNotes
            coverageNotes
            if !subject.suggestions.isEmpty { SubjectSourceText(title: "Suggestions", entries: subject.suggestions) }
          }.padding(.vertical, 6)
        }.font(.subheadline)
      }
      Text(!individualApprovals ? "Assistant notes are advisory; approval is yours." : approvalLabel == "description" ? "Agent notes are advisory. Approve your description when you are satisfied; another agent rewrite is optional." : "Review the object's identity, type and links here. Detailed appearance is developed later in Review visual designs, with a separate approval.")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        Label(approved ? "Approved" : "Needs review", systemImage: approved ? "checkmark.seal.fill" : "pencil")
          .font(.caption).foregroundStyle(approved ? .green : .secondary)
        Spacer()
        Button("Save changes") { Task {
          saving = true; error = nil
          defer { saving = false }
          do {
            if !(await onSave(try editedSubject())) {
              error = "Could not save. See the workflow error below."
            }
          } catch { self.error = error.localizedDescription }
        } }.disabled(!changed || approved || !editable)
      }
      if individualApprovals || approved {
      Button(approved ? "Unlock \(approvalLabel)" : (changed ? "Save & approve \(approvalLabel)" : "Approve \(approvalLabel)")) {
        Task { await approveDescription() }
      }.disabled(!editable || (!approved &&
        (name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
         description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)))
      }
      if changed { Text(individualApprovals ? "Approval saves and locks your edits." : "Save edits, then approve all subjects above.").font(.caption).foregroundStyle(.secondary) }
      if individualApprovals {
        SubjectSourceText(title: "Evidence", entries: subject.evidence)
        SubjectSourceText(title: "Suggestions", entries: subject.suggestions)
      } else if !subject.evidence.isEmpty {
        DisclosureGroup("Source evidence · \(subject.evidence.count)") {
          SubjectSourceText(title: "", entries: subject.evidence)
        }.font(.subheadline)
      }
      if let error { Text(error).foregroundStyle(.red).font(.caption) }
    }.padding(12).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
      .disabled(saving)
      .sheet(item: $sheetContext) { context in
      ReferenceSheetGenerator(context: context, referencePaths: imagePaths) { asset in
        guard let onReferenceSave else { return false }
        let paths = imagePaths.contains(asset.path) ? imagePaths : imagePaths + [asset.path]
        guard paths.count <= 8, await onReferenceSave(subject.id, paths) else { return false }
        imagePaths = paths
        return true
      }
    }
  }
  private var attention: DirectorSubjectAttention {
    DirectorSubjectAttention(subject: subject, coverageProposal: coverageProposal, currentDescription: description)
  }
  @ViewBuilder private var coverageNotes: some View {
    if let proposal = coverageProposal {
      DisclosureGroup("Proposed coverage changes · saved approval preserved") {
        ForEach(proposal.relationships ?? []) { link in
          Text("\(link.role.label) → \(inventory.first(where: { $0.id == link.targetID })?.name ?? link.targetID) · \(link.placement)").font(.caption)
        }
        Text("Unlock the relevant descriptions, then run coverage review again to apply these changes. Existing approved data has not changed.").font(.caption)
      }
    }
    ObjectCoverageReviewView(report: coverageProposal?.coverageReview ?? subject.coverageReview,
      candidates: libraryCandidates, selectedMatch: selectedLibraryMatch, locked: approved || !editable || changed,
      onSelect: onLibrarySelect, onCreate: onCreateObject)
  }
  @ViewBuilder private var descriptionNotes: some View {
    if let review = subject.descriptionReview {
      Label(review.reviewedDescription != description || imagePaths != savedReferencePaths || subject.referenceAssetKeys != review.referenceAssets
            ? "Agent notes refer to an earlier description or image selection"
            : (review.isReady(for: description) ? "Agent review complete" : "Agent review flagged issues"),
            systemImage: review.isReady(for: description) ? "checkmark.shield" : "exclamationmark.triangle")
        .font(.caption).foregroundStyle(.secondary)
      if !review.proposedDetails.isEmpty {
        VStack(alignment: .leading, spacing: 6) {
          Text("Proposed design details — approve or revise").font(.subheadline.bold())
          Text("These additions are included in the description but are not established source or image facts.")
            .font(.caption)
          SubjectSourceText(title: "", entries: review.proposedDetails)
        }.padding(10).background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
      }
      if let observations = review.referenceDetails, !observations.isEmpty {
        SubjectSourceText(title: "Observed in reference images", entries: observations)
      }
      SubjectSourceText(title: "Review issues", entries: review.issues)
    }
  }
  @ViewBuilder private var relationshipNotes: some View {
    if let review = subject.relationshipReview {
      SubjectSourceText(title: "Relationship review", entries: review.issues)
      SubjectSourceText(title: "Suggested missing objects", entries: review.missingObjects)
    }
  }
  private func moveReference(_ path: String, by offset: Int) {
    guard let index = imagePaths.firstIndex(of: path), imagePaths.indices.contains(index + offset) else { return }
    imagePaths.swapAt(index, index + offset)
  }
  private var descriptionTargets: [DescriptionLinkTarget] {
    let ids = Set((relationships ?? []).map(\.targetID))
    return inventory.filter { ids.contains($0.id) }.map {
      DescriptionLinkTarget(id: $0.id, name: $0.name, aliases: $0.aliases, description: $0.description)
    }
  }
  private func editedSubject() throws -> WorkflowSubjectProposal {
    var value = try subject.editing(name: name, description: description)
    value.kind = kind
    value.relationships = relationships
    return value
  }
  private var linkedObjects: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Linked objects").font(.subheadline.bold())
      ForEach(relationships ?? []) { link in
        VStack(alignment: .leading) {
          HStack {
            Button(inventory.first(where: { $0.id == link.targetID })?.name ?? link.targetID) { onNavigate(link.targetID) }
              .help(inventory.first(where: { $0.id == link.targetID })?.description ?? "Linked object is unavailable.")
            Spacer()
            Button("Remove") { relationships?.removeAll { $0.id == link.id } }.disabled(approved || !editable)
          }
          HStack {
            Picker("Role", selection: Binding(get: { link.role }, set: { role in updateLink(link.id) { $0.role = role } })) {
              ForEach(ObjectRelationshipRole.allCases) { Text($0.label).tag($0) }
            }
            TextField("Placement / state", text: Binding(get: { link.placement }, set: { text in updateLink(link.id) { $0.placement = String(text.prefix(300)) } }))
          }.disabled(approved || !editable)
        }
      }
      Menu("Link object…") {
        ForEach(inventory.filter { $0.id != subject.id }) { target in
          Button(target.name) {
            var links = relationships ?? []
            links.append(WorkflowObjectRelationship(id: UUID().uuidString, targetID: target.id, role: .uses, placement: ""))
            relationships = links
          }
        }
      }.disabled(approved || !editable || (relationships ?? []).count >= 32)
      if individualApprovals { relationshipNotes }
    }
  }
  private func updateLink(_ id: String, edit: (inout WorkflowObjectRelationship) -> Void) {
    guard let i = relationships?.firstIndex(where: { $0.id == id }) else { return }
    edit(&relationships![i])
  }
  private func approveDescription() async {
    saving = true; error = nil
    defer { saving = false }
    do {
      if !approved && changed {
        guard await onSave(try editedSubject()) else {
          error = "Could not save your edits. The description was not approved."
          return
        }
      }
      if !(await onApproval(subject.id, !approved)) {
        error = "Could not update approval. See the workflow error below."
      }
    } catch { self.error = error.localizedDescription }
  }
}

struct SubjectSourceText: View {
  let title: String
  let entries: [String]
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      Text(title).font(.subheadline.bold())
      if entries.isEmpty { Text("None").font(.caption).foregroundStyle(.secondary) }
      ForEach(entries.indices, id: \.self) { index in
        Text(entries[index]).font(.caption).textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
  }
}
