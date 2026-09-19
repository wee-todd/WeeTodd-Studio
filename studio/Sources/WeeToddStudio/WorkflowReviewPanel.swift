import StudioCore
import SwiftUI

/// Approval is a human decision on the exact saved output, separate from validation.
struct WorkflowReviewPanel: View {
  @Binding var draft: DirectorReviewDraft
  var bindReferences: ([String]) throws -> [String]
  let step: WorkflowRunSummary.Step
  let busy: Bool
  var referenceBindings: [String: String] = [:]
  var onDescriptionReview: ((String, [String]) async -> Bool)? = nil
  var onReferenceSave: ((String, [String]) async -> Bool)? = nil
  var subjectScope = ""
  var onCoverageReview: (() async -> Bool)? = nil
  var libraryCandidates: [LibraryObjectCandidate] = []
  var librarySelections: [String: LibraryObjectMatch] = [:]
  var onLibrarySelect: (String, LibraryObjectMatch?) -> Void = { _, _ in }
  var onCreateObject: (String, MissingObjectProposal) -> Void = { _, _ in }
  var allowKindEditing = false
  var subjectApprovalLabel = "description"
  var preserveApprovedCast = false
  var focused = false
  var dynamicMusicTiming = false
  var subjectNames: [String: String] = [:]
  var storyContext: WorkflowStoryOutline?
  var promptPreview: H3PromptPreview?
  let onReview: (String, String?, [String: JSONValue]?, String?, DirectorShotRepairScope?) async -> Bool
  @State private var editingClip: WorkflowClipDraft?
  @State private var editingStory = false
  @State private var repairingClip: WorkflowClipDraft?
  @State private var editingJSON = false
  @State private var localError: String?
  @State private var creativeDirty = false
  @State private var showDiscardDraft = false
  private var outputs: [String: JSONValue] { step.outputs ?? [:] }
  private var hasUnsavedDraft: Bool { creativeDirty || draft.hasUnsavedChanges(outputs: outputs, referenceBindings: referenceBindings) }
  private var approvalTitle: String {
    if let subjects { return "Approve all \(subjects.count) subjects" }
    if let plan { return "Approve all \(plan.clips.count) shots" }
    return creativeBrief == nil ? "Approve step" : "Approve brief"
  }
  private func decode<T: Decodable>(_ type: T.Type, key: String) -> T? {
    guard let value = outputs[key], let data = try? JSONEncoder().encode(value) else { return nil }
    return try? JSONDecoder().decode(type, from: data)
  }
  private var plan: WorkflowClipPlan? { decode(WorkflowClipPlan.self, key: "clips") }
  private var story: WorkflowStoryOutline? { decode(WorkflowStoryOutline.self, key: "story") }
  private var subjects: [WorkflowSubjectProposal]? { decode([WorkflowSubjectProposal].self, key: "subjects") }
  private var creativeBrief: CreativeBrief? { decode(CreativeBrief.self, key: "creative_brief") }
  private var h3Preview: H3PromptPreview? { decode(H3PromptPreview.self, key: "h3_prompts") ?? decode(H3PromptPreview.self, key: "prompt_plan") }

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Label(step.approved == true ? "Approved by you" : "Not yet approved",
              systemImage: step.approved == true ? "lock.fill" : "person.crop.circle.badge.questionmark")
          .font(.headline)
        Spacer()
        if step.approved == true {
          Button("Unlock") { submit("unapprove") }
        } else {
          Button(approvalTitle) { submit("approve") }
            .buttonStyle(.borderedProminent)
            .disabled(!DirectorReviewPresentation.canApprove(step, hasUnsavedDraft: hasUnsavedDraft, briefReady: creativeBrief?.isReady ?? true))
        }
      }
      Text("Approval locks these saved choices. Changing their inputs makes dependent results stale.")
        .font(.caption).foregroundStyle(.secondary)
      if hasUnsavedDraft {
        Label("Save or discard your draft edits before approving this review.", systemImage: "pencil.circle")
          .font(.caption).foregroundStyle(.orange)
        Button("Discard draft edits…", role: .destructive) { showDiscardDraft = true }
          .font(.caption)
      }
      if step.status == "stale" {
        Label("Inputs changed. Run this step again before reviewing.", systemImage: "exclamationmark.triangle")
          .foregroundStyle(.orange)
      }
      if let creativeBrief {
        CreativeBriefReviewView(draft: $draft.brief, value: creativeBrief, approved: step.approved == true, busy: busy,
          dynamicMusicTiming: dynamicMusicTiming,
          onSave: { values in await onReview("edit", nil, values, nil, nil) },
          onDirtyChange: { creativeDirty = $0 })
      } else if let h3Preview {
        H3PromptReviewView(preview: h3Preview)
      } else if let subjects {
        WorkflowSubjectReviewView(draft: $draft, bindReferences: bindReferences, subjects: subjects, step: step, referenceBindings: referenceBindings,
          onDescriptionReview: onDescriptionReview, onReferenceSave: onReferenceSave, subjectScope: subjectScope, onCoverageReview: onCoverageReview,
          libraryCandidates: libraryCandidates, librarySelections: librarySelections,
          onLibrarySelect: onLibrarySelect, onCreateObject: onCreateObject, allowKindEditing: allowKindEditing,
          approvalLabel: subjectApprovalLabel, individualApprovals: !focused, onSave: { edited in
          do {
            var updated = subjects
            guard let index = updated.firstIndex(where: { $0.id == edited.id }) else { return false }
            updated[index] = edited
            var values = outputs
            values["subjects"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(updated))
            let saved = await onReview("edit", nil, values, nil, nil)
            if saved { draft.json = nil }; return saved
          } catch { localError = error.localizedDescription; return false }
        }, onApproval: { id, approved in
          await onReview(approved ? "approve" : "unapprove", id, nil, nil, nil)
        })
      } else if let plan {
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 12) {
            if let storyContext {
              DisclosureGroup("Story beats") {
                ForEach(storyContext.beats.indices, id: \.self) { i in Text("\(i + 1). \(storyContext.beats[i])").font(.callout) }
              }
            }
            ForEach(plan.clips) { clip in clipCard(clip, plan: plan) }
            if let promptPreview {
              DisclosureGroup("Generated prompt drafts") { H3PromptReviewView(preview: promptPreview).frame(minHeight: 300) }
            }
          }
        }
      } else if let story {
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            Text("Characters").font(.headline)
            ForEach(story.characters) { character in
              Text("\(subjectNames[character.id] ?? character.id): \(character.description)")
            }
            Text("Story beats").font(.headline)
            ForEach(story.beats.indices, id: \.self) { i in Text("\(i + 1). \(story.beats[i])") }
          }.frame(maxWidth: .infinity, alignment: .leading)
        }
        Button("Edit story…") { editingStory = true }.disabled(step.approved == true || step.status == "stale")
      } else {
        ScrollView { WorkflowTextPreviewView(text: pretty(outputs)).font(.system(.callout, design: .monospaced)) }
        Button("Edit output…") { editingJSON = true }.disabled(step.approved == true || step.status != "completed")
      }
      if let localError { WorkflowTextPreviewView(text: localError).foregroundStyle(.red).font(.caption) }
    }.disabled(busy)
      .confirmationDialog("Discard unsaved review edits?", isPresented: $showDiscardDraft) {
        Button("Discard draft edits", role: .destructive) {
          draft.subjects = [:]; draft.referencePaths = [:]; draft.brief = nil
          draft.clips = [:]; draft.story = nil; draft.json = nil; creativeDirty = false
        }
      } message: { Text("The saved review output stays intact. This removes your unsaved text, reference, and shot edits in this review.") }
      .sheet(item: $editingClip) { clip in
        WorkflowClipEditSheet(clip: Binding(get: { draft.clips[clip.id] ?? clip }, set: { draft.clips[clip.id] = $0 }),
          characters: plan?.characters ?? [], subjectNames: subjectNames) { edited in
          do {
            guard var plan else { return false }
            try plan.replace(edited)
            let saved = await onReview("edit", nil, try plan.outputs(), nil, nil)
            if saved { draft.clips.removeValue(forKey: edited.id) }; return saved
          } catch { localError = error.localizedDescription; return false }
        }
      }
      .sheet(isPresented: $editingStory) {
        if let story { WorkflowStoryEditSheet(story: Binding(get: { draft.story ?? story }, set: { draft.story = $0 }), preserveApprovedCast: preserveApprovedCast) { updated in
          guard let value = try? JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(updated)) else { return false }
          let saved = await onReview("edit", nil, ["story": value], nil, nil)
          if saved { draft.story = nil }; return saved
        } }
      }
      .sheet(item: $repairingClip) { clip in
        WorkflowRepairSheet(instruction: Binding(get: { draft.repairs[clip.id] ?? "" }, set: { draft.repairs[clip.id] = $0 }),
          scope: Binding(get: { draft.repairScopes?[clip.id] ?? .action }, set: { value in
            if draft.repairScopes == nil { draft.repairScopes = [:] }; draft.repairScopes?[clip.id] = value
          }), clip: clip) { instruction, scope in
          if draft.repairBaselines == nil { draft.repairBaselines = [:] }
          if draft.repairScopes == nil { draft.repairScopes = [:] }
          draft.repairScopes?[clip.id] = scope
          draft.repairBaselines?[clip.id] = clip
          let saved = await onReview("repair", clip.id, nil, instruction, scope)
          if saved { draft.repairs.removeValue(forKey: clip.id) }; return saved
        }
      }
      .sheet(isPresented: $editingJSON) {
        WorkflowJSONEditSheet(text: Binding(get: { draft.json ?? pretty(outputs) }, set: { draft.json = $0 }), onSave: { text in
          do {
            let values = try JSONDecoder().decode([String: JSONValue].self, from: Data(text.utf8))
            let saved = await onReview("edit", nil, values, nil, nil)
            if saved { draft.json = nil }; return saved
          } catch { localError = error.localizedDescription; return false }
        })
      }
  }
  private func clipCard(_ clip: WorkflowClipDraft, plan: WorkflowClipPlan) -> some View {
    let item = step.items?[clip.id]
    let approved = item?.approved == true
    return VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text("Shot \((plan.clips.firstIndex(where: { $0.id == clip.id }) ?? 0) + 1)").font(.headline)
        Text(String(format: "%.1f–%.1f s", Double(clip.startFrame) / Double(plan.fps),
                    Double(clip.startFrame + clip.frameCount) / Double(plan.fps))).font(.caption)
        Spacer()
        Text(approved ? "Approved" : (item?.status ?? "Pending").capitalized).font(.caption)
      }
      Text(clip.action).textSelection(.enabled)
      Text("Cast: " + clip.characters.map { subjectNames[$0] ?? $0 }.joined(separator: ", ")).font(.caption)
      Text("Start: \(clip.startState)").font(.caption)
      Text("End: \(clip.endState)").font(.caption)
      Text("\(subjectNames[clip.location] ?? clip.location) · \(clip.continuity == "continue" ? "Continues previous clip" : "Cut")")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        Button("Edit…") { editingClip = clip }.disabled(approved || step.status == "stale")
        if !focused || approved {
          Button(approved ? "Unlock" : "Approve") { submit(approved ? "unapprove" : "approve", clip.id) }
            .disabled(item?.status != "completed" || step.status == "stale" || hasUnsavedDraft)
        }
        Button("Correct…") { repairingClip = clip }.disabled(approved || step.status == "stale" || hasUnsavedDraft)
      }
      if let before = draft.repairBaselines?[clip.id] {
        let changes = (draft.repairScopes?[clip.id] ?? .all).changes(from: before, to: clip)
        if !changes.isEmpty {
          DisclosureGroup("Changes since correction · \(changes.count) fields") {
            ForEach(changes) { change in
              VStack(alignment: .leading, spacing: 3) {
                Text(change.field + (change.withinScope ? "" : " · related change")).font(.caption.bold())
                Text("Before: " + change.before).font(.caption).foregroundStyle(.secondary)
                Text("After: " + change.after).font(.caption).textSelection(.enabled)
              }.padding(.vertical, 3)
            }
          }
        }
      }
      DisclosureGroup("Shot details") {
        Text("ID: \(clip.id) · frame \(clip.startFrame) · \(clip.frameCount) frames")
          .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
      }.font(.caption)
    }.padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
  }
  private func submit(_ action: String, _ item: String? = nil) {
    guard action != "approve" || !hasUnsavedDraft else { return }
    Task { _ = await onReview(action, item, nil, nil, nil) }
  }
  private func pretty(_ value: [String: JSONValue]) -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return (try? String(data: encoder.encode(value), encoding: .utf8)) ?? ""
  }
}

private struct WorkflowClipEditSheet: View {
  @Environment(\.dismiss) private var dismiss
  @Binding var clip: WorkflowClipDraft
  let characters: [WorkflowCharacter]
  let subjectNames: [String: String]
  let onSave: (WorkflowClipDraft) async -> Bool
  @State private var saving = false
  @State private var failed = false
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Edit \(clip.id)").font(.title2)
      Text("Action"); TextEditor(text: $clip.action).frame(height: 65)
      Text("Starting state"); TextEditor(text: $clip.startState).frame(height: 55)
        .disabled(clip.continuity == "continue")
      Text("Ending state"); TextEditor(text: $clip.endState).frame(height: 55)
      TextField("Location", text: $clip.location)
      Picker("Connection", selection: $clip.continuity) {
        Text("Cut").tag("cut"); Text("Continue previous clip").tag("continue")
      }.disabled(clip.startFrame == 0)
      DisclosureGroup("Visible characters · \(clip.characters.count)") {
        Text("Select only characters visible in this shot.").font(.caption).foregroundStyle(.secondary)
        ScrollView {
          VStack(alignment: .leading, spacing: 6) {
            ForEach(characters) { character in
              Toggle(subjectNames[character.id] ?? character.id, isOn: Binding(
                get: { clip.characters.contains(character.id) },
                set: { selected in
                  clip.characters.removeAll { $0 == character.id }
                  if selected { clip.characters.append(character.id) }
                }
              ))
            }
          }.frame(maxWidth: .infinity, alignment: .leading)
        }.frame(height: min(140, CGFloat(characters.count * 26)))
      }.disabled(saving)
      Text("A changed ending can require repairs to following continuous clips. Timing is controlled by movie settings.")
        .font(.caption).foregroundStyle(.secondary)
      if failed { Text("Could not save. Close this editor to see the workflow error.").foregroundStyle(.red) }
      HStack {
        Button("Cancel") { dismiss() }.disabled(saving)
        Spacer()
        Button("Save changes") { Task {
          saving = true
          if await onSave(clip) { dismiss() } else { failed = true }
          saving = false
        } }.disabled(saving).buttonStyle(.borderedProminent)
      }
    }.padding(24).frame(width: 560).interactiveDismissDisabled(saving)
  }
}

private struct WorkflowStoryEditSheet: View {
  @Environment(\.dismiss) private var dismiss
  @Binding var story: WorkflowStoryOutline
  var preserveApprovedCast = false
  let onSave: (WorkflowStoryOutline) async -> Bool
  @State private var saving = false
  @State private var failed = false
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Edit story").font(.title2)
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          Text("Character identities").font(.headline)
          ForEach(story.characters.indices, id: \.self) { i in
            Text(story.characters[i].id).font(.caption)
            if preserveApprovedCast {
              Text(story.characters[i].description).font(.callout).textSelection(.enabled)
            } else { TextEditor(text: $story.characters[i].description).frame(height: 55) }
          }
          Text("Chronological beats").font(.headline)
          ForEach(story.beats.indices, id: \.self) { i in
            HStack {
              Text("\(i + 1)")
              TextEditor(text: $story.beats[i]).frame(height: 55)
            }
          }
        }
      }
      Text("Changes reopen approval and invalidate dependent clip plans.").font(.caption).foregroundStyle(.secondary)
      if failed { Text("Could not save. Close this editor to see the workflow error.").foregroundStyle(.red) }
      HStack {
        Button("Cancel") { dismiss() }.disabled(saving)
        Spacer()
        Button("Save changes") { Task {
          saving = true
          if await onSave(story) { dismiss() } else { failed = true }
          saving = false
        } }.disabled(saving).buttonStyle(.borderedProminent)
      }
    }.padding(24).frame(width: 560, height: 580).interactiveDismissDisabled(saving)
  }
}

private struct WorkflowJSONEditSheet: View {
  @Environment(\.dismiss) private var dismiss
  @Binding var text: String
  let onSave: (String) async -> Bool
  @State private var saving = false
  @State private var failed = false
  var body: some View {
    VStack(alignment: .leading) {
      Text("Edit step output").font(.title2)
      Text("Keep the output ports and value types required by this workflow.").font(.caption)
      TextEditor(text: $text).font(.system(.body, design: .monospaced))
      if failed { Text("Could not save. Close this editor to see the workflow error.").foregroundStyle(.red) }
      HStack {
        Button("Cancel") { dismiss() }.disabled(saving)
        Spacer()
        Button("Save changes") { Task {
          saving = true
          if await onSave(text) { dismiss() } else { failed = true }
          saving = false
        } }.disabled(saving)
      }
    }.padding(24).frame(width: 600, height: 500).interactiveDismissDisabled(saving)
  }
}

private struct WorkflowRepairSheet: View {
  @Environment(\.dismiss) private var dismiss
  @Binding var instruction: String
  @Binding var scope: DirectorShotRepairScope
  let clip: WorkflowClipDraft
  let onRepair: (String, DirectorShotRepairScope) async -> Bool
  @State private var working = false
  @State private var failed = false
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Correct this shot").font(.title2)
      Text(clip.action).foregroundStyle(.secondary)
      Picker("Change only", selection: $scope) {
        ForEach(DirectorShotRepairScope.allCases) { Text($0.label).tag($0) }
      }
      Text(scope.preservedDescription).font(.caption).foregroundStyle(.secondary)
      Text("What should change?")
      TextEditor(text: $instruction).frame(height: 110)
      Text("The repair keeps this clip's assigned story beat. Dependent continuous clips are rechecked; unaffected work is reused.")
        .font(.caption).foregroundStyle(.secondary)
      if failed { Text("Repair failed. Close this editor to inspect the workflow error.").foregroundStyle(.red) }
      HStack {
        Button("Cancel") { dismiss() }.disabled(working)
        Spacer()
        if working { ProgressView().controlSize(.small) }
        Button("Correct \(scope == .all ? "shot" : scope.label.lowercased())") { Task {
          working = true
          if await onRepair(instruction, scope) { dismiss() } else { failed = true }
          working = false
        } }.disabled(working || instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      }
    }.padding(24).frame(width: 520).interactiveDismissDisabled(working)
  }
}
