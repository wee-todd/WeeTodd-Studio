import AppKit
import StudioCore
import SwiftUI

struct ProjectPlanningView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @State private var page = "shots"
  @State private var selectedShotIDs = Set<UUID>()
  @State private var maximumClipSeconds = 15.0
  private var selectedShot: UUID? {
    get { plan.shots.first { selectedShotIDs.contains($0.id) }?.id }
    nonmutating set { selectedShotIDs = newValue.map { [$0] } ?? [] }
  }
  @State private var selectedSubject: UUID?
  @State private var extracting = false
  @State private var libraryOpen = false
  @State private var producing = false
  @State private var splitting = false
  @State private var splitFrames = 1
  @State private var splitBoundary = ""
  @State private var generationSettingsOpen = false
  @State private var generationEngine: Engine = .ltx25
  @State private var generationWidth = 1344
  @State private var generationHeight = 768
  @State private var editingProjectID: UUID?
  @State private var error: String?
  @State private var sheetContext: ReferenceSheetContext?
  var plan: ProjectPlanning { store.planning }

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Shot List").font(.title2)
        Text(store.project.name).foregroundStyle(.secondary)
        Spacer()
        Button("Production Library…") { libraryOpen = true }
        Button("Export shot list…") { export() }
        Button("Done") { dismiss() }
      }
      Picker("View", selection: $page) {
        Text("Shots (\(plan.shots.count))").tag("shots")
        Text("Production objects (\(plan.subjects.count))").tag("subjects")
        Text("Original script").tag("script")
      }.pickerStyle(.segmented)
      if !plan.reviewNotes.isEmpty {
        DisclosureGroup("Review notes (\(plan.reviewNotes.count))") {
          ForEach(plan.reviewNotes, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
        }
      }
      if page == "script" { script }
      else if page == "subjects" { subjects }
      else { shots }
      if let error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
      Text("Approvals belong to the content you reviewed. Create reference opens optional image generation. Changes mark affected records for review; timeline clips stay unchanged.")
        .font(.caption).foregroundStyle(.secondary)
    }.padding(22).frame(width: 1140, height: 770)
      .sheet(isPresented: $libraryOpen) { ProductionLibraryView().environmentObject(store) }
      .sheet(isPresented: $producing) { MusicVideoProductionView().environmentObject(store) }
      .sheet(isPresented: $splitting) { splitEditor }
      .sheet(isPresented: $generationSettingsOpen) { generationEditor }
      .sheet(item: $sheetContext) { context in
        let subject = plan.subjects.first { store.project.id.uuidString + ":" + $0.id.uuidString == context.subjectKey }
        ReferenceSheetGenerator(context: context, referencePaths: (subject?.referenceAssetIDs ?? []).compactMap { id in store.allAssets.first { $0.id == id }?.path }) { asset in
          guard let subject, let current = store.planning.subjects.first(where: { $0.id == subject.id }),
            current.details == context.description, current.name == context.name,
            asset.generation?.referenceSheet?.subjectKey == context.subjectKey,
            asset.generation?.referenceSheet?.description == current.details else {
            error = "The subject changed. Review it before attaching the generated image."; return false
          }
          do {
            var updated = store.project
            if let appearance = asset.generation?.referenceSheet?.characterDefinition?.appearance,
              let index = updated.planning?.subjects.firstIndex(where: { $0.id == subject.id }) {
              updated.planning?.subjects[index].applyAcceptedAppearance(appearance, sourceDescription: current.details)
            }
            _ = try updated.attachPlanningReference(asset, subjectID: subject.id)
            store.change { $0 = updated }; return true
          } catch { self.error = error.localizedDescription; return false }
        }.environmentObject(store)
      }
      .sheet(isPresented: $extracting) {
        WorkflowView(initialBuiltin: "weetodd.subject-inventory-reviewed", initialInputs: ["brief": .string(plan.sourceText)], onProjectImport: { page = "subjects" })
          .environmentObject(store)
      }
  }
  private var script: some View {
    VStack(alignment: .leading) {
      Text("Keep the complete source here, including dialogue, camera direction and sound.")
      TextEditor(text: Binding(get: { plan.sourceText }, set: { value in store.changePlanning { $0.sourceText = value } }))
        .font(.body).border(Color(nsColor: .separatorColor))
      Button("Identify characters, props and locations…") { extracting = true }
        .disabled(plan.sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || store.bridge.busy)
      Text("Identification creates proposals with source excerpts. Review each description before approving it. Editing the script does not automatically rewrite existing records.")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
  private var subjects: some View {
    HSplitView {
      VStack(alignment: .leading) {
        HStack {
          Menu("Add subject") {
            ForEach(PlanningSubjectKind.allCases) { kind in
              Button(kind.label) {
                let item = PlanningSubject(name: "New \(kind.rawValue)", kind: kind)
                store.changePlanning { $0.subjects.append(item) }; selectedSubject = item.id
              }
            }
          }
          Button("Identify from script…") { extracting = true }.disabled(plan.sourceText.isEmpty || store.bridge.busy)
        }
        List(selection: $selectedSubject) {
          ForEach(PlanningSubjectKind.reviewOrder) { kind in
            Section(kind.groupTitle) {
              ForEach(plan.subjects.filter { $0.kind == kind }.sorted {
                $0.name.localizedStandardCompare($1.name) == .orderedAscending
              }) { subject in
                VStack(alignment: .leading, spacing: 4) {
                  Text(subject.name).font(.headline)
                  Text(subjectStatus(subject)).font(.caption).foregroundStyle(.secondary)
                }.tag(subject.id)
              }
            }
          }
        }
      }.frame(minWidth: 290, idealWidth: 320)
      if let id = selectedSubject, let subject = plan.subjects.first(where: { $0.id == id }) {
        subjectEditor(subject).frame(minWidth: 470)
      } else { empty("Select a subject to review its identity and reference images.") }
    }
  }
  private var shots: some View {
    HSplitView {
      VStack(alignment: .leading) {
        HStack {
          Button("Add shot") {
            let shot = PlanningShot(name: "Shot \(plan.shots.count + 1)", frameCount: min(120, max(1, plan.frameRate)) * 5)
            store.changePlanning { $0.shots.append(shot) }; selectedShot = shot.id
          }
          Button { moveShot(-1) } label: { Image(systemName: "arrow.up") }.help("Move selected shot earlier")
          Button { moveShot(1) } label: { Image(systemName: "arrow.down") }.help("Move selected shot later")
          Spacer()
          TextField("FPS", value: Binding(get: { plan.frameRate }, set: { value in store.changePlanning { $0.frameRate = value } }), format: .number).frame(width: 50)
          Text("FPS").font(.caption)
        }
        HStack {
          Button("Split…") {
            guard let shot = plan.shots.first(where: { $0.id == selectedShot }) else { return }
            splitFrames = max(1, shot.frameCount / 2); splitBoundary = ""
            editingProjectID = store.project.id; splitting = true
          }.disabled(selectedShotIDs.count != 1 || selectedShot.flatMap { id in plan.shots.first { $0.id == id } }.map { $0.linkedClipID != nil || $0.combinedShots != nil || $0.frameCount < 2 } ?? true)
          Button("Combine") { attempt {
            var value = plan
            let id = try value.combineShots(selectedShotIDs, maximumSeconds: maximumClipSeconds)
            store.change { $0.planning = value }; selectedShot = id
          } }.disabled(selectedShotIDs.count < 2)
          Button("Restore original shots") { attempt {
            guard let id = selectedShot else { return }
            var value = plan; try value.uncombineShot(id)
            store.change { $0.planning = value }; selectedShot = nil
          } }.disabled(selectedShot.flatMap { id in plan.shots.first { $0.id == id }?.combinedShots } == nil)
          Text("Max s").font(.caption)
          TextField("15", value: $maximumClipSeconds, format: .number).frame(width: 45)
        }
        Button("Generation settings for selection…") {
          guard let shot = plan.shots.first(where: { selectedShotIDs.contains($0.id) }) else { return }
          generationEngine = shot.engine
          generationWidth = shot.generationWidth ?? store.project.clips.first?.generationWidth ?? 768
          generationHeight = shot.generationHeight ?? store.project.clips.first?.generationHeight ?? 512
          editingProjectID = store.project.id; generationSettingsOpen = true
        }.disabled(selectedShotIDs.isEmpty || plan.shots.contains { selectedShotIDs.contains($0.id) && $0.linkedClipID != nil })
        Table(plan.shots, selection: $selectedShotIDs) {
          TableColumn("Shot") { shot in Text(shot.name) }.width(min: 90, ideal: 120)
          TableColumn("Start") { shot in Text(String(format: "%.2f s", Double(plan.startFrame(of: shot.id)) / Double(max(1, plan.frameRate)))) }.width(65)
          TableColumn("Frames") { shot in Text(String(shot.frameCount)) }.width(55)
          TableColumn("Review") { shot in Text(shotStatus(shot)).font(.caption) }.width(min: 85, ideal: 100)
        }
        if plan.shots.contains(where: { $0.musicSource != nil }) {
          Button("Realign song intervals to shot lengths") { attempt {
            var value = store.project
            let count = try value.realignPlanningMusicIntervals()
            store.change { $0 = value }
            store.notice = count == 0 ? "Song intervals already match the shot lengths." : "Realigned \(count) song intervals. Review affected shots, then explicitly reuse any trimmed timeline takes."
          } }.disabled(store.operationBusy || store.productionRunning)
            .help("Preserve the complete song while moving cuts between shots. Trim existing timeline clips first; affected shots require review.")
        }
        Button("Add approved selection to timeline") { attempt {
          var value = store.project; try value.applyPlanningShots(selectedShotIDs)
          store.change { $0 = value }
          store.notice = "Added approved shots and their song intervals. Produce the timeline or render individual clips."
        } }.disabled(selectedShotIDs.isEmpty)
        if let clip = store.selectedClip, !clip.sourcePath.isEmpty {
          Button("Reuse selected timeline take: " + clip.name) { attempt {
            guard let id = selectedShot else { return }
            var value = store.project
            try value.reuseTimelineClip(clip.id, forPlanningShot: id)
            store.change { $0 = value }
            store.notice = "Linked the existing take to the reviewed shot; its footage and versions are preserved."
          } }.disabled(selectedShotIDs.count != 1)
        }
        Button("Add approved plan and produce…") { attempt {
          let pending = Set(plan.shots.filter { $0.linkedClipID == nil }.map(\.id))
          if !pending.isEmpty {
            var value = store.project; try value.applyPlanningShots(pending)
            store.change { $0 = value }
          }
          producing = true
        } }.disabled(store.operationBusy || store.productionRunning || plan.shots.isEmpty)
        Text("Split keeps the song interval intact; review each new action and boundary. Combine retains original shots and frame references for restoration.").font(.caption).foregroundStyle(.secondary)
      }.frame(minWidth: 430, idealWidth: 490)
      if let id = selectedShot, let shot = plan.shots.first(where: { $0.id == id }) {
        shotEditor(shot).frame(minWidth: 470)
      } else { empty("Select a shot to edit its action, subjects and first/last-frame descriptions.") }
    }
  }
  private var splitEditor: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Split planned shot").font(.title2)
      if let shot = plan.shots.first(where: { $0.id == selectedShot }) {
        Text("\(shot.name) · \(shot.frameCount) frames")
        TextField("First part frames", value: $splitFrames, format: .number)
        Text(String(format: "First part %.3f s · second part %.3f s", Double(splitFrames) / Double(max(1, plan.frameRate)), Double(shot.frameCount - splitFrames) / Double(max(1, plan.frameRate)))).font(.caption)
        Text("State at the cut")
        TextEditor(text: $splitBoundary).frame(height: 90).border(Color.secondary)
        Text("Both parts keep the original direction for revision. Outer frame images stay with their respective ends; the new cut starts without an image. Song timing and the natural ending stay intact.").font(.caption)
      }
      if let error { Text(error).foregroundStyle(.red) }
      HStack {
        Button("Cancel") { splitting = false; error = nil }
        Spacer()
        Button("Split shot") { attempt {
          guard store.project.id == editingProjectID, let id = selectedShot else { throw StudioError.invalid("The movie changed. Reopen the shot editor.") }
          var value = plan
          let next = try value.splitShot(id, afterFrames: splitFrames, boundary: splitBoundary)
          store.change { $0.planning = value }; selectedShot = next; splitting = false
        } }.disabled(splitBoundary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      }
    }.padding(22).frame(width: 520)
  }
  private var generationEditor: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Generation settings · \(selectedShotIDs.count) shots").font(.title2)
      Picker("Model", selection: $generationEngine) {
        ForEach([Engine.ltx25, .ltx23, .h3, .drawThings]) { Text($0.label).tag($0) }
      }
      HStack {
        Text("Render size")
        TextField("Width", value: $generationWidth, format: .number)
        Text("×")
        TextField("Height", value: $generationHeight, format: .number)
      }
      Text("Applies to the selected unapplied shots and marks them for review. Song timing is preserved. Model-specific conditioning is checked before rendering.").font(.caption)
      if let error { Text(error).foregroundStyle(.red) }
      HStack {
        Button("Cancel") { generationSettingsOpen = false; error = nil }
        Spacer()
        Button("Apply settings") { attempt {
          guard store.project.id == editingProjectID else { throw StudioError.invalid("The movie changed. Reopen generation settings.") }
          var value = plan
          try value.setGenerationSettings(selectedShotIDs, engine: generationEngine, width: generationWidth, height: generationHeight)
          store.change { $0.planning = value }; generationSettingsOpen = false
        } }
      }
    }.padding(22).frame(width: 480)
  }
  private func subjectEditor(_ subject: PlanningSubject) -> some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 12) {
        HStack {
          Text(subjectStatus(subject)).font(.headline)
          Spacer()
          Button(plan.isSubjectApproved(subject.id) ? "Unlock description" : "Approve description") {
            attempt {
              var value = plan
              if value.isSubjectApproved(subject.id) { value.subjects[value.subjects.firstIndex(where: { $0.id == subject.id })!].approvedRevision = nil }
              else { try value.approveSubject(subject.id) }
              store.change { $0.planning = value }
            }
          }
        }
        VStack(alignment: .leading, spacing: 10) {
          TextField("Name", text: subjectBinding(subject.id, \.name))
          Text("ID: \(subject.id.uuidString)").font(.system(.caption2, design: .monospaced))
            .foregroundStyle(.secondary).textSelection(.enabled)
          Picker("Kind", selection: Binding(get: { subject.kind }, set: { kind in
            store.changePlanning { value in
              guard let i = value.subjects.firstIndex(where: { $0.id == subject.id }) else { return }
              value.subjects[i].kind = kind
              if kind != .set { value.subjects[i].environmentID = nil }
            }
          })) {
            ForEach(PlanningSubjectKind.allCases) { Text($0.label).tag($0) }
          }
          TextField("Aliases (comma separated)", text: subjectBinding(subject.id, \.aliases))
          Text("Identity / appearance").font(.caption).foregroundStyle(.secondary)
          LinkedDescriptionEditor(text: subjectBinding(subject.id, \.details), targets: descriptionTargets(subject),
            mentions: subject.descriptionMentions ?? [], sourceDescription: subject.mentionSourceDescription,
            editable: !plan.isSubjectApproved(subject.id), accessibilityLabel: "Object description", onNavigate: { selectedSubject = UUID(uuidString: $0) })
            .frame(height: 150)
        }.disabled(plan.isSubjectApproved(subject.id))
        Text("Hover over a linked name to read its description. Command-click while editing to open it.")
          .font(.caption).foregroundStyle(.secondary)
        objectLinks(subject)
        if let report = subject.descriptionReview {
          Text(report.isReady(for: subject.details) ? "Agent review complete. Project approval is your decision." :
            "Agent review needs attention or predates your edits. Project approval records your own review.")
            .font(.caption).foregroundStyle(.secondary)
          SubjectSourceText(title: "Proposed design details", entries: report.proposedDetails)
          if let details = report.referenceDetails, !details.isEmpty {
            SubjectSourceText(title: "Observed in reference images", entries: details)
          }
          SubjectSourceText(title: "Agent review issues", entries: report.issues)
        }
        SubjectSourceText(title: "Evidence", entries: subject.evidence.isEmpty ? [] : [subject.evidence])
        SubjectSourceText(title: "Suggestions", entries: subject.suggestions.isEmpty ? [] : [subject.suggestions])
        Divider()
        Text("Reference images").font(.headline)
        Text("Reference approval is separate from the description. Imported files are referenced in place.").font(.caption).foregroundStyle(.secondary)
        ForEach(Array(subject.referenceAssetIDs.enumerated()), id: \.element) { index, aid in
          HStack {
            if let asset = store.project.assets.first(where: { $0.id == aid }) {
              PreviewableImage(path: asset.path, title: "\(asset.name) · Reference \(index + 1)").frame(width: 86, height: 64).clipped()
              Text("\(asset.name) · Reference \(index + 1)").lineLimit(2)
            } else { Label("Missing image", systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
            Spacer()
            Button { store.changePlanning { p in
              guard let i = p.subjects.firstIndex(where: { $0.id == subject.id }) else { return }
              p.subjects[i].referenceAssetIDs.removeAll { $0 == aid }
            } } label: { Image(systemName: "xmark") }
          }
        }
        HStack {
          Button("Import images…") { store.importPlanningReferences(subjectID: subject.id) }
          Button(subject.kind == .character ? "Create character sheet…" : "Create reference…") {
            sheetContext = ReferenceSheetContext(subjectKey: store.project.id.uuidString + ":" + subject.id.uuidString,
              name: subject.name, kind: subject.kind, description: subject.details,
              linkedDefinitions: ReferenceSheetLinks.definitions(subject: subject, inventory: plan.subjects))
            if let appearance = subject.characterAppearance {
              var definition = CharacterSheetDefinition.newDraft(); definition.appearance = appearance
              sheetContext?.characterDefinition = definition
            }
          }.disabled(store.bridge.busy || subject.details.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
          Menu("Choose project/global image") {
            ForEach(store.allAssets.filter { $0.kind == .image && !subject.referenceAssetIDs.contains($0.id) }) { asset in
              Button(asset.name) { store.linkPlanningReference(asset, subjectID: subject.id) }
            }
          }
        }
        HStack {
          Text(plan.areReferencesApproved(subject.id, assets: store.project.assets) ? "References approved" : "References need review").font(.caption)
          Spacer()
          Button("Approve references") { attempt {
            var value = plan; try value.approveReferences(subject.id, assets: store.project.assets)
            store.change { $0.planning = value }
          } }.disabled(subject.referenceAssetIDs.isEmpty)
        }
        Text("Create character sheet uses local Draw Things and an installed four-panel LoRA. Other subjects offer reference templates. Inspect a candidate and use it as a reference; approve your references separately.")
          .font(.caption).foregroundStyle(.secondary)
        Menu("Merge into another subject") {
          ForEach(plan.subjects.filter { $0.id != subject.id && $0.kind == subject.kind }) { target in
            Button(target.name) { attempt {
              var value = plan; try value.mergeSubject(subject.id, into: target.id)
              store.change { $0.planning = value }; selectedSubject = target.id
            } }
          }
        }.disabled(plan.isSubjectApproved(subject.id))
        Button("Remove subject") { attempt { var value = plan; try value.removeSubject(subject.id); store.change { $0.planning = value }; selectedSubject = nil } }
          .disabled(plan.isSubjectApproved(subject.id))
      }.padding(12)
    }
  }
  private func descriptionTargets(_ subject: PlanningSubject) -> [DescriptionLinkTarget] {
    let ids = Set((subject.relationships ?? []).map(\.targetID) + [subject.environmentID].compactMap { $0 })
    return plan.subjects.filter { ids.contains($0.id) }.map {
      DescriptionLinkTarget(id: $0.id.uuidString, name: $0.name,
        aliases: $0.aliases.components(separatedBy: ","), description: $0.details)
    }
  }
  private func objectLinks(_ subject: PlanningSubject) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Divider()
      if let origin = subject.libraryOrigin {
        Text("Library version \(origin.version) · " + (origin.definitionRevision == subject.revision ? "Pinned definition" : "Movie variation"))
          .font(.caption).foregroundStyle(.secondary)
      }
      TextField("Tags (comma separated)", text: Binding(get: { (subject.tags ?? []).joined(separator: ", ") }, set: { text in
        subjectBinding(subject.id, \.tags).wrappedValue = Array(Set(text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }.filter { !$0.isEmpty })).sorted()
      }))
      if subject.kind == .set {
        Picker("Environment", selection: subjectBinding(subject.id, \.environmentID)) {
          Text("Choose environment").tag(UUID?.none)
          ForEach(plan.subjects.filter { $0.kind == .environment }) { Text($0.name).tag(Optional($0.id)) }
        }.disabled(plan.isSubjectApproved(subject.id))
      }
      if subject.kind == .environment {
        ForEach(plan.subjects.filter { $0.kind == .set && $0.environmentID == subject.id }) { set in
          Button("Set: " + set.name) { selectedSubject = set.id }
        }
      }
      Text("Linked objects / placements").font(.headline)
      Text("Link reusable appearance by ID. Use placement notes for this instance’s position or state.").font(.caption).foregroundStyle(.secondary)
      ForEach(subject.relationships ?? []) { link in
        VStack(alignment: .leading, spacing: 5) {
          HStack {
            Button(plan.subjects.first(where: { $0.id == link.targetID })?.name ?? "Missing object") { selectedSubject = link.targetID }
              .help(plan.subjects.first(where: { $0.id == link.targetID })?.details ?? "Linked object is unavailable.")
            Text(link.targetID.uuidString).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
            Spacer()
            Button { editLink(subject.id, link.id) { _ in nil } } label: { Image(systemName: "xmark") }
              .disabled(plan.isSubjectApproved(subject.id))
          }
          HStack {
            Picker("Role", selection: Binding(get: { link.role }, set: { role in editLink(subject.id, link.id) { var x = $0; x.role = role; return x } })) {
              ForEach(ObjectRelationshipRole.allCases) { Text($0.label).tag($0) }
            }
            TextField("Placement / state", text: Binding(get: { link.placement }, set: { text in editLink(subject.id, link.id) { var x = $0; x.placement = String(text.prefix(300)); return x } }))
          }.disabled(plan.isSubjectApproved(subject.id))
        }
      }
      Menu("Link object…") {
        ForEach(plan.subjects.filter { $0.id != subject.id }) { target in
          Button("\(target.name) · \(target.kind.label)") {
            var links = subject.relationships ?? []; links.append(ObjectRelationship(targetID: target.id, role: subject.kind == .set ? .contains : .uses))
            subjectBinding(subject.id, \.relationships).wrappedValue = links
          }
        }
      }.disabled(plan.isSubjectApproved(subject.id) || (subject.relationships ?? []).count >= 32)
      if let review = subject.relationshipReview {
        SubjectSourceText(title: "Relationship review", entries: review.issues)
        SubjectSourceText(title: "Suggested missing objects", entries: review.missingObjects)
      }
    }
  }
  private func editLink(_ subjectID: UUID, _ linkID: UUID, edit: (ObjectRelationship) -> ObjectRelationship?) {
    store.changePlanning { value in
      guard let i = value.subjects.firstIndex(where: { $0.id == subjectID }) else { return }
      value.subjects[i].relationships = value.subjects[i].relationships?.compactMap { $0.id == linkID ? edit($0) : $0 }
    }
  }
  private func shotEditor(_ shot: PlanningShot) -> some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Text(shotStatus(shot)).font(.headline)
          Spacer()
          Button(plan.isShotApproved(shot.id, assets: store.project.assets) ? "Unlock shot" : "Approve shot") { attempt {
            var value = plan
            if value.isShotApproved(shot.id, assets: store.project.assets) { value.shots[value.shots.firstIndex(where: { $0.id == shot.id })!].approvedRevision = nil }
            else { try value.approveShot(shot.id, assets: store.project.assets) }
            store.change { $0.planning = value }
          } }
        }
        VStack(alignment: .leading, spacing: 10) {
          TextField("Name", text: shotBinding(shot.id, \.name))
          Picker("Model", selection: shotBinding(shot.id, \.engine)) {
            ForEach([Engine.ltx25, .ltx23, .h3, .drawThings]) { Text($0.label).tag($0) }
          }
          if let width = shot.generationWidth, let height = shot.generationHeight {
            Text("Render size: \(width) × \(height)").font(.caption)
          }
          if let source = shot.musicSource {
            Text("Song: \(source.start, specifier: "%.3f")–\(source.start + source.duration, specifier: "%.3f") s").font(.caption)
          }
          endpointPicker("First image", shot: shot, key: \.firstAssetID)
          endpointPicker("Last image", shot: shot, key: \.lastAssetID)
          HStack {
            TextField("Frames", value: shotBinding(shot.id, \.frameCount), format: .number)
            Text(String(format: "%.2f seconds", Double(shot.frameCount) / Double(max(1, plan.frameRate))))
            Picker("Transition", selection: shotBinding(shot.id, \.continuity)) { Text("Cut").tag("cut"); Text("Continue").tag("continue") }
          }
          editor("Action", shotBinding(shot.id, \.action), height: 65)
          editor("Original shot / detailed direction", shotBinding(shot.id, \.direction), height: 95)
          editor("First frame", shotBinding(shot.id, \.firstFrame), height: 65)
          editor("Last frame", shotBinding(shot.id, \.lastFrame), height: 65)
          if shot.continuity == "continue" { Button("Use previous ending as first frame") {
            guard let i = plan.shots.firstIndex(where: { $0.id == shot.id }), i > 0 else { return }
            shotBinding(shot.id, \.firstFrame).wrappedValue = plan.shots[i-1].lastFrame
          } }
          DisclosureGroup("Camera, dialogue and sound") {
            editor("Camera", shotBinding(shot.id, \.camera), height: 60)
            editor("Dialogue", shotBinding(shot.id, \.dialogue), height: 60)
            editor("Sound", shotBinding(shot.id, \.sound), height: 60)
          }
          Text("Subjects in this shot").font(.headline)
          ForEach(plan.subjects) { subject in
            Toggle("\(subject.name) · \(subject.kind.label)", isOn: Binding(get: { shot.subjectIDs.contains(subject.id) }, set: { enabled in
              var ids = shot.subjectIDs.filter { $0 != subject.id }; if enabled { ids.append(subject.id) }
              shotBinding(shot.id, \.subjectIDs).wrappedValue = ids
            }))
          }
          ForEach(shot.subjectIDs.filter { id in !plan.subjects.contains { $0.id == id } }, id: \.self) { id in
            Button("Remove missing subject link") { shotBinding(shot.id, \.subjectIDs).wrappedValue = shot.subjectIDs.filter { $0 != id } }
          }
        }.disabled(plan.isShotApproved(shot.id, assets: store.project.assets))
        let resolvedIDs = Set((try? plan.resolvedObjects(shot.subjectIDs).map(\.id)) ?? shot.subjectIDs)
        ForEach((shot.appearanceOverrides ?? []).filter { !resolvedIDs.contains($0.subjectID) }) { state in
          HStack {
            Text("Unlinked appearance override: " + state.state).font(.caption).foregroundStyle(.orange)
            Button("Remove override") { shotBinding(shot.id, \.appearanceOverrides).wrappedValue = shot.appearanceOverrides?.filter { $0.subjectID != state.subjectID } }
              .accessibilityLabel("Remove appearance override for \(plan.subjects.first(where: { $0.id == state.subjectID })?.name ?? "missing object")")
              .accessibilityIdentifier("remove-appearance-override-\(shot.id.uuidString)-\(state.subjectID.uuidString)")
          }
        }
        DisclosureGroup("Resolved objects and references") {
          if let snapshot = try? plan.resolvedSnapshot(shot.subjectIDs, assets: store.allAssets, requireApproval: false) {
            ForEach(snapshot.subjects) { object in
              Text("\(object.name) · \(object.kind.label) · \(object.referenceAssetIDs.count) references").font(.caption)
              TextField("Shot-only appearance / state (e.g. night, rain, coat unbuttoned)", text: Binding(get: {
                shot.appearanceOverrides?.first(where: { $0.subjectID == object.id })?.state ?? ""
              }, set: { text in
                var states = shot.appearanceOverrides?.filter { $0.subjectID != object.id } ?? []
                if !text.isEmpty { states.append(ObjectStateOverride(subjectID: object.id, state: String(text.prefix(2000)))) }
                shotBinding(shot.id, \.appearanceOverrides).wrappedValue = states
              }))
                .accessibilityLabel("\(object.name) appearance in \(shot.name)")
                .accessibilityIdentifier("shot-appearance-\(shot.id.uuidString)-\(object.id.uuidString)")
                .disabled(plan.isShotApproved(shot.id, assets: store.project.assets))
            }
            Text("Only this shot’s linked dependencies are included. Sibling sets are excluded.").font(.caption).foregroundStyle(.secondary)
          }
        }
        ForEach(plan.issues(for: shot.id, assets: store.project.assets), id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
        Text("Subject references remain in the production library. Assign first/last images here; timed frames and model-specific reference adapters remain available in the timeline inspector.")
          .font(.caption).foregroundStyle(.secondary)
        Button("Remove shot") { store.changePlanning { $0.shots.removeAll { $0.id == shot.id } }; selectedShot = nil }
          .disabled(plan.isShotApproved(shot.id, assets: store.project.assets))
      }.padding(12)
    }
  }
  private func endpointPicker(_ title: String, shot: PlanningShot, key: WritableKeyPath<PlanningShot, UUID?>) -> some View {
    HStack {
      Picker(title, selection: shotBinding(shot.id, key)) {
        Text("None").tag(nil as UUID?)
        ForEach(store.project.assets.filter { $0.kind == .image }) { image in Text(image.name).tag(Optional(image.id)) }
      }
      Button("Import…") {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let asset = MediaAsset(name: url.deletingPathExtension().lastPathComponent, kind: .image, path: url.path)
        store.change { project in
          guard let i = project.planning?.shots.firstIndex(where: { $0.id == shot.id }) else { return }
          project.assets.append(asset); project.planning?.shots[i][keyPath: key] = asset.id
        }
      }
    }
  }
  private func subjectBinding<T>(_ id: UUID, _ key: WritableKeyPath<PlanningSubject, T>) -> Binding<T> {
    let fallback = plan.subjects.first(where: { $0.id == id })![keyPath: key]
    return Binding(get: { plan.subjects.first(where: { $0.id == id })?[keyPath: key] ?? fallback }, set: { value in
      store.changePlanning { p in if let i = p.subjects.firstIndex(where: { $0.id == id }) { p.subjects[i][keyPath: key] = value } }
    })
  }
  private func shotBinding<T>(_ id: UUID, _ key: WritableKeyPath<PlanningShot, T>) -> Binding<T> {
    let fallback = plan.shots.first(where: { $0.id == id })![keyPath: key]
    return Binding(get: { plan.shots.first(where: { $0.id == id })?[keyPath: key] ?? fallback }, set: { value in
      store.changePlanning { p in if let i = p.shots.firstIndex(where: { $0.id == id }) { p.shots[i][keyPath: key] = value } }
    })
  }
  private func subjectStatus(_ s: PlanningSubject) -> String { plan.isSubjectApproved(s.id) ? "Approved" : s.approvedRevision == nil ? "Draft" : "Needs review" }
  private func shotStatus(_ s: PlanningShot) -> String { plan.isShotApproved(s.id, assets: store.project.assets) ? "Approved" : s.approvedRevision == nil ? "Draft" : "Needs review" }
  private func editor(_ title: String, _ value: Binding<String>, height: CGFloat) -> some View {
    VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); TextEditor(text: value).font(.body).frame(height: height).border(Color(nsColor: .separatorColor)) }
  }
  private func empty(_ text: String) -> some View { Text(text).foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity).padding(30) }
  private func attempt(_ work: () throws -> Void) { do { try work(); error = nil } catch { self.error = error.localizedDescription } }
  private func moveShot(_ offset: Int) {
    guard let id = selectedShot, let i = plan.shots.firstIndex(where: { $0.id == id }), plan.shots.indices.contains(i + offset) else { return }
    store.changePlanning { $0.shots.swapAt(i, i + offset) }
  }
  private func export() {
    let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "shot-list.json"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    attempt {
      let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(plan.exportDocument(assets: store.allAssets)).write(to: url, options: .atomic)
    }
  }
}
