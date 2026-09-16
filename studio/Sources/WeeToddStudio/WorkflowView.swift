import AppKit
import CryptoKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

/// Both built-ins and imported definitions use the shared Python validator and executor.
struct WorkflowView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  var sessionScope = "movie"
  var initialBuiltin = "weetodd.guided-movie-planning"
  var initialInputs: [String: JSONValue] = [:]
  var initialImages: [PromptAssistantImage] = []
  var onPrompt: ((String) -> Void)?
  var onProjectImport: (() -> Void)?
  private var addedToProject: Bool {
    get { director.state.addedToProject }
    nonmutating set { director.state.addedToProject = newValue }
  }
  @AppStorage("qwen35PromptModelPath") private var promptModel = ""
  @AppStorage("lastWorkflowJobPath") private var lastJobPath = ""
  @StateObject private var director = DirectorSessionController()
  @State private var documentTarget: DirectorDocumentTarget?
  @State private var setupModel: String?
  @State private var catalog: [[String: JSONValue]] = []
  private var definition: [String: JSONValue] {
    get { director.state.definition }
    nonmutating set { director.state.definition = newValue }
  }
  private var libraryCandidates: [LibraryObjectCandidate] {
    get { director.state.libraryCandidates }
    nonmutating set { director.state.libraryCandidates = newValue }
  }
  private var catalogFrozen: Bool {
    get { director.state.catalogFrozen }
    nonmutating set { director.state.catalogFrozen = newValue }
  }
  private var structuredInputs: [String: JSONValue] {
    get { director.state.structuredInputs }
    nonmutating set { director.state.structuredInputs = newValue }
  }
  private var librarySelections: [String: LibraryObjectMatch] {
    get { director.state.librarySelections }
    nonmutating set { director.state.librarySelections = newValue }
  }
  private var fields: [String: String] {
    get { director.state.fields }
    nonmutating set { director.state.fields = newValue }
  }
  private var imageInputs: [String: [PromptAssistantImage]] {
    get { director.state.imageInputs }
    nonmutating set { director.state.imageInputs = newValue }
  }
  private var modelPaths: [String: String] {
    get { director.state.modelPaths }
    nonmutating set { director.state.modelPaths = newValue }
  }
  private var runID: UUID {
    get { director.state.runID }
    nonmutating set { director.state.runID = newValue }
  }
  private var runDirectory: String {
    get { director.state.runDirectory }
    nonmutating set { director.state.runDirectory = newValue }
  }
  private var result: WorkflowRunSummary? {
    get { director.state.result }
    nonmutating set { director.state.result = newValue }
  }
  private var selectedStep: String {
    get { director.state.selectedStep }
    nonmutating set { director.state.selectedStep = newValue }
  }
  @State private var running = false
  @State private var error: String?
  private var tokens: Int {
    get { director.state.tokens }
    nonmutating set { director.state.tokens = newValue }
  }
  private var showWorkflowInputs: Bool {
    get { director.state.showWorkflowInputs }
    nonmutating set { director.state.showWorkflowInputs = newValue }
  }
  @State private var showHistory = false
  @State private var showTechnicalInspector = false
  @State private var inspectorShowsInputs = false
  @State private var inspectorShowsHistory = false
  @State private var inspectorSetupModel: String?
  private var reviewAssetBindings: [String: String] {
    get { director.state.reviewAssetBindings }
    nonmutating set { director.state.reviewAssetBindings = newValue }
  }
  private var reviewingStructuredOutput: Bool {
    ["subjects", "creative_brief", "h3_prompts", "story", "clips"].contains {
      result?.steps[selectedStep]?.outputs?[$0] != nil
    }
  }
  private var inputSpecs: [String: JSONValue] { definition["inputs"]?.workflowObject ?? [:] }
  private var modelSpecs: [String: JSONValue] { definition["models"]?.workflowObject ?? [:] }
  private var steps: [[String: JSONValue]] {
    guard case .array(let list) = definition["steps"] else { return [] }
    return list.compactMap(\.workflowObject)
  }
  private var busy: Bool { running || store.operationBusy }
  private var guided: Bool { definition["id"]?.workflowText == "weetodd.guided-movie-planning" }
  private var legacyInventory: Bool { definition["id"]?.workflowText == "weetodd.subject-inventory" }
  private var presentation: DirectorReviewPresentation { DirectorReviewPresentation(definition: definition, result: result) }
  private var guidedImportReady: Bool { presentation.canImport }
  private var preparation: DirectorPreparationState {
    DirectorPreparationState(requiredModels: Array(modelSpecs.keys), bindings: modelPaths, brief: fields["brief"] ?? "")
  }
  private var hasUnsavedReviews: Bool {
    director.state.reviews.contains { id, draft in
      draft.hasUnsavedChanges(outputs: result?.steps[id]?.outputs ?? [:], referenceBindings: reviewAssetBindings)
    }
  }

  var body: some View {
    Group {
      if guided { focusedBody } else { advancedBody }
    }.padding(24).frame(width: 1120, height: 750)
      .sheet(isPresented: $showTechnicalInspector, onDismiss: {
        if let name = inspectorSetupModel { inspectorSetupModel = nil; setupModel = name }
      }) { technicalInspector }
      .sheet(isPresented: $showHistory) {
        WorkflowHistoryView(directory: runDirectory, workflowBusy: busy) { id in
          selectedStep = id; showWorkflowInputs = true
        }
      }
      .sheet(isPresented: Binding(get: { setupModel != nil }, set: { if !$0 { setupModel = nil } })) {
        AssistantModelSetupView(store: store, currentModelPath: setupModel.flatMap { modelPaths[$0] } ?? promptModel) { path in
          if let name = setupModel { modelPaths[name] = path; promptModel = path }
        }
      }
      .interactiveDismissDisabled(running)
      .task { await loadCatalog() }
      .onDisappear { if running { store.bridge.cancel() }; Task { try? await director.flush() } }
  }

  private var advancedBody: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Director").font(.title2)
        Spacer()
        Button("Execution history…") { showHistory = true }.disabled(runDirectory.isEmpty)
        if reviewingStructuredOutput {
          Button(showWorkflowInputs ? "Focus on review" : "Inputs and steps") { showWorkflowInputs.toggle() }
        }
        Menu("Workflows") {
          ForEach(catalog.indices, id: \.self) { index in
            Button(catalog[index]["name"]?.workflowText ?? "Workflow") { configure(catalog[index]) }
          }
        }.disabled(busy)
        Button("Import definition…") { importDefinition() }.disabled(busy)
        Button("Open job…") { openJob() }.disabled(busy)
        Button("Resume last") { loadJob(URL(fileURLWithPath: lastJobPath)) }
          .disabled(busy || lastJobPath.isEmpty)
        Button("Done") { dismiss() }.disabled(running)
      }
      Text("Plan your movie: refine the brief, review objects and shot plans, then import approved planning. Timeline generation remains a separate step.").font(.callout).foregroundStyle(.secondary)
      Text("Drafts are kept locally across sessions. Save updates a reviewable result; approval is always your explicit choice.").font(.caption).foregroundStyle(.secondary)
      Text(definition["name"]?.workflowText ?? "Loading workflows…").font(.headline)
      Text(definition["description"]?.workflowText ?? "")
        .font(.callout).foregroundStyle(.secondary)
      if legacyInventory {
        HStack {
          Text("This saved workflow only extracts subjects. Run remaining will not enrich descriptions or classify environments and sets.")
            .font(.caption).foregroundStyle(.orange)
          Button("Start guided workflow…") {
            let original = fields["brief"] ?? ""
            if let next = catalog.first(where: { $0["id"]?.workflowText == "weetodd.guided-movie-planning" }) {
              configure(next); fields["brief"] = original
            }
          }.disabled(busy)
        }
      }
      HSplitView {
        if !reviewingStructuredOutput || showWorkflowInputs {
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            ForEach(modelSpecs.keys.sorted(), id: \.self) { name in
              VStack(alignment: .leading, spacing: 4) {
                Text("Model · \(name)").font(.headline)
                Text("Qwen3.5 \(modelSpecs[name]?.workflowObject?["variant"]?.workflowText ?? "") · local")
                  .font(.caption).foregroundStyle(.secondary)
                HStack {
                  Text(modelPaths[name].map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Choose installed model")
                    .font(.caption).lineLimit(2).help(modelPaths[name] ?? "")
                  Spacer()
                  Button("Set up assistant…") { setupModel = name }
                }
              }
            }
            Divider()
            if guided {
              CreativeIntakeView(fields: Binding(get: { fields }, set: { fields = $0; result = nil; addedToProject = false; catalogFrozen = false }))
              inputField("images")
              inputField("library")
            } else {
              ForEach(inputSpecs.keys.sorted(), id: \.self) { name in inputField(name) }
            }
            Picker("Tokens per call", selection: Binding(get: { tokens }, set: { tokens = $0 })) {
              ForEach([256, 512, 1024], id: \.self) { Text(String($0)).tag($0) }
            }
            Text("Text and selected images stay local. Files are referenced in place. Each weighted step unloads its model when done.")
              .font(.caption).foregroundStyle(.secondary)
          }.padding(.trailing, 12)
        }.frame(minWidth: 280, idealWidth: 340, maxWidth: 430).disabled(busy)
        VStack(alignment: .leading) {
          Text("Steps").font(.headline)
          List(selection: Binding(get: { selectedStep }, set: { selectedStep = $0 })) {
            ForEach(steps, id: \.workflowID) { step in
              let id = step.workflowID
              VStack(alignment: .leading, spacing: 4) {
                Text(step["name"]?.workflowText ?? id)
                Text(result?.steps[id]?.approved == true ? "Approved" : result?.steps[id]?.status ?? "Pending").font(.caption).foregroundStyle(.secondary)
              }.tag(id)
            }
          }
          .disabled(busy)
          Button("Regenerate selected") { Task { await run(regenerate: selectedStep) } }
            .disabled(busy || selectedStep.isEmpty)
          Text("Regeneration invalidates dependent steps. Completed work is reused when its inputs are unchanged.")
            .font(.caption).foregroundStyle(.secondary)
        }.frame(minWidth: 180, idealWidth: 220, maxWidth: 260)
        }
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Text(selectedStep.isEmpty ? "Workflow result" : "Step result").font(.headline)
            Spacer()
            Button("All outputs (JSON)") { selectedStep = "" }.disabled(busy)
          }
          reviewContent
          if result?.needsAttention == true {
            Label("Review needs attention. Model reviews may be mistaken; inspect the results.", systemImage: "exclamationmark.triangle")
              .font(.caption).foregroundStyle(.orange)
          }
          Text("Movie workflows produce plans and endpoint descriptions. Create reference generates images only when you request it; timeline generation remains separate.")
            .font(.caption).foregroundStyle(.secondary)
          Text("Structure validation checks the data. Story approval is your separate review of its content.")
            .font(.caption).foregroundStyle(.secondary)
        }.frame(minWidth: 320, maxWidth: .infinity)
      }
      if let failure = director.persistenceError { Text("Draft could not be saved: " + failure).font(.callout).foregroundStyle(.red) }
      if let error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
      HStack {
        if running {
          ProgressView().controlSize(.small)
          WorkflowLiveProgress(bridge: store.bridge)
          Button("Pause") { store.bridge.cancel() }
        } else {
          Button("Run next") { Task { await run(maxSteps: 1) } }.disabled(busy || definition.isEmpty)
          Button("Run remaining") { Task { await run() } }.buttonStyle(.borderedProminent)
            .disabled(busy || definition.isEmpty)
          if let result { Text(String(format: "%@ · %.1f s total", result.status.capitalized, result.totalSeconds)).font(.caption) }
        }
        Spacer()
        if let result, result.steps.values.contains(where: { $0.status == "completed" && ($0.outputs?["story"] != nil || $0.outputs?["clips"] != nil || $0.outputs?["subjects"] != nil) }) {
          Button(addedToProject ? "Added to project" : "Add to project") {
            do {
              try validateTarget(); try store.importPlanning(result, sourceID: runDirectory, sourceText: fields["brief"] ?? "", referenceBindings: reviewAssetBindings, librarySelections: librarySelections)
              documentTarget = DirectorDocumentTarget(store: store); addedToProject = true
              if let onProjectImport { onProjectImport(); dismiss() }
            } catch { self.error = error.localizedDescription }
          }.disabled(busy || (guided && !guidedImportReady))
            .help(guided && !guidedImportReady ? "Review the creative brief, assets, shots and final prompt drafts before importing this guided plan." : "Import the reviewed plan")
        }
        Button("Export job…") { exportJob() }.disabled(busy || definition.isEmpty)
        Button("Reveal records") { NSWorkspace.shared.open(URL(fileURLWithPath: runDirectory)) }
          .disabled(runDirectory.isEmpty || result == nil)
        if onPrompt != nil {
          Button("Use proposal") {
            do { try validateTarget(); if let prompt = result?.prompt { onPrompt?(prompt); dismiss() } } catch { self.error = error.localizedDescription }
          }.disabled(busy || result?.prompt == nil)
        }
      }
    }
  }

  @ViewBuilder private func inputField(_ name: String) -> some View {
    let spec = inputSpecs[name]?.workflowObject ?? [:]
    let type = spec["type"]?.workflowText ?? "text"
    VStack(alignment: .leading, spacing: 5) {
      Text(spec["label"]?.workflowText ?? name).font(.headline)
      if type == "image_list" {
        ForEach(imageInputs[name] ?? []) { image in
          HStack {
            WorkspaceImage(path: image.path).frame(width: 58, height: 42).clipped()
            Text(image.label).font(.caption).lineLimit(2)
            Spacer()
            Button { imageInputs[name]?.removeAll { $0.id == image.id }; result = nil } label: { Image(systemName: "xmark") }
          }
        }
        Button("Add reference images…") { addImages(name) }
      } else if type == "object_catalog" {
        Text("\(libraryCandidates.count) movie/global candidates · metadata only").font(.caption)
        Button("Refresh reusable object matches") { do { try refreshObjectCatalog(updateExecutionInputs: true) } catch { self.error = error.localizedDescription } }
          .disabled(result != nil)
      } else if type == "subject_list" {
        Text("Structured inventory loaded from the saved job.").font(.caption).foregroundStyle(.secondary)
      } else if type == "boolean" {
        Toggle("Enabled", isOn: Binding(get: { fields[name] == "true" }, set: { fields[name] = $0 ? "true" : "false"; result = nil }))
      } else {
        let binding = Binding(get: { fields[name] ?? "" }, set: { fields[name] = $0; result = nil })
        if type == "text" { TextEditor(text: binding).frame(minHeight: 70, maxHeight: 120) }
        else { TextField(type == "integer" ? "Whole number" : "Number", text: binding) }
      }
    }
  }
  private var outputText: String {
    let value = selectedStep.isEmpty ? result?.outputs : result?.steps[selectedStep]?.outputs
    guard let value, let data = try? JSONEncoder.pretty.encode(value), let text = String(data: data, encoding: .utf8) else {
      return result?.steps[selectedStep]?.error ?? "Run a step, then inspect its output here before continuing."
    }
    return text
  }
  private func configure(_ value: [String: JSONValue]) {
    addedToProject = false; libraryCandidates = []; librarySelections = [:]; catalogFrozen = false; structuredInputs = [:]
    director.state.executionStarted = false
    definition = value; fields = [:]; imageInputs = [:]; modelPaths = [:]; reviewAssetBindings = [:]
    result = nil; director.state.reviews = [:]; selectedStep = ""; error = nil; runID = UUID()
    runDirectory = store.dataDirectory.appendingPathComponent("Workflows/Runs/\(runID.uuidString)").path
    for (name, raw) in inputSpecs {
      let spec = raw.workflowObject ?? [:]
      fields[name] = spec["default"]?.workflowText ?? ""
      if ["subject_list", "object_catalog"].contains(spec["type"]?.workflowText ?? "") { structuredInputs[name] = spec["default"] }
      if spec["type"]?.workflowText == "image_list" { imageInputs[name] = [] }
    }
    for name in modelSpecs.keys { if !promptModel.isEmpty { modelPaths[name] = promptModel } }
    if guided {
      fields["frame_rate"] = String(Int(store.project.settings.fps.rounded()))
      let settings = store.project.settings
      fields["presentation"] = settings.width == settings.height ? "Square" : settings.width > settings.height ? "Widescreen" : "Vertical"
      if fields["brief"]?.isEmpty != false { fields["brief"] = store.planning.sourceText }
    }
  }
  private func loadCatalog() async {
    guard documentTarget == nil else { return }
    documentTarget = DirectorDocumentTarget(store: store)
    do {
      let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
      let imageIdentity = initialImages.map { ["path": $0.path, "label": $0.label, "enabled": String($0.enabled)] }
      let discriminator = sessionScope + initialBuiltin
        + String(decoding: try encoder.encode(initialInputs), as: UTF8.self)
        + String(decoding: try encoder.encode(imageIdentity), as: UTF8.self)
      let key = SHA256.hash(data: Data(discriminator.utf8)).map { String(format: "%02x", $0) }.joined()
      try await director.open(store.directorSessionURL(key: key))
      if result != nil { await director.refreshCheckpoint() }
    } catch { self.error = error.localizedDescription; return }
    do {
      let response = try await store.bridge.invoke("workflow-catalog", runtime: store.runtime, payload: [:])
      catalog = try JSONDecoder().decode([[String: JSONValue]].self, from: JSONSerialization.data(withJSONObject: response["definitions"] ?? []))
      guard let first = catalog.first(where: { $0["id"]?.workflowText == initialBuiltin }) ?? catalog.first else { return }
      try validateTarget(); if !definition.isEmpty { return }; configure(first)
      for (name, value) in initialInputs { fields[name] = value.workflowText; if ["subject_list", "object_catalog"].contains(inputSpecs[name]?.workflowObject?["type"]?.workflowText ?? "") { structuredInputs[name] = value } }
      if inputSpecs["images"] != nil { imageInputs["images"] = initialImages }
    } catch { self.error = error.localizedDescription }
  }
  private func job() throws -> WorkflowJob {
    var inputs = [String: JSONValue](), assets = reviewAssetBindings
    if inputSpecs.values.contains(where: { $0.workflowObject?["type"]?.workflowText == "object_catalog" }), !catalogFrozen { try refreshObjectCatalog(updateExecutionInputs: true) }
    for (name, spec) in inputSpecs {
      let type = spec.workflowObject?["type"]?.workflowText ?? "text"
      if type == "image_list" {
        let images = imageInputs[name] ?? []
        guard images.count <= 8 else { throw StudioError.invalid("Use at most eight images per input.") }
        inputs[name] = .array(images.enumerated().map { index, image in
          let ref = "asset:\(name)-\(index + 1)"; assets[ref] = image.path; return .string(ref)
        })
      } else if type == "object_catalog" {
        inputs[name] = structuredInputs[name] ?? .array([])
      } else if type == "subject_list" {
        guard let value = structuredInputs[name] else { throw StudioError.invalid("This workflow needs a subject inventory.") }; inputs[name] = value
      } else { inputs[name] = try WorkflowJob.input(fields[name] ?? "", type: type) }
    }
    var request = WorkflowJob(definition: definition, inputs: inputs, models: modelPaths,
                       assets: assets, runDirectory: runDirectory, maxTokens: tokens)
    request.librarySelections = librarySelections
    request.coverageLibraryCandidates = libraryCandidates
    return request
  }
  private func run(maxSteps: Int? = nil, regenerate: String? = nil, preserveResult: Bool = false) async {
    running = true; error = nil
    // Keep reviewable results visible while work runs or fails.
    defer { running = false }
    do {
      try validateTarget(); var request = try job()
      director.state.executionStarted = true
      let saved = store.dataDirectory.appendingPathComponent("Workflows/Jobs/\(runID.uuidString).json")
      try FileManager.default.createDirectory(at: saved.deletingLastPathComponent(), withIntermediateDirectories: true)
      try JSONEncoder.pretty.encode(request).write(to: saved, options: .atomic)
      lastJobPath = saved.path
      request.maxSteps = maxSteps; request.regenerate = regenerate
      try await director.execute { try await store.bridge.invoke("workflow-run", runtime: store.runtime, payload: try request.object()) }
      pruneLibrarySelections()
      error = result?.error
      if guided, presentation.mode == .focused {
        if case .review(let id) = presentation.nextAction { selectedStep = id }
        else { selectedStep = presentation.stepID(for: .shots) ?? result?.preferredReviewStepID ?? "" }
      } else { selectedStep = result?.preferredReviewStepID ?? "" }
    } catch { self.error = error.localizedDescription }
  }
  private func pruneLibrarySelections() {
    let subjects = (try? result?.subjectsForImport()) ?? []
    librarySelections = librarySelections.filter { id, match in subjects.first(where: { $0.id == id })?.validatesReuse(match) == true }
    if guided {
      for subject in subjects {
        if let definition = subject.reusedDefinition {
          var match = definition.libraryMatch; match.sourceRevision = subject.reuseSelectionRevision
          librarySelections[subject.id] = match
        }
      }
    }
  }
  private func refreshObjectCatalog(updateExecutionInputs: Bool = false) throws {
    let objectText = result?.steps[selectedStep]?.outputs?["subjects"].flatMap { try? JSONEncoder().encode($0) }.map { String(decoding: $0, as: UTF8.self) } ?? ""
    libraryCandidates = try store.objectCatalog(matching: (fields["brief"] ?? "") + " " + objectText)
    if updateExecutionInputs {
      let value = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(libraryCandidates))
      for (name, spec) in inputSpecs where spec.workflowObject?["type"]?.workflowText == "object_catalog" { structuredInputs[name] = value }
      catalogFrozen = true
    }
  }
  private func persistReviewJob() throws {
    let saved = store.dataDirectory.appendingPathComponent("Workflows/Jobs/\(runID.uuidString).json")
    try FileManager.default.createDirectory(at: saved.deletingLastPathComponent(), withIntermediateDirectories: true)
    try JSONEncoder.pretty.encode(job()).write(to: saved, options: .atomic); lastJobPath = saved.path
  }
  private func selectLibraryMatch(_ id: String, _ match: LibraryObjectMatch?) {
    if guided {
      Task {
        if await review("apply_library_definition", item: id, outputs: nil, instruction: nil,
          coverageCatalog: libraryCandidates, libraryChoice: match) {
          pruneLibrarySelections()
          do { try persistReviewJob() } catch { self.error = error.localizedDescription }
        }
      }
      return
    }
    var chosen = match
    if let subject = try? result?.subjectsForImport().first(where: { $0.id == id }) {
      chosen?.sourceRevision = subject.reuseSelectionRevision
    }
    librarySelections[id] = chosen
    do { try persistReviewJob() } catch { self.error = error.localizedDescription }
  }
  private func createMissingObject(_ owner: String, _ suggestion: MissingObjectProposal) {
    do { try validateTarget() } catch { self.error = error.localizedDescription; return }
    let key = "coverage:" + runDirectory + ":" + owner + ":" + suggestion.name.lowercased()
    guard !store.planning.subjects.contains(where: { $0.sourceKey == key }) else { return }
    var object = PlanningSubject(name: suggestion.name, kind: suggestion.kind)
    object.details = suggestion.description; object.evidence = suggestion.evidence.joined(separator: "\n"); object.sourceKey = key
    store.changePlanning { $0.subjects.append(object) }; documentTarget = DirectorDocumentTarget(store: store)
  }
  private func reviewObjectCoverage() async -> Bool {
    do { try refreshObjectCatalog(); try persistReviewJob() } catch { self.error = error.localizedDescription; return false }
    return await review("review_object_coverage", item: nil, outputs: nil, instruction: nil, coverageCatalog: libraryCandidates)
  }
  private func applyReferences(_ id: String, _ paths: [String], action: String, stepID: String? = nil, expectedRevision: String? = nil) async -> Bool {
    guard expectedRevision == nil || expectedRevision == result?.revision else { error = "This reference request belongs to an earlier review. Reopen the subject before attaching it."; return false }
    do { try validateTarget() } catch { self.error = error.localizedDescription; return false }
    do {
      let references = try bindReferences(paths)
      let saved = store.dataDirectory.appendingPathComponent("Workflows/Jobs/\(runID.uuidString).json")
      try FileManager.default.createDirectory(at: saved.deletingLastPathComponent(), withIntermediateDirectories: true)
      try JSONEncoder.pretty.encode(job()).write(to: saved, options: .atomic)
      lastJobPath = saved.path
      return await review(action, item: id, outputs: nil, instruction: nil, references: references, stepID: stepID)
    } catch { self.error = error.localizedDescription; return false }
  }
  private func review(_ action: String, item: String?, outputs: [String: JSONValue]?, instruction: String?, fieldScope: DirectorShotRepairScope? = nil, references: [String]? = nil, coverageCatalog: [LibraryObjectCandidate]? = nil, libraryChoice: LibraryObjectMatch? = nil, stepID: String? = nil) async -> Bool {
    do { try validateTarget() } catch { self.error = error.localizedDescription; return false }
    guard let revision = result?.revision else { return false }
    let sid = stepID ?? selectedStep
    if action == "approve", director.state.reviews[sid]?.hasUnsavedChanges(outputs: result?.steps[sid]?.outputs ?? [:], referenceBindings: reviewAssetBindings) == true {
      error = "Save or discard this review's draft edits before approval."; return false
    }
    running = true; error = nil
    do {
      try persistReviewJob(); var payload = try job().object()
      var mutation: [String: Any] = ["action": action, "stepID": sid, "expectedRevision": revision]
      if let item { mutation["itemID"] = item }
      if let instruction { mutation["instruction"] = instruction }
      if let fieldScope { mutation["fieldScope"] = fieldScope.rawValue }
      if let references { mutation["referenceAssets"] = references }
      if let outputs { mutation["outputs"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(outputs)) }
      if let coverageCatalog { mutation["library"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(coverageCatalog)) }
      if action == "apply_library_definition" {
        if let libraryChoice {
          mutation["libraryChoice"] = ["objectID": libraryChoice.objectID, "packageID": libraryChoice.packageID,
            "version": libraryChoice.version, "scope": libraryChoice.scope ?? "global", "definitionRevision": libraryChoice.definitionRevision ?? ""] as [String: Any]
        } else { mutation["libraryChoice"] = NSNull() }
      }
      payload["review"] = mutation
      try await director.execute { try await store.bridge.invoke("workflow-review", runtime: store.runtime, payload: payload) }
      pruneLibrarySelections()
      running = false
      if action == "repair" {
        await run(maxSteps: 1, preserveResult: true); selectedStep = sid
        return result?.repairCompleted(stepID: sid, itemID: item ?? "") == true
      }
      return true
    } catch {
      self.error = error.localizedDescription; running = false
      return false
    }
  }
  private func addImages(_ name: String) {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
    guard panel.runModal() == .OK else { return }
    let existing = imageInputs[name] ?? []
    guard existing.count + panel.urls.count <= 8 else { error = "Use at most eight images per input."; return }
    imageInputs[name] = existing + panel.urls.map { PromptAssistantImage(path: $0.path, label: $0.deletingPathExtension().lastPathComponent) }
    result = nil
  }
  private func importDefinition() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      running = true; defer { running = false }
      do {
        try validateTarget()
        let response = try await store.bridge.invoke("workflow-validate", runtime: store.runtime, payload: ["definitionPath": url.path])
        let report = response["report"] as? [String: Any]
        guard report?["valid"] as? Bool == true else {
          throw StudioError.invalid((report?["issues"] as? [[String: Any]])?.first?["message"] as? String ?? "Invalid workflow definition")
        }
        let value = try JSONDecoder().decode([String: JSONValue].self, from: JSONSerialization.data(withJSONObject: response["definition"] ?? [:]))
        try validateTarget(); configure(value)
      } catch { self.error = error.localizedDescription }
    }
  }
  private func exportJob() {
    do {
      var payload = try job().object()
      payload["runtime"] = ["drawThingsHelperPath": store.runtime.drawThingsHelperPath]
      let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "workflow-job.json"
      guard panel.runModal() == .OK, let url = panel.url else { return }
      try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  private func openJob() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
    guard panel.runModal() == .OK, let url = panel.url else { return }; loadJob(url)
  }
  private func loadJob(_ url: URL) {
    Task {
      running = true; defer { running = false }
      do {
        try validateTarget()
        try await director.importJob(url) { definition in
          let encoded = try JSONEncoder().encode(definition)
          let response = try await store.bridge.invoke("workflow-validate", runtime: store.runtime,
            payload: ["definition": try JSONSerialization.jsonObject(with: encoded)])
          let report = response["report"] as? [String: Any]
          guard report?["valid"] as? Bool == true else {
            throw StudioError.invalid((report?["issues"] as? [[String: Any]])?.first?["message"] as? String ?? "Invalid workflow definition")
          }
          try validateTarget()
        }
        pruneLibrarySelections(); lastJobPath = url.path; error = nil
      } catch { self.error = error.localizedDescription }
    }
  }
  private func validateTarget() throws {
    guard let documentTarget else { throw StudioError.invalid("Director is still opening.") }
    try documentTarget.validate(store: store)
  }
  private func reviewDraft(_ step: String) -> Binding<DirectorReviewDraft> {
    Binding(get: { director.state.reviews[step] ?? DirectorReviewDraft() }, set: { director.state.reviews[step] = $0 })
  }
  private func bindReferences(_ paths: [String]) throws -> [String] {
    try DirectorReferenceAssets.bind(paths, in: &director.state.reviewAssetBindings)
  }

}

private struct WorkflowLiveProgress: View {
  @ObservedObject var bridge: Bridge
  var body: some View { Text(bridge.message).font(.caption).lineLimit(2) }
}
private extension JSONValue {
  var workflowText: String {
    switch self {
    case .string(let text): return text
    case .integer(let n): return String(n)
    case .number(let n): return String(n)
    case .boolean(let b): return b ? "true" : "false"
    default: return ""
    }
  }
  var workflowObject: [String: JSONValue]? { if case .object(let value) = self { return value }; return nil }
}
private extension Dictionary where Key == String, Value == JSONValue {
  var workflowID: String { self["id"]?.workflowText ?? "" }
}
private extension JSONEncoder {
  static var pretty: JSONEncoder { let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; return encoder }
}

private extension WorkflowView {
  var focusedBody: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(alignment: .top) {
        VStack(alignment: .leading, spacing: 4) {
          Text("Director").font(.title2.bold())
          Text("Shape the brief, review the subjects, then approve your shots.").font(.callout).foregroundStyle(.secondary)
        }
        Spacer()
        Button("Inspector…") { showTechnicalInspector = true }
        Menu("Workflow") {
          ForEach(catalog.indices, id: \.self) { index in
            Button("Start " + (catalog[index]["name"]?.workflowText ?? "workflow")) { configure(catalog[index]) }
          }
          Divider()
          Button("Import definition…") { importDefinition() }
          Button("Open job…") { openJob() }
          Button("Resume last") { loadJob(URL(fileURLWithPath: lastJobPath)) }.disabled(lastJobPath.isEmpty)
          Button("Export job…") { exportJob() }
        }.disabled(busy)
        Button("Done") { dismiss() }.disabled(running)
      }
      HStack(spacing: 12) {
        ForEach(DirectorReviewPhase.allCases) { phase in
          Button {
            if let id = presentation.stepID(for: phase) { selectedStep = id }
          } label: {
            HStack {
              Image(systemName: presentation.isApproved(phase) ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(presentation.isApproved(phase) ? Color.green : Color.secondary)
              Text(phase.label).font(.headline)
              Spacer()
              if presentation.stepID(for: phase) == selectedStep && !selectedStep.isEmpty {
                Text("Reviewing").font(.caption).foregroundStyle(.secondary)
              }
            }.padding(10).frame(maxWidth: .infinity)
              .background(presentation.phase(for: selectedStep) == phase && !selectedStep.isEmpty ? Theme.raised : Color.clear,
                in: RoundedRectangle(cornerRadius: 8))
          }.buttonStyle(.plain).disabled(busy || presentation.stepID(for: phase) == nil)
        }
      }
      if DirectorReviewMode.supports(definition), !director.state.reviewModeLocked {
        Picker("Review style", selection: Binding(get: { presentation.mode }, set: { mode in
          do { try director.state.setReviewMode(mode) } catch { self.error = error.localizedDescription }
        })) {
          ForEach(DirectorReviewMode.allCases) { Text($0.label).tag($0) }
        }.pickerStyle(.segmented).frame(maxWidth: 540).disabled(busy)
        Text(presentation.mode == .focused ? "Pause at Brief, Subjects and Shots. All intermediate work remains available in the inspector." : "Pause at all eight decisions, including classification, reuse, visual design, story and final prompts.")
          .font(.caption).foregroundStyle(.secondary)
      } else {
        Text("\(presentation.mode == .focused ? "Focused" : "Detailed") review · this job keeps its saved approval stages")
          .font(.caption).foregroundStyle(.secondary)
      }
      Divider()
      if result == nil {
        HSplitView {
          ScrollView {
            VStack(alignment: .leading, spacing: 16) {
              CreativeIntakeView(fields: Binding(get: { fields }, set: { fields = $0; addedToProject = false; catalogFrozen = false }))
              inputField("images")
              inputField("library")
            }.padding(.trailing, 16)
          }.frame(minWidth: 600)
          VStack(alignment: .leading, spacing: 14) {
            Text("Start with your story").font(.headline)
            Text("Director will prepare a brief for your decisions, develop reusable subjects, and plan the shots. Your approval is required at each review.")
              .font(.callout).foregroundStyle(.secondary)
            ForEach(modelSpecs.keys.sorted(), id: \.self) { name in
              Text(modelPaths[name].map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Choose a local assistant")
                .font(.caption).lineLimit(2)
              Button("Set up assistant…") { setupModel = name }
            }
            Text("Text and references stay local. Drafts are kept across sessions.").font(.caption).foregroundStyle(.secondary)
            Spacer()
          }.padding(.leading, 12).frame(minWidth: 250, maxWidth: 330)
        }.disabled(busy)
      } else {
        reviewContent.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
      }
      if let failure = director.persistenceError {
        Text("Draft could not be saved: " + failure).font(.caption).foregroundStyle(.red)
      }
      if let error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
      Divider()
      HStack {
        if running {
          ProgressView().controlSize(.small)
          WorkflowLiveProgress(bridge: store.bridge)
          Spacer()
          Button("Pause") { store.bridge.cancel() }
        } else {
          VStack(alignment: .leading, spacing: 3) {
            Text(nextActionText).font(.callout)
            Text("Save keeps your edits. Approve locks saved choices. Add to project imports the reviewed plan.")
              .font(.caption).foregroundStyle(.secondary)
          }
          Spacer()
          if hasUnsavedReviews, let unsavedID = steps.map(\.workflowID).first(where: { id in
            director.state.reviews[id]?.hasUnsavedChanges(outputs: result?.steps[id]?.outputs ?? [:], referenceBindings: reviewAssetBindings) == true
          }), unsavedID != selectedStep {
            Button("Review unsaved edits") { selectedStep = unsavedID }.disabled(busy)
          }
          switch presentation.nextAction {
          case .start:
            if let name = preparation.missingModel {
              Button("Set up assistant…") { setupModel = name }.buttonStyle(.borderedProminent).disabled(busy)
            } else {
              Button("Prepare brief") { Task { await run() } }.buttonStyle(.borderedProminent)
                .disabled(busy || definition.isEmpty || !preparation.canPrepare)
            }
          case .review(let id):
            if selectedStep != id {
              Button("Review \(presentation.phase(for: id).label.lowercased())") { selectedStep = id }.buttonStyle(.borderedProminent).disabled(busy)
            }
          case .continuePlanning:
            Button("Continue planning") { Task { await run() } }.buttonStyle(.borderedProminent).disabled(busy || hasUnsavedReviews)
          case .addToProject:
            Button(addedToProject ? "Added to project" : "Add to project") { addReviewedPlan() }
              .buttonStyle(.borderedProminent).disabled(busy || addedToProject || hasUnsavedReviews)
          }
        }
      }
    }
  }
  var nextActionText: String {
    if hasUnsavedReviews { return "Next: save or discard your draft edits." }
    switch presentation.nextAction {
    case .start:
      if preparation.missingModel != nil { return "Next: set up a local assistant." }
      return preparation.hasBrief ? "Next: prepare your creative brief." : "Next: write your movie idea."
    case .review(let id): return "Next: review and approve \(presentation.phase(for: id).label.lowercased())."
    case .continuePlanning: return "Next: continue to the next review."
    case .addToProject: return addedToProject ? "This plan has been added to the project." : "Your reviewed plan is ready to add to the project."
    }
  }
  @ViewBuilder var reviewContent: some View {
    if let step = result?.steps[selectedStep], step.outputs != nil, result?.revision != nil {
      ForEach(step.warnings ?? [], id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
      let reviewedStep = selectedStep
      let reviewedRevision = result?.revision
      WorkflowReviewPanel(draft: reviewDraft(selectedStep), bindReferences: bindReferences, step: step, busy: busy && !store.referenceSheetOpen,
        referenceBindings: reviewAssetBindings,
        onDescriptionReview: { id, paths in await applyReferences(id, paths, action: "review_description", stepID: reviewedStep, expectedRevision: reviewedRevision) },
        onReferenceSave: { id, paths in await applyReferences(id, paths, action: "set_reference_assets", stepID: reviewedStep, expectedRevision: reviewedRevision) },
        subjectScope: runDirectory, onCoverageReview: reviewObjectCoverage,
        libraryCandidates: libraryCandidates, librarySelections: librarySelections,
        onLibrarySelect: selectLibraryMatch, onCreateObject: createMissingObject,
        allowKindEditing: (guided && selectedStep == "subjects_coverage") || steps.first(where: { $0.workflowID == selectedStep })?["operation"]?.workflowText == "project.classify_subjects@1",
        subjectApprovalLabel: guided && selectedStep == "classify" ? "classification" : (guided && selectedStep == "inventory" ? "inventory item" : "description"),
        preserveApprovedCast: guided, focused: guided && presentation.mode == .focused,
        subjectNames: Dictionary(((try? result?.subjectsForImport()) ?? []).map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first }),
        storyContext: decodeOutput(WorkflowStoryOutline.self, stepID: "story", key: "story"),
        promptPreview: decodeOutput(H3PromptPreview.self, stepID: "prompt_preview", key: "h3_prompts")) { action, item, outputs, instruction, scope in
          await review(action, item: item, outputs: outputs, instruction: instruction, fieldScope: scope, stepID: reviewedStep)
        }.id(selectedStep)
    } else {
      ScrollView {
        Text(guided && selectedStep.isEmpty ? "Continue planning to prepare your next review." : outputText)
          .font(.callout).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(10)
      }.background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
    }
  }
  func decodeOutput<T: Decodable>(_ type: T.Type, stepID: String, key: String) -> T? {
    guard let value = result?.steps[stepID]?.outputs?[key], let data = try? JSONEncoder().encode(value) else { return nil }
    return try? JSONDecoder().decode(type, from: data)
  }
  func addReviewedPlan() {
    guard let result, guidedImportReady, !hasUnsavedReviews else { return }
    do {
      try validateTarget()
      try store.importPlanning(result, sourceID: runDirectory, sourceText: fields["brief"] ?? "", referenceBindings: reviewAssetBindings, librarySelections: librarySelections)
      documentTarget = DirectorDocumentTarget(store: store); addedToProject = true
      if let onProjectImport { onProjectImport(); dismiss() }
    } catch { self.error = error.localizedDescription }
  }
  var technicalInspector: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Director inspector").font(.title2)
        Spacer()
        Button("Done") { showTechnicalInspector = false }
      }
      Text("Execution steps, saved outputs and local records. Selecting a step changes the review shown in Director.")
        .font(.caption).foregroundStyle(.secondary)
      HSplitView {
        VStack(alignment: .leading, spacing: 10) {
          List(selection: Binding(get: { selectedStep }, set: { selectedStep = $0 })) {
            ForEach(steps, id: \.workflowID) { step in
              VStack(alignment: .leading, spacing: 3) {
                Text(step["name"]?.workflowText ?? step.workflowID)
                Text(result?.steps[step.workflowID]?.approved == true ? "Approved" : result?.steps[step.workflowID]?.status ?? "Pending")
                  .font(.caption).foregroundStyle(.secondary)
              }.tag(step.workflowID)
            }
          }.disabled(busy)
          Button("Review selected step") { showTechnicalInspector = false }.disabled(selectedStep.isEmpty)
          Button("Regenerate selected") { Task { await run(regenerate: selectedStep) } }.disabled(busy || selectedStep.isEmpty || hasUnsavedReviews)
          Button("Run one step") { Task { await run(maxSteps: 1) } }.disabled(busy || hasUnsavedReviews)
          Picker("Tokens per call", selection: Binding(get: { tokens }, set: { tokens = $0 })) {
            ForEach([256, 512, 1024], id: \.self) { Text(String($0)).tag($0) }
          }.disabled(busy)
          Text("Regeneration invalidates dependent results and approvals. Saved drafts remain available.").font(.caption).foregroundStyle(.secondary)
        }.frame(minWidth: 240, maxWidth: 320)
        VStack(alignment: .leading, spacing: 8) {
          Picker("Inspector content", selection: $inspectorShowsInputs) {
            Text("Saved output").tag(false); Text("Inputs and models").tag(true)
          }.pickerStyle(.segmented)
          if inspectorShowsInputs {
            ScrollView {
              VStack(alignment: .leading, spacing: 14) {
                ForEach(modelSpecs.keys.sorted(), id: \.self) { name in
                  Text("Local assistant · " + name).font(.headline)
                  Text(modelPaths[name] ?? "No model selected").font(.caption).textSelection(.enabled)
                  Button("Set up assistant…") { inspectorSetupModel = name; showTechnicalInspector = false }
                }
                Text("Changing creative inputs regenerates affected results and reopens their review.")
                  .font(.caption).foregroundStyle(.secondary)
                ForEach(inputSpecs.keys.sorted(), id: \.self) { name in inputField(name) }
              }
            }.disabled(busy || hasUnsavedReviews)
          } else {
            HStack {
              Text(selectedStep.isEmpty ? "All outputs" : "Saved output · \(selectedStep)").font(.headline)
              Spacer()
              Button("All outputs") { selectedStep = "" }.disabled(busy)
            }
            ScrollView {
              Text(outputText).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
          }
          if let result { Text(String(format: "%@ · %.1f s total", result.status.capitalized, result.totalSeconds)).font(.caption) }
        }.padding(.leading, 12).frame(minWidth: 450)
      }
      HStack {
        Button("Execution history…") { inspectorShowsHistory = true }.disabled(runDirectory.isEmpty)
        Button("Reveal records") { NSWorkspace.shared.open(URL(fileURLWithPath: runDirectory)) }.disabled(runDirectory.isEmpty || result == nil)
        Button("Export job…") { exportJob() }.disabled(busy)
        Spacer()
        Text("Definition \(definition["version"]?.workflowText ?? "") · \(presentation.requiredStepIDs.count) required reviews")
          .font(.caption).foregroundStyle(.secondary)
      }
    }.padding(24).frame(width: 940, height: 650)
      .sheet(isPresented: $inspectorShowsHistory) {
        WorkflowHistoryView(directory: runDirectory, workflowBusy: busy) { id in selectedStep = id }
      }
  }
}
