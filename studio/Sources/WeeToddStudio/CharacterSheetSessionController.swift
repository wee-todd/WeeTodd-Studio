import AppKit
import Combine
import Foundation
import StudioCore

/// Owns one character document, never borrows the movie's image draft or autosave path.
@MainActor final class CharacterSheetSessionController: ObservableObject, Identifiable {
  nonisolated let id: UUID
  @Published var document: CharacterSheetDocument
  @Published var running = false
  @Published var status = "Ready"
  @Published var error: String?
  @Published var previewPath: String?
  @Published var catalog: [String: Any] = [:]
  private(set) var catalogFailure: (connectionID: String, message: String)?
  let bridge: Bridge
  let store: StudioStore
  let storage: CharacterSheetDocumentStore
  var onUse: ((MediaAsset) async -> Bool)?
  private var undoStack: [CharacterSheetDocument] = []
  private var redoStack: [CharacterSheetDocument] = []
  private var revisionClock: Int
  private var observation: AnyCancellable?
  var cancelled = false
  var workTask: Task<Void, Never>?
  var compiled: CompiledCharacterSheet { CharacterSheetCompiler.compile(document.definition) }
  var canUndo: Bool { !undoStack.isEmpty && !running }
  var canRedo: Bool { !redoStack.isEmpty && !running }
  var connection: DrawThingsConnection? { localConnections.first { $0.id == document.draft.profileID } }
  var localConnections: [DrawThingsConnection] {
    store.drawThingsConnections.filter { $0.route == "grpc" && ["localhost", "127.0.0.1", "::1"].contains($0.host.lowercased()) }
  }
  init(document: CharacterSheetDocument, store: StudioStore, storage: CharacterSheetDocumentStore, bridge: Bridge? = nil) {
    self.id = document.id; self.document = document; self.store = store; self.storage = storage
    self.bridge = bridge ?? store.bridge.independent(); self.revisionClock = document.revision
    if self.document.draft.profileID.isEmpty { self.document.draft.profileID = localConnections.first?.id ?? "" }
    observation = self.bridge.objectWillChange.throttle(for: .milliseconds(500), scheduler: RunLoop.main, latest: true)
      .sink { [weak self] _ in self?.objectWillChange.send() }
  }
  func save() {
    do { try storage.save(document) } catch { self.error = error.localizedDescription }
  }
  func edit(_ body: (inout CharacterSheetDocument) -> Void) {
    guard !running else { return }
    undoStack.append(document); if undoStack.count > 40 { undoStack.removeFirst() }; redoStack.removeAll()
    body(&document); advanceRevision(); save()
  }
  private func advanceRevision() {
    revisionClock = max(revisionClock, document.revision) + 1
    document.revision = revisionClock
  }
  func undo() {
    guard canUndo, let previous = undoStack.popLast() else { return }
    redoStack.append(document); document = previous; advanceRevision(); save()
  }
  func redo() {
    guard canRedo, let next = redoStack.popLast() else { return }
    undoStack.append(document); document = next; advanceRevision(); save()
  }
  func setText(_ path: String, _ text: String) {
    edit { doc in
      let old = doc.definition.entry(at: path)
      var entry = Self.entry(text, path: path, source: .userAuthored)
      entry.revision = (old?.revision ?? 0) + 1
      doc.definition.setEntry(entry, at: path)
    }
  }
  static func entry(_ text: String, path: String, source: CharacterFieldSource) -> CharacterFieldEntry {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .init(state: .unspecified, source: source) }
    let kind = CharacterFieldCatalog.shared.field(for: path)?.valueKind
    let value: CharacterFieldValue
    switch kind {
    case .color:
      let colors: [String: (Double, Double, Double)] = ["blue": (0,0,1), "brown": (0.35,0.18,0.08), "green": (0,0.5,0), "black": (0,0,0), "amber": (1,0.6,0), "gray": (0.5,0.5,0.5), "hazel": (0.5,0.4,0.2)]
      let rgb = colors[text.lowercased()] ?? (0.5,0.5,0.5)
      value = .color(.init(red: rgb.0, green: rgb.1, blue: rgb.2, displayValue: text))
    case .measurement:
      let parts = text.split(separator: " ")
      if parts.count == 2, let number = Decimal(string: String(parts[0])) { value = .measurement(value: number, unit: String(parts[1])) }
      else { value = .text(text) } // Retain incomplete edits; compiler explains the expected type.
    case .orderedChoices: value = .orderedChoices(text.split(separator: ",").map { .init(id: String($0).trimmingCharacters(in: .whitespaces), displayValue: String($0).trimmingCharacters(in: .whitespaces)) })
    case .choice: value = .choice(id: text.lowercased(), displayValue: text)
    default: value = .text(text)
    }
    return .value(value, source: source)
  }
  func setState(_ path: String, _ state: CharacterFieldState) {
    edit { $0.definition.setEntry(.init(state: state, source: .userAuthored), at: path) }
  }
  func setRequired(_ path: String, _ required: Bool) {
    edit { if required { $0.definition.settings.requiredFieldPaths.insert(path) } else { $0.definition.settings.requiredFieldPaths.remove(path) } }
  }
  func selectStyle(_ id: String) { edit { $0.definition.settings.stylePresetID = id; $0.definition.settings.stylePresetVersion = 1 } }
  func addRecord(_ collection: String) {
    edit { doc in
      let item = CharacterRepeatableRecord(order: 0)
      switch collection {
      case "garments": var record = item; record.order = doc.definition.appearance.garments.count; doc.definition.appearance.garments.append(record)
      case "accessories": var record = item; record.order = doc.definition.appearance.accessories.count; doc.definition.appearance.accessories.append(record)
      case "features": var record = item; record.order = doc.definition.appearance.features.count; doc.definition.appearance.features.append(record)
      case "surfaces": var record = item; record.order = doc.definition.appearance.surfaces.count; doc.definition.appearance.surfaces.append(record)
      default: break
      }
    }
  }
  func removeRecord(_ collection: String, id: UUID) {
    edit { doc in
      switch collection {
      case "garments": doc.definition.appearance.garments.removeAll { $0.id == id }
      case "accessories": doc.definition.appearance.accessories.removeAll { $0.id == id }
      case "features": doc.definition.appearance.features.removeAll { $0.id == id }
      case "surfaces": doc.definition.appearance.surfaces.removeAll { $0.id == id }
      default: break
      }
      doc.definition.settings.requiredFieldPaths = doc.definition.settings.requiredFieldPaths.filter { !$0.contains(id.uuidString) }
    }
  }
  func applyProposals(batchID: UUID, selectedIDs: Set<String>) {
    guard !running, let index = document.proposals.firstIndex(where: { $0.id == batchID }) else { return }
    let batch = document.proposals[index]
    guard batch.documentID == id, batch.revision == document.revision else {
      document.proposals[index].stale = true; error = "The character changed. Rerun this analysis before applying its proposals."; save(); return
    }
    if !batch.sourcePath.isEmpty, (try? CharacterArtifactHash.file(batch.sourcePath)) != batch.sourceHash {
      document.proposals[index].stale = true; error = "The source image changed. Analyze it again."; save(); return
    }
    edit { doc in
      var rejected: [CharacterFieldProposal] = []
      var diagnostics = batch.diagnostics ?? []
      for proposal in batch.proposals where selectedIDs.contains(proposal.id) {
        func reject(_ code: String, _ message: String) {
          rejected.append(proposal)
          diagnostics.append(.init(code: code, proposalID: proposal.id,
            field: proposal.field, message: message))
        }
        guard !diagnostics.contains(where: { $0.proposalID == proposal.id }) else {
          rejected.append(proposal); continue
        }
        guard proposal.state == "value" else {
          reject("proposal.state", "Only populated value proposals can be applied."); continue
        }
        guard let field = CharacterFieldCatalog.shared.field(for: proposal.field),
          batch.role == "text" ? (2...8).contains(field.section)
            : field.extractionRoles.contains(batch.role) else {
          reject("proposal.field", "This field is not eligible for the extraction role."); continue
        }
        if proposal.field == "style.presetID" {
          guard (try? CharacterStylePresetRegistry.resolve(id: proposal.value, version: 1,
            appearance: doc.definition.appearance)) != nil else {
            reject("proposal.value", "The proposed style preset is unavailable."); continue
          }
          doc.definition.settings.stylePresetID = proposal.value; continue
        }
        var entry = Self.entry(proposal.value, path: proposal.field, source: batch.role == "text" ? .legacyMapping : .imageAnalysis)
        entry.revision = (doc.definition.entry(at: proposal.field)?.revision ?? 0) + 1
        entry.evidence = .init(summary: proposal.evidence, sourceRole: batch.role, sourceAssetHash: batch.sourceHash, uncertaintyReason: proposal.uncertainty)
        let issues = CharacterFieldCatalog.shared.validate(entry, at: proposal.field,
          appearance: doc.definition.appearance)
        guard issues.isEmpty else {
          reject("proposal.value", issues.map(\.message).joined(separator: " ")); continue
        }
        doc.definition.setEntry(entry, at: proposal.field)
      }
      if rejected.isEmpty { doc.proposals.removeAll { $0.id == batchID } }
      else if let current = doc.proposals.firstIndex(where: { $0.id == batchID }) {
        doc.proposals[current].proposals = rejected
        doc.proposals[current].diagnostics = diagnostics
      }
    }
  }
  func launch(_ operation: @escaping @MainActor () async throws -> Void) {
    guard !running, !store.operationBusy else { error = "Wait for the current local inference job to finish."; return }
    running = true; store.characterDirectorBusy = true; cancelled = false; error = nil
    workTask = Task { @MainActor [weak self] in
      guard let self else { return }
      defer { self.running = false; self.store.characterDirectorBusy = false; self.workTask = nil; self.save() }
      do { try await operation(); if !self.cancelled { self.status = "Ready for review" } }
      catch { self.error = self.cancelled ? nil : error.localizedDescription; self.status = self.cancelled ? "Cancelled" : "Needs attention" }
    }
  }
  func cancel() { cancelled = true; bridge.cancel(); workTask?.cancel() }
  func checkCancellation() throws { if cancelled || Task.isCancelled { throw CancellationError() } }
  func refreshCatalog() async {
    guard let connection else { error = "Add a Draw Things Local connection in Studio Settings."; return }
    catalog = [:]; catalogFailure = nil; error = nil
    let discovery = bridge.independent()
    do {
      let result = try await discovery.invoke("dt-discover", runtime: store.runtime, payload: ["connection": try connection.object()])
      guard self.connection?.id == connection.id else { return }
      guard !(result["models"] as? [[String: Any]] ?? []).isEmpty else {
        throw StudioError.invalid("Draw Things returned no models. Turn on Enable Model Browsing in its local server settings, then refresh the model catalog.")
      }
      catalog = result; selectCatalogDefaults(); save()
    } catch {
      if self.connection?.id == connection.id {
        catalogFailure = (connection.id, error.localizedDescription)
        self.error = error.localizedDescription
      }
    }
  }
  var models: [(id: String, name: String)] {
    let rules = catalog["capabilities"] as? [String: Any] ?? [:]
    return (catalog["models"] as? [[String: Any]] ?? []).compactMap { row in
      guard let id = row["id"] as? String, let name = row["name"] as? String,
        ((rules[id] as? [String: Any])?["operations"] as? [String: Any])?["image"] != nil else { return nil }
      return (id, name)
    }.sorted { $0.name < $1.name }
  }
  func loras(model: String) -> [(id: String, name: String)] {
    (catalog["loras"] as? [[String: Any]] ?? []).compactMap { row in
      guard let id = row["id"] as? String, let name = row["name"] as? String,
        (row["compatibleModelIDs"] as? [String] ?? []).contains(model) else { return nil }
      return (id, name)
    }.sorted { $0.name < $1.name }
  }
  static func isCharacterDetailLoRA(_ item: (id: String, name: String)) -> Bool {
    let key = (item.id + item.name).lowercased().filter { $0.isLetter || $0.isNumber }
    return key.contains("highresolution9b") || key.contains("hichresolution9b")
  }
  private func selectCatalogDefaults() {
    func key(_ text: String) -> String { text.lowercased().filter { $0.isLetter || $0.isNumber } }
    func unique(_ values: [(id: String, name: String)], _ term: String) -> String? {
      let matches = values.filter { key($0.name + $0.id).contains(term) }; return matches.count == 1 ? matches[0].id : nil
    }
    if document.draft.modelID.isEmpty { document.draft.modelID = unique(models, "krea2turbo") ?? "" }
    if document.draft.characterSheetLoRAID == nil { document.draft.characterSheetLoRAID = unique(loras(model: document.draft.modelID), "krea2characterdesign4viewv1") }
    if document.refinement.modelID.isEmpty {
      let matches = models.filter { key($0.name + $0.id).contains("klein") && key($0.name + $0.id).contains("9b") && !key($0.name + $0.id).contains("base") && !key($0.name + $0.id).contains("kv") }
      if matches.count == 1 { document.refinement.modelID = matches[0].id }
    }
    let options = loras(model: document.refinement.modelID)
    if document.refinement.detailLoRAID.isEmpty {
      let matches = options.filter(Self.isCharacterDetailLoRA)
      if matches.count == 1 { document.refinement.detailLoRAID = matches[0].id }
    }
    if document.refinement.headLoRAID.isEmpty { document.refinement.headLoRAID = unique(options, "bfsheadv1fluxklein9bstep3750rank64") ?? "" }
  }
  func importSource(_ url: URL, role: CharacterSourceRole) async {
    guard !running else { return }
    do {
      let destination = storage.directory(id: id).appendingPathComponent("Sources/\(UUID().uuidString).\(url.pathExtension)")
      try await Task.detached {
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: url, to: destination)
      }.value
      edit { $0.sources[role.rawValue] = destination.path; if role == .face { $0.headReference = nil } }
      if role == .face { launch { [self] in try await prepareHead() } }
    } catch { self.error = error.localizedDescription }
  }
  func prepareHead(selection: CharacterHeadSelection? = nil) async throws {
    guard let path = document.sources["face"] else { throw StudioError.invalid("Choose a reference face first.") }
    status = "Removing reference background…"
    let captured = path
    let result = try await CharacterHeadPreparation().prepare(source: URL(fileURLWithPath: path), selection: selection,
      outputDirectory: storage.directory(id: id).appendingPathComponent("Heads/\(UUID().uuidString)"))
    try checkCancellation()
    guard document.sources["face"] == captured else { throw StudioError.invalid("The reference face changed.") }
    document.headReference = result; save()
  }
  static func extractionDiagnostics(_ result: [String: Any]) -> [CharacterProposalDiagnostic] {
    (result["diagnostics"] as? [[String: Any]] ?? []).compactMap { value in
      guard let message = value["message"] as? String, !message.isEmpty else { return nil }
      let field = value["field"] as? String
      return .init(code: value["code"] as? String ?? "proposal.invalid",
        field: field, message: field.map { "\($0): \(message)" } ?? message)
    }
  }

  func analyze(role: String, modelPath: String) async throws {
    status = "Analyzing \(role)…"
    let captured = document.revision
    var preparationDiagnostics: [CharacterProposalDiagnostic] = []
    let sourcePath = role == "style" && document.styleUsesCharacterImage ? document.sources["character"] : document.sources[role]
    var payload: [String: Any] = ["role": role, "modelPath": modelPath,
      "capturedTarget": ["documentID": id.uuidString, "revision": captured],
      "fields": try JSONSerialization.jsonObject(with: CharacterFieldCatalog.shared.catalogJSON()) as? [String: Any] ?? [:],
      "cacheDirectory": storage.root.appendingPathComponent("Analysis Cache").path]
    let catalogObject = try JSONSerialization.jsonObject(with: CharacterFieldCatalog.shared.catalogJSON()) as? [String: Any]
    payload["fields"] = catalogObject?["fields"] ?? []
    let fields = CharacterFieldCatalog.shared.fields.filter { field in
      role == "text" ? (2...8).contains(field.section) : field.extractionRoles.contains(role)
    }.filter { role == "text" || !["identity.authoredAge", "identity.authoredSexGender", "identity.authoredAncestry"].contains($0.key) }
    var requested = fields.filter { !$0.key.contains("[]") }.map(\.key)
    if role != "style" {
      let groups: [(String, [CharacterRepeatableRecord], Int)] = [
        ("garments", document.definition.appearance.garments, 3), ("surfaces", document.definition.appearance.surfaces, 3),
        ("accessories", document.definition.appearance.accessories, 2), ("features", document.definition.appearance.features, 2)]
      for (name, records, limit) in groups {
        var ids = Array(records.prefix(limit).map(\.id))
        while ids.count < limit { ids.append(UUID()) }
        for id in ids { for field in fields where field.key.hasPrefix(name + "[].") {
          requested.append(field.key.replacingOccurrences(of: "[]", with: "[" + id.uuidString + "]"))
        } }
      }
    }
    payload["requestedFields"] = requested
    if role == "text" { payload["sourceText"] = document.originalDescription }
    else {
      guard let sourcePath else { throw StudioError.invalid("Choose an image for \(role) analysis.") }
      let hash = try await Task.detached { try CharacterArtifactHash.file(sourcePath) }.value
      payload["sourceImage"] = ["path": sourcePath, "label": role, "sha256": hash]
      if role == "character" {
        status = "Preparing bounded face and scalp detail…"
        let detail = try await CharacterDetailImagePreparation().prepare(
          source: URL(fileURLWithPath: sourcePath),
          outputDirectory: storage.directory(id: id).appendingPathComponent("Analysis Inputs"))
        try checkCancellation()
        guard detail.sourceSHA256 == hash else { throw StudioError.invalid("The character image changed during analysis preparation.") }
        payload["sourceDetailImages"] = detail.sourceDetailImages.map {
          ["path": $0.path, "label": $0.label, "sha256": $0.sha256]
        }
        preparationDiagnostics = detail.diagnostics.map { .init(code: "image.detail", message: $0) }
        status = "Analyzing character…"
      }
    }
    let result = try await bridge.invoke("character-analyze", runtime: store.runtime, payload: payload)
    try checkCancellation()
    var proposals: [CharacterFieldProposal] = []
    var diagnostics = preparationDiagnostics + Self.extractionDiagnostics(result)
    let rawProposals = result["proposals"] as? [Any]
    if rawProposals == nil {
      diagnostics.append(.init(code: "proposal.schema",
        message: "The extraction result did not contain a proposal list."))
    }
    for (offset, raw) in (rawProposals ?? []).enumerated() {
      guard let value = raw as? [String: Any] else {
        diagnostics.append(.init(code: "proposal.schema",
          message: "Proposal \(offset + 1) was not an object.")); continue
      }
      let proposalID = value["id"] as? String ?? UUID().uuidString
      guard let field = value["field"] as? String, let text = value["value"] as? String else {
        diagnostics.append(.init(code: "proposal.schema", proposalID: proposalID,
          field: value["field"] as? String,
          message: "Proposal \(offset + 1) did not contain a text field and value.")); continue
      }
      let state = value["state"] as? String ?? "value"
      let proposal = CharacterFieldProposal(id: proposalID, field: field, value: text,
        state: state, evidence: value["evidence"] as? String ?? (value["evidence"] as? [String: Any])?["summary"] as? String ?? "",
        uncertainty: value["uncertainty"] as? String ?? "")
      proposals.append(proposal)
      guard state == "value" else {
        diagnostics.append(.init(code: "proposal.state", proposalID: proposalID, field: field,
          message: "Only populated value proposals can be applied.")); continue
      }
      guard let catalogField = CharacterFieldCatalog.shared.field(for: field),
        role == "text" ? (2...8).contains(catalogField.section)
          : catalogField.extractionRoles.contains(role) else {
        diagnostics.append(.init(code: "proposal.field", proposalID: proposalID, field: field,
          message: "This field is not eligible for the extraction role.")); continue
      }
      let typed = Self.entry(text, path: field, source: role == "text" ? .legacyMapping : .imageAnalysis)
      let issues = CharacterFieldCatalog.shared.validate(typed, at: field,
        appearance: document.definition.appearance)
      diagnostics += issues.map { .init(code: "proposal.value", proposalID: proposalID,
        field: field, message: $0.message) }
    }
    let metadataValue = result["metadata"] as? [String: Any]
    let metadata: CharacterExtractionMetadata? = metadataValue.flatMap { value in
      guard let schema = value["schemaVersion"] as? Int,
        let extraction = value["extractionVersion"] as? Int,
        let prompt = value["promptVersion"] as? Int,
        let fingerprint = value["modelFingerprint"] as? String else { return nil }
      return .init(schemaVersion: schema, extractionVersion: extraction,
        promptVersion: prompt, modelFingerprint: fingerprint)
    }
    var batch = CharacterProposalBatch(documentID: id, revision: captured, sourcePath: sourcePath ?? "",
      sourceHash: result["sourceHash"] as? String ?? "", role: role, proposals: proposals,
      metadata: metadata, diagnostics: diagnostics.isEmpty ? nil : diagnostics)
    batch.stale = document.revision != captured; document.proposals.append(batch); save()
  }
}
