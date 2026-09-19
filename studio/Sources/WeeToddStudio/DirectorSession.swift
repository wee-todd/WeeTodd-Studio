import Foundation
import StudioCore
import Combine

/// Drafts belong to the Director session, never to approved workflow outputs.
struct DirectorReviewDraft: Codable {
  var selectedSubject: String?
  var subjects: [String: WorkflowSubjectProposal] = [:]
  var referencePaths: [String: [String]] = [:]
  var brief: DirectorBriefDraft?
  var clips: [String: WorkflowClipDraft] = [:]
  var story: WorkflowStoryOutline?
  var json: String?
  var repairs: [String: String] = [:]
  var repairScopes: [String: DirectorShotRepairScope]?
  var repairBaselines: [String: WorkflowClipDraft]?
}
struct DirectorBriefDraft: Codable, Equatable {
  var baseline: CreativeBrief
  var answers: [String: String]
  var preferences: CreativeBriefPreferences
  var duration: String
  var clipDuration: String
  var frameRate: String
  init(_ brief: CreativeBrief) {
    baseline = brief; preferences = brief.preferences
    answers = Dictionary(brief.questions.map { ($0.id, $0.answer) }, uniquingKeysWith: { first, _ in first })
    duration = Self.number(brief.preferences.durationSeconds)
    clipDuration = Self.number(brief.preferences.targetClipSeconds)
    frameRate = String(brief.preferences.frameRate)
  }
  static func number(_ value: Double) -> String {
    let text = String(value); return text.hasSuffix(".0") ? String(text.dropLast(2)) : text
  }
}
struct DirectorSession: Codable {
  var version = 1
  var addedToProject = false
  var definition: [String: JSONValue] = [:]
  var libraryCandidates: [LibraryObjectCandidate] = []
  var catalogFrozen = false
  var structuredInputs: [String: JSONValue] = [:]
  var librarySelections: [String: LibraryObjectMatch] = [:]
  var fields: [String: String] = [:]
  var imageInputs: [String: [PromptAssistantImage]] = [:]
  var modelPaths: [String: String] = [:]
  var runID = UUID()
  var runDirectory = ""
  var result: WorkflowRunSummary?
  var selectedStep = ""
  var tokens = 1024
  var showWorkflowInputs = false
  var reviewAssetBindings: [String: String] = [:]
  var reviews: [String: DirectorReviewDraft] = [:]
  var executionStarted: Bool?
  var reviewModeLocked: Bool { executionStarted == true || result != nil }
  var hasUnsavedReviews: Bool {
    reviews.contains { id, draft in
      draft.hasUnsavedChanges(outputs: result?.steps[id]?.outputs ?? [:], referenceBindings: reviewAssetBindings)
    }
  }
  mutating func changeInputs(_ update: (inout DirectorSession) -> Void) throws {
    // Compare with the saved outputs before invalidating them. Merely opening an
    // editor populates clean caches; those are not unsaved user changes.
    guard !hasUnsavedReviews else {
      throw StudioError.invalid("Save or discard your review edits before changing workflow inputs.")
    }
    for id in Array(reviews.keys) {
      reviews[id]?.subjects = [:]; reviews[id]?.referencePaths = [:]
      reviews[id]?.brief = nil; reviews[id]?.clips = [:]
      reviews[id]?.story = nil; reviews[id]?.json = nil
      // Navigation and unsubmitted repair instructions remain the user's work.
    }
    update(&self)
    result = nil
  }
  mutating func setReviewMode(_ mode: DirectorReviewMode) throws {
    definition = try mode.applying(to: definition, hasStarted: reviewModeLocked)
  }
}

/// File I/O and decoding never run on the UI actor. Limits match the workflow writer.
actor DirectorSessionFiles {
  static let jobImportLimit = 2 * 1024 * 1024
  static let draftLimit = 8 * 1024 * 1024
  private var savedSequence: [String: Int] = [:]
  func read<T: Decodable>(_ type: T.Type, at url: URL, limit: Int) throws -> T {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: limit + 1) ?? Data()
    guard data.count <= limit else { throw StudioError.invalid("\(url.lastPathComponent) exceeds its \(limit / 1024 / 1024) MiB limit.") }
    return try JSONDecoder().decode(type, from: data)
  }
  func save(_ state: DirectorSession, to url: URL, sequence: Int) throws {
    guard sequence >= savedSequence[url.path, default: -1] else { return }
    let data = try JSONEncoder().encode(state)
    guard data.count <= Self.draftLimit else { throw StudioError.invalid("Director drafts exceed 8 MiB. The previous saved draft is preserved.") }
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
    savedSequence[url.path] = sequence
  }
  func checkpoint(in directory: String) throws -> WorkflowRunSummary? {
    let url = URL(fileURLWithPath: directory).appendingPathComponent("run.json")
    guard !directory.isEmpty, FileManager.default.fileExists(atPath: url.path) else { return nil }
    return try WorkflowCheckpoint.read(WorkflowRunSummary.self, at: url)
  }
  func candidate(at url: URL) throws -> DirectorSession {
    let saved = try read(WorkflowJob.self, at: url, limit: Self.jobImportLimit)
    guard saved.definition["format"]?.directorText == "weetodd-workflow-v1",
          let id = saved.definition["id"]?.directorText, !id.isEmpty,
          case .array(let steps) = saved.definition["steps"], !steps.isEmpty,
          (1...1024).contains(saved.maxTokens), !saved.runDirectory.isEmpty else {
      throw StudioError.invalid("The saved workflow job has an invalid definition, token limit, or run directory.")
    }
    var candidate = DirectorSession()
    candidate.executionStarted = true
    candidate.definition = saved.definition; candidate.runDirectory = saved.runDirectory
    candidate.modelPaths = saved.models; candidate.tokens = saved.maxTokens
    candidate.reviewAssetBindings = saved.assets; candidate.librarySelections = saved.librarySelections ?? [:]
    let specs = saved.definition["inputs"]?.directorObject ?? [:]
    // The runner fills omitted inputs from their declared defaults before checkpointing.
    var effectiveInputs = saved.inputs
    for (name, spec) in specs where effectiveInputs[name] == nil {
      if let value = spec.directorObject?["default"] { effectiveInputs[name] = value }
    }
    for (name, value) in effectiveInputs {
      let type = specs[name]?.directorObject?["type"]?.directorText ?? "text"
      if type == "image_list" {
        guard case .array(let refs) = value, refs.count <= 8 else { throw StudioError.invalid("Use at most eight image bindings.") }
        candidate.imageInputs[name] = try refs.map { ref in
          guard let path = saved.assets[ref.directorText], !path.isEmpty else { throw StudioError.invalid("Job has missing image bindings.") }
          return PromptAssistantImage(path: path, label: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent)
        }
      } else if ["subject_list", "object_catalog"].contains(type) {
        candidate.structuredInputs[name] = value
        if type == "object_catalog" {
          candidate.libraryCandidates = try JSONDecoder().decode([LibraryObjectCandidate].self, from: JSONEncoder().encode(value))
          candidate.catalogFrozen = true
        }
      } else { candidate.fields[name] = value.directorText }
    }
    if let candidates = saved.coverageLibraryCandidates { candidate.libraryCandidates = candidates; candidate.catalogFrozen = true }
    let checkpointURL = URL(fileURLWithPath: saved.runDirectory).appendingPathComponent("run.json")
    if FileManager.default.fileExists(atPath: checkpointURL.path) {
      let envelope = try WorkflowCheckpoint.read([String: JSONValue].self, at: checkpointURL)
      guard envelope["format"]?.directorText == "weetodd-workflow-run-v1", envelope["workflowID"]?.directorText == id else {
        throw StudioError.invalid("Run directory belongs to a different workflow.")
      }
      guard envelope["definition"] == .object(saved.definition), envelope["inputs"] == .object(effectiveInputs) else {
        throw StudioError.invalid("The checkpoint's workflow definition or inputs differ from this job. The current Director session was preserved. Choose its matching saved job or a new run directory.")
      }
      candidate.result = try JSONDecoder().decode(WorkflowRunSummary.self, from: JSONEncoder().encode(envelope))
    }
    candidate.selectedStep = candidate.result?.preferredReviewStepID ?? ""
    return candidate
  }
}
@MainActor final class DirectorSessionController: ObservableObject {
  @Published var state = DirectorSession() { didSet { scheduleSave() } }
  @Published private(set) var persistenceError: String?
  private let files = DirectorSessionFiles()
  private var url: URL?
  private var sequence = 0
  private var pending: Task<Void, Never>?
  func open(_ url: URL) async throws {
    // Never overwrite an unreadable existing draft with a fresh session.
    let exists = FileManager.default.fileExists(atPath: url.path)
    let loaded = exists ? try await files.read(DirectorSession.self, at: url, limit: DirectorSessionFiles.draftLimit) : DirectorSession()
    guard loaded.version == 1 else { throw StudioError.invalid("This Director session version is not supported.") }
    self.url = url; state = loaded
  }
  func importJob(_ url: URL, validateDefinition: ([String: JSONValue]) async throws -> Void) async throws {
    let previousRun = state.runID
    let candidate = try await files.candidate(at: url)
    try await validateDefinition(candidate.definition)
    guard state.runID == previousRun else { throw StudioError.invalid("The Director session changed while opening a job.") }
    state = candidate
  }
  func refreshCheckpoint() async {
    let directory = state.runDirectory, run = state.runID
    if let saved = try? await files.checkpoint(in: directory), state.runID == run, state.runDirectory == directory {
      state.result = saved
    }
  }
  func flush() async throws {
    pending?.cancel()
    guard let url else { return }
    sequence += 1
    do { try await files.save(state, to: url, sequence: sequence); persistenceError = nil }
    catch { persistenceError = error.localizedDescription; throw error }
  }
  private func scheduleSave() {
    guard url != nil else { return }
    pending?.cancel()
    pending = Task { [weak self] in
      do { try await Task.sleep(nanoseconds: 200_000_000); try Task.checkCancellation(); try await self?.flush() }
      catch is CancellationError {} catch { self?.persistenceError = error.localizedDescription }
    }
  }
}

/// UUID alone cannot identify a reopened document. Planning revisions protect imports too.
struct DirectorDocumentTarget {
  let sessionID: UUID
  let projectID: UUID
  let planning: Data
  @MainActor init(store: StudioStore) {
    sessionID = store.documentSessionID; projectID = store.project.id
    planning = Self.revision(store)
  }
  @MainActor func validate(store: StudioStore, checkPlanning: Bool = true) throws {
    guard sessionID == store.documentSessionID, projectID == store.project.id,
          !checkPlanning || planning == Self.revision(store) else {
      throw StudioError.invalid("The movie or its planning inputs changed. Reopen Director for the current movie; your draft is retained.")
    }
  }
  @MainActor private static func revision(_ store: StudioStore) -> Data {
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    return (try? encoder.encode(store.planning)) ?? Data()
  }
}
private extension JSONValue {
  var directorObject: [String: JSONValue]? { if case .object(let value) = self { return value }; return nil }
  var directorText: String {
    switch self {
    case .string(let text): return text
    case .integer(let n): return String(n)
    case .number(let n): return String(n)
    case .boolean(let flag): return flag ? "true" : "false"
    default: return ""
    }
  }
}

@MainActor final class ReferenceWorkspaceLease {
  let id = UUID()
  let previousDraft: DrawThingsImageDraft?
  private let previousPreview: String?
  private let previousEstimate: [String: Any]?
  private let sessionID: UUID
  private let subjectKey: String?
  init(store: StudioStore, subjectKey: String? = nil) {
    sessionID = store.documentSessionID; self.subjectKey = subjectKey
    if let previous = store.activeReferenceLease, previous.sessionID == sessionID {
      // A closing editor may disappear after its replacement has already opened.
      previousDraft = previous.previousDraft
      previousPreview = previous.previousPreview; previousEstimate = previous.previousEstimate
    } else {
      previousDraft = store.imageDraft; previousPreview = store.imagePreviewPath; previousEstimate = store.imageEstimate
    }
    store.activeReferenceLease = self
  }
  func validate(store: StudioStore) throws {
    guard store.activeReferenceLease === self, store.documentSessionID == sessionID,
          subjectKey == nil || store.imageDraft?.referenceSheet?.subjectKey == subjectKey else {
      throw StudioError.invalid("The movie or reference workspace changed. Reopen this subject before attaching a candidate.")
    }
  }
  @discardableResult func restore(store: StudioStore) -> Bool {
    guard (try? validate(store: store)) != nil else { return false }
    store.persistImageWorkspace(); store.restoringImageWorkspace = true
    store.imageDraft = previousDraft; store.imagePreviewPath = previousPreview; store.imageEstimate = previousEstimate
    store.restoringImageWorkspace = false; store.referenceSheetOpen = false
    store.activeReferenceLease = nil; store.persistImageWorkspace()
    return true
  }
}

enum DirectorReferenceAssets {
  static func bind(_ paths: [String], in bindings: inout [String: String]) throws -> [String] {
    guard paths.count <= 8, Set(paths).count == paths.count, paths.allSatisfy({ !$0.isEmpty }) else {
      throw StudioError.invalid("Choose at most eight distinct reference images.")
    }
    return paths.map { path in
      if let key = bindings.keys.sorted().first(where: { bindings[$0] == path }) { return key }
      let key = "description-ref:\(UUID().uuidString)"; bindings[key] = path; return key
    }
  }
}

extension DirectorSessionController {
  /// A failed weighted operation can still have checkpointed completed steps.
  func execute(_ operation: () async throws -> [String: Any]) async throws {
    let run = state.runID
    do {
      let response = try await operation()
      let result = try JSONDecoder().decode(WorkflowRunSummary.self, from: JSONSerialization.data(withJSONObject: response))
      guard state.runID == run else { throw StudioError.invalid("The Director session changed. Its newer results were preserved.") }
      state.result = result
    } catch {
      if state.runID == run { await refreshCheckpoint() }
      throw error
    }
  }
}


extension DirectorReviewDraft {
  /// A hidden or closed editor still owns a draft; navigation never makes approval safe.
  func hasUnsavedChanges(outputs: [String: JSONValue], referenceBindings: [String: String]) -> Bool {
    func decode<T: Decodable>(_ type: T.Type, _ key: String) -> T? {
      guard let value = outputs[key], let data = try? JSONEncoder().encode(value) else { return nil }
      return try? JSONDecoder().decode(type, from: data)
    }
    let savedSubjects = decode([WorkflowSubjectProposal].self, "subjects") ?? []
    if subjects.contains(where: { id, value in savedSubjects.first(where: { $0.id == id }) != value }) { return true }
    if referencePaths.contains(where: { id, paths in
      guard let subject = savedSubjects.first(where: { $0.id == id }) else { return true }
      return paths != subject.referenceAssetKeys.compactMap { referenceBindings[$0] }
    }) { return true }
    let plan = decode(WorkflowClipPlan.self, "clips")
    if clips.contains(where: { id, value in plan?.clips.first(where: { $0.id == id }) != value }) { return true }
    if let story, (try? JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(story))) != outputs["story"] { return true }
    if let json, (try? JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))) != outputs { return true }
    return brief?.hasChanges == true
  }
}
extension DirectorBriefDraft {
  var hasChanges: Bool {
    preferences != baseline.preferences || baseline.questions.contains { answers[$0.id] != $0.answer }
      || duration != Self.number(baseline.preferences.durationSeconds)
      || clipDuration != Self.number(baseline.preferences.targetClipSeconds)
      || frameRate != String(baseline.preferences.frameRate)
  }
}
