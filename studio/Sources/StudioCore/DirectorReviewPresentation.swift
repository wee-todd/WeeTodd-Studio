import Foundation

public enum DirectorReviewMode: String, CaseIterable, Identifiable {
  case focused, detailed
  public var id: String { rawValue }
  public var label: String { self == .focused ? "Focused · 3 reviews" : "Detailed · 8 reviews" }
  public static func supports(_ definition: [String: JSONValue]) -> Bool {
    (definition["id"] == .string("weetodd.guided-movie-planning") && definition["version"] == .string("1.1.0")) ||
      (definition["id"] == .string("weetodd.music-video-planning") && [JSONValue.string("1.0.0"), .string("1.1.0")].contains(definition["version"] ?? .null))
  }
  /// Choose gates before the run has any saved execution identity. Saved jobs keep their definition.
  public func applying(to definition: [String: JSONValue], hasStarted: Bool) throws -> [String: JSONValue] {
    guard !hasStarted, Self.supports(definition), case .array(let steps) = definition["steps"] else {
      throw StudioError.invalid("Review mode is fixed for this job. Start a new guided workflow to choose its reviews.")
    }
    let required: Set<String> = self == .focused ? ["creative_brief", "subjects_coverage", "clips"] :
      ["creative_brief", "classify", "inventory", "design", "subjects_coverage", "story", "clips", "prompt_preview"]
    var updated = definition
    updated["steps"] = .array(steps.map { raw in
      guard case .object(var step) = raw, case .string(let id) = step["id"] else { return raw }
      step["requiresApproval"] = .boolean(required.contains(id)); return .object(step)
    })
    return updated
  }
}

public enum DirectorReviewPhase: String, CaseIterable, Identifiable {
  case brief, subjects, shots
  public var id: String { rawValue }
  public var label: String { rawValue.capitalized }
}
public enum DirectorNextAction: Equatable {
  case start, review(String), continuePlanning, addToProject
}

/// Navigation and approval decisions always use the job's saved step list, including older gates.
public struct DirectorReviewPresentation {
  public let definition: [String: JSONValue]
  public let result: WorkflowRunSummary?
  public init(definition: [String: JSONValue], result: WorkflowRunSummary?) {
    self.definition = definition; self.result = result
  }
  private var steps: [(id: String, required: Bool)] {
    guard case .array(let values) = definition["steps"] else { return [] }
    return values.compactMap { value in
      guard case .object(let step) = value, case .string(let id) = step["id"] else { return nil }
      return (id, step["requiresApproval"] == .boolean(true))
    }
  }
  public var requiredStepIDs: [String] { steps.filter(\.required).map(\.id) }
  public var mode: DirectorReviewMode { requiredStepIDs == ["creative_brief", "subjects_coverage", "clips"] ? .focused : .detailed }
  public func phase(for stepID: String) -> DirectorReviewPhase {
    if ["describe", "creative_brief", "resolve"].contains(stepID) { return .brief }
    if ["subjects", "classify", "links", "inventory", "design", "subjects_coverage"].contains(stepID) { return .subjects }
    return .shots
  }
  public func stepID(for phase: DirectorReviewPhase) -> String? {
    let candidates = steps.filter { self.phase(for: $0.id) == phase }
    if let awaiting = result?.awaitingStep, candidates.contains(where: { $0.id == awaiting }),
       result?.steps[awaiting]?.approved != true { return awaiting }
    if let gate = candidates.first(where: { $0.required && result?.steps[$0.id]?.status == "completed" && result?.steps[$0.id]?.approved != true }) { return gate.id }
    if phase == .brief, result?.steps["creative_brief"]?.status == "completed" { return "creative_brief" }
    // The shot plan remains the review surface after the compiled prompt preview finishes.
    if phase == .shots, result?.steps["clips"]?.status == "completed" { return "clips" }
    return candidates.last(where: { result?.steps[$0.id]?.status == "completed" })?.id
  }
  public func isApproved(_ phase: DirectorReviewPhase) -> Bool {
    let gates = requiredStepIDs.filter { self.phase(for: $0) == phase }
    return !gates.isEmpty && gates.allSatisfy { result?.steps[$0]?.status == "completed" && result?.steps[$0]?.approved == true }
  }
  public var canImport: Bool {
    result?.canImportGuidedPlan(requiredSteps: requiredStepIDs, allSteps: steps.map(\.id)) == true
  }
  public var nextAction: DirectorNextAction {
    guard let result else { return .start }
    if let gate = requiredStepIDs.first(where: { result.steps[$0]?.status == "completed" && result.steps[$0]?.approved != true }) { return .review(gate) }
    return canImport ? .addToProject : .continuePlanning
  }
  public static func canApprove(_ step: WorkflowRunSummary.Step, hasUnsavedDraft: Bool, briefReady: Bool = true) -> Bool {
    step.status == "completed" && !hasUnsavedDraft && briefReady &&
      (step.items?.values.allSatisfy { $0.status == "completed" } ?? true)
  }
}

public struct DirectorShotChange: Identifiable {
  public var id: String { field }
  public let field: String
  public let before: String
  public let after: String
  public let withinScope: Bool
}
public enum DirectorShotRepairScope: String, Codable, CaseIterable, Identifiable {
  case action, states, location, characters, all
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .action: return "Action"
    case .states: return "Start and end states"
    case .location: return "Location and connection"
    case .characters: return "Cast"
    case .all: return "Whole shot"
    }
  }
  public var preservedDescription: String {
    let fields: [(Self, String)] = [(.action, "action"), (.states, "start/end states"), (.location, "location/connection"), (.characters, "cast")]
    let unchanged = fields.filter { self != .all && $0.0 != self }.map(\.1)
    return (["Shot ID and timing"] + unchanged).joined(separator: ", ") + " stay unchanged. Supplied dialogue is preserved."
  }
  public func changes(from before: WorkflowClipDraft, to after: WorkflowClipDraft) -> [DirectorShotChange] {
    let values: [(String, String, String, Self?)] = [
      ("Action", before.action, after.action, .action), ("Starting state", before.startState, after.startState, .states),
      ("Ending state", before.endState, after.endState, .states), ("Location", before.location, after.location, .location),
      ("Connection", before.continuity, after.continuity, .location), ("Cast", before.characters.joined(separator: ", "), after.characters.joined(separator: ", "), .characters),
      ("Shot ID", before.id, after.id, nil), ("Start frame", String(before.startFrame), String(after.startFrame), nil),
      ("Frame count", String(before.frameCount), String(after.frameCount), nil)]
    return values.filter { $0.1 != $0.2 }.map {
      DirectorShotChange(field: $0.0, before: $0.1, after: $0.2, withinScope: $0.3 != nil && (self == .all || self == $0.3))
    }
  }
}

/// Readiness only checks declared bindings; the execution validator still checks the actual model.
public struct DirectorPreparationState {
  public let missingModel: String?
  public let hasBrief: Bool
  public var canPrepare: Bool { missingModel == nil && hasBrief }
  public init(requiredModels: [String], bindings: [String: String], brief: String) {
    missingModel = requiredModels.sorted().first { (bindings[$0] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    hasBrief = !brief.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }
}

/// A collapsed report must still disclose that issues, conflicts, or design choices need review.
public struct DirectorSubjectAttention {
  public let issues: [String]
  public let proposedDetailCount: Int
  public let missingObjectCount: Int
  public let hasOutdatedNotes: Bool
  public let needsAttention: Bool
  public let hasNotes: Bool
  public init(subject: WorkflowSubjectProposal, coverageProposal: WorkflowSubjectProposal? = nil, currentDescription: String? = nil) {
    let coverage = coverageProposal?.coverageReview ?? subject.coverageReview
    let description = currentDescription ?? subject.description
    var seen = Set<String>()
    issues = ((subject.descriptionReview?.issues ?? []) + (subject.relationshipReview?.issues ?? []) + (coverage?.issues ?? []))
      .filter { seen.insert($0).inserted }
    proposedDetailCount = subject.descriptionReview?.proposedDetails.count ?? 0
    missingObjectCount = Set((subject.relationshipReview?.missingObjects ?? []) + (coverage?.missingObjects.map(\.name) ?? [])).count
    let reviews = [subject.descriptionReview.map { ($0.status, $0.reviewedDescription) },
      subject.relationshipReview.map { ($0.status, $0.reviewedDescription) }, coverage.map { ($0.status, $0.reviewedDescription) }].compactMap { $0 }
    hasOutdatedNotes = reviews.contains { $0.1 != description }
    needsAttention = !issues.isEmpty || proposedDetailCount > 0 || missingObjectCount > 0 || reviews.contains { $0.0 != "ready" }
    hasNotes = !reviews.isEmpty || !subject.suggestions.isEmpty || coverageProposal != nil
  }
  public var summary: String {
    var parts: [String] = []
    if !issues.isEmpty { parts.append("\(issues.count) \(issues.count == 1 ? "issue" : "issues")") }
    if proposedDetailCount > 0 { parts.append("\(proposedDetailCount) proposed \(proposedDetailCount == 1 ? "detail" : "details")") }
    if missingObjectCount > 0 { parts.append("\(missingObjectCount) missing object \(missingObjectCount == 1 ? "suggestion" : "suggestions")") }
    return parts.isEmpty ? (needsAttention ? "Assistant flagged this review for attention" : "Assistant notes available") : parts.joined(separator: " · ")
  }
}
