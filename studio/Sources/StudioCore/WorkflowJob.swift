import Foundation

/// Local executable job wraps a portable definition with explicitly chosen file bindings.
public struct WorkflowJob: Codable {
  public var definition: [String: JSONValue]
  public var inputs: [String: JSONValue]
  public var models: [String: String]
  public var assets: [String: String]
  public var runDirectory: String
  public var maxTokens: Int
  public var maxSteps: Int?
  public var regenerate: String?
  public var coverageLibraryCandidates: [LibraryObjectCandidate]?
  public var librarySelections: [String: LibraryObjectMatch]?
  public init(definition: [String: JSONValue], inputs: [String: JSONValue], models: [String: String],
              assets: [String: String], runDirectory: String, maxTokens: Int = 1024) {
    self.definition = definition; self.inputs = inputs; self.models = models; self.assets = assets
    self.runDirectory = runDirectory; self.maxTokens = maxTokens
  }
  public static func input(_ text: String, type: String) throws -> JSONValue {
    switch type {
    case "text": return .string(text)
    case "integer":
      guard let value = Int(text) else { throw StudioError.invalid("Enter a whole number.") }
      return .integer(value)
    case "number":
      guard let value = Double(text), value.isFinite else { throw StudioError.invalid("Enter a finite number.") }
      return .number(value)
    case "boolean": return .boolean(text == "true")
    case "subject_list", "object_catalog":
      guard let data = text.data(using: .utf8), data.count <= 2_000_000,
            let value = try? JSONDecoder().decode(JSONValue.self, from: data), case .array = value else {
        throw StudioError.invalid("Choose a valid structured object inventory.")
      }
      return value
    default: throw StudioError.invalid("Unsupported workflow input type: \(type)")
    }
  }
}

public struct WorkflowRunSummary: Codable {
  public struct Step: Codable {
    public struct Item: Codable {
      public var status: String
      public var approved: Bool?
    }
    public var name: String
    public var status: String
    public var outputs: [String: JSONValue]?
    public var error: String?
    public var seconds: Double?
    public var approved: Bool?
    public var items: [String: Item]?
    public var warnings: [String]?
    public var coverageReviewReport: ObjectCoverageProposalReport?
  }
  public var status: String
  public var totalSeconds: Double
  public var steps: [String: Step]
  public var outputs: [String: JSONValue]
  public var error: String?
  public var revision: String?
  public var awaitingStep: String?
  public func canImportGuidedPlan(requiredSteps: [String], allSteps: [String]) -> Bool {
    !allSteps.isEmpty && allSteps.allSatisfy { steps[$0]?.status == "completed" }
      && !requiredSteps.isEmpty && requiredSteps.allSatisfy { steps[$0]?.status == "completed" && steps[$0]?.approved == true }
      && allSteps.contains { steps[$0]?.outputs?["h3_prompts"] != nil && steps[$0]?.status == "completed" }
  }
  public var preferredReviewStepID: String {
    awaitingStep ?? steps.keys.sorted().first {
      steps[$0]?.status == "completed" && steps[$0]?.outputs?["h3_prompts"] != nil
    } ?? preferredSubjectStepID ?? ""
  }
  public func repairCompleted(stepID: String, itemID: String) -> Bool {
    guard steps[stepID]?.status == "completed", steps[stepID]?.items?[itemID]?.status == "completed" else { return false }
    return status == "completed" || status == "paused" || (status == "awaiting_approval" && awaitingStep == stepID)
  }
  public var prompt: String? {
    if case .string(let text) = outputs["prompt"] { return text }
    return nil
  }
  public var needsAttention: Bool {
    if case .object(let review) = outputs["review"], case .string("needs_attention") = review["status"] { return true }
    return false
  }
}
