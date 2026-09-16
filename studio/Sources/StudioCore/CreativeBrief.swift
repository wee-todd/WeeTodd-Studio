import Foundation

/// Source evidence and host-assigned questions remain unchanged when answers are saved.
public struct CreativeBrief: Codable, Equatable {
  public struct Fact: Codable, Equatable {
    public let text: String
    public let evidence: String
  }

  public struct Question: Codable, Equatable, Identifiable {
    public let id: String
    public let prompt: String
    public let options: [String]
    public var answer: String
    public let requiresExplicitChoice: Bool

    public var isAnswered: Bool {
      let text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
      return !text.isEmpty && (!requiresExplicitChoice || !Self.isDelegatedAnswer(text))
    }

    public static func isDelegatedAnswer(_ answer: String) -> Bool {
      let normalized = answer.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }.joined(separator: " ")
      return ["director decide", "you decide", "surprise me", "anything", "either", "whatever"]
        .contains { (" " + normalized + " ").contains(" " + $0 + " ") }
    }
  }

  public let sourceText: String
  public let facts: [Fact]
  public private(set) var questions: [Question]
  public private(set) var preferences: CreativeBriefPreferences
  public let referenceObservations: [JSONValue]

  public var validationIssues: [String] {
    var issues = preferences.validationIssues
    if Set(questions.map(\.id)).count != questions.count { issues.append("Question IDs are duplicated. Reopen a valid workflow checkpoint.") }
    if questions.contains(where: { $0.answer.count > 2000 }) { issues.append("Keep each answer within 2,000 characters.") }
    for question in questions where !question.isAnswered {
      issues.append(question.requiresExplicitChoice
        ? "Make an explicit choice: \(question.prompt)" : "Answer: \(question.prompt)")
    }
    return issues
  }
  public var isReady: Bool { validationIssues.isEmpty }

  /// Incomplete answers can be saved for later; approval requires `isReady`.
  public func editing(answers: [String: String], preferences: CreativeBriefPreferences) throws -> Self {
    guard Set(answers.keys).isSubset(of: Set(questions.map(\.id))) else {
      throw StudioError.invalid("The brief's questions changed. Reload before saving answers.")
    }
    try preferences.validate()
    var updated = self
    for index in updated.questions.indices {
      if let answer = answers[updated.questions[index].id] { updated.questions[index].answer = answer }
    }
    updated.preferences = preferences
    return updated
  }

  public func outputs() throws -> [String: JSONValue] {
    try preferences.validate()
    return ["creative_brief": try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(self))]
  }
}

public struct CreativeBriefPreferences: Codable, Equatable {
  public var durationSeconds: Double
  public var targetClipSeconds: Double
  public var frameRate: Int
  public var visualStyle: String
  public var presentation: String
  public var cameraStyle: String
  public var audioStyle: String
  public var designPolicy: String
  public var constraints: String

  public var validationIssues: [String] {
    var issues: [String] = []
    if !durationSeconds.isFinite || !(1...3600).contains(durationSeconds) {
      issues.append("Movie duration must be between 1 and 3,600 seconds.")
    }
    if !targetClipSeconds.isFinite || !(1...60).contains(targetClipSeconds) {
      issues.append("Target clip duration must be between 1 and 60 seconds.")
    }
    if !(1...120).contains(frameRate) { issues.append("Frame rate must be between 1 and 120 fps.") }
    for (name, text) in [("Visual style", visualStyle), ("Presentation", presentation),
                         ("Camera style", cameraStyle), ("Audio style", audioStyle),
                         ("Design policy", designPolicy), ("Constraints", constraints)] where text.unicodeScalars.count > 2000 {
      issues.append("\(name) must use at most 2,000 characters.")
    }
    return issues
  }

  public func validate() throws {
    if let issue = validationIssues.first { throw StudioError.invalid(issue) }
  }
}

/// Read-only H3 text preview; rendering and approval remain workflow decisions.
public struct H3PromptPreview: Codable, Equatable {
  public struct Prompt: Codable, Equatable, Identifiable {
    public let clipID: String
    public let durationSeconds: Double
    public let integratedMultimodalDescription: String
    public let overallSoundscape: String
    public let nonDiegeticMusic: String
    public let prompt: String
    public let referenceAssets: [String]
    public let subjectIDs: [String]
    public var id: String { clipID }

    enum CodingKeys: String, CodingKey {
      case clipID, durationSeconds, prompt, referenceAssets, subjectIDs
      case integratedMultimodalDescription = "integrated_multimodal_description"
      case overallSoundscape = "overall_soundscape"
      case nonDiegeticMusic = "non_diegetic_music"
    }
  }
  public let status: String
  public let warnings: [String]
  public let prompts: [Prompt]
}
