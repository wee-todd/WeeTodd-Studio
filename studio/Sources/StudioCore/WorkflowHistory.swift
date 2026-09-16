import Foundation

/// Read-only projection of bounded checkpoint telemetry, including older checkpoints.
public struct WorkflowHistorySnapshot: Decodable {
  public struct Definition: Decodable {
    public struct Step: Decodable, Identifiable {
      public var id: String
      public var name: String
      public var operation: String
    }
    public var steps: [Step]
  }
  public struct Step: Decodable {
    public var status: String
    public var seconds: Double?
    public var error: String?
    public var savedResponses: Int
    public var legacyTurns: [WorkflowModelTurn]
    public init(from decoder: Decoder) throws {
      let raw = try JSONValue(from: decoder)
      guard case .object(let fields) = raw else { throw StudioError.invalid("Invalid step record") }
      if case .string(let value) = fields["status"] { status = value } else { status = "unknown" }
      if case .string(let value) = fields["error"] { error = value }
      switch fields["seconds"] {
      case .number(let value): seconds = value
      case .integer(let value): seconds = Double(value)
      default: seconds = nil
      }
      func count(_ value: JSONValue) -> Int {
        switch value {
        case .object(let fields):
          return fields.reduce(0) { result, pair in
            if pair.key == "calls", case .array(let calls) = pair.value { return result + calls.count }
            return result + count(pair.value)
          }
        case .array(let values): return values.reduce(0) { $0 + count($1) }
        default: return 0
        }
      }
      savedResponses = count(raw)
      func collect(_ value: JSONValue, path: String) -> [WorkflowModelTurn] {
        switch value {
        case .object(let fields):
          var result: [WorkflowModelTurn] = []
          for key in fields.keys.sorted() {
            if key == "calls", case .array(let calls) = fields[key] {
              for (index, call) in calls.enumerated() {
                if case .object(let fields) = call,
                   let turn = WorkflowModelTurn.legacy(fields, path: "\(path)/calls/\(index + 1)") { result.append(turn) }
              }
            } else if key == "lastRejectedResponse", case .object(let call) = fields[key],
                      let turn = WorkflowModelTurn.legacy(call, path: path + "/lastRejectedResponse") {
              result.append(turn)
            } else if let child = fields[key] { result += collect(child, path: path + "/" + key) }
          }
          return result
        case .array(let values): return values.enumerated().flatMap { collect($0.element, path: "\(path)/\($0.offset)") }
        default: return []
        }
      }
      legacyTurns = collect(raw, path: "Saved checkpoint")
    }
  }
  public struct Entry: Decodable, Identifiable {
    public struct Event: Decodable, Identifiable {
      public var id: String
      public var kind: String
      public var message: String
      public var purpose: String?
      public var status: String?
      public var seconds: Double?
      public var error: String?
      public var requestKey: String?
    }
    public var id: String
    public var stepID: String
    public var name: String
    public var operation: String
    public var reason: String
    public var startedAt: Double
    public var status: String
    public var seconds: Double
    public var modelCalls: Int
    public var reusedCalls: Int
    public var retries: Int
    public var events: [Event]
    public var eventsOmitted: Int?
    public var message: String?
    public var error: String?
    public var timingIncomplete: Bool?
    public func elapsed(at date: Date) -> Double {
      status == "running" ? max(seconds, date.timeIntervalSince1970 - startedAt) : seconds
    }
  }
  public var status: String
  public var totalSeconds: Double
  public var definition: Definition
  public var steps: [String: Step]
  public var executionHistory: [Entry]
  public var historyOmitted: Int
  enum CodingKeys: String, CodingKey { case status, totalSeconds, definition, steps, executionHistory, historyOmitted }
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    status = try c.decode(String.self, forKey: .status)
    totalSeconds = try c.decode(Double.self, forKey: .totalSeconds)
    definition = try c.decode(Definition.self, forKey: .definition)
    steps = try c.decode([String: Step].self, forKey: .steps)
    executionHistory = try c.decodeIfPresent([Entry].self, forKey: .executionHistory) ?? []
    historyOmitted = try c.decodeIfPresent(Int.self, forKey: .historyOmitted) ?? 0
  }
}

public enum WorkflowOperationDescription {
  public static func text(_ operation: String) -> String {
    switch operation {
    case "vision.describe@1": return "Reads each supplied reference image and records visible details and uncertainty. With no images, this step completes without a model call."
    case "text.plan_edits@1": return "Breaks your instructions into individual prompt edits and identifies details to preserve."
    case "text.apply_edits@1": return "Applies each planned edit in a separate model call, carrying the latest draft into the next edit."
    case "text.check_edits@1": return "Checks the revised prompt against the requested edits and reports concerns without rewriting it."
    case "project.identify_subjects@1", "project.identify_creative_subjects@1": return "Extracts characters, locations and reusable objects from bounded sections of your source. The app validates evidence and assigns IDs."
    case "project.classify_subjects@1": return "Classifies extracted objects, including environments and sets, while preserving their IDs for review."
    case "project.link_subjects@1": return "Looks for relationships between known objects and links them by ID rather than duplicating their descriptions."
    case "project.review_subjects@1", "project.review_creative_subjects@1": return "Writes fuller visual descriptions and critiques them. Each object can require several writing, checking or repair calls; proposed details remain subject to your approval."
    case "project.review_object_coverage@1": return "Checks each object against the inventory for missing objects, invalid links and reusable library matches. Bounded corrective attempts can add model calls."
    case "movie.prepare_creative_brief@1": return "Combines your story, creative preferences and image observations, then identifies choices needing clarification."
    case "movie.resolve_creative_brief@1": return "Applies your approved answers to create resolved planning inputs. This is an app operation, not a generation call."
    case "movie.plan_story@1", "movie.plan_story@2", "movie.plan_treatment@1": return "Plans the story progression for the requested length. Longer plans are split into bounded writing tasks. The treatment uses approved subjects and creative choices."
    case "movie.plan_beats@1", "movie.plan_creative_beats@1": return "Allocates exact project frames and writes individual shot actions and start/end states. Each shot may require a corrective attempt."
    case "movie.allocate_clips@1": return "Allocates project frames and scopes story actions to clips using app logic. No model call."
    case "movie.plan_endpoints@1": return "Writes first/last-frame descriptions for each clip, reusing continuous endpoints where appropriate. It does not generate images."
    case "movie.plan_endpoints@2": return "Assembles endpoint descriptions from approved clip states and continuity links. No model call or image generation."
    case "movie.check_plan@1", "movie.check_plan@2": return "Checks frame coverage, duration, object IDs and continuity, then flags structural or pacing concerns. No model call."
    case "movie.compile_h3_prompts@1": return "Combines approved descriptions and shot plans into H3 prompt drafts. No model call or video generation."
    default: return "Runs the registered operation \(operation). Inspect its recorded activity for the work performed."
    }
  }
}
