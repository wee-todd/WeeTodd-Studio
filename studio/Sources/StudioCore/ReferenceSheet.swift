import Foundation

/// Builds reference context from explicit placements, never from name mentions alone.
public enum ReferenceSheetLinks {
  public static func definitions(relationships: [WorkflowObjectRelationship], inventory: [WorkflowSubjectProposal]) -> String {
    format(relationships, targets: inventory.map {
      DescriptionLinkTarget(id: $0.id, name: $0.name, description: $0.description)
    })
  }
  public static func definitions(subject: PlanningSubject, inventory: [PlanningSubject]) -> String {
    var links = (subject.relationships ?? []).map {
      WorkflowObjectRelationship(id: $0.id.uuidString, targetID: $0.targetID.uuidString, role: $0.role, placement: $0.placement)
    }
    if let environment = subject.environmentID?.uuidString,
       !links.contains(where: { $0.targetID == environment && $0.role == .located_in }) {
      links.append(WorkflowObjectRelationship(id: "environment:" + environment, targetID: environment,
        role: .located_in, placement: "Parent environment of this set."))
    }
    return format(links, targets: inventory.map {
      DescriptionLinkTarget(id: $0.id.uuidString, name: $0.name, description: $0.details)
    })
  }
  private static func format(_ relationships: [WorkflowObjectRelationship], targets: [DescriptionLinkTarget]) -> String {
    relationships.map { link in
      let target = targets.first { $0.id == link.targetID }
      return "Link \(link.id) → \(link.targetID) · \(target?.name ?? "Missing object")\n"
        + "Role: \(link.role.rawValue). Placement / state: \(link.placement.isEmpty ? "Unspecified." : link.placement)\n"
        + "Separate object definition: \(target?.description ?? "Definition unavailable; do not invent its appearance.")"
    }.joined(separator: "\n\n")
  }
}

public enum ReferenceSheetTemplate: String, Codable, CaseIterable, Identifiable {
  case character, portrait, prop, environment, set, clothing, custom
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .character: return "Character · front, close-up, profile"
    case .portrait: return "Character · portrait"
    case .prop: return "Prop · three views"
    case .environment: return "Environment · establishing views"
    case .set: return "Set · spatial continuity"
    case .clothing: return "Clothing · front and back"
    case .custom: return "Custom reference"
    }
  }
  public static func suggested(for kind: PlanningSubjectKind) -> Self {
    switch kind {
    case .character: return .character
    case .prop: return .prop
    case .environment: return .environment
    case .set, .location: return .set
    case .clothing, .outfit: return .clothing
    }
  }
  public var layout: String {
    switch self {
    case .character:
      return "Production character reference sheet on a clean neutral backdrop with soft even lighting. Three clearly separated views of the SAME character: left full-body front view; center head-and-shoulders face close-up showing facial structure and distinctive identity features; right full-body side profile at 90 degrees. Keep the two body views at equal scale, entire body and feet visible. Enlarge the center close-up for facial detail. Keep identity, anatomy, proportions, clothing and accessories identical across all views. Adapt stance to the subject's species; a quadruped stays on four legs. No text labels."
    case .portrait:
      return "Production portrait reference of the same character, head and shoulders, neutral expression, soft even light and a plain backdrop. Make facial structure, eyes, hair or fur and distinctive identity features clear. No text labels."
    case .prop:
      return "Production prop reference sheet of ONE consistent object on a plain neutral background. Three clearly separated views: front, three-quarter and side, equal scale and matching proportions, materials, colors and construction. Show the whole object and its distinctive details. No people, unrelated objects, captions or text labels."
    case .environment:
      return "Environment reference sheet: a wide establishing view plus two complementary views of the SAME environment. Preserve architecture, geography, time of day, palette and lighting across views. Make landmarks and relationships between major spaces clear. No story action, people, captions or text labels."
    case .set:
      return "Set continuity reference sheet: one wide view and two reverse or complementary camera angles of the SAME physical set. Preserve floor plan, doors, windows, fixtures, scale, materials, lighting and object placement across views. Show navigable space and spatial relationships clearly. No story action, people or text labels."
    case .clothing:
      return "Wardrobe reference sheet of ONE consistent outfit or garment, front and back plus a material detail. Preserve silhouette, seams, fastenings, fabric, color and accessories. Neutral background and even light. Use an unadorned display form if needed; do not invent a new character. No text labels."
    case .custom:
      return "Create a clear reusable production reference image of this subject, preserving its defined identity and physical details."
    }
  }
}

/// Records the exact subject definition used to make a candidate. This never grants approval.
public struct ReferenceSheetContext: Codable, Equatable, Identifiable {
  public var subjectKey: String
  public var name: String
  public var kind: PlanningSubjectKind
  public var description: String
  public var linkedDefinitions: String
  public var template: ReferenceSheetTemplate
  public var style = "Realistic cinematic"
  public var direction = ""
  public var id: String { subjectKey }
  public init(subjectKey: String, name: String, kind: PlanningSubjectKind, description: String,
              linkedDefinitions: String = "") {
    self.subjectKey = subjectKey; self.name = name; self.kind = kind; self.description = description
    self.linkedDefinitions = linkedDefinitions; self.template = .suggested(for: kind)
  }
  public var prompt: String {
    var text = template.layout + "\nVisual style: " + style + "\nSubject: " + name + ".\n" + description
    text += "\nUse the subject description as the authoritative baseline identity and appearance. Preserve ordinary authored clothing and equipment. Linked definitions provide context, not an instruction to display every linked object: honor each relationship's role and placement / state. Do not incorporate conditional or future transfers, damage, held objects or poses into the baseline unless explicitly requested in the sheet instructions. Preserve explicitly defined state variants."
    text += "\nIf visual references are supplied, preserve the depicted subject's identity and design while applying the requested views and sheet layout; temporary image states must not override the defined baseline or relationship conditions."
    if !linkedDefinitions.isEmpty { text += "\nLinked objects and relationship conditions (separate definitions):\n" + linkedDefinitions }
    if !direction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { text += "\nView / pose / camera instructions: " + direction }
    return text
  }
  public func apply(to draft: inout DrawThingsImageDraft) {
    draft.name = name + " · " + template.label
    draft.prompt = prompt
    let framingErrors = template == .character ? "cropped full-body views" : "cropped subject"
    draft.negativePrompt = "inconsistent identity, mismatched views, inconsistent materials, \(framingErrors), distorted anatomy, unwanted text, watermark, clutter"
    draft.width = template == .portrait ? 768 : 1280
    draft.height = template == .portrait ? 1024 : 768
    draft.referenceSheet = self
  }
}
