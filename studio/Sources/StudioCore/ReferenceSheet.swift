import Foundation

public enum ReferenceSheetTemplate: String, Codable, CaseIterable, Identifiable {
  case character, portrait, prop, environment, set, clothing, custom
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .character: return "Character · three views"
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
      return "Production character reference sheet on a clean neutral backdrop with soft even lighting. Three full-body views of the SAME character, equal scale, entire body and feet visible: left front view; center true three-quarter view with torso, hips and feet rotated about 40 degrees; right true side profile at 90 degrees. Keep identity, anatomy, proportions, clothing and accessories identical across all views. Adapt stance to the subject's species; a quadruped stays on four legs. No text labels."
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
    text += "\nIf visual references are supplied, preserve the depicted subject's identity and design while applying the requested views and sheet layout."
    if !linkedDefinitions.isEmpty { text += "\nLinked objects (separate definitions; include only where relevant):\n" + linkedDefinitions }
    if !direction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { text += "\nView / pose / camera instructions: " + direction }
    return text
  }
  public func apply(to draft: inout DrawThingsImageDraft) {
    draft.name = name + " · " + template.label
    draft.prompt = prompt
    draft.negativePrompt = "inconsistent identity, mismatched views, inconsistent materials, cropped subject, distorted anatomy, unwanted text, watermark, clutter"
    draft.width = template == .portrait ? 768 : 1280
    draft.height = template == .portrait ? 1024 : 768
    draft.referenceSheet = self
  }
}
