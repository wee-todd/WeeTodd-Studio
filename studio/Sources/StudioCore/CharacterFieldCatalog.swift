import Foundation

public enum CharacterFieldValueKind: String, Codable { case text, choice, measurement, color, orderedChoices }
public enum CharacterFieldOwnership: String, Codable { case appearance, settings }
public struct CharacterFieldDefinition: Codable, Equatable {
  public var key: String; public var section: Int; public var label: String
  public var valueKind: CharacterFieldValueKind; public var suggestions: [String]
  public var applicability: [String]; public var defaultRequired: Bool
  public var defaultValue: CharacterFieldValue?; public var canBeAbsent: Bool
  public var extractionRoles: [String]; public var maxLength: Int
  public var ownership: CharacterFieldOwnership
}
public struct ResolvedCharacterField {
  public var path: String; public var catalog: CharacterFieldDefinition; public var entry: CharacterFieldEntry?
}
public struct CharacterFieldValidationIssue: Codable, Equatable {
  public var code: String; public var fieldPath: String; public var message: String
  public init(_ code: String, _ path: String, _ message: String) { self.code = code; fieldPath = path; self.message = message }
}

public struct CharacterFieldCatalog: Codable, Equatable {
  public var schemaVersion: Int
  public var fields: [CharacterFieldDefinition]
  public static let shared = CharacterFieldCatalog(schemaVersion: 1, fields: makeFields())

  public func field(for path: String) -> CharacterFieldDefinition? {
    if let exact = fields.first(where: { $0.key == path }) { return exact }
    guard let parsed = CharacterFieldPath(path) else { return nil }
    return fields.first { $0.key == "\(parsed.collection)[].\(parsed.field)" }
  }
  public func resolvedField(path: String, appearance: CharacterAppearance) -> ResolvedCharacterField? {
    guard let catalog = field(for: path) else { return nil }
    return .init(path: path, catalog: catalog, entry: appearance.entry(at: path))
  }
  public func validate(_ entry: CharacterFieldEntry, at path: String,
                       appearance: CharacterAppearance) -> [CharacterFieldValidationIssue] {
    guard let definition = field(for: path) else { return [.init("field.unknown", path, "This field is not in schema version \(schemaVersion).")] }
    if entry.state == .explicitlyAbsent && !definition.canBeAbsent {
      return [.init("state.absenceNotAllowed", path, "This field cannot be explicitly absent.")]
    }
    if entry.state == .notApplicable && definition.applicability.isEmpty {
      return [.init("state.notApplicableNotAllowed", path, "Not applicable is not valid for this field.")]
    }
    guard entry.state == .value, let value = entry.value else { return [] }
    var issues: [CharacterFieldValidationIssue] = []
    let strings: [String]
    switch value {
    case .text(let text): strings = [text]; if definition.valueKind != .text && definition.valueKind != .choice { issues.append(.init("value.type", path, "Use the field's expected value type.")) }
    case .choice(_, let display): strings = [display]; if definition.valueKind != .choice && definition.valueKind != .text { issues.append(.init("value.type", path, "Use the field's expected value type.")) }
    case .measurement(let number, let unit):
      strings = [unit]
      if definition.valueKind != .measurement { issues.append(.init("value.type", path, "Use the field's expected value type.")) }
      if !["mm", "cm", "m", "in", "ft"].contains(unit.lowercased()) { issues.append(.init("measurement.unit", path, "Use mm, cm, m, in or ft.")) }
      if NSDecimalNumber(decimal: number) == .notANumber || number <= 0 { issues.append(.init("measurement.value", path, "Use a positive finite measurement.")) }
    case .color(let color):
      strings = [color.displayValue]
      if definition.valueKind != .color { issues.append(.init("value.type", path, "Use a color value.")) }
      if ![color.red, color.green, color.blue, color.alpha].allSatisfy({ $0.isFinite && (0...1).contains($0) }) {
        issues.append(.init("color.component", path, "Color components must be finite values from 0 through 1."))
      }
    case .orderedChoices(let values): strings = values.map(\.displayValue); if definition.valueKind != .orderedChoices { issues.append(.init("value.type", path, "Use ordered choices.")) }
    }
    for text in strings {
      if text.count > definition.maxLength { issues.append(.init("value.tooLong", path, "Keep this value within \(definition.maxLength) characters.")) }
      if text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) ||
          text.range(of: #"<\s*(lora|lyco|embedding):"#, options: [.regularExpression, .caseInsensitive]) != nil {
        issues.append(.init("value.controlToken", path, "Runtime control tokens and control characters are not allowed."))
      }
    }
    if path.hasSuffix(".side"), let side = strings.first?.lowercased(),
       !["character-left", "character-right", "midline", "bilateral", "unresolved"].contains(side) {
      issues.append(.init("side.invalid", path, "Choose character-left, character-right, midline, bilateral or unresolved."))
    }
    return issues
  }
  public func catalogJSON(prettyPrinted: Bool = false) throws -> Data {
    let encoder = JSONEncoder(); encoder.outputFormatting = prettyPrinted ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
    return try encoder.encode(self)
  }

  private static func makeFields() -> [CharacterFieldDefinition] {
    var result: [CharacterFieldDefinition] = []
    func add(_ section: Int, _ keys: [String], kind: CharacterFieldValueKind = .text,
             suggestions: [String] = [], applicability: [String] = [], required: Bool = false,
             absent: Bool = false, ownership: CharacterFieldOwnership = .appearance) {
      for key in keys { result.append(.init(key: key, section: section,
        label: key.split(separator: ".").last.map(String.init) ?? key, valueKind: kind,
        suggestions: suggestions, applicability: applicability, defaultRequired: required,
        defaultValue: nil, canBeAbsent: absent,
        extractionRoles: ownership == .appearance
          ? (key.hasPrefix("identity.authored") ? ["text"] : ["character", "text"])
          : ["style"],
        maxLength: 240, ownership: ownership)) }
    }
    add(2, ["identity.species", "identity.type"], kind: .choice,
      suggestions: ["human", "humanoid", "animal", "robot", "creature"], required: true)
    add(2, ["identity.authoredAge", "identity.authoredSexGender", "identity.authoredAncestry", "identity.designClass"])
    add(3, ["body.plan"], kind: .choice, suggestions: ["biped", "quadruped", "winged", "serpentine"])
    add(3, ["body.height", "body.scale"], kind: .measurement)
    add(3, ["body.build", "body.shoulders", "body.torso", "body.arms", "body.legs", "body.musculature", "body.massDistribution", "body.posture"])
    add(3, ["body.pose"], ownership: .settings)
    add(4, ["head.shape", "face.shape", "face.jaw", "face.cheekbones", "face.noseMuzzleBeak", "face.mouth", "face.lips", "eyes.shape", "eyes.spacing", "brows.shape", "ears.shape"])
    add(4, ["eyes.color"], kind: .color, suggestions: ["Blue", "Brown", "Hazel", "Green", "Black", "Amber", "Gray"])
    add(4, ["face.covering"], kind: .choice, suggestions: ["skin", "fur", "scales", "metal"], applicability: ["anatomy"])
    add(4, ["face.expression"], ownership: .settings)
    add(5, ["hair.color"], kind: .color, applicability: ["hair"], absent: true)
    add(5, ["hair.length", "hair.texture", "hair.style", "hair.hairline", "hair.facialHair"], applicability: ["hair"], absent: true)
    add(5, ["covering.pattern", "covering.growthDirection"], applicability: ["bodyCovering"])
    add(6, ["garments[].type", "garments[].bodyRegion", "garments[].layer", "garments[].fit", "garments[].cut", "garments[].color", "garments[].closures", "garments[].seams", "garments[].trim", "garments[].condition", "garments[].placement"])
    add(7, ["accessories[].type", "accessories[].color", "accessories[].placement", "accessories[].side", "accessories[].orientation", "accessories[].shape", "accessories[].condition"])
    add(7, ["features[].type", "features[].shape", "features[].color", "features[].placement", "features[].side", "features[].orientation", "features[].asymmetry"])
    add(8, ["surfaces[].target", "surfaces[].material", "surfaces[].finish", "surfaces[].texture", "surfaces[].pattern", "surfaces[].wear", "surfaces[].detailScale"])
    add(9, ["style.presetID"], kind: .choice, suggestions: CharacterStylePresetRegistry.all.map(\.id), required: true, ownership: .settings)
    add(9, ["style.paletteTreatment", "style.edgeTreatment", "style.textureTreatment", "style.finish", "style.contrast", "style.detailLevel"], ownership: .settings)
    add(10, ["camera.projection", "camera.lensAppearance", "camera.elevation"], ownership: .settings)
    add(11, ["lighting.preset", "lighting.key", "lighting.fill", "lighting.rim", "lighting.exposure", "lighting.shadows", "lighting.color"], ownership: .settings)
    return result.sorted { ($0.section, $0.key) < ($1.section, $1.key) }
  }
}
