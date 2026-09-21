import Foundation

public struct CharacterSheetSettings: Codable, Equatable {
  public var stylePresetID: String
  public var stylePresetVersion: Int
  public var fields: [String: CharacterFieldEntry]
  public var requiredFieldPaths: Set<String>

  public init(stylePresetID: String, stylePresetVersion: Int = 1,
              fields: [String: CharacterFieldEntry] = [:], requiredFieldPaths: Set<String> = []) {
    self.stylePresetID = stylePresetID; self.stylePresetVersion = stylePresetVersion
    self.fields = fields; self.requiredFieldPaths = requiredFieldPaths
  }
  private enum CodingKeys: String, CodingKey { case stylePresetID, stylePresetVersion, fields, requiredFieldPaths }
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    stylePresetID = try values.decode(String.self, forKey: .stylePresetID)
    stylePresetVersion = try values.decodeIfPresent(Int.self, forKey: .stylePresetVersion) ?? 1
    fields = try values.decodeIfPresent([String: CharacterFieldEntry].self, forKey: .fields) ?? [:]
    requiredFieldPaths = try values.decodeIfPresent(Set<String>.self, forKey: .requiredFieldPaths) ?? []
  }
  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(stylePresetID, forKey: .stylePresetID)
    try values.encode(stylePresetVersion, forKey: .stylePresetVersion)
    try values.encode(fields, forKey: .fields)
    try values.encode(requiredFieldPaths.sorted(), forKey: .requiredFieldPaths)
  }
  public static func newDraft() -> Self {
    var settings = Self(stylePresetID: "photograph", requiredFieldPaths: ["identity.species", "identity.type", "style.presetID"])
    settings.setText("body.pose", "neutral species-appropriate pose", source: .templateDefault)
    settings.setText("face.expression", "neutral relaxed expression", source: .templateDefault)
    settings.setText("camera.projection", "orthographic-like with minimal perspective", source: .templateDefault)
    settings.setText("camera.lensAppearance", "approximately 70–100mm full-frame equivalent appearance", source: .templateDefault)
    settings.setText("camera.elevation", "neutral eye-level", source: .templateDefault)
    settings.setText("lighting.preset", "soft neutral studio lighting", source: .templateDefault)
    settings.setText("lighting.key", "large diffused key", source: .templateDefault)
    settings.setText("lighting.fill", "gentle fill", source: .templateDefault)
    settings.setText("lighting.rim", "subtle rim separation", source: .templateDefault)
    settings.setText("lighting.exposure", "balanced exposure", source: .templateDefault)
    settings.setText("lighting.shadows", "minimal shadows", source: .templateDefault)
    settings.setText("lighting.color", "neutral light", source: .templateDefault)
    return settings
  }
  public func entry(at path: String) -> CharacterFieldEntry? { fields[path] }
  public mutating func setEntry(_ entry: CharacterFieldEntry, at path: String) { fields[path] = entry }
  public mutating func setText(_ path: String, _ value: String, source: CharacterFieldSource = .userAuthored) {
    fields[path] = .value(.text(value), source: source)
  }
  public mutating func setChoice(_ path: String, id: String, displayValue: String,
                                 source: CharacterFieldSource = .userAuthored) {
    fields[path] = .value(.choice(id: id, displayValue: displayValue), source: source)
  }
  public mutating func setState(_ path: String, _ state: CharacterFieldState,
                                source: CharacterFieldSource = .userAuthored) {
    fields[path] = .init(state: state, source: source)
  }
}

public struct CharacterSheetDefinition: Codable, Equatable {
  public var schemaVersion: Int
  public var appearance: CharacterAppearance
  public var settings: CharacterSheetSettings
  public init(schemaVersion: Int = 1, appearance: CharacterAppearance = .init(), settings: CharacterSheetSettings) {
    self.schemaVersion = schemaVersion; self.appearance = appearance; self.settings = settings
  }
  private enum CodingKeys: String, CodingKey { case schemaVersion, appearance, settings }
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    schemaVersion = try values.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
    appearance = try values.decode(CharacterAppearance.self, forKey: .appearance)
    settings = try values.decode(CharacterSheetSettings.self, forKey: .settings)
  }
  public static func newDraft() -> Self {
    var appearance = CharacterAppearance()
    appearance.fields["identity.species"] = .init(state: .unspecified, source: .templateDefault)
    appearance.fields["identity.type"] = .init(state: .unspecified, source: .templateDefault)
    return .init(appearance: appearance, settings: .newDraft())
  }
  public func entry(at path: String) -> CharacterFieldEntry? { settings.fields[path] ?? appearance.entry(at: path) }
  public mutating func setEntry(_ entry: CharacterFieldEntry, at path: String) {
    if CharacterFieldCatalog.shared.field(for: path)?.ownership == .settings { settings.setEntry(entry, at: path) }
    else { appearance.setEntry(entry, at: path) }
  }
}
