import Foundation

public struct CharacterStyleClause: Codable, Equatable {
  public var id: String; public var text: String; public var applicability: String?
  public init(id: String, text: String, applicability: String? = nil) {
    self.id = id; self.text = text; self.applicability = applicability
  }
}
public struct CharacterStylePreset: Codable, Equatable, Identifiable {
  public var id: String; public var version: Int; public var label: String; public var baseClauses: [CharacterStyleClause]
}
public struct ResolvedCharacterStyle: Codable, Equatable {
  public var id: String; public var version: Int; public var label: String; public var clauses: [CharacterStyleClause]
}
public enum CharacterStylePresetError: Error, Equatable { case unknownPreset(String, Int) }

public enum CharacterStylePresetRegistry {
  public static let all: [CharacterStylePreset] = [
    preset("photograph", "Photograph", ["high-resolution photorealistic studio character photography", "physically plausible materials", "realistic global illumination", "high dynamic range", "neutral color reproduction", "subtle natural surface imperfections", "professional character reference photography"]),
    preset("cinematicPhotograph", "Cinematic Photograph", ["high-resolution photorealistic studio character photography", "controlled cinematic tonal response", "film-like highlight roll-off", "restrained photographic grain", "physically plausible materials", "preserved local colors", "professional character reference photography"]),
    preset("realistic3D", "Realistic 3D", ["high-resolution physically based 3D character rendering", "physically plausible material response", "fine geometric and surface detail", "neutral color reproduction", "realistic light transport"]),
    preset("stylized3D", "Stylized 3D", ["stylized 3D character rendering", "controlled simplified surface shading", "clean material separation", "deliberate edge definition", "preserved authored proportions and local colors"]),
    preset("anime", "Anime", ["anime character reference illustration", "clean controlled linework", "flat local colors", "restrained cel shading", "precise feature placement", "preserved authored proportions"]),
    preset("comic", "Comic", ["comic character reference illustration", "controlled ink outlines", "clear shape separation", "restrained graphic shadow shapes", "preserved local colors and authored proportions"]),
    preset("oilPainting", "Oil Painting", ["oil-painted character reference", "controlled visible brush texture", "opaque pigment layering", "clearly resolved contour and facial structure", "preserved local colors and authored proportions"]),
    preset("conceptArt", "Concept Art", ["production character concept illustration", "precise construction detail", "controlled edges", "readable material separation", "preserved silhouette, local colors and authored proportions"]),
    preset("clay", "Clay", ["studio clay-maquette character rendering", "matte clay surface treatment", "subtle sculpted surface texture", "clearly resolved anatomical and garment forms", "preserved silhouette, authored proportions and local color boundaries"]),
    preset("sculpture", "Sculpture", ["studio sculpture character study", "clearly resolved sculpted forms", "controlled surface finish", "precise facial and garment construction", "preserved silhouette, authored proportions and local color boundaries"]),
  ]
  public static func resolve(id: String, version: Int, appearance: CharacterAppearance) throws -> ResolvedCharacterStyle {
    guard let preset = all.first(where: { $0.id == id && $0.version == version }) else { throw CharacterStylePresetError.unknownPreset(id, version) }
    var clauses = preset.baseClauses
    if ["photograph", "cinematicPhotograph", "realistic3D"].contains(id) {
      if appearance.entry(at: "body.plan")?.state == .value { clauses.insert(.init(id: "anatomy", text: "realistic anatomy within the supplied body plan", applicability: "bodyPlan"), at: min(1, clauses.count)) }
      if appearance.entry(at: "face.covering")?.value?.displayString.lowercased() == "skin" { clauses.insert(.init(id: "skin", text: "natural skin texture with pores and fine variation", applicability: "naturalSkin"), at: min(2, clauses.count)) }
      let hasHair = ["hair.color", "hair.length", "hair.texture", "hair.style", "hair.hairline", "hair.facialHair"].contains { appearance.entry(at: $0)?.state == .value }
      if hasHair { clauses.append(.init(id: "hair", text: "realistic hair strands", applicability: "hair")) }
      let hasTextile = appearance.surfaces.contains { record in record.fields.values.contains { $0.displayString.lowercased().contains("textile") || $0.displayString.lowercased().contains("woven") || $0.displayString.lowercased().contains("knit") } }
      if hasTextile { clauses.append(.init(id: "textile", text: "fine textile detail", applicability: "textile")) }
    }
    return .init(id: preset.id, version: preset.version, label: preset.label, clauses: clauses)
  }
  private static func preset(_ id: String, _ label: String, _ clauses: [String]) -> CharacterStylePreset {
    .init(id: id, version: 1, label: label, baseClauses: clauses.enumerated().map { .init(id: "base.\($0.offset)", text: $0.element) })
  }
}
