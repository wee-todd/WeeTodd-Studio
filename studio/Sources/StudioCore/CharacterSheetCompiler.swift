import Foundation

public struct CharacterSheetDiagnostic: Codable, Equatable {
  public enum Severity: String, Codable { case error, warning }
  public var code: String; public var fieldPath: String; public var message: String; public var severity: Severity
  public init(code: String, fieldPath: String, message: String, severity: Severity = .error) {
    self.code = code; self.fieldPath = fieldPath; self.message = message; self.severity = severity
  }
}
public struct CharacterSheetSection: Equatable {
  public var index: Int; public var label: String; public var text: String; public var range: Range<String.Index>
}
public struct CompiledCharacterSheet: Equatable {
  public var prompt: String; public var sections: [CharacterSheetSection]
  public var diagnostics: [CharacterSheetDiagnostic]; public var canGenerate: Bool
  public var compilerVersion: Int; public var inputDigest: String; public var presetSnapshot: ResolvedCharacterStyle?
}

public enum CharacterSheetCompiler {
  public static let compilerVersion = 2
  public static let sectionLabels = [
    "Required LoRA trigger / layout", "Character identity", "Body / proportions", "Face / head", "Hair",
    "Clothing", "Accessories / distinctive features", "Materials / surface detail", "Style / rendering",
    "Camera / lens", "Lighting", "Turnaround consistency rules", "Composition / framing"
  ]

  public static func compile(_ definition: CharacterSheetDefinition) -> CompiledCharacterSheet {
    let catalog = CharacterFieldCatalog.shared
    var diagnostics: [CharacterSheetDiagnostic] = []
    if definition.schemaVersion != catalog.schemaVersion {
      diagnostics.append(.init(code: "schema.unsupported", fieldPath: "schemaVersion",
        message: "This character sheet schema version is not supported."))
    }
    for (name, count) in [("garments", definition.appearance.garments.count),
                          ("accessories", definition.appearance.accessories.count),
                          ("features", definition.appearance.features.count),
                          ("surfaces", definition.appearance.surfaces.count)] where count > 32 {
      diagnostics.append(.init(code: "records.tooMany", fieldPath: name,
        message: "Keep each repeatable character category to 32 records or fewer."))
    }
    for path in definition.settings.requiredFieldPaths.sorted() {
      if path == "style.presetID" {
        if definition.settings.stylePresetID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
          diagnostics.append(.init(code: "required.missing", fieldPath: path, message: "Choose a style preset."))
        }
      } else {
        let entry = definition.entry(at: path)
        let field = catalog.field(for: path)
        let resolves: Bool
        switch entry?.state {
        case .value: resolves = entry?.value?.displayString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        case .explicitlyAbsent: resolves = field?.canBeAbsent == true
        case .notApplicable: resolves = field?.applicability.isEmpty == false
        default: resolves = false
        }
        if !resolves {
          diagnostics.append(.init(code: "required.missing", fieldPath: path, message: "Complete this required character field."))
        }
      }
    }
    for field in catalog.fields where !field.key.contains("[]") {
      if let entry = definition.entry(at: field.key) {
        diagnostics += catalog.validate(entry, at: field.key, appearance: definition.appearance).map {
          .init(code: $0.code, fieldPath: $0.fieldPath, message: $0.message)
        }
      }
    }
    for (collection, records) in [("garments", definition.appearance.orderedGarments),
                                  ("accessories", definition.appearance.orderedAccessories),
                                  ("features", definition.appearance.orderedFeatures),
                                  ("surfaces", definition.appearance.orderedSurfaces)] {
      for record in records { for (field, entry) in record.fields {
        let path = "\(collection)[\(record.id.uuidString)].\(field)"
        diagnostics += catalog.validate(entry, at: path, appearance: definition.appearance).map {
          .init(code: $0.code, fieldPath: $0.fieldPath, message: $0.message)
        }
      } }
    }
    let namedTargets = repeatableTargetNames(definition.appearance)
    for surface in definition.appearance.orderedSurfaces {
      guard let targetID = surfaceTargetID(surface.fields["target"]?.value), namedTargets[targetID] == nil else { continue }
      diagnostics.append(.init(code: "surface.targetMissing",
        fieldPath: "surfaces[\(surface.id.uuidString)].target",
        message: "Choose an existing garment or accessory as this material target."))
    }

    let preset: ResolvedCharacterStyle?
    do { preset = try CharacterStylePresetRegistry.resolve(id: definition.settings.stylePresetID,
      version: definition.settings.stylePresetVersion, appearance: definition.appearance) }
    catch {
      preset = nil
      diagnostics.append(.init(code: "preset.unknown", fieldPath: "style.presetID", message: "Choose an available style preset version."))
    }
    if definition.settings.stylePresetID == "photograph" {
      let conflicting = ["style.paletteTreatment", "style.edgeTreatment", "style.textureTreatment", "style.finish"]
        .first { path in
          guard let text = definition.settings.fields[path]?.value?.displayString.lowercased() else { return false }
          return ["3d", "cgi", "anime", "comic", "paint", "illustration", "cartoon"].contains { text.contains($0) }
        }
      if let conflicting { diagnostics.append(.init(code: "style.conflict", fieldPath: conflicting,
        message: "This rendering refinement conflicts with Photograph.")) }
    }

    var sectionText = Array(repeating: "", count: 13)
    sectionText[0] = ReferenceSheetTemplate.characterSheetPrefix
    for section in 2...11 {
      var clauses: [String] = []
      if section == 9 { clauses += preset?.clauses.map(\.text) ?? [] }
      let scalpOrder = ["hair.scalpCoverage": 0, "hair.hairline": 1, "hair.style": 2]
      let scalarFields = catalog.fields.filter { $0.section == section && !$0.key.contains("[]") }
        .sorted { (scalpOrder[$0.key] ?? 3, $0.key) < (scalpOrder[$1.key] ?? 3, $1.key) }
      for field in scalarFields {
        guard !(section == 9 && definition.settings.stylePresetID == "photograph" && diagnostics.contains(where: { $0.code == "style.conflict" && $0.fieldPath == field.key })) else { continue }
        if let clause = render(definition.entry(at: field.key), key: field.key) { clauses.append(clause) }
      }
      if section == 6 { clauses += renderRecords(definition.appearance.orderedGarments, collection: "garments") }
      if section == 7 {
        clauses += renderRecords(definition.appearance.orderedAccessories, collection: "accessories")
        clauses += renderRecords(definition.appearance.orderedFeatures, collection: "features")
      }
      if section == 8 { clauses += renderSurfaces(definition.appearance.orderedSurfaces, namedTargets: namedTargets) }
      sectionText[section - 1] = clauses.joined(separator: ", ")
    }
    let quadruped = definition.entry(at: "body.plan")?.value?.displayString.lowercased().contains("quadruped") == true
    sectionText[11] = quadruped
      ? "Keep the same identity, anatomy, proportions, face, covering, outfit, accessories, colors, materials and anatomical sides across every view; the quadruped stays on four legs in equivalent neutral stances."
      : "Keep the same identity, anatomy, proportions, face, hair or covering, outfit, accessories, colors, materials and anatomical sides across every view; use equivalent neutral stances and expressions without redesign, aging or beautification."
    sectionText[12] = "one row of four distinct panels, clearly separated: complete full-body front, side and back views at equal scale, plus an enlarged matching facial or head close-up; keep heads and extremities visible, center each subject on a plain solid white backdrop, with no scenery or panel labels."

    var prompt = ""; var sections: [CharacterSheetSection] = []
    for index in sectionText.indices {
      if index > 0 { prompt.append("\n") }
      let start = prompt.endIndex; prompt.append(sectionText[index]); let end = prompt.endIndex
      sections.append(.init(index: index + 1, label: sectionLabels[index], text: sectionText[index], range: start..<end))
    }
    return .init(prompt: prompt, sections: sections, diagnostics: diagnostics,
      canGenerate: !diagnostics.contains { $0.severity == .error }, compilerVersion: compilerVersion,
      inputDigest: digest(definition, preset: preset), presetSnapshot: preset)
  }

  private static func render(_ entry: CharacterFieldEntry?, key: String) -> String? {
    guard let entry else { return nil }
    switch entry.state {
    case .unspecified, .notApplicable: return nil
    case .value:
      let text = entry.value?.displayString.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      guard !text.isEmpty else { return nil }
      return "\(humanLabel(key)): \(text)"
    case .explicitlyAbsent:
      let noun = key == "hair.facialHair" ? "facial hair" : humanLabel(key)
      return "no \(noun)"
    }
  }
  private static func renderRecords(_ records: [CharacterRepeatableRecord], collection: String) -> [String] {
    records.map { record in
      let fields = record.fields.sorted { $0.key < $1.key }.compactMap { key, entry -> String? in
        guard let value = render(entry, key: key) else { return nil }
        return value
      }
      return fields.isEmpty ? nil : "\(collection.dropLast()): " + fields.joined(separator: ", ")
    }.compactMap { $0 }
  }
  private static func renderSurfaces(_ records: [CharacterRepeatableRecord],
                                     namedTargets: [UUID: String]) -> [String] {
    records.compactMap { record in
      let fields = record.fields.sorted { $0.key < $1.key }.compactMap { key, entry -> String? in
        if key == "target", let id = surfaceTargetID(entry.value) {
          guard let name = namedTargets[id] else { return nil }
          return "target: \(name)"
        }
        return render(entry, key: key)
      }
      return fields.isEmpty ? nil : "surface: " + fields.joined(separator: ", ")
    }
  }
  private static func repeatableTargetNames(_ appearance: CharacterAppearance) -> [UUID: String] {
    var names: [UUID: String] = [:]
    for record in appearance.orderedGarments + appearance.orderedAccessories {
      let name = record.fields["type"]?.value?.displayString.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      if !name.isEmpty { names[record.id] = name }
    }
    return names
  }
  private static func surfaceTargetID(_ value: CharacterFieldValue?) -> UUID? {
    switch value {
    case .text(let raw): return UUID(uuidString: raw)
    case .choice(let id, let displayValue): return UUID(uuidString: id) ?? UUID(uuidString: displayValue)
    default: return nil
    }
  }
  private static func humanLabel(_ key: String) -> String {
    let last = key.replacingOccurrences(of: ".", with: " ")
    return last.replacingOccurrences(of: "([a-z])([A-Z])", with: "$1 $2", options: .regularExpression).lowercased()
  }
  private static func digest(_ definition: CharacterSheetDefinition, preset: ResolvedCharacterStyle?) -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    struct DigestInput: Codable { let definition: CharacterSheetDefinition; let preset: ResolvedCharacterStyle? }
    let data = (try? encoder.encode(DigestInput(definition: definition, preset: preset))) ?? Data()
    var hash: UInt64 = 14695981039346656037
    for byte in data { hash ^= UInt64(byte); hash &*= 1099511628211 }
    return String(format: "%016llx", hash)
  }
}
