import Foundation

public struct CharacterChoiceValue: Codable, Equatable, Hashable {
  public var id: String
  public var displayValue: String
  public init(id: String, displayValue: String) { self.id = id; self.displayValue = displayValue }
}

public struct CharacterColorValue: Codable, Equatable {
  public var red: Double
  public var green: Double
  public var blue: Double
  public var alpha: Double
  public var displayValue: String
  public init(red: Double, green: Double, blue: Double, alpha: Double = 1, displayValue: String) {
    self.red = red; self.green = green; self.blue = blue; self.alpha = alpha; self.displayValue = displayValue
  }
}

public enum CharacterFieldValue: Codable, Equatable {
  case text(String)
  case choice(id: String, displayValue: String)
  case measurement(value: Decimal, unit: String)
  case color(CharacterColorValue)
  case orderedChoices([CharacterChoiceValue])

  public var displayString: String {
    switch self {
    case .text(let value): return value
    case .choice(_, let displayValue): return displayValue
    case .measurement(let value, let unit): return "\(value) \(unit)"
    case .color(let value): return value.displayValue
    case .orderedChoices(let values): return values.map(\.displayValue).joined(separator: ", ")
    }
  }
}

public enum CharacterFieldState: String, Codable, Equatable { case unspecified, value, explicitlyAbsent, notApplicable }
public enum CharacterFieldSource: String, Codable, Equatable {
  case templateDefault, userAuthored, imageAnalysis, legacyMapping, imported
}

public struct CharacterFieldEvidence: Codable, Equatable {
  public var summary: String
  public var sourceRole: String?
  public var sourceAssetHash: String?
  public var visibility: String?
  public var confidence: Double?
  public var uncertaintyReason: String?
  public init(summary: String, sourceRole: String? = nil, sourceAssetHash: String? = nil,
              visibility: String? = nil, confidence: Double? = nil, uncertaintyReason: String? = nil) {
    self.summary = summary; self.sourceRole = sourceRole; self.sourceAssetHash = sourceAssetHash
    self.visibility = visibility; self.confidence = confidence; self.uncertaintyReason = uncertaintyReason
  }
}

public struct CharacterFieldEntry: Codable, Equatable {
  public var state: CharacterFieldState
  public var value: CharacterFieldValue?
  public var source: CharacterFieldSource
  public var revision: Int
  public var evidence: CharacterFieldEvidence?

  public init(state: CharacterFieldState = .unspecified, value: CharacterFieldValue? = nil,
              source: CharacterFieldSource = .userAuthored, revision: Int = 0,
              evidence: CharacterFieldEvidence? = nil) {
    self.state = state; self.value = state == .value ? value : nil
    self.source = source; self.revision = revision; self.evidence = evidence
  }
  public static func value(_ value: CharacterFieldValue, source: CharacterFieldSource = .userAuthored,
                           revision: Int = 0, evidence: CharacterFieldEvidence? = nil) -> Self {
    .init(state: .value, value: value, source: source, revision: revision, evidence: evidence)
  }
  public var displayString: String { value?.displayString ?? "" }
}

public struct CharacterRepeatableRecord: Codable, Equatable, Identifiable {
  public var id: UUID
  public var order: Int
  public var fields: [String: CharacterFieldEntry]
  public init(id: UUID = UUID(), order: Int, fields: [String: CharacterFieldEntry] = [:]) {
    self.id = id; self.order = order; self.fields = fields
  }
  public func entry(_ key: String) -> CharacterFieldEntry? { fields[key] }
}
public typealias CharacterGarment = CharacterRepeatableRecord
public typealias CharacterAccessory = CharacterRepeatableRecord
public typealias CharacterFeature = CharacterRepeatableRecord
public typealias CharacterSurface = CharacterRepeatableRecord

public struct CharacterAppearance: Codable, Equatable {
  public var fields: [String: CharacterFieldEntry]
  public var garments: [CharacterGarment]
  public var accessories: [CharacterAccessory]
  public var features: [CharacterFeature]
  public var surfaces: [CharacterSurface]

  public init(fields: [String: CharacterFieldEntry] = [:], garments: [CharacterGarment] = [],
              accessories: [CharacterAccessory] = [], features: [CharacterFeature] = [],
              surfaces: [CharacterSurface] = []) {
    self.fields = fields; self.garments = garments; self.accessories = accessories
    self.features = features; self.surfaces = surfaces
  }
  private enum CodingKeys: String, CodingKey { case fields, garments, accessories, features, surfaces }
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    fields = try values.decodeIfPresent([String: CharacterFieldEntry].self, forKey: .fields) ?? [:]
    garments = try values.decodeIfPresent([CharacterGarment].self, forKey: .garments) ?? []
    accessories = try values.decodeIfPresent([CharacterAccessory].self, forKey: .accessories) ?? []
    features = try values.decodeIfPresent([CharacterFeature].self, forKey: .features) ?? []
    surfaces = try values.decodeIfPresent([CharacterSurface].self, forKey: .surfaces) ?? []
  }
  public var orderedGarments: [CharacterGarment] { Self.ordered(garments) }
  public var orderedAccessories: [CharacterAccessory] { Self.ordered(accessories) }
  public var orderedFeatures: [CharacterFeature] { Self.ordered(features) }
  public var orderedSurfaces: [CharacterSurface] { Self.ordered(surfaces) }
  private static func ordered(_ records: [CharacterRepeatableRecord]) -> [CharacterRepeatableRecord] {
    records.sorted { ($0.order, $0.id.uuidString) < ($1.order, $1.id.uuidString) }
  }

  public func entry(at path: String) -> CharacterFieldEntry? {
    if let direct = fields[path] { return direct }
    guard let parsed = CharacterFieldPath(path) else { return nil }
    let records: [CharacterRepeatableRecord]
    switch parsed.collection { case "garments": records = garments; case "accessories": records = accessories
    case "features": records = features; case "surfaces": records = surfaces; default: return nil }
    return records.first { $0.id == parsed.id }?.fields[parsed.field]
  }
  public mutating func setEntry(_ entry: CharacterFieldEntry, at path: String) {
    guard let parsed = CharacterFieldPath(path) else { fields[path] = entry; return }
    switch parsed.collection {
    case "garments": Self.setRecordEntry(&garments, parsed: parsed, entry: entry)
    case "accessories": Self.setRecordEntry(&accessories, parsed: parsed, entry: entry)
    case "features": Self.setRecordEntry(&features, parsed: parsed, entry: entry)
    case "surfaces": Self.setRecordEntry(&surfaces, parsed: parsed, entry: entry)
    default: fields[path] = entry
    }
  }
  private static func setRecordEntry(_ records: inout [CharacterRepeatableRecord], parsed: CharacterFieldPath,
                                     entry: CharacterFieldEntry) {
    if let index = records.firstIndex(where: { $0.id == parsed.id }) { records[index].fields[parsed.field] = entry }
    else { records.append(.init(id: parsed.id, order: records.count, fields: [parsed.field: entry])) }
  }
  public mutating func setText(_ path: String, _ text: String, source: CharacterFieldSource = .userAuthored) {
    setEntry(.value(.text(text), source: source), at: path)
  }
  public mutating func setChoice(_ path: String, id: String, displayValue: String,
                                 source: CharacterFieldSource = .userAuthored) {
    setEntry(.value(.choice(id: id, displayValue: displayValue), source: source), at: path)
  }
  public mutating func setMeasurement(_ path: String, value: Decimal, unit: String,
                                      source: CharacterFieldSource = .userAuthored) {
    setEntry(.value(.measurement(value: value, unit: unit), source: source), at: path)
  }
  public mutating func setColor(_ path: String, red: Double, green: Double, blue: Double, alpha: Double = 1,
                                displayValue: String, source: CharacterFieldSource = .userAuthored) {
    setEntry(.value(.color(.init(red: red, green: green, blue: blue, alpha: alpha, displayValue: displayValue)),
      source: source), at: path)
  }
  public mutating func setOrderedChoices(_ path: String, _ values: [CharacterChoiceValue],
                                         source: CharacterFieldSource = .userAuthored) {
    setEntry(.value(.orderedChoices(values), source: source), at: path)
  }
  public mutating func setState(_ path: String, _ state: CharacterFieldState,
                                source: CharacterFieldSource = .userAuthored) {
    setEntry(.init(state: state, source: source), at: path)
  }
}

struct CharacterFieldPath {
  let collection: String; let id: UUID; let field: String
  init?(_ path: String) {
    guard let open = path.firstIndex(of: "["), let close = path.firstIndex(of: "]"), open < close,
          open > path.startIndex else { return nil }
    let dot = path.index(after: close)
    guard dot < path.endIndex, path[dot] == "." else { return nil }
    let fieldStart = path.index(after: dot)
    guard fieldStart < path.endIndex,
          let id = UUID(uuidString: String(path[path.index(after: open)..<close])) else { return nil }
    collection = String(path[..<open]); self.id = id; field = String(path[fieldStart...])
  }
}
