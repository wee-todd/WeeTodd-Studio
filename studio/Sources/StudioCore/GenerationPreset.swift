import Foundation

/// An absent key preserves a draft value; automatic explicitly clears an override.
public indirect enum GenerationPresetValue: Codable, Equatable {
  case automatic, bool(Bool), integer(Int64), unsigned(UInt64), number(Double), string(String)
  case array([GenerationPresetValue]), object([String: GenerationPresetValue])
  public static func == (lhs: Self, rhs: Self) -> Bool {
    switch (lhs, rhs) {
    case (.automatic, .automatic): return true
    case (.bool(let a), .bool(let b)): return a == b
    case (.integer(let a), .integer(let b)): return a == b
    case (.unsigned(let a), .unsigned(let b)): return a == b
    case (.integer(let a), .unsigned(let b)), (.unsigned(let b), .integer(let a)): return UInt64(exactly: a) == b
    case (.number(let a), .unsigned(let b)), (.unsigned(let b), .number(let a)): return UInt64(exactly: a) == b
    case (.number(let a), .number(let b)): return a == b
    case (.integer(let a), .number(let b)), (.number(let b), .integer(let a)): return Int64(exactly: b) == a
    case (.string(let a), .string(let b)): return a == b
    case (.array(let a), .array(let b)): return a == b
    case (.object(let a), .object(let b)): return a == b
    default: return false
    }
  }
  public init(from decoder: Decoder) throws {
    let value = try decoder.singleValueContainer()
    if value.decodeNil() { self = .automatic }
    else if let decoded = try? value.decode(Bool.self) { self = .bool(decoded) }
    else if let decoded = try? value.decode(Int64.self) { self = .integer(decoded) }
    else if let decoded = try? value.decode(UInt64.self) { self = .unsigned(decoded) }
    else if let decoded = try? value.decode(Double.self) { self = .number(decoded) }
    else if let decoded = try? value.decode(String.self) { self = .string(decoded) }
    else if let decoded = try? value.decode([Self].self) { self = .array(decoded) }
    else { self = .object(try value.decode([String: Self].self)) }
  }
  public func encode(to encoder: Encoder) throws {
    var value = encoder.singleValueContainer()
    switch self {
    case .automatic: try value.encodeNil()
    case .bool(let v): try value.encode(v)
    case .integer(let v): try value.encode(v)
    case .unsigned(let v): try value.encode(v)
    case .number(let v): guard v.isFinite else { throw StudioError.invalid("Preset numbers must be finite.") }; try value.encode(v)
    case .string(let v): try value.encode(v)
    case .array(let v): try value.encode(v)
    case .object(let v): try value.encode(v)
    }
  }
  public static func encoded<T: Encodable>(_ value: T) throws -> Self { try JSONDecoder().decode(Self.self, from: JSONEncoder().encode(value)) }
  public func decoded<T: Decodable>(_ type: T.Type = T.self) throws -> T { try JSONDecoder().decode(type, from: JSONEncoder().encode(self)) }
  public var display: String {
    switch self {
    case .automatic: return "Automatic"
    case .bool(let v): return v ? "On" : "Off"
    case .integer(let v): return String(v)
    case .unsigned(let v): return String(v)
    case .number(let v): return String(v)
    case .string(let v): return v.isEmpty ? "Empty" : v
    case .array(let v): return v.isEmpty ? "None" : v.map(\.display).joined(separator: ", ")
    case .object(let v): return v.keys.sorted().map { $0 + ": " + v[$0]!.display }.joined(separator: "; ")
    }
  }
}

public struct GenerationPresetContext: Codable, Equatable, Hashable {
  public var workspace: GenerationWorkspaceKind
  public var task: String
  public init(workspace: GenerationWorkspaceKind, task: String) { self.workspace = workspace; self.task = task }
}
public struct GenerationPresetAdapter: Codable, Equatable {
  public var reference: GenerationLoRARef
  public var strength: Double
  public var enabled: Bool
  public var fields: [String: GenerationPresetValue] = [:]
  public init(reference: GenerationLoRARef, strength: Double = 1, enabled: Bool = true) {
    self.reference = reference; self.strength = strength; self.enabled = enabled
  }
  private enum CodingKeys: String, CodingKey { case reference, strength, enabled, fields }
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    reference = try c.decode(GenerationLoRARef.self, forKey: .reference)
    strength = try c.decode(Double.self, forKey: .strength); enabled = try c.decode(Bool.self, forKey: .enabled)
    fields = try c.decodeIfPresent([String: GenerationPresetValue].self, forKey: .fields) ?? [:]
  }
}
public struct GenerationPresetSnapshot: Codable, Equatable {
  public var schemaVersion = 1
  public var context: GenerationPresetContext
  public var model: GenerationModelRef?
  /// Nil is an older preset with no stack field; [] explicitly clears the stack.
  public var adapters: [GenerationPresetAdapter]?
  public var fields: [String: GenerationPresetValue]
  public init(context: GenerationPresetContext, model: GenerationModelRef? = nil,
    adapters: [GenerationPresetAdapter]? = nil, fields: [String: GenerationPresetValue] = [:]) {
    self.context = context; self.model = model; self.adapters = adapters; self.fields = fields
  }
  public func validate() throws {
    guard schemaVersion == 1, !context.task.isEmpty, context.task.count <= 128, fields.count <= 128,
      fields.keys.allSatisfy({ !$0.isEmpty && $0.count <= 128 }), (adapters?.count ?? 0) <= 16 else { throw StudioError.invalid("Unsupported or invalid preset settings.") }
    try model?.validate()
    for adapter in adapters ?? [] {
      try adapter.reference.validate()
      guard adapter.strength.isFinite, (-10...10).contains(adapter.strength) else { throw StudioError.invalid("Use a finite supported LoRA strength.") }
    }
    guard Set((adapters ?? []).map { $0.reference.id }).count == (adapters?.count ?? 0),
      try JSONEncoder().encode(self).count <= 64 * 1024 else { throw StudioError.invalid("A preset must fit within 64 KiB and contain distinct LoRAs.") }
  }
}
public struct StudioGenerationPreset: Codable, Equatable, Identifiable {
  public enum Origin: String, Codable { case supplied, user }
  public var id: String
  public var name: String
  public var revision: Int
  public var origin: Origin
  public var snapshot: GenerationPresetSnapshot
  public init(id: String = UUID().uuidString, name: String, revision: Int = 1, origin: Origin = .user, snapshot: GenerationPresetSnapshot) {
    self.id = id; self.name = name; self.revision = revision; self.origin = origin; self.snapshot = snapshot
  }
}
public struct PresetSelectionState: Codable, Equatable {
  public var presetID: String
  public var appliedRevision: Int
  public var baseline: GenerationPresetSnapshot
  public init(_ preset: StudioGenerationPreset, baseline: GenerationPresetSnapshot? = nil) {
    presetID = preset.id; appliedRevision = preset.revision; self.baseline = baseline ?? preset.snapshot
  }
  public func isModified(_ current: GenerationPresetSnapshot) -> Bool { !GenerationPresetDiff.changes(from: baseline, to: current).isEmpty }
}
