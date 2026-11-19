import Foundation

/// Small explicit field projections. Callers own the whitelist and typed validation.
public enum GenerationPresetFields {
  private static let nestedOwnedKeys: [String: Set<String>] = [
    "ltx25Guidance": ["mode", "experimentalEnabled", "audioCFG", "stgScale", "videoRescale", "audioRescale", "modalityScale", "stgBlocks", "sigmas"]
  ]
  public static func capture<T: Encodable>(_ value: T, keys: Set<String>) throws -> [String: GenerationPresetValue] {
    guard case .object(let values) = try GenerationPresetValue.encoded(value) else { throw StudioError.invalid("Invalid preset recipe object.") }
    return Dictionary(uniqueKeysWithValues: keys.map { key in
      var value = values[key] ?? .automatic
      if let owned = nestedOwnedKeys[key], case .object(var fields) = value {
        for field in owned where fields[field] == nil { fields[field] = .automatic }
        value = .object(fields)
      }
      return (key, value)
    })
  }
  public static func applying<T: Codable>(_ fields: [String: GenerationPresetValue], to value: T, allowed: Set<String>) throws -> T {
    guard Set(fields.keys).isSubset(of: allowed), case .object(var values) = try GenerationPresetValue.encoded(value) else {
      throw StudioError.invalid("This preset contains unsupported recipe fields. Update Studio or choose a compatible preset.")
    }
    for (key, value) in fields {
      if let owned = nestedOwnedKeys[key], case .object(let updates) = value, !Set(updates.keys).isSubset(of: owned) {
        throw StudioError.invalid("This preset contains unsupported nested recipe fields.")
      }
    }
    func merge(_ old: GenerationPresetValue, _ new: GenerationPresetValue) -> GenerationPresetValue {
      guard case .object(var previous) = old, case .object(let updates) = new else { return new }
      for (key, value) in updates { previous[key] = previous[key].map { merge($0, value) } ?? value }
      return .object(previous)
    }
    values.merge(fields) { merge($0, $1) }
    return try GenerationPresetValue.object(values).decoded(T.self)
  }
}

extension GenerationLibrarySnapshot {
  public func nativePresetAdapters(_ members: [LoRAMember], model: GenerationModelRef, context: GenerationLibraryContext) throws -> [GenerationPresetAdapter] {
    try members.map { member in
      guard let entry = loras(for: context, model: model).first(where: {
        if case .nativeFile(let path, _, _, _) = $0.binding { return path == member.asset.path }; return false
      }) else { throw StudioError.invalid("Link \(member.asset.name) in Models & LoRAs before saving this preset.") }
      _ = try resolve(entry.reference, for: model, in: context)
      var result = GenerationPresetAdapter(reference: entry.reference, strength: member.strength, enabled: member.isEnabled)
      if member.asset.loraModel == .h3 { result.fields["h3Settings"] = try .encoded(member.h3Settings) }
      return result
    }
  }
  public func nativePresetMembers(_ adapters: [GenerationPresetAdapter], model: GenerationModelRef, context: GenerationLibraryContext, existing: [LoRAMember] = []) throws -> [LoRAMember] {
    try adapters.map { adapter in
      guard case .nativeFile(let path, let profile, let layout, let grid) = try resolve(adapter.reference, for: model, in: context),
        let entry = loraEntries.first(where: { $0.reference == adapter.reference }),
        let family = LoRAModel(rawValue: adapter.reference.family), Set(adapter.fields.keys).isSubset(of: ["h3Settings"]) else {
        throw StudioError.invalid("Choose compatible native LoRAs for this preset.")
      }
      let previous = existing.first { $0.asset.path == path && $0.asset.loraModel == family && $0.asset.loraProfile == profile && $0.asset.loraLayout == layout && $0.asset.loraAdalnInputGrid == grid }
      var asset = previous?.asset ?? MediaAsset(name: entry.name, kind: .lora, path: path, scope: .global)
      asset.loraModel = family; asset.loraProfile = profile; asset.loraLayout = layout; asset.loraAdalnInputGrid = grid
      let settings: H3LoRASettings? = try adapter.fields["h3Settings"].map { try $0.decoded(H3LoRASettings?.self) } ?? previous?.h3Settings
      var member = previous ?? LoRAMember(asset: asset, strength: adapter.strength, enabled: adapter.enabled, h3Settings: settings)
      member.asset = asset; member.strength = adapter.strength; member.h3Settings = settings
      if member.isEnabled != adapter.enabled { member.enabled = adapter.enabled }
      return member
    }
  }
}
