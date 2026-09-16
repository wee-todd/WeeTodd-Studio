import Foundation

/// Training provenance filters candidates; the renderer validates tensor targets and shapes.
public enum LoRAModel: String, Codable, CaseIterable, Identifiable {
  case h3, ltx23, ltx25
  public var id: String { rawValue }
  public var label: String { Engine(rawValue: rawValue)!.label }
  public func supports(_ engine: Engine) -> Bool {
    rawValue == engine.rawValue || (self == .ltx23 && engine == .ltx25)
  }
}

public enum LoRAGroupApplicationMode: String, CaseIterable, Identifiable {
  case add, replace
  public var id: String { rawValue }
  public var label: String {
    self == .add ? "Add to current stack" : "Replace current stack"
  }
}

public struct LoRAMember: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var asset: MediaAsset
  public var strength: Double
  public var enabled: Bool?
  public var isEnabled: Bool { enabled ?? true }
  public init(asset: MediaAsset, strength: Double = 1, enabled: Bool? = nil) {
    self.asset = asset
    self.strength = strength
    self.enabled = enabled
  }
  public var fileKey: String {
    URL(fileURLWithPath: asset.path).standardizedFileURL.resolvingSymlinksInPath().path
  }
  public func validate(for engine: Engine) throws {
    guard asset.kind == .lora, asset.loraModel?.supports(engine) == true else {
      throw StudioError.invalid(
        "Choose a LoRA trained for \(engine.label). LTX 2.5 also accepts compatible LTX 2.3 adapters."
      )
    }
    guard !asset.path.isEmpty,
      URL(fileURLWithPath: asset.path).pathExtension.lowercased() == "safetensors"
    else {
      throw StudioError.invalid("Link a SafeTensors LoRA file for \(asset.name).")
    }
    guard strength.isFinite, (0...2).contains(strength) else {
      throw StudioError.invalid("LoRA strength must be a finite number from 0 to 2.")
    }
  }
}

public struct LoRAGroup: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var name: String
  public var engine: Engine
  public var members: [LoRAMember]
  public init(name: String, engine: Engine, members: [LoRAMember] = []) {
    self.name = name
    self.engine = engine
    self.members = members
  }
  public func validate() throws {
    guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !members.isEmpty else {
      throw StudioError.invalid("Name the group and add at least one LoRA.")
    }
    for member in members { try member.validate(for: engine) }
    guard Set(members.map(\.fileKey)).count == members.count else {
      throw StudioError.invalid("A group can contain each LoRA file only once.")
    }
  }
  public func supports(_ target: Engine) -> Bool {
    engine == target && (try? validate()) != nil
  }
}

extension StudioProject {
  /// Applying a library item creates clip-owned descriptors, independent of future library edits.
  public mutating func applyLoRAs(
    _ members: [LoRAMember], to clipID: UUID, groupName: String? = nil,
    mode: LoRAGroupApplicationMode = .add
  ) throws {
    guard let index = clips.firstIndex(where: { $0.id == clipID }) else {
      throw StudioError.invalid("Select a clip first.")
    }
    let clip = clips[index]
    for member in members { try member.validate(for: clip.engine) }
    let existing = Set(
      clip.attachments.filter { $0.role == .lora }.compactMap { attachment in
        assets.first { $0.id == attachment.assetID }.map { LoRAMember(asset: $0).fileKey }
      })
    let incoming = members.map(\.fileKey)
    guard Set(incoming).count == incoming.count,
      mode == .replace || existing.isDisjoint(with: incoming) else {
      throw StudioError.invalid(
        "The selection repeats a LoRA file or overlaps the current stack. Remove duplicates or choose Replace current stack."
      )
    }
    let applicationID = groupName == nil ? nil : UUID()
    if mode == .replace { clips[index].attachments.removeAll { $0.role == .lora } }
    for member in members {
      var asset = member.asset
      asset.id = UUID()
      asset.scope = .clip
      asset.owner = clipID
      assets.append(asset)
      var attachment = Attachment(assetID: asset.id, role: .lora)
      attachment.strength = member.strength
      attachment.enabled = member.enabled
      attachment.loraGroupName = groupName
      attachment.loraGroupID = applicationID
      clips[index].attachments.append(attachment)
    }
  }

  public func loraGroupSnapshot(
    for clipID: UUID, name: String, additionalAssets: [MediaAsset] = []
  ) throws -> LoRAGroup {
    guard let clip = clips.first(where: { $0.id == clipID }) else {
      throw StudioError.invalid("Select a clip first.")
    }
    let sources = assets + additionalAssets
    let members = try clip.attachments.filter { $0.role == .lora }.map { attachment in
      guard let asset = sources.first(where: { $0.id == attachment.assetID }) else {
        throw StudioError.invalid("Relink the missing LoRA before saving this stack.")
      }
      return LoRAMember(asset: asset, strength: attachment.strength, enabled: attachment.enabled)
    }
    let group = LoRAGroup(name: name, engine: clip.engine, members: members)
    try group.validate()
    return group
  }
}
