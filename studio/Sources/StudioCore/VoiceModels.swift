import Foundation

public struct InstalledVoiceModel: Codable, Equatable, Identifiable {
  public var id: UUID
  public var engine: VoiceEngine
  public var name: String
  public var path: String
  public var kind: String?
  public var supportsInstructions: Bool { engine == .qwen3TTS && kind == "custom_voice" }
  public init(id: UUID = UUID(), engine: VoiceEngine, name: String, path: String, kind: String? = nil) {
    self.id = id; self.engine = engine; self.name = name; self.path = path; self.kind = kind
  }
  public var variantName: String {
    let prefix = engine.label + " · "
    return name.hasPrefix(prefix) ? String(name.dropFirst(prefix.count)) : name
  }
}

public struct VoiceModelSettings: Codable, Equatable {
  public var models: [InstalledVoiceModel] = []
  public var lastFamily: VoiceEngine = .qwen3TTS
  public var selectedModels: [String: UUID] = [:]
  public init() {}
  public func preferred(for engine: VoiceEngine) -> InstalledVoiceModel? {
    let family = models.filter { $0.engine == engine }
    return family.first { $0.id == selectedModels[engine.rawValue] } ?? family.first
  }
  @discardableResult public mutating func register(_ value: InstalledVoiceModel) -> InstalledVoiceModel {
    var model = value
    model.path = URL(fileURLWithPath: model.path).standardizedFileURL.path
    if let index = models.firstIndex(where: { $0.path == model.path }) {
      model.id = models[index].id; models[index] = model
    } else { models.append(model) }
    if selectedModels[model.engine.rawValue] == nil { selectedModels[model.engine.rawValue] = model.id }
    return model
  }
}

extension VoiceEngine {
  public var label: String { self == .fishS2Pro ? "Fish S2 Pro" : "Qwen3-TTS" }
}
