import Foundation

public enum MusicModelKind: String, Codable, CaseIterable {
  case generator, instrumental, sheetsage
  public var label: String {
    switch self { case .generator: return "YuE2 music model"
    case .instrumental: return "Instrumental LoRA"
    case .sheetsage: return "SheetSage2 reference encoder" }
  }
}
public struct InstalledMusicModel: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var kind: MusicModelKind
  public var name: String
  public var path: String
  public var decoderPath: String?
  public init(kind: MusicModelKind, name: String, path: String, decoderPath: String? = nil) {
    self.kind = kind; self.name = name; self.path = path; self.decoderPath = decoderPath
  }
}
public struct MusicModelSettings: Codable, Equatable {
  public var models: [InstalledMusicModel] = []
  public var selectedModelID: UUID?
  public init() {}
  @discardableResult public mutating func register(_ model: InstalledMusicModel) -> InstalledMusicModel {
    if let found = models.first(where: { $0.path == model.path && $0.kind == model.kind && $0.decoderPath == model.decoderPath }) { return found }
    models.append(model)
    if model.kind == .generator { selectedModelID = model.id }
    return model
  }
  public func request(for draft: MusicDraft) throws -> MusicRequest {
    var resolved = draft
    if let id = draft.modelID {
      guard let model = models.first(where: { $0.id == id && $0.kind == .generator }) else {
        throw StudioError.invalid("This music model is unavailable. Add or relink it in Runtime Settings.")
      }
      resolved.modelPath = model.path; resolved.vaePath = model.decoderPath ?? ""
    } else if draft.modelPath.isEmpty {
      guard let model = models.first(where: { $0.id == selectedModelID && $0.kind == .generator })
        ?? models.first(where: { $0.kind == .generator }) else {
        throw StudioError.invalid("Add a YuE2 model in Runtime Settings before generating music.")
      }
      resolved.modelPath = model.path; resolved.vaePath = model.decoderPath ?? ""
    }
    var request = try resolved.request()
    if let id = draft.adapterID {
      guard draft.instrumental, let model = models.first(where: { $0.id == id && $0.kind == .instrumental }) else {
        throw StudioError.invalid("Choose an installed instrumental LoRA and enable Instrumental.")
      }
      let strength = draft.adapterStrength ?? 1
      guard strength.isFinite, (0...2).contains(strength) else { throw StudioError.invalid("LoRA strength must be between 0 and 2.") }
      request.arLoraPath = model.path; request.arLoraStrength = strength
    }
    return request
  }
}
public struct MusicSection: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name: String
  public var start: Int?
  public var end: Int?
  public init(name: String, start: Int? = nil, end: Int? = nil) { self.name = name; self.start = start; self.end = end }
  public static let names = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"]
}
public struct MusicArrangement: Codable, Equatable {
  public enum Mode: String, Codable, CaseIterable { case automatic, ordered, timed }
  public var mode: Mode = .automatic
  public var sections: [MusicSection] = []
  public init(mode: Mode = .automatic, sections: [MusicSection] = []) { self.mode = mode; self.sections = sections }
  public func prompt() throws -> String {
    if mode == .automatic { return "[instrumental]" }
    guard !sections.isEmpty, sections.count <= 64 else { throw StudioError.invalid("Add between 1 and 64 arrangement sections.") }
    var previousEnd = 0
    return try sections.map { section in
      guard MusicSection.names.contains(section.name) else { throw StudioError.invalid("Choose a supported section name.") }
      if mode == .ordered { return "[\(section.name)]" }
      guard let start = section.start, let end = section.end, start >= previousEnd, end > start, end <= 983 else {
        throw StudioError.invalid("Give every section a start and end in seconds, in order without overlaps (up to 983 seconds).")
      }
      previousEnd = end
      func timestamp(_ seconds: Int) -> String { String(format: "%d:%02d", seconds / 60, seconds % 60) }
      return "[\(section.name) \(timestamp(start))-\(timestamp(end))]"
    }.joined(separator: "\n")
  }
}
public struct MusicReference: Codable, Equatable {
  public var path: String
  public var sourceIn: Double = 0
  public var duration: Double = 30
  public var encoderID: UUID?
  public var melodyOnly = true
  public var analysis: MusicReferenceAnalysis?
  public init(path: String) { self.path = path }
}
public struct MusicReferenceAnalysis: Codable, Equatable {
  public var abc: String
  public var artifacts: String
  public var sourceSHA256: String
  public var sourceIn: Double
  public var duration: Double
  public var melodyOnly: Bool
  public var modelSHA256: String?
  public var eventCount: Int
  public init(abc: String, artifacts: String, sourceSHA256: String, sourceIn: Double, duration: Double,
              melodyOnly: Bool, modelSHA256: String? = nil, eventCount: Int) {
    self.abc = abc; self.artifacts = artifacts; self.sourceSHA256 = sourceSHA256
    self.sourceIn = sourceIn; self.duration = duration; self.melodyOnly = melodyOnly
    self.modelSHA256 = modelSHA256; self.eventCount = eventCount
  }
}

extension StudioProject {
  public mutating func mapMusicReferencePaths(_ transform: (String) throws -> String) rethrows {
    if let reference = musicDraft?.reference {
      musicDraft?.reference?.path = try transform(reference.path)
      if let analysis = reference.analysis { musicDraft?.reference?.analysis?.artifacts = try transform(analysis.artifacts) }
    }
    for index in assets.indices {
      if let reference = assets[index].musicGeneration?.draft.reference {
        assets[index].musicGeneration?.draft.reference?.path = try transform(reference.path)
        if let analysis = reference.analysis {
          assets[index].musicGeneration?.draft.reference?.analysis?.artifacts = try transform(analysis.artifacts)
        }
      }
    }
  }
}
