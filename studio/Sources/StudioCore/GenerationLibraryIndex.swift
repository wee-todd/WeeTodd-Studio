import Foundation

/// Additive local bindings, separate from portable presets and from model weights.
public struct GenerationLibraryIndex: Codable, Equatable {
  public var version = 1
  public var rootDirectory: String? = nil
  public private(set) var snapshot = GenerationLibrarySnapshot()
  private var modelAliases: [String: String] = [:]
  private var loraAliases: [String: String] = [:]
  public init() {}

  public mutating func refreshAvailability(models: (GenerationModelEntry) -> GenerationLibraryAvailability,
    loras: (GenerationLoRAEntry) -> GenerationLibraryAvailability) {
    for index in snapshot.modelEntries.indices { snapshot.modelEntries[index].availability = models(snapshot.modelEntries[index]) }
    for index in snapshot.loraEntries.indices { snapshot.loraEntries[index].availability = loras(snapshot.loraEntries[index]) }
  }
  public mutating func refreshLocalModelIdentities() {
    for i in snapshot.modelEntries.indices {
      guard let files = snapshot.modelEntries[i].identityFiles, !files.isEmpty, snapshot.modelEntries[i].availability == .installed else { continue }
      snapshot.modelEntries[i].reference.sourceIdentity = GenerationLibraryFileIdentity.signature(files)
    }
    for i in snapshot.loraEntries.indices {
      guard snapshot.loraEntries[i].reference.sourceIdentity != nil, snapshot.loraEntries[i].availability == .installed,
        case .nativeFile(let path, _, _, _) = snapshot.loraEntries[i].binding else { continue }
      snapshot.loraEntries[i].reference.sourceIdentity = GenerationLibraryFileIdentity.signature([path])
    }
  }

  @discardableResult public mutating func registerModel(_ candidate: GenerationModelEntry, alias: String) throws -> GenerationModelRef {
    try candidate.reference.validate(); try validateAlias(alias)
    if modelAliases[alias] == nil, let existing = snapshot.modelEntries.first(where: { $0.id == candidate.id }),
      existing.reference != candidate.reference { throw StudioError.invalid("This model ID belongs to a different source. Relink its existing entry explicitly.") }
    var updated = self
    var entry = candidate
    if let id = modelAliases[alias] { entry.reference.id = id }
    if let index = updated.snapshot.modelEntries.firstIndex(where: { $0.id == entry.id }) { updated.snapshot.modelEntries[index] = entry }
    else { updated.snapshot.modelEntries.append(entry) }
    updated.modelAliases[alias] = entry.id
    try updated.validate(); self = updated; return entry.reference
  }

  @discardableResult public mutating func registerLoRA(_ candidate: GenerationLoRAEntry, alias: String) throws -> GenerationLoRARef {
    try candidate.reference.validate(); try validateAlias(alias)
    if loraAliases[alias] == nil, let existing = snapshot.loraEntries.first(where: { $0.id == candidate.id }),
      existing.reference != candidate.reference { throw StudioError.invalid("This LoRA ID belongs to a different source. Relink its existing entry explicitly.") }
    var updated = self
    var entry = candidate
    if let id = loraAliases[alias] { entry.reference.id = id }
    if let index = updated.snapshot.loraEntries.firstIndex(where: { $0.id == entry.id }) { updated.snapshot.loraEntries[index] = entry }
    else { updated.snapshot.loraEntries.append(entry) }
    updated.loraAliases[alias] = entry.id
    try updated.validate(); self = updated; return entry.reference
  }

  public func modelReference(alias: String) -> GenerationModelRef? {
    guard let id = modelAliases[alias] else { return nil }; return snapshot.modelEntries.first { $0.id == id }?.reference
  }
  public func loraReference(alias: String) -> GenerationLoRARef? {
    guard let id = loraAliases[alias] else { return nil }; return snapshot.loraEntries.first { $0.id == id }?.reference
  }

  public static func read(from file: URL) throws -> Self {
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
    let data = try handle.read(upToCount: 8 * 1024 * 1024 + 1) ?? Data()
    guard data.count <= 8 * 1024 * 1024 else { throw StudioError.invalid("The model library index exceeds 8 MiB.") }
    let value = try JSONDecoder().decode(Self.self, from: data)
    try value.validate(); return value
  }
  public func write(to file: URL) throws {
    try validate()
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(self)
    guard data.count <= 8 * 1024 * 1024 else { throw StudioError.invalid("The model library index exceeds 8 MiB.") }
    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: file, options: .atomic)
  }
  private func validateAlias(_ alias: String) throws {
    guard !alias.isEmpty, alias.utf8.count <= 8192, !alias.utf8.contains(0) else { throw StudioError.invalid("Invalid model-library source link.") }
  }
  private func validate() throws {
    guard rootDirectory == nil || rootDirectory.map({ $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0) }) == true else { throw StudioError.invalid("Invalid shared model-library folder.") }
    guard version == 1, snapshot.modelEntries.count + snapshot.loraEntries.count <= 4096,
      modelAliases.count + loraAliases.count <= 8192,
      Set(snapshot.modelEntries.map(\.id)).count == snapshot.modelEntries.count,
      Set(snapshot.loraEntries.map(\.id)).count == snapshot.loraEntries.count else {
      throw StudioError.invalid("Invalid or unsupported model-library index.")
    }
    for entry in snapshot.modelEntries {
      try entry.reference.validate()
      guard (entry.identityFiles?.count ?? 0) <= 4096, (entry.identityFiles ?? []).allSatisfy({ $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0) }) else { throw StudioError.invalid("Invalid model component identities.") }
      guard !entry.name.isEmpty, entry.name.utf8.count <= 1024, !entry.workspaces.isEmpty,
        !entry.tasks.isEmpty, entry.tasks.count <= 64 else { throw StudioError.invalid("Invalid model-library entry.") }
    }
    for entry in snapshot.loraEntries {
      try entry.reference.validate()
      guard !entry.name.isEmpty, entry.name.utf8.count <= 1024 else { throw StudioError.invalid("Invalid LoRA-library entry.") }
    }
    for (alias, id) in modelAliases { try validateAlias(alias); guard snapshot.modelEntries.contains(where: { $0.id == id }) else { throw StudioError.invalid("Model-library link has no entry.") } }
    for (alias, id) in loraAliases { try validateAlias(alias); guard snapshot.loraEntries.contains(where: { $0.id == id }) else { throw StudioError.invalid("LoRA-library link has no entry.") } }
  }
}
