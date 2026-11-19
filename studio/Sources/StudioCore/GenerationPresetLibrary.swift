import Foundation

public struct GenerationPresetLibrary {
  private struct Envelope: Codable { var version = 1; var records: [StudioGenerationPreset] = []; var migrations: Set<String> = [] }
  private var envelope = Envelope()
  private var defaults: [StudioGenerationPreset]
  public init(defaults: [StudioGenerationPreset] = []) { self.defaults = defaults }
  public var presets: [StudioGenerationPreset] {
    let overrides = Set(envelope.records.map(\.id))
    return (defaults.filter { !overrides.contains($0.id) } + envelope.records).sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  public func preset(id: String) -> StudioGenerationPreset? { presets.first { $0.id == id } }
  public mutating func replaceDefaults(_ records: [StudioGenerationPreset]) throws {
    var candidate = self
    let userNames = Set(envelope.records.map { $0.name.lowercased() })
    let userIDs = Set(envelope.records.map(\.id))
    candidate.defaults = records.filter { userIDs.contains($0.id) || !userNames.contains($0.name.lowercased()) }
    try candidate.validate(); self = candidate
  }
  public func migrated(_ key: String) -> Bool { envelope.migrations.contains(key) }
  public mutating func markMigrated(_ key: String) throws {
    guard key.count <= 128, envelope.migrations.count < 512 else { throw StudioError.invalid("Invalid preset migration marker.") }
    envelope.migrations.insert(key)
  }
  @discardableResult public mutating func create(name: String, snapshot: GenerationPresetSnapshot, id: String = UUID().uuidString) throws -> StudioGenerationPreset {
    let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
    try validateName(normalized)
    guard preset(id: id) == nil else { throw StudioError.invalid("A preset with this identity already exists.") }
    let value = StudioGenerationPreset(id: id, name: normalized, snapshot: snapshot)
    var candidate = self; candidate.envelope.records.append(value); try candidate.validate(); self = candidate
    return value
  }
  @discardableResult public mutating func update(id: String, expectedRevision: Int, snapshot: GenerationPresetSnapshot) throws -> StudioGenerationPreset {
    guard var value = preset(id: id), value.revision == expectedRevision else { throw StudioError.invalid("The preset changed. Review its latest values before updating.") }
    guard value.revision < Int.max else { throw StudioError.invalid("Preset revision limit reached.") }
    value.revision += 1; value.snapshot = snapshot
    var candidate = self
    candidate.envelope.records.removeAll { $0.id == id }; candidate.envelope.records.append(value)
    try candidate.validate(); self = candidate; return value
  }
  private func validateName(_ name: String, excluding: String? = nil) throws {
    guard !name.isEmpty, name.count <= 80, !name.unicodeScalars.contains(where: { $0.value < 32 }),
      !presets.contains(where: { $0.id != excluding && $0.name.trimmingCharacters(in: .whitespacesAndNewlines).caseInsensitiveCompare(name) == .orderedSame }) else {
      throw StudioError.invalid("Use a unique preset name from 1 to 80 characters.")
    }
  }
  private func validate() throws {
    guard envelope.version == 1, presets.count <= 512, Set(presets.map(\.id)).count == presets.count,
      envelope.migrations.count <= 512 else { throw StudioError.invalid("Unsupported or oversized preset library.") }
    for record in presets {
      guard GenerationLibraryLimits.identifier(record.id), record.revision > 0 else { throw StudioError.invalid("Invalid preset identity or revision.") }
      try validateName(record.name, excluding: record.id); try record.snapshot.validate()
    }
  }
  public func write(to file: URL) throws {
    try validate()
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(envelope)
    guard data.count <= 8 * 1024 * 1024 else { throw StudioError.invalid("The preset library exceeds 8 MiB.") }
    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: file, options: .atomic)
  }
  public static func read(from file: URL, defaults: [StudioGenerationPreset] = []) throws -> Self {
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
    let data = try handle.read(upToCount: 8 * 1024 * 1024 + 1) ?? Data()
    guard data.count <= 8 * 1024 * 1024 else { throw StudioError.invalid("The preset library exceeds 8 MiB.") }
    var value = Self(defaults: defaults); value.envelope = try JSONDecoder().decode(Envelope.self, from: data)
    try value.validate(); return value
  }
}
