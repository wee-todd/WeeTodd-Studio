import Drama3Contracts
import Foundation

/// User-owned samples and versioned profiles. Derived inference caches live elsewhere.
public actor VoiceLibrary {
  public let root: URL
  public init(root: URL) { self.root = root }
  public func save(name: String, samples: [VoiceReference], description: String) async throws
    -> VoiceProfileSnapshot
  {
    guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !samples.isEmpty,
      samples.count <= 64, description.count <= 8000
    else { throw StudioError.invalid("Name the voice and add up to 64 reference samples.") }
    var archived: [VoiceReference] = []
    for sample in samples {
      archived.append(
        try await VoiceSampleArchive.archive(sample, in: root.appendingPathComponent("Samples")))
    }
    let profile = VoiceProfileSnapshot(
      name: name, engine: .fishDrama3, samples: archived, description: description)
    try write(profile)
    return profile
  }
  public func update(_ reference: VoiceProfileReference, name: String, description: String) throws
    -> VoiceProfileSnapshot
  {
    var p = try load(reference)
    p.reference.version += 1
    p.name = name
    p.description = description
    try p.validate()
    try write(p)
    return p
  }
  public func load(_ reference: VoiceProfileReference, requireSamples: Bool = true) throws
    -> VoiceProfileSnapshot
  {
    let url = profileURL(reference)
    guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 1024 * 1024
    else { throw StudioError.invalid("Saved voice metadata exceeds its limit.") }
    let value = try JSONDecoder().decode(VoiceProfileSnapshot.self, from: Data(contentsOf: url))
    try value.validate()
    guard value.reference == reference,
      !requireSamples
        || value.samples.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
      throw StudioError.invalid(
        "Saved voice samples are unavailable. Restore or import its retained samples.")
    }
    if requireSamples {
      for sample in value.samples where !sample.sourceHash.isEmpty {
        guard try Drama3AudioPreparation.fileDigest(sample.path) == sample.sourceHash else {
          throw StudioError.invalid(
            "Retained voice audio changed. Restore its original sample before reusing this saved version."
          )
        }
      }
    }
    return value
  }
  public func list() throws -> [VoiceProfileSnapshot] {
    let directory = root.appendingPathComponent("Profiles")
    guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
    let folders = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
    var values: [VoiceProfileSnapshot] = []
    for folder in folders.prefix(10000) {
      guard let id = UUID(uuidString: folder.lastPathComponent) else { continue }
      let versions =
        (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))
        ?? []
      if let version = versions.compactMap({ Int($0.deletingPathExtension().lastPathComponent) })
        .max(), let p = try? load(.init(id: id, version: version), requireSamples: false)
      {
        values.append(p)
      }
    }
    return values.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  public func adopt(_ profile: VoiceProfileSnapshot) async throws -> VoiceProfileSnapshot {
    try profile.validate()
    if let old = try? load(profile.reference) {
      guard old.fingerprint == profile.fingerprint else {
        throw StudioError.invalid("This voice ID already has different retained samples.")
      }
      return old
    }
    var p = profile
    for i in p.samples.indices {
      p.samples[i] = try await VoiceSampleArchive.archive(
        p.samples[i], in: root.appendingPathComponent("Samples"))
    }
    try write(p)
    return p
  }
  private func profileURL(_ reference: VoiceProfileReference) -> URL {
    root.appendingPathComponent("Profiles/\(reference.id.uuidString)/\(reference.version).json")
  }
  private func write(_ p: VoiceProfileSnapshot) throws {
    let url = profileURL(p.reference)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard !FileManager.default.fileExists(atPath: url.path) else {
      throw StudioError.invalid("A saved voice version cannot be replaced.")
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(p).write(to: url, options: .atomic)
  }
}
