import CryptoKit
import Foundation

public struct VoiceProfileReference: Codable, Equatable, Hashable, Sendable {
  public var id: UUID
  public var version: Int
  public init(id: UUID = UUID(), version: Int = 1) {
    self.id = id
    self.version = version
  }
}
public struct VoiceProfileSnapshot: Codable, Equatable {
  public var reference: VoiceProfileReference
  public var name: String
  public var engine: VoiceEngine
  public var samples: [VoiceReference]
  public var description: String
  public init(
    reference: VoiceProfileReference = .init(), name: String, engine: VoiceEngine = .fishDrama3,
    samples: [VoiceReference], description: String = ""
  ) {
    self.reference = reference
    self.name = name
    self.engine = engine
    self.samples = samples
    self.description = description
  }
  public var fingerprint: String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    var portable = self
    for i in portable.samples.indices {
      portable.samples[i].path = ""
      portable.samples[i].assetID = nil
    }
    return SHA256.hash(data: (try? encoder.encode(portable)) ?? Data()).map {
      String(format: "%02x", $0)
    }.joined()
  }
  public func validate() throws {
    guard reference.version > 0, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      samples.count <= 64, !samples.isEmpty
    else { throw StudioError.invalid("Choose a saved voice with retained samples.") }
    for sample in samples { try sample.validate(transcriptRequired: true) }
  }
  public mutating func mapPaths(_ transform: (String) throws -> String) rethrows {
    for i in samples.indices { samples[i].path = try transform(samples[i].path) }
  }
}
public struct CharacterVoiceBinding: Codable, Equatable {
  public var profile: VoiceProfileSnapshot
  public init(profile: VoiceProfileSnapshot) { self.profile = profile }
}
