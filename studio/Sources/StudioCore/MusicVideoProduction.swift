import CryptoKit
import Foundation

/// Links a saved movie to an immutable production snapshot and its resumable outputs.
public struct MusicVideoProduction: Codable, Equatable {
  public var jobDirectory: String
  public var inputFingerprint: String
  public var executionFingerprint: String?
  public var applied = false
  public init(jobDirectory: String, inputFingerprint: String, executionFingerprint: String? = nil) {
    self.jobDirectory = jobDirectory; self.inputFingerprint = inputFingerprint
    self.executionFingerprint = executionFingerprint
  }
}

public struct MusicVideoProductionStatus: Decodable {
  public struct Unit: Decodable, Identifiable {
    public var id: String
    public var name: String
    public var clipIDs: [String]
    public var status: String
    public var attempts: Int
    public var error: String?
  }
  public var jobDirectory: String
  public var projectID: UUID
  public var status: String
  public var units: [Unit]
  public var outputPath: String?
  public var resolvedProjectPath: String?
  public var error: String?
}

extension StudioProject {
  public func productionInputFingerprint() throws -> String {
    var snapshot = self; snapshot.production = nil
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    return SHA256.hash(data: try encoder.encode(snapshot)).map { String(format: "%02x", $0) }.joined()
  }

  /// Apply only the generated take fields; the song, plan, objects and edit remain owned by Studio.
  public mutating func applyProduction(_ resolved: StudioProject, expectedFingerprint: String) throws {
    guard try productionInputFingerprint() == expectedFingerprint,
      resolved.id == id, resolved.clips.map(\.id) == clips.map(\.id),
      resolved.audio == audio, resolved.audioTracks == audioTracks,
      resolved.settings == settings, production?.applied != true else {
      throw StudioError.invalid("The movie changed since production started. The output is saved; create a new production for this edit.")
    }
    for (original, result) in zip(clips, resolved.clips) {
      guard abs(original.duration - result.duration) < 0.000001,
        result.sourceIn.isFinite, result.sourceIn >= 0, !result.sourcePath.isEmpty,
        result.versions.starts(with: original.versions) else {
        throw StudioError.invalid("Production takes do not match the reviewed shot timing and version history.")
      }
    }
    for index in clips.indices {
      clips[index].sourcePath = resolved.clips[index].sourcePath
      clips[index].sourceIn = resolved.clips[index].sourceIn
      clips[index].versions = resolved.clips[index].versions
      clips[index].renderedSignature = ""
      clips[index].validatedSignature = ""
    }
    production?.applied = true
  }
}
