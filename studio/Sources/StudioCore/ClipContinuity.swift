import CryptoKit
import Foundation

/// Nil on legacy clips means independent generation.
public struct ClipContinuity: Codable, Equatable {
  public var mode: String
  public var sourceClipID: UUID?
  public var saveContext: Bool
  public var boundaryImagePolicy: String
  public init(mode: String = "independent", sourceClipID: UUID? = nil, saveContext: Bool = false,
    boundaryImagePolicy: String = "balanced") {
    self.mode = mode; self.sourceClipID = sourceClipID; self.saveContext = saveContext
    self.boundaryImagePolicy = boundaryImagePolicy
  }
  private enum CodingKeys: String, CodingKey { case mode, sourceClipID, saveContext, boundaryImagePolicy }
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    mode = try values.decodeIfPresent(String.self, forKey: .mode) ?? "independent"
    sourceClipID = try values.decodeIfPresent(UUID.self, forKey: .sourceClipID)
    saveContext = try values.decodeIfPresent(Bool.self, forKey: .saveContext) ?? false
    boundaryImagePolicy = try values.decodeIfPresent(String.self, forKey: .boundaryImagePolicy) ?? "balanced"
  }
  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(mode, forKey: .mode)
    try values.encodeIfPresent(sourceClipID, forKey: .sourceClipID)
    try values.encode(saveContext, forKey: .saveContext)
    // Preserve legacy clip fingerprints when this scene-only setting is unused/defaulted.
    if boundaryImagePolicy != "balanced" {
      try values.encode(boundaryImagePolicy, forKey: .boundaryImagePolicy)
    }
  }
}

public struct ContinuationArtifact: Codable, Equatable {
  public var manifest: String
  public var manifestSHA256: String
  public var payloadSHA256: String
  public init(manifest: String, manifestSHA256: String, payloadSHA256: String) {
    self.manifest = manifest; self.manifestSHA256 = manifestSHA256; self.payloadSHA256 = payloadSHA256
  }
  private enum CodingKeys: String, CodingKey {
    case manifest
    case manifestSHA256 = "manifest_sha256"
    case payloadSHA256 = "payload_sha256"
  }
}

extension Clip {
  public var continuityMode: String { continuity?.mode ?? "independent" }
  public var activeRenderVersion: RenderVersion? { versions.last { $0.path == sourcePath } }
  public var supportsNativeContinuity: Bool { [.h3, .ltx23, .ltx25].contains(engine) }
}

extension StudioProject {
  public func earlierContinuitySources(for clip: Clip) -> [Clip] {
    guard let index = clips.firstIndex(where: { $0.id == clip.id }) else { return [] }
    return Array(clips.prefix(index))
  }

  /// Requiring an earlier clip makes circular or forward dependencies impossible.
  public func continuitySource(for clip: Clip) throws -> Clip? {
    guard clip.supportsNativeContinuity, clip.continuityMode != "independent" else { return nil }
    if clip.continuityMode == "scene" {
      let members = try continuousSceneMembers(for: clip)
      guard let index = members.firstIndex(where: { $0.id == clip.id }), index > 0 else { return nil }
      return members[index - 1]
    }
    guard ["frame", "motion"].contains(clip.continuityMode) else {
      throw StudioError.invalid("Choose a supported connection in Clip Continuity.")
    }
    let earlier = earlierContinuitySources(for: clip)
    if let selected = clip.continuity?.sourceClipID {
      guard let source = earlier.first(where: { $0.id == selected }) else {
        throw StudioError.invalid("Choose a continuity source earlier in the timeline; the selected source was removed or moved.")
      }
      return source
    }
    guard let source = earlier.last else {
      throw StudioError.invalid("Add and render an earlier clip, or choose Independent.")
    }
    return source
  }

  public func shouldSaveContinuityContext(for clip: Clip) -> Bool {
    guard clip.engine == .h3 else { return false }
    if clip.continuity?.saveContext == true { return true }
    return clips.contains { candidate in
      candidate.engine == .h3 && candidate.continuityMode == "motion"
        && (try? continuitySource(for: candidate))?.id == clip.id
    }
  }

  /// Only cheap filesystem metadata is read on the UI thread. Python verifies artifact hashes.
  public func continuityDependencyFingerprint(for clip: Clip) -> String {
    if isContinuousSceneMember(clip) {
      do { return try continuousSceneInputFingerprint(for: clip) }
      catch {
        return SHA256.hash(data: Data(error.localizedDescription.utf8)).map { String(format: "%02x", $0) }.joined()
      }
    }
    guard clip.supportsNativeContinuity else { return "" }
    let saveContext = shouldSaveContinuityContext(for: clip)
    guard clip.continuityMode != "independent" || saveContext else { return "" }
    var parts = ["saveContext:\(saveContext)"]
    do {
      if let source = try continuitySource(for: clip) {
        let version = source.activeRenderVersion
        parts += [source.id.uuidString, source.engine.rawValue, source.sourcePath,
          String(source.sourceIn), String(source.duration), String(source.settings(in: self).fps),
          String(source.generationWidth), String(source.generationHeight), version?.id.uuidString ?? "imported",
          String(describing: version?.usableSourceIn), String(describing: version?.usableDuration)]
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        parts.append((try? encoder.encode(version?.continuationArtifact).base64EncodedString()) ?? "")
        var paths = [source.sourcePath]
        if let manifest = version?.continuationArtifact?.manifest {
          paths += [manifest, URL(fileURLWithPath: manifest).deletingLastPathComponent().appendingPathComponent("latents.safetensors").path]
        }
        for path in paths {
          let attributes = try? FileManager.default.attributesOfItem(atPath: path)
          parts.append(path + "|" + String(describing: attributes?[.modificationDate]) + "|"
            + String(describing: attributes?[.size]) + "|" + String(describing: attributes?[.systemFileNumber]))
        }
      }
    } catch { parts.append(error.localizedDescription) }
    return SHA256.hash(data: Data(parts.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
  }

  public func continuityIssues(for clip: Clip) -> [String] {
    if isContinuousSceneMember(clip) { return continuousSceneIssues(for: clip) }
    guard clip.supportsNativeContinuity, clip.continuityMode != "independent" else { return [] }
    do {
      guard let source = try continuitySource(for: clip) else { return [] }
      if source.sourcePath.isEmpty || !FileManager.default.fileExists(atPath: source.sourcePath) {
        return ["Render and accept the continuity source first, or relink its movie."]
      }
      if !clip.extensionDirection.isEmpty {
        return ["Remove the separate video extension setting before enabling cross-clip continuity."]
      }
      if clip.continuityMode == "motion" {
        if clip.engine == .h3 {
          guard source.engine == .h3, let version = source.activeRenderVersion,
            version.continuationArtifact != nil else {
            return ["Render and accept an H3 source take with Save motion context enabled first."]
          }
          guard let start = version.usableSourceIn, let duration = version.usableDuration,
            abs(source.sourceIn + source.duration - start - duration) <= 0.5 / 24 else {
            return ["H3 motion context must end at the source take’s untrimmed terminal frame. Restore its endpoint or choose Match previous frame."]
          }
        } else if clip.attachments.contains(where: { $0.role != .lora }) {
          return ["LTX motion continuation cannot combine endpoint, audio, or reference attachments. Choose Match previous frame to preserve those controls."]
        }
      }
      return []
    } catch { return [error.localizedDescription] }
  }
}

extension ProjectStorage {
  /// Keep the immutable manifest beside its fixed-name tensor payload without rewriting hashes.
  public static func collectContinuationArtifact(_ artifact: ContinuationArtifact, to directory: URL) throws -> ContinuationArtifact {
    let source = URL(fileURLWithPath: artifact.manifest)
    let payload = source.deletingLastPathComponent().appendingPathComponent("latents.safetensors")
    guard FileManager.default.fileExists(atPath: source.path), FileManager.default.fileExists(atPath: payload.path) else {
      throw StudioError.invalid("Relink the continuation manifest and latents.safetensors before collecting this project.")
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    do {
      try FileManager.default.copyItem(at: source, to: directory.appendingPathComponent("manifest.json"))
      try FileManager.default.copyItem(at: payload, to: directory.appendingPathComponent("latents.safetensors"))
    } catch {
      try? FileManager.default.removeItem(at: directory)
      throw error
    }
    var result = artifact
    result.manifest = directory.appendingPathComponent("manifest.json").path
    return result
  }
}
