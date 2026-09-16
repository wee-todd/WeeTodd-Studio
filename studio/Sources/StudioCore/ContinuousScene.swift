import CryptoKit
import Foundation

/// One editable shot range in a scene's single decoded movie.
public struct ContinuousSceneMember: Codable, Equatable {
  public var clipID: UUID
  public var sourceIn: Double
  public var duration: Double

  public init(clipID: UUID, sourceIn: Double, duration: Double) {
    self.clipID = clipID; self.sourceIn = sourceIn; self.duration = duration
  }

  private enum CodingKeys: String, CodingKey {
    case clipID = "clip_id"
    case sourceIn = "source_in"
    case duration
  }
}

extension StudioProject {
  /// Includes the independent first shot when the next shot connects to it.
  public func isContinuousSceneMember(_ clip: Clip) -> Bool {
    guard let index = clips.firstIndex(where: { $0.id == clip.id }) else {
      return clip.continuityMode == "scene"
    }
    return clips[index].continuityMode == "scene"
      || (index + 1 < clips.count && clips[index + 1].continuityMode == "scene")
  }

  /// Disconnects one shot without changing its media or the remaining scene's instructions.
  /// Uses structural links so a saved scene with incompatible models can also be repaired.
  @discardableResult
  public mutating func separateContinuousSceneMember(clipID: UUID) -> Bool {
    guard let index = clips.firstIndex(where: { $0.id == clipID }),
      isContinuousSceneMember(clips[index]) else { return false }
    var first = index
    while first > 0 && clips[first].continuityMode == "scene" { first -= 1 }
    let leader = clips[first]
    if clips[index].continuityMode == "scene" {
      clips[index].continuity?.mode = "independent"
      clips[index].continuity?.sourceClipID = nil
    }
    let following = index + 1
    if following < clips.count && clips[following].continuityMode == "scene" {
      clips[following].continuity?.mode = "independent"
      clips[following].continuity?.sourceClipID = nil
      // The right-hand group gets a new leader; retain its effective shared settings.
      if following + 1 < clips.count && clips[following + 1].continuityMode == "scene" {
        clips[following].soundscape = leader.soundscape
        clips[following].music = leader.music
        clips[following].continuity?.boundaryImagePolicy = leader.continuity?.boundaryImagePolicy ?? "balanced"
      }
    }
    return true
  }

  /// Returns the maximal contiguous scene, or an empty list for an ordinary clip.
  /// A scene never depends on an already accepted movie from its predecessor.
  public func continuousSceneMembers(for clip: Clip) throws -> [Clip] {
    guard let index = clips.firstIndex(where: { $0.id == clip.id }) else {
      throw StudioError.invalid("The selected scene shot was removed from this movie.")
    }
    guard isContinuousSceneMember(clip) else { return [] }
    var first = index
    while clips[first].continuityMode == "scene" {
      guard first > 0 else {
        throw StudioError.invalid("The first scene shot must use Independent continuity.")
      }
      first -= 1
    }
    var last = index
    while last + 1 < clips.count && clips[last + 1].continuityMode == "scene" { last += 1 }
    let members = Array(clips[first...last])
    guard members.count >= 2, members.count <= 6 else {
      throw StudioError.invalid("A continuous scene must contain two to six shots.")
    }
    guard members.allSatisfy({ $0.engine == .ltx25 }) else {
      throw StudioError.invalid("Every continuous scene shot must use local LTX 2.5.")
    }
    guard members[0].continuityMode == "independent" else {
      throw StudioError.invalid("Set the first scene shot to Independent before connecting the following shots.")
    }
    let memberIDs = Set(members.map(\.id))
    guard memberIDs.count == members.count,
      clips.filter({ memberIDs.contains($0.id) }).count == members.count else {
      throw StudioError.invalid("Scene shots must have unique IDs; a duplicate shot ID was found.")
    }
    for offset in 1..<members.count {
      if let sourceID = members[offset].continuity?.sourceClipID,
        sourceID != members[offset - 1].id {
        throw StudioError.invalid("A continuous scene shot must connect to its immediately preceding shot. A source was removed or moved.")
      }
    }
    guard members.allSatisfy({ $0.duration.isFinite && $0.duration > 0 }),
      members.reduce(0, { $0 + $1.duration }) <= 30 + 0.000001 else {
      throw StudioError.invalid("Continuous scene shots need positive durations totaling at most 30 seconds.")
    }
    return members
  }

  /// UI identity uses cheap file metadata. Native preflight verifies actual component and input hashes.
  /// The caller supplies its complete asset library and resolved runtime identity when available.
  public func continuousSceneInputFingerprint(
    for clip: Clip, assets: [MediaAsset]? = nil, runtimeIdentity: String = ""
  ) throws -> String {
    let members = try continuousSceneMembers(for: clip)
    guard !members.isEmpty else { return "" }
    let availableAssets = assets ?? self.assets
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    var parts = ["continuous-scene-v1", runtimeIdentity]
    for member in members {
      parts.append(member.generationFingerprint)
      parts.append(try encoder.encode(member.settings(in: self)).base64EncodedString())
      parts.append(GenerationSelection.assetFingerprint(for: member, assets: availableAssets))
      let referencedIDs = Set(member.attachments.map(\.assetID))
      for asset in availableAssets.filter({ referencedIDs.contains($0.id) })
        .sorted(by: { $0.id.uuidString < $1.id.uuidString }) {
        let attributes = try? FileManager.default.attributesOfItem(atPath: asset.path)
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970
        parts.append(asset.path + "|" + String(describing: modified) + "|"
          + String(describing: attributes?[.size]) + "|" + String(describing: attributes?[.systemFileNumber]))
      }
    }
    return SHA256.hash(data: try encoder.encode(parts)).map { String(format: "%02x", $0) }.joined()
  }

  public func continuousSceneIssues(for clip: Clip) -> [String] {
    do {
      let members = try continuousSceneMembers(for: clip)
      guard !members.isEmpty else { return [] }
      if members.contains(where: { !["balanced", "strict"].contains($0.continuity?.boundaryImagePolicy ?? "balanced") }) {
        return ["Choose Automatic or Strict boundary image guidance for this continuous scene."]
      }
      if members.contains(where: { !$0.extensionDirection.isEmpty }) {
        return ["Remove separate video extension settings before generating a continuous scene."]
      }
      if members.contains(where: { member in
        member.attachments.contains { ![.first, .last, .keyframe, .lora].contains($0.role) }
      }) {
        return ["Continuous scenes support first, last and keyframe images plus ordinary LoRAs. Remove reference, audio-driver and control inputs."]
      }
      return []
    } catch { return [error.localizedDescription] }
  }

  /// Validates a complete result before changing the value. The caller checks the captured
  /// document session and input fingerprint before invoking this method for a late render.
  public mutating func acceptContinuousScene(
    versions: [UUID: RenderVersion], members: [ContinuousSceneMember]
  ) throws {
    let indices = try continuousSceneIndices(for: members)
    guard Set(versions.keys) == Set(members.map(\.clipID)),
      let firstVersion = versions[members[0].clipID], !firstVersion.path.isEmpty else {
      throw StudioError.invalid("Accept every scene shot together with its complete rendered take.")
    }
    let suppliedTakeIDs = Set(versions.values.compactMap(\.sceneTakeID))
    guard suppliedTakeIDs.count <= 1 else {
      throw StudioError.invalid("These scene shot versions belong to different takes.")
    }
    let takeID = suppliedTakeIDs.first ?? UUID()
    guard !clips.contains(where: { $0.versions.contains(where: { $0.sceneTakeID == takeID }) }) else {
      throw StudioError.invalid("This scene take was already accepted. Select it from Versions to restore it.")
    }
    let frameRates = Set(versions.values.compactMap(\.sceneFrameRate))
    guard frameRates.count <= 1,
      frameRates.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 120 }) else {
      throw StudioError.invalid("The scene take has inconsistent or invalid frame rates.")
    }
    var updated = clips
    for (offset, index) in indices.enumerated() {
      let member = members[offset]
      guard var version = versions[member.clipID], version.path == firstVersion.path,
        version.sceneMembers == nil || version.sceneMembers == members,
        !updated[index].versions.contains(where: { $0.id == version.id }) else {
        throw StudioError.invalid("The scene result is incomplete, inconsistent, or already accepted.")
      }
      // Imported and legacy accepted media may have no version entry yet.
      if !updated[index].sourcePath.isEmpty,
        !updated[index].versions.contains(where: { $0.path == updated[index].sourcePath }) {
        updated[index].versions.append(RenderVersion(
          path: updated[index].sourcePath, seed: updated[index].seed, prompt: updated[index].prompt,
          recipePath: "", usableSourceIn: updated[index].sourceIn, usableDuration: updated[index].duration))
      }
      version.sceneMembers = members; version.sceneTakeID = takeID
      version.sceneFrameRate = frameRates.first
      version.usableSourceIn = member.sourceIn; version.usableDuration = member.duration
      updated[index].versions.append(version)
      Self.applySceneRange(member, version: version, to: &updated[index])
    }
    clips = updated
  }

  /// Restores a historical take as a group. It cannot silently revive removed or reordered shots.
  public mutating func activateContinuousSceneVersion(selectedClipID: UUID, version: RenderVersion) throws {
    guard let selected = clips.first(where: { $0.id == selectedClipID }),
      let stored = selected.versions.first(where: { $0.id == version.id }), stored == version,
      let members = stored.sceneMembers, let takeID = stored.sceneTakeID,
      members.contains(where: { $0.clipID == selectedClipID }) else {
      throw StudioError.invalid("Choose a complete scene take belonging to the selected shot.")
    }
    let indices = try continuousSceneIndices(for: members)
    var updated = clips
    for (offset, index) in indices.enumerated() {
      let matches = updated[index].versions.filter { $0.sceneTakeID == takeID }
      guard matches.count == 1, let memberVersion = matches.first,
        !memberVersion.path.isEmpty, memberVersion.path == stored.path,
        memberVersion.sceneMembers == members,
        memberVersion.sceneFrameRate == stored.sceneFrameRate,
        memberVersion.usableSourceIn == members[offset].sourceIn,
        memberVersion.usableDuration == members[offset].duration else {
        throw StudioError.invalid("This scene take is missing a shot version or contains inconsistent source ranges.")
      }
      Self.applySceneRange(members[offset], version: memberVersion, to: &updated[index])
    }
    clips = updated
  }

  private func continuousSceneIndices(for members: [ContinuousSceneMember]) throws -> [Int] {
    guard let first = members.first,
      let clip = clips.first(where: { $0.id == first.clipID }),
      try continuousSceneMembers(for: clip).map(\.id) == members.map(\.clipID) else {
      throw StudioError.invalid("The scene membership changed. Restore the complete shot order before selecting this take.")
    }
    var end = 0.0
    var indices: [Int] = []
    for member in members {
      guard member.sourceIn.isFinite, member.duration.isFinite,
        member.sourceIn >= 0, member.duration > 0, abs(member.sourceIn - end) < 0.000001,
        let index = clips.firstIndex(where: { $0.id == member.clipID }) else {
        throw StudioError.invalid("The scene take has missing shots, gaps, overlaps, or invalid source ranges.")
      }
      end = member.sourceIn + member.duration
      indices.append(index)
    }
    guard end.isFinite, end <= 30 + 0.000001 else {
      throw StudioError.invalid("A continuous scene take must be at most 30 seconds long.")
    }
    return indices
  }

  private static func applySceneRange(_ member: ContinuousSceneMember, version: RenderVersion, to clip: inout Clip) {
    clip.sourcePath = version.path
    clip.sourceIn = member.sourceIn
    clip.duration = member.duration
    clip.renderedSignature = ""
    clip.validatedSignature = ""
    clip.motionResult = nil
  }
}
