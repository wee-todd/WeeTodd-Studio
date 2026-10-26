import Foundation
import StudioCore

@MainActor extension StudioStore {
  func loadLoRAPromptMetadata() {
    let file = dataDirectory.appendingPathComponent("lora-prompt-metadata.json")
    guard FileManager.default.fileExists(atPath: file.path) else { return }
    do { loraPromptLibrary = try JSONDecoder().decode([String: LoRAPromptMetadata].self, from: Data(contentsOf: file)) }
    catch { self.error = "Could not read LoRA triggers and notes: \(error.localizedDescription)" }
  }
  func loRAPromptMetadata(key: String, fallback: LoRAPromptMetadata? = nil) -> LoRAPromptMetadata {
    loraPromptLibrary[key] ?? fallback ?? LoRAPromptMetadata()
  }
  @discardableResult func saveLoRAPromptMetadata(_ metadata: LoRAPromptMetadata, key: String) -> Bool {
    var updated = loraPromptLibrary; updated[key] = metadata
    do {
      let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
      try encoder.encode(updated).write(to: dataDirectory.appendingPathComponent("lora-prompt-metadata.json"), options: .atomic)
      loraPromptLibrary = updated
      return true
    } catch { self.error = "Could not save LoRA triggers and notes: \(error.localizedDescription)"; return false }
  }
  func resolvedLoRAMembers(_ members: [LoRAMember], preservingSnapshots: Bool = false) -> [LoRAMember] {
    members.map { member in
      var value = member
      if !preservingSnapshots || value.asset.loraPrompt == nil {
        value.asset.loraPrompt = loRAPromptMetadata(key: LoRAPromptKey.native(value.asset.path), fallback: value.asset.loraPrompt)
      }
      return value
    }
  }
  func insertLoRAPromptGuidance(_ guidance: [LoRAPromptMetadata], into draft: inout DrawThingsImageDraft) throws {
    let prompt = LoRAPromptInsertion.applying(guidance, to: draft.prompt)
    guard prompt == draft.prompt || !draft.hasManagedCharacterPrompt else {
      throw StudioError.invalid("Character Director owns this prompt. Add the trigger to the character fields and rebuild, or leave this LoRA's trigger list empty.")
    }
    draft.prompt = prompt
  }
  func drawThingsPromptMetadata(profileID: String, modelID: String) -> LoRAPromptMetadata {
    let item = (drawThingsCatalogs[profileID]?["loras"] as? [[String: Any]])?.first { $0["id"] as? String == modelID }
    let prefix = (item?["promptPrefix"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    return loRAPromptMetadata(key: LoRAPromptKey.drawThings(profileID: profileID, modelID: modelID),
      fallback: LoRAPromptMetadata(triggers: prefix.isEmpty ? [] : [prefix]))
  }
  func applyDrawThingsLoRAGroup(_ group: DrawThingsLoRAGroup, mode: LoRAGroupApplicationMode, imageWorkspace: Bool = false) {
    guard var selection = drawThingsLoRASelection(imageWorkspace: imageWorkspace) else { return }
    var resolved = group
    for index in resolved.members.indices where resolved.members[index].promptMetadata == nil {
      resolved.members[index].promptMetadata = drawThingsPromptMetadata(profileID: group.profileID, modelID: resolved.members[index].modelID)
    }
    do {
      try selection.apply(resolved, mode: mode)
      let guidance = resolved.members.filter(\.isEnabled).compactMap(\.promptMetadata)
      if var draft = imageWorkspace ? imageDraft : nil {
        draft.loras = selection.loras
        try insertLoRAPromptGuidance(guidance, into: &draft)
        imageDraft = draft; imageEstimate = nil
      } else if !imageWorkspace {
        editClip { clip in
          clip.drawThings = selection
          clip.prompt = LoRAPromptInsertion.applying(guidance, to: clip.prompt)
        }
      }
    } catch { self.error = error.localizedDescription }
  }
  func setNativeLoRAEnabled(_ attachmentID: UUID, enabled: Bool) {
    guard let clipID = selectedClipID,
      let clipIndex = project.clips.firstIndex(where: { $0.id == clipID }),
      let index = project.clips[clipIndex].attachments.firstIndex(where: { $0.id == attachmentID && $0.role == .lora })
    else { return }
    let attachment = project.clips[clipIndex].attachments[index]
    var updated = project
    if enabled, var asset = allAssets.first(where: { $0.id == attachment.assetID }) {
      if asset.loraPrompt == nil {
        asset.loraPrompt = loRAPromptMetadata(key: LoRAPromptKey.native(asset.path))
        if asset.scope == .clip, asset.owner == clipID,
          let assetIndex = updated.assets.firstIndex(where: { $0.id == asset.id }) {
          updated.assets[assetIndex] = asset
        } else {
          asset.id = UUID(); asset.scope = .clip; asset.owner = clipID
          updated.assets.append(asset)
          updated.clips[clipIndex].attachments[index].assetID = asset.id
        }
      }
      if let metadata = asset.loraPrompt {
        updated.clips[clipIndex].prompt = LoRAPromptInsertion.applying([metadata], to: updated.clips[clipIndex].prompt)
      }
    }
    updated.clips[clipIndex].attachments[index].enabled = enabled
    change { $0 = updated }
  }
  func setDrawThingsLoRAEnabled(_ modelID: String, enabled: Bool, imageWorkspace: Bool = false) {
    guard let selection = drawThingsLoRASelection(imageWorkspace: imageWorkspace) else { return }
    guard let index = selection.loras.firstIndex(where: { $0.modelID == modelID }) else {
      if enabled { addDrawThingsLoRA(profileID: selection.profileID, modelID: modelID, imageWorkspace: imageWorkspace) }
      return
    }
    let metadata = selection.loras[index].promptMetadata
      ?? drawThingsPromptMetadata(profileID: selection.profileID, modelID: modelID)
    if var draft = imageWorkspace ? imageDraft : nil {
      draft.loras[index].enabled = enabled
      do {
        if enabled {
          draft.loras[index].promptMetadata = metadata
          try insertLoRAPromptGuidance([metadata], into: &draft)
        }
        imageDraft = draft; imageEstimate = nil
      } catch { self.error = error.localizedDescription }
    } else if !imageWorkspace {
      editClip { clip in
        clip.drawThings?.loras[index].enabled = enabled
        if enabled {
          clip.drawThings?.loras[index].promptMetadata = metadata
          clip.prompt = LoRAPromptInsertion.applying([metadata], to: clip.prompt)
        }
      }
    }
  }
}
