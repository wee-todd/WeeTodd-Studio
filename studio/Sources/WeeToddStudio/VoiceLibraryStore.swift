import AppKit
import StudioCore

@MainActor extension StudioStore {
  var savedVoiceLibrary: VoiceLibrary {
    VoiceLibrary(root: dataDirectory.appendingPathComponent("Voice Library"))
  }
  func refreshSavedVoices() async {
    do { voiceLibraryProfiles = try await savedVoiceLibrary.list() } catch {
      self.error = error.localizedDescription
    }
  }
  func saveClonedVoice(name: String, description: String) async {
    guard !operationBusy, !voiceLibraryBusy, let draft = project.voiceDraft else { return }
    voiceLibraryBusy = true
    defer { voiceLibraryBusy = false }
    let session = documentSessionID
    let projectID = project.id
    do {
      var samples = draft.dramaReferences ?? []
      if let reference = draft.reference { samples.insert(reference, at: 0) }
      if samples.isEmpty { samples = draft.voiceProfile?.samples ?? [] }
      let profile = try await savedVoiceLibrary.save(
        name: name, samples: samples, description: description)
      await refreshSavedVoices()
      guard project.id == projectID, documentSessionID == session else {
        notice = "Saved voice in the Voice Library. The destination movie changed."
        return
      }
      voiceLibraryBusy = false
      if project.voiceDraft == draft { useSavedVoice(profile, characterID: draft.characterID) }
      notice =
        "Saved \(profile.name) with retained audio samples. It is ready for character assignment."
    } catch { self.error = error.localizedDescription }
  }
  func updateSavedVoice(_ reference: VoiceProfileReference, name: String, description: String) async
  {
    guard !operationBusy else { return }
    voiceLibraryBusy = true
    defer { voiceLibraryBusy = false }
    do {
      _ = try await savedVoiceLibrary.update(reference, name: name, description: description)
      await refreshSavedVoices()
      notice =
        "Saved a new voice version. Existing character assignments retain their reviewed version."
    } catch { self.error = error.localizedDescription }
  }
  func useSavedVoice(_ profile: VoiceProfileSnapshot, characterID: UUID? = nil) {
    guard !operationBusy else { return }
    if project.voiceDraft == nil { openVoice() }
    selectVoiceFamily(.fishDrama3)
    change { p in
      p.voiceDraft?.usesDialogue = false
      p.voiceDraft?.dramaMode = "speak"
      p.voiceDraft?.dramaParentTakeID = nil
      p.voiceDraft?.dramaSourceAudio = nil
      p.voiceDraft?.dramaEditStart = nil
      p.voiceDraft?.dramaEditEnd = nil
      p.voiceDraft?.dramaEditOperation = nil
      p.voiceDraft?.voiceProfile = profile
      p.voiceDraft?.characterID = characterID
      p.voiceDraft?.reference = profile.samples.first
      p.voiceDraft?.dramaReferences = Array(profile.samples.dropFirst())
      p.voiceDraft?.referenceMode = .audioAndTranscript
    }
  }
  func useCharacterVoice(_ id: UUID) {
    guard let character = planning.subjects.first(where: { $0.id == id && $0.kind == .character }),
      let voice = character.voiceBinding
    else {
      error = "Assign a saved voice to this character first."
      return
    }
    useSavedVoice(voice.profile, characterID: id)
  }
  @discardableResult
  func retainAndUseCharacterVoice(_ profile: VoiceProfileSnapshot) async -> Bool {
    guard !operationBusy, !voiceLibraryBusy else { return false }
    voiceLibraryBusy = true
    defer { voiceLibraryBusy = false }
    let session = documentSessionID
    let projectID = project.id
    do {
      let retained = try await savedVoiceLibrary.adopt(profile)
      await refreshSavedVoices()
      guard documentSessionID == session, project.id == projectID else {
        notice = "Retained the character voice in the Voice Library. The destination movie changed."
        return false
      }
      voiceLibraryBusy = false
      useSavedVoice(retained)
      openVoice()
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
}
