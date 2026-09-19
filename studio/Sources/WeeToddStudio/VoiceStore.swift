import AppKit
import AVFoundation
import StudioCore

@MainActor extension StudioStore {
  var voiceTakes: [MediaAsset] { project.assets.filter { $0.voiceGeneration != nil } }
  var installedVoiceModels: [InstalledVoiceModel] { runtime.voiceModels?.models ?? [] }
  var familyVoiceModels: [InstalledVoiceModel] {
    installedVoiceModels.filter { $0.engine == (project.voiceDraft?.engine ?? .qwen3TTS) }
  }
  var selectedVoiceModel: InstalledVoiceModel? {
    guard let draft = project.voiceDraft else { return nil }
    if let id = draft.modelID { return familyVoiceModels.first { $0.id == id } }
    return runtime.voiceModels?.preferred(for: draft.engine)
  }
  func voiceModelAvailable(_ model: InstalledVoiceModel) -> Bool {
    FileManager.default.fileExists(atPath: URL(fileURLWithPath: model.path).appendingPathComponent("config.json").path)
  }
  func persistVoiceModels(_ settings: VoiceModelSettings) throws {
    var updated = runtime; updated.voiceModels = settings
    try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
    try JSONEncoder().encode(updated).write(to: dataDirectory.appendingPathComponent("runtime.json"), options: .atomic)
    runtime = updated
  }
  func openVoice() {
    if project.voiceDraft == nil {
      var draft = VoiceDraft(); draft.engine = runtime.voiceModels?.lastFamily ?? .qwen3TTS
      draft.modelID = runtime.voiceModels?.preferred(for: draft.engine)?.id
      change { $0.voiceDraft = draft }
    }
    migrateVoiceModelLocation()
    if let model = selectedVoiceModel { adaptVoiceMode(to: model) }
    showVoice = true
  }
  /// Preserve older projects while moving machine-specific configuration into Runtime settings.
  func migrateVoiceModelLocation() {
    guard let draft = project.voiceDraft, !draft.modelPath.isEmpty else { return }
    var settings = runtime.voiceModels ?? VoiceModelSettings()
    let model = settings.models.first { $0.path == draft.modelPath && $0.engine == draft.engine }
      ?? settings.register(InstalledVoiceModel(engine: draft.engine,
        name: URL(fileURLWithPath: draft.modelPath).lastPathComponent, path: draft.modelPath))
    settings.lastFamily = draft.engine; settings.selectedModels[draft.engine.rawValue] = model.id
    do { try persistVoiceModels(settings) }
    catch { self.error = error.localizedDescription; return }
    change { $0.voiceDraft?.modelID = model.id; $0.voiceDraft?.modelPath = "" }
  }
  func selectVoiceFamily(_ engine: VoiceEngine) {
    guard !operationBusy else { return }
    var settings = runtime.voiceModels ?? VoiceModelSettings(); settings.lastFamily = engine
    do { try persistVoiceModels(settings) }
    catch { self.error = error.localizedDescription; return }
    change {
      $0.voiceDraft?.engine = engine; $0.voiceDraft?.modelID = settings.preferred(for: engine)?.id
      $0.voiceDraft?.modelPath = ""; $0.voiceDraft?.precision = "auto"
      if engine == .fishS2Pro { $0.voiceDraft?.language = "auto" }
      if $0.voiceDraft?.dialogue != nil {
        for index in $0.voiceDraft!.dialogue!.speakers.indices {
          let mode = $0.voiceDraft!.dialogue!.speakers[index].referenceMode
          if (engine == .fishS2Pro && mode == .speakerIdentityOnly) || (engine == .qwen3TTS && mode == .synthetic) {
            $0.voiceDraft?.dialogue?.speakers[index].referenceMode = .audioAndTranscript
          }
        }
      }
      if (engine == .fishS2Pro && $0.voiceDraft?.referenceMode == .speakerIdentityOnly)
        || (engine == .qwen3TTS && $0.voiceDraft?.referenceMode == .synthetic) {
        $0.voiceDraft?.referenceMode = .audioAndTranscript
      }
    }
    if let model = selectedVoiceModel { adaptVoiceMode(to: model) }
  }
  func adaptVoiceMode(to model: InstalledVoiceModel) {
    func compatible(_ mode: VoiceReferenceMode) -> VoiceReferenceMode {
      if model.supportsInstructions { return .customVoice }
      if mode == .customVoice || (model.engine == .fishS2Pro && mode == .speakerIdentityOnly)
        || (model.engine == .qwen3TTS && mode == .synthetic) { return .audioAndTranscript }
      return mode
    }
    change {
      if let mode = $0.voiceDraft?.referenceMode { $0.voiceDraft?.referenceMode = compatible(mode) }
      if $0.voiceDraft?.dialogue != nil {
        for index in $0.voiceDraft!.dialogue!.speakers.indices {
          $0.voiceDraft!.dialogue!.speakers[index].referenceMode = compatible($0.voiceDraft!.dialogue!.speakers[index].referenceMode)
        }
      }
    }
  }
  func selectVoiceModel(_ id: UUID?) {
    guard !operationBusy, let id, let model = familyVoiceModels.first(where: { $0.id == id }) else { return }
    runtime.voiceModels?.selectedModels[model.engine.rawValue] = id
    runtime.voiceModels?.lastFamily = model.engine; saveRuntime(reloadProfiles: false)
    change { $0.voiceDraft?.modelID = id; $0.voiceDraft?.modelPath = ""; $0.voiceDraft?.precision = "auto" }
    adaptVoiceMode(to: model)
  }
  func registerVoiceModel(_ result: [String: Any]) throws {
    guard let value = result["model"] as? [String: Any],
      let path = value["path"] as? String, !path.isEmpty,
      let rawEngine = value["engine"] as? String, let engine = VoiceEngine(rawValue: rawEngine),
      let name = value["name"] as? String, !name.isEmpty else {
      throw StudioError.invalid("Model inspection did not return a supported speech model.")
    }
    var settings = runtime.voiceModels ?? VoiceModelSettings()
    settings.register(InstalledVoiceModel(engine: engine, name: name, path: path, kind: value["kind"] as? String))
    try persistVoiceModels(settings)
  }
  func addVoiceModel(at url: URL) async {
    guard !operationBusy else { return }
    do {
      let result = try await bridge.invoke("voice-inspect", runtime: runtime, payload: ["voice": ["model_path": url.path]])
      try registerVoiceModel(result)
      notice = "Speech model added to Runtime settings. It is available in every movie."
    } catch { self.error = error.localizedDescription }
  }
  func removeVoiceModel(_ model: InstalledVoiceModel) {
    guard !operationBusy else { return }
    runtime.voiceModels?.models.removeAll { $0.id == model.id }
    if runtime.voiceModels?.selectedModels[model.engine.rawValue] == model.id {
      runtime.voiceModels?.selectedModels.removeValue(forKey: model.engine.rawValue)
    }
    saveRuntime(reloadProfiles: false)
    notice = "Model removed from Runtime settings. Its files remain on disk."
  }
  func chooseVoiceModel() {
    let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
    panel.title = "Add an installed Fish S2 Pro or Qwen3-TTS model"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await addVoiceModel(at: url) }
  }
  func chooseVoiceReference() {
    let panel = NSOpenPanel(); panel.title = "Choose a voice sample or video soundtrack"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let session = documentSessionID, draftID = project.voiceDraft?.id
    Task {
      do {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard let _ = try await asset.loadTracks(withMediaType: .audio).first, duration.isFinite, duration > 0 else {
          throw StudioError.invalid("Choose a file with an audio track.")
        }
        guard documentSessionID == session, project.voiceDraft?.id == draftID else { return }
        change { $0.voiceDraft?.reference = VoiceReference(path: url.path, duration: min(10, duration)) }
      } catch { self.error = error.localizedDescription }
    }
  }
  func referenceFromAsset(_ asset: MediaAsset) {
    change { $0.voiceDraft?.reference = VoiceReference(assetID: asset.id, path: asset.path, duration: min(10, max(0.1, asset.duration))) }
  }
  func referenceFromClip() {
    guard let clip = selectedClip, !clip.sourcePath.isEmpty else { return }
    change { $0.voiceDraft?.reference = VoiceReference(path: clip.sourcePath, start: clip.sourceIn, duration: min(10, clip.duration)) }
  }
  func downloadVoiceModel(_ modelID: String) {
    guard !operationBusy else { return }
    let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
    panel.title = "Choose a speech model library"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        let result = try await bridge.invoke("voice-download", runtime: runtime, payload: ["model_id": modelID, "directory": url.path])
        guard let path = result["model_path"] as? String else { throw StudioError.invalid("Speech setup did not return a model folder.") }
        await addVoiceModel(at: URL(fileURLWithPath: path))
      } catch { self.error = error.localizedDescription }
    }
  }
  func inspectVoiceModel(_ model: InstalledVoiceModel) async {
    guard !operationBusy else { return }
    do {
      let result = try await bridge.invoke("voice-inspect", runtime: runtime, payload: ["voice": ["engine": model.engine.rawValue, "model_path": model.path]])
      try registerVoiceModel(result)
      notice = result["description"] as? String ?? "Speech model checked."
    } catch { self.error = error.localizedDescription }
  }
  func generateVoice() async {
    guard !operationBusy else { return }
    migrateVoiceModelLocation()
    if let model = selectedVoiceModel { adaptVoiceMode(to: model) }
    guard let draft = project.voiceDraft else { return }
    let session = documentSessionID, projectID = project.id
    let job = dataDirectory.appendingPathComponent("Voice/\(UUID().uuidString)")
    do {
      guard let model = selectedVoiceModel else {
        throw StudioError.invalid("Choose an installed speech model. Add models in Runtime settings.")
      }
      var resolved = draft; resolved.modelPath = model.path; resolved.precision = "auto"
      let request = try resolved.request()
      let result = try await bridge.invoke(draft.usesDialogue == true ? "voice-dialogue" : "voice-generate", runtime: runtime, payload: ["voice": try request.object()], output: job)
      guard let path = result["audio"] as? String, let duration = result["duration"] as? Double,
        let rate = result["sample_rate"] as? Int, let frames = result["frames"] as? Int,
        duration.isFinite, duration > 0, rate == (draft.engine == .fishS2Pro ? 44100 : 24000), frames > 0,
        abs(duration - Double(frames) / Double(rate)) < 1 / Double(rate), !path.isEmpty,
        result["channels"] as? Int == 1, let artifacts = result["artifacts"] as? String else {
        throw StudioError.invalid("Speech did not return a verified audio take.")
      }
      guard project.id == projectID, documentSessionID == session else { notice = "Voice take saved at \(path). The destination movie changed."; return }
      var asset = MediaAsset(name: draft.name, kind: .audio, path: path); asset.duration = duration
      asset.voiceGeneration = VoiceGeneration(draft: draft, request: request, artifacts: artifacts, sampleRate: rate, frames: frames, truncated: result["truncated"] as? Bool ?? false)
      change { $0.assets.append(asset) }
      if project.voiceDraft == draft { selectedVoiceAssetID = asset.id }
      notice = asset.voiceGeneration!.truncated ? "Voice take saved with a token-limit warning. Audition before placing it." : "Voice take saved. Audition it, then add it to the selected clip."
    } catch { self.error = error.localizedDescription }
  }
  func placeVoiceTake(_ asset: MediaAsset) {
    guard let clipID = selectedClipID else { error = "Select the destination video clip first."; return }
    do {
      var updated = project; let region = try updated.placeVoice(asset, clipID: clipID)
      change { $0 = updated }; selectedAudioID = region.id
      notice = "Voice added on its own track. Music remains independently editable."
    } catch { self.error = error.localizedDescription }
  }
  func saveVoicePreset() {
    guard let draft = project.voiceDraft, let reference = draft.reference else { return }
    change { if $0.voicePresets == nil { $0.voicePresets = [] }; $0.voicePresets?.append(VoicePreset(name: draft.name, reference: reference)) }
  }
}
