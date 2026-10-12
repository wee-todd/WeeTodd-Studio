import AppKit
import Foundation
import StudioCore
import UniformTypeIdentifiers

@MainActor extension StudioStore {
  var installedMusicModels: [InstalledMusicModel] { runtime.musicModels?.models ?? [] }
  var selectedMusicModel: InstalledMusicModel? {
    let id = project.musicDraft?.modelID ?? runtime.musicModels?.selectedModelID
    if let id { return installedMusicModels.first { $0.id == id && $0.kind == .generator } }
    return installedMusicModels.first { $0.kind == .generator }
  }
  func resolvedMusicRequest(_ draft: MusicDraft) throws -> MusicRequest {
    try (runtime.musicModels ?? MusicModelSettings()).request(for: draft)
  }
  func persistMusicModels(_ settings: MusicModelSettings) throws {
    var updated = runtime; updated.musicModels = settings
    try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
    try JSONEncoder().encode(updated).write(to: dataDirectory.appendingPathComponent("runtime.json"), options: .atomic)
    runtime = updated
  }
  func migrateMusicModelLocation() {
    guard let draft = project.musicDraft, !draft.modelPath.isEmpty else { return }
    var settings = runtime.musicModels ?? MusicModelSettings()
    let model = settings.register(InstalledMusicModel(kind: .generator,
      name: "YuE2 · " + URL(fileURLWithPath: draft.modelPath).lastPathComponent,
      path: draft.modelPath, decoderPath: draft.vaePath.isEmpty ? nil : draft.vaePath))
    do {
      try persistMusicModels(settings)
      change { $0.musicDraft?.modelID = model.id; $0.musicDraft?.modelPath = ""; $0.musicDraft?.vaePath = "" }
    } catch { self.error = error.localizedDescription }
  }
  func selectMusicModel(_ id: UUID?) {
    guard let id, installedMusicModels.contains(where: { $0.id == id && $0.kind == .generator }) else { return }
    var settings = runtime.musicModels ?? MusicModelSettings(); settings.selectedModelID = id
    do { try persistMusicModels(settings) }
    catch { self.error = error.localizedDescription; return }
    change { $0.musicDraft?.modelID = id; $0.musicDraft?.modelPath = ""; $0.musicDraft?.vaePath = ""; $0.musicDraft?.precision = "auto" }
    musicModelStatus = nil
  }
  func registerMusicModel(_ result: [String: Any], replacing id: UUID? = nil) throws {
    guard let model = result["model"] as? [String: Any], let raw = model["kind"] as? String,
      let kind = MusicModelKind(rawValue: raw), let path = model["path"] as? String, !path.isEmpty,
      let name = model["name"] as? String else { throw StudioError.invalid("Model inspection returned no supported music model.") }
    var settings = runtime.musicModels ?? MusicModelSettings()
    var installed = InstalledMusicModel(kind: kind, name: name, path: path, decoderPath: model["decoder_path"] as? String)
    if let id, let index = settings.models.firstIndex(where: { $0.id == id && $0.kind == kind }) {
      installed.id = id; installed.name = settings.models[index].name; settings.models[index] = installed
    } else { settings.register(installed) }
    try persistMusicModels(settings)
  }
  func addMusicModel(_ kind: MusicModelKind, separateDecoder: Bool = false, replacing id: UUID? = nil) {
    guard !operationBusy else { return }
    let panel = NSOpenPanel(); panel.title = "Add \(kind.label)"
    panel.canChooseDirectories = true; panel.canChooseFiles = kind != .generator
    guard panel.runModal() == .OK, let url = panel.url else { return }
    var payload: [String: Any] = ["path": url.path, "kind": kind.rawValue]
    if separateDecoder {
      let decoder = NSOpenPanel(); decoder.title = "Choose the separate YuE2 decoder folder"
      decoder.canChooseDirectories = true; decoder.canChooseFiles = false
      guard decoder.runModal() == .OK, let folder = decoder.url else { return }
      payload["decoder_path"] = folder.path
    }
    Task {
      do {
        let result = try await bridge.invoke("music-extension-inspect", runtime: runtime, payload: payload)
        try registerMusicModel(result, replacing: id)
        notice = "Music model registered in Runtime Settings. Its files are shared across movies."
      } catch { self.error = error.localizedDescription }
    }
  }
  func downloadMusicExtension(_ kind: MusicModelKind) {
    guard !operationBusy else { return }
    let panel = NSOpenPanel(); panel.title = "Choose library for \(kind.label) · CC BY-NC 4.0"
    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        let result = try await bridge.invoke("music-extension-download", runtime: runtime,
          payload: ["directory": url.path, "kind": kind.rawValue])
        try registerMusicModel(result)
        notice = "Verified \(kind.label) ready."
      } catch { self.error = error.localizedDescription }
    }
  }
  func removeMusicModel(_ model: InstalledMusicModel) {
    var settings = runtime.musicModels ?? MusicModelSettings()
    settings.models.removeAll { $0.id == model.id }
    if settings.selectedModelID == model.id { settings.selectedModelID = nil }
    do { try persistMusicModels(settings) }
    catch { self.error = error.localizedDescription }
  }
  func chooseMusicReference() {
    guard !operationBusy else { return }
    let panel = NSOpenPanel(); panel.title = "Choose reference music"; panel.allowedContentTypes = [.audio, .movie]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    var reference = MusicReference(path: url.path)
    reference.encoderID = installedMusicModels.first(where: { $0.kind == .sheetsage })?.id
    change { $0.musicDraft?.reference = reference }
  }
  func analyzeMusicReference() async {
    guard !operationBusy, let draft = project.musicDraft, let reference = draft.reference else { return }
    let session = documentSessionID, projectID = project.id
    do {
      let model = installedMusicModels.first { $0.id == reference.encoderID && $0.kind == .sheetsage }
        ?? (reference.encoderID == nil ? installedMusicModels.first { $0.kind == .sheetsage } : nil)
      guard let model else { throw StudioError.invalid("Add or select a SheetSage2 encoder in Runtime Settings.") }
      let job = dataDirectory.appendingPathComponent("MusicReferences/\(UUID().uuidString)")
      let result = try await bridge.invoke("music-reference-analyze", runtime: runtime, payload: [
        "audio_path": reference.path, "source_in": reference.sourceIn, "duration": reference.duration,
        "model_path": model.path, "melody_only": reference.melodyOnly], output: job)
      guard let abc = result["abc"] as? String, !abc.isEmpty,
        let artifacts = result["artifacts"] as? String, let hash = result["source_sha256"] as? String else {
        throw StudioError.invalid("Reference analysis returned no usable score.")
      }
      guard session == documentSessionID, projectID == project.id, project.musicDraft == draft,
        installedMusicModels.contains(model) else {
        notice = "Reference score saved at \(artifacts). The draft or model settings changed; current work was preserved."; return
      }
      let analysis = MusicReferenceAnalysis(abc: abc, artifacts: artifacts, sourceSHA256: hash,
        sourceIn: reference.sourceIn, duration: reference.duration, melodyOnly: reference.melodyOnly,
        modelSHA256: result["model_sha256"] as? String, eventCount: (result["events"] as? [Any])?.count ?? 0)
      change { $0.musicDraft?.reference?.analysis = analysis }
      notice = "Reference analyzed. Review the score, then choose Use score to apply it."
    } catch { self.error = error.localizedDescription }
  }
  func useMusicReferenceScore() {
    guard let reference = project.musicDraft?.reference, let result = reference.analysis,
      reference.sourceIn == result.sourceIn, reference.duration == result.duration,
      reference.melodyOnly == result.melodyOnly else {
      error = "The reference settings changed. Analyze again before using its score."; return
    }
    change { $0.musicDraft?.abc = result.abc; $0.musicDraft?.cot = result.melodyOnly ? "melody" : "full" }
    notice = "Reference score applied. Generate music to create your new arrangement."
  }
}
