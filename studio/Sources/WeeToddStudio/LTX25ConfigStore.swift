import Foundation
import StudioCore

@MainActor extension StudioStore {
  var ltx25Configs: [LTX25GenerationConfig] { LTX25GenerationConfig.builtIns + savedLTX25Configs }
  var selectedLTX25Config: LTX25GenerationConfig? {
    ltx25Configs.first { $0.id == selectedClip?.generationSelection?.ltx25ConfigID }
  }
  var ltx25ConfigIsModified: Bool {
    guard let config = selectedLTX25Config else { return selectedClip?.generationSelection?.isModified == true }
    return !config.matches(selectedClip?.generationSelection)
  }

  func loadLTX25Configs() {
    let file = dataDirectory.appendingPathComponent("ltx25-configs.json")
    guard FileManager.default.fileExists(atPath: file.path) else { return }
    do {
      let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
      guard (attributes[.size] as? NSNumber)?.intValue ?? Int.max <= 1024 * 1024 else {
        throw StudioError.invalid("The LTX config library exceeds 1 MiB.")
      }
      let configs = try JSONDecoder().decode([LTX25GenerationConfig].self, from: Data(contentsOf: file))
      guard configs.count <= 1000, Set(configs.map(\.id)).count == configs.count,
        configs.allSatisfy({ !$0.isBuiltIn }) else {
        throw StudioError.invalid("The LTX config library has invalid or duplicate identities.")
      }
      try configs.forEach { try $0.validate() }
      savedLTX25Configs = configs
      ltx25ConfigLibraryError = nil
    } catch {
      ltx25ConfigLibraryError = "Could not read the LTX 2.5 config library: " + error.localizedDescription
      self.error = ltx25ConfigLibraryError
    }
  }

  func applyLTX25Config(_ config: LTX25GenerationConfig) {
    guard var clip = selectedClip else { return }
    do {
      try clip.applyLTX25Config(config, in: project)
      editClip { $0 = clip }
    } catch { self.error = error.localizedDescription }
  }

  func resetLTX25Config() {
    let fallback = selectedClip?.generationSelection?.ltx25Guidance == nil
      ? LTX25GenerationConfig.distilled : selectedClip?.generationSelection?.ltx25Guidance?.mode == .guidedHQ
        ? .devHQ : .dev
    applyLTX25Config(selectedLTX25Config ?? fallback)
  }

  @discardableResult func saveLTX25Config(name: String) async -> Bool {
    guard let clip = selectedClip, clip.engine == .ltx25 else { return false }
    let session = documentSessionID
    do {
      var request = try payload()
      request["runtime"] = ["profilesDirectory": runtime.profilesDirectory]
      let config = try await Task.detached(priority: .userInitiated) {
        try NativeLTXPreparation.configSnapshot(name: name, clip: clip, request: request)
      }.value
      guard documentSessionID == session, selectedClip == clip else {
        throw StudioError.invalid("The selected clip changed while reading its settings. Save the current config again.")
      }
      guard config.supports(clip, in: project) else {
        throw StudioError.invalid("This clip's inputs or continuity do not support the current LTX config.")
      }
      try writeLTX25Configs(savedLTX25Configs + [config])
      applyLTX25Config(config)
      notice = "Saved LTX 2.5 config: " + config.name
      return true
    } catch { self.error = error.localizedDescription; return false }
  }

  func deleteLTX25Config(_ id: String) {
    do { try writeLTX25Configs(savedLTX25Configs.filter { $0.id != id }) }
    catch { self.error = error.localizedDescription }
  }

  private func writeLTX25Configs(_ configs: [LTX25GenerationConfig]) throws {
    if let ltx25ConfigLibraryError { throw StudioError.invalid(ltx25ConfigLibraryError) }
    guard configs.count <= 1000 else { throw StudioError.invalid("Limit the LTX config library to 1,000 entries.") }
    let data = try JSONEncoder().encode(configs)
    guard data.count <= 1024 * 1024 else { throw StudioError.invalid("The LTX config library exceeds 1 MiB.") }
    try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
    try data.write(to: dataDirectory.appendingPathComponent("ltx25-configs.json"), options: .atomic)
    savedLTX25Configs = configs
  }
}
