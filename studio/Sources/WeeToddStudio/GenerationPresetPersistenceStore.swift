import Foundation
import StudioCore

@MainActor extension StudioStore {
  var generationPresetsFile: URL { dataDirectory.appendingPathComponent("generation-presets.json") }
  func loadGenerationPresets() {
    guard FileManager.default.fileExists(atPath: generationPresetsFile.path) else { return }
    do {
      generationPresets = try .read(from: generationPresetsFile)
      generationPresetStorageIdentity = GenerationLibraryFileIdentity.signature([generationPresetsFile.path])
    }
    catch {
      do { try FileManager.default.moveItem(at: generationPresetsFile, to: dataDirectory.appendingPathComponent("generation-presets-unreadable-\(UUID().uuidString).json")) }
      catch { generationPresetsWritable = false }
      generationPresetError = "The preset library could not be read. Its original file was preserved."
    }
  }
  func persistGenerationPresets(_ candidate: GenerationPresetLibrary) throws {
    guard generationPresetsWritable else { throw StudioError.invalid("The original preset library could not be preserved. Resolve its file permissions before saving.") }
    let identity = FileManager.default.fileExists(atPath: generationPresetsFile.path)
      ? GenerationLibraryFileIdentity.signature([generationPresetsFile.path]) : nil
    guard identity == generationPresetStorageIdentity else { throw StudioError.invalid("The preset library changed on disk. Reload it and review the latest values before saving.") }
    try candidate.write(to: generationPresetsFile)
    generationPresetStorageIdentity = GenerationLibraryFileIdentity.signature([generationPresetsFile.path])
    generationPresets = candidate; generationPresetError = nil
  }
  func migrateLegacyRestorationPresets() throws {
    guard !videoUpscalePresets.isEmpty else { return }
    var candidate = generationPresets, changed = false
    for legacy in videoUpscalePresets {
      let marker = "flash-v1:" + legacy.id.uuidString
      let existing = candidate.preset(id: legacy.id.uuidString)
      let unresolved = existing?.snapshot.model?.id.hasPrefix("catalog.flash.") == true
      if candidate.migrated(marker), !unresolved { continue }
      try legacy.settings.validateExecution()
      if existing == nil || unresolved {
        var state = restorationPresetState; state.tool = .flashVSR; state.flash = legacy.settings
        let matches = generationLibrary.modelEntries.filter { entry in
          guard entry.availability == .installed, case .restoration("flashVSR", let components) = entry.binding else { return false }
          return components["variant"] == legacy.settings.variant.rawValue && components["decoder"] == legacy.settings.decoder.rawValue
        }
        let configured = restorationModel(for: state).flatMap { ref in matches.first { $0.reference == ref } }
        guard let model = (configured ?? (matches.count == 1 ? matches.first : nil))?.reference else { continue }
        let snapshot = GenerationPresetSnapshot(context: .init(workspace: .restoration, task: "flashVSR"), model: model, adapters: [],
          fields: try GenerationPresetFields.capture(legacy.settings, keys: RestorationPresetRecipe.flashKeys))
        if let existing { _ = try candidate.update(id: existing.id, expectedRevision: existing.revision, snapshot: snapshot) }
        else { try candidate.create(name: legacy.name, snapshot: snapshot, id: legacy.id.uuidString) }
      }
      try candidate.markMigrated(marker); changed = true
    }
    guard changed else { return }
    try persistGenerationPresets(candidate)
  }
  func migrateLegacyLTXPresets() throws {
    var candidate = generationPresets, changed = false
    for config in savedLTX25Configs {
      let marker = "ltx-config:" + config.id
      if candidate.migrated(marker) { continue }
      let task = config.selection.ltx25AdapterBinding?.task ?? config.selection.task
      var project = StudioProject(), clip = Clip(engine: .ltx25)
      clip.profileID = config.componentBindingID ?? "auto"; clip.generationSelection = config.selection; clip.generationSelection?.task = task
      project.clips = [clip]
      guard var snapshot = try? ClipPresetRecipe.capture(from: clip, project: project, resolvedLibrary: generationLibrary) else { continue }
      // Old configs did not own the project's seed, negative guidance or ordinary LoRA stack.
      snapshot.fields.removeValue(forKey: "seed"); snapshot.fields.removeValue(forKey: "negativePrompt"); snapshot.adapters = nil
      if candidate.preset(id: config.id) == nil {
        var name = String(config.name.prefix(80))
        if candidate.presets.contains(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame }) { name = String(config.name.prefix(65)) + " · " + config.id.suffix(8) }
        try candidate.create(name: name, snapshot: snapshot, id: config.id)
      }
      try candidate.markMigrated(marker); changed = true
    }
    if changed { try persistGenerationPresets(candidate) }
  }
}
