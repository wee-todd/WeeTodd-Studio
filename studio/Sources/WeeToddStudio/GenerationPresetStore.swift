import Foundation
import StudioCore

enum GenerationPresetTarget: Hashable {
  case image, clip(UUID), ripple(UUID), restoration(NativeUpscalerTool), finishing(UUID)
}
struct GenerationPresetActions {
  var hasSelection, canSaveNew, canUpdate, isModified: Bool
  var changeCount: Int
  var issue: String?
}
struct GenerationPresetReviewToken: Identifiable {
  let id = UUID()
  var target: GenerationPresetTarget
  var targetToken: GenerationTargetToken
  var presetID: String
  var presetRevision: Int
  var name: String
  var before, after: GenerationPresetSnapshot
  var changes: [GenerationPresetChange] { GenerationPresetDiff.changes(from: before, to: after) }
}

@MainActor extension StudioStore {
  func ensurePresetTarget(_ target: GenerationPresetTarget) throws {
    let valid: Bool
    switch target {
    case .image: valid = imageDraft != nil
    case .clip(let id), .finishing(let id): valid = selectedClipID == id && project.clips.contains { $0.id == id }
    case .ripple(let id): valid = (selectedClipID == id || rippleClipID == id) && project.clips.contains { $0.id == id && $0.rippleDraft != nil }
    case .restoration(let tool): valid = videoUpscaleTool == tool
    }
    guard valid else { throw StudioError.invalid("The active destination changed. Open its preset controls again.") }
  }
  func presetSelection(for target: GenerationPresetTarget) -> PresetSelectionState? {
    switch target {
    case .image: return imagePresetSelection
    case .clip(let id): return project.generationPresetSelections?["clip:" + id.uuidString]
    case .ripple(let id): return project.generationPresetSelections?["ripple:" + id.uuidString]
    case .restoration(let tool): return restorationPresetSelections[tool.rawValue]
    case .finishing(let id): return project.generationPresetSelections?["finishing:" + id.uuidString]
    }
  }
  func setPresetSelection(_ selection: PresetSelectionState?, for target: GenerationPresetTarget) throws {
    try ensurePresetTarget(target)
    switch target {
    case .image: setImagePresetSelection(selection)
    case .clip: setClipPresetSelection(selection)
    case .ripple(let id): setRipplePresetSelection(selection, clipID: id)
    case .restoration: setRestorationPresetSelection(selection)
    case .finishing: setFinishingPresetSelection(selection)
    }
  }
  func clearPresetSelection(for target: GenerationPresetTarget) throws { try setPresetSelection(nil, for: target) }
  func presetTargetToken(for target: GenerationPresetTarget) throws -> GenerationTargetToken {
    try ensurePresetTarget(target)
    switch target {
    case .image: return try imagePresetTargetToken()
    case .clip: return try clipPresetTargetToken()
    case .ripple(let id): return try ripplePresetTargetToken(clipID: id)
    case .restoration: return try restorationPresetTargetToken()
    case .finishing: return try finishingPresetTargetToken()
    }
  }
  func capturePreset(for target: GenerationPresetTarget) throws -> GenerationPresetSnapshot {
    try ensurePresetTarget(target)
    switch target {
    case .image: return try captureImagePreset()
    case .clip: return try captureClipPreset()
    case .ripple(let id): return try captureRipplePreset(clipID: id)
    case .restoration: return try captureRestorationPreset()
    case .finishing: return try captureFinishingPreset()
    }
  }
  func presetActions(for target: GenerationPresetTarget) -> GenerationPresetActions {
    let selected = presetSelection(for: target)
    do {
      let current = try capturePreset(for: target)
      let count = selected.map { GenerationPresetDiff.changes(from: $0.baseline, to: current).count } ?? 0
      let saved = selected.flatMap { generationPresets.preset(id: $0.presetID) }
      let canUpdate = saved.map { count > 0 && !GenerationPresetDiff.changes(from: $0.snapshot, to: current).isEmpty } ?? false
      return .init(hasSelection: selected != nil, canSaveNew: true, canUpdate: canUpdate, isModified: count > 0, changeCount: count,
        issue: selected != nil && saved == nil ? "The selected preset is missing. Save as new or choose No preset." : nil)
    } catch { return .init(hasSelection: selected != nil, canSaveNew: false, canUpdate: false, isModified: false, changeCount: 0, issue: error.localizedDescription) }
  }
  func applyPreset(_ preset: StudioGenerationPreset, for target: GenerationPresetTarget, expectedTarget token: GenerationTargetToken) async throws {
    try ensurePresetTarget(target)
    switch target {
    case .image: try await applyImagePreset(preset, expectedTarget: token)
    case .clip: try await applyClipPreset(preset, expectedTarget: token)
    case .ripple(let id): try await applyRipplePreset(preset, expectedTarget: token, clipID: id)
    case .restoration: try await applyRestorationPreset(preset, expectedTarget: token)
    case .finishing: try await applyFinishingPreset(preset, expectedTarget: token)
    }
  }
  @discardableResult func saveNewPreset(name: String, for target: GenerationPresetTarget, expectedTarget token: GenerationTargetToken) async throws -> StudioGenerationPreset {
    try Task.checkCancellation()
    guard try presetTargetToken(for: target) == token else { throw StudioError.invalid("The draft changed. Save its current values from the preset controls.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try presetTargetToken(for: target) == token else { throw StudioError.invalid("The target or installed model changed while checking the preset.") }
    let snapshot = try capturePreset(for: target)
    var candidate = generationPresets
    let record = try candidate.create(name: name, snapshot: snapshot)
    try persistGenerationPresets(candidate)
    try setPresetSelection(.init(record, baseline: snapshot), for: target)
    return record
  }
  func beginPresetUpdate(for target: GenerationPresetTarget) throws -> GenerationPresetReviewToken {
    guard let selected = presetSelection(for: target), let saved = generationPresets.preset(id: selected.presetID) else { throw StudioError.invalid("Choose a saved preset or Save as new preset.") }
    let token = try presetTargetToken(for: target), current = try capturePreset(for: target)
    guard !GenerationPresetDiff.changes(from: saved.snapshot, to: current).isEmpty else { throw StudioError.invalid("No changes to save.") }
    return .init(target: target, targetToken: token, presetID: saved.id, presetRevision: saved.revision, name: saved.name, before: saved.snapshot, after: current)
  }
  func confirmPresetUpdate(_ review: GenerationPresetReviewToken) async throws {
    try Task.checkCancellation()
    guard try presetTargetToken(for: review.target) == review.targetToken else { throw StudioError.invalid("The draft changed after review. Review its current values before updating.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try presetTargetToken(for: review.target) == review.targetToken,
      presetSelection(for: review.target)?.presetID == review.presetID,
      let saved = generationPresets.preset(id: review.presetID), saved.revision == review.presetRevision, saved.snapshot == review.before,
      try capturePreset(for: review.target) == review.after, !review.changes.isEmpty else {
      throw StudioError.invalid("The preset, target, model or draft changed after review. Review the latest values before updating.")
    }
    var candidate = generationPresets
    let record = try candidate.update(id: saved.id, expectedRevision: saved.revision, snapshot: review.after)
    try persistGenerationPresets(candidate)
    try setPresetSelection(.init(record, baseline: review.after), for: review.target)
  }
}
