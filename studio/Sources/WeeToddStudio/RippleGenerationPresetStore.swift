import Foundation
import StudioCore

@MainActor extension StudioStore {
  var ripplePresetSelection: PresetSelectionState? {
    guard let id = rippleClipID else { return nil }; return project.generationPresetSelections?["ripple:" + id.uuidString]
  }
  func setRipplePresetSelection(_ selection: PresetSelectionState?, clipID: UUID? = nil) {
    guard let clip = project.clips.first(where: { $0.id == (clipID ?? rippleClipID) }) else { return }; let key = "ripple:" + clip.id.uuidString
    change { project in
      if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
      project.generationPresetSelections?[key] = selection
      if project.generationPresetSelections?.isEmpty == true { project.generationPresetSelections = nil }
    }
  }
  func ripplePresetTargetToken(clipID: UUID? = nil) throws -> GenerationTargetToken {
    guard let clip = project.clips.first(where: { $0.id == (clipID ?? rippleClipID) }), let draft = clip.rippleDraft else { throw StudioError.invalid("Open a Ripple draft first.") }
    struct Revision: Encodable { var draft: RippleDraft; var selection: PresetSelectionState? }
    return .init(documentSessionID: documentSessionID, workspaceKey: "ripple:" + project.id.uuidString + ":" + clip.id.uuidString,
      semanticRevision: try GenerationTargetToken.revision(Revision(draft: draft, selection: project.generationPresetSelections?["ripple:" + clip.id.uuidString])), memberIDs: [clip.id])
  }
  func legacyRippleLibraryModel() -> GenerationModelRef? {
    generationLibrary.models(for: RipplePresetRecipe.context).filter { entry in
      guard case .ripple(let profile, let adapter) = entry.binding else { return false }
      return adapter == runtime.rippleAdapterPath && (runtime.rippleProfileID == nil || runtime.rippleProfileID == "" || runtime.rippleProfileID == "auto" || profile == runtime.rippleProfileID)
    }.sorted { a, b in
      guard case .ripple(let first, _) = a.binding, case .ripple(let second, _) = b.binding else { return a.id < b.id }; return first < second
    }.first?.reference
  }
  func captureRipplePreset(clipID: UUID? = nil) throws -> GenerationPresetSnapshot {
    guard let draft = project.clips.first(where: { $0.id == (clipID ?? rippleClipID) })?.rippleDraft else { throw StudioError.invalid("Open a Ripple draft first.") }
    return try RipplePresetRecipe.capture(from: draft, resolvedLibrary: generationLibrary, legacyModel: legacyRippleLibraryModel())
  }
  func rippleRuntime(for draft: RippleDraft) throws -> RuntimeSettings {
    guard let model = draft.libraryModel else { return runtime }
    guard case .ripple(let profile, let adapter) = try RipplePresetRecipe.binding(model, library: generationLibrary) else { throw StudioError.invalid("Relink the Ripple model.") }
    var result = runtime; result.rippleProfileID = profile; result.rippleAdapterPath = adapter; return result
  }
  func applyRipplePreset(_ preset: StudioGenerationPreset, expectedTarget target: GenerationTargetToken, clipID: UUID? = nil, associate: Bool = true) async throws {
    guard try ripplePresetTargetToken(clipID: clipID) == target else { throw StudioError.invalid("The Ripple draft changed. Select the preset again.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try ripplePresetTargetToken(clipID: clipID) == target, let clip = project.clips.first(where: { $0.id == (clipID ?? rippleClipID) }), let draft = clip.rippleDraft else { throw StudioError.invalid("The Ripple target changed while checking its model.") }
    let candidate = try RipplePresetRecipe.candidate(applying: preset.snapshot, to: draft, resolvedLibrary: generationLibrary)
    let baseline = try RipplePresetRecipe.capture(from: candidate, resolvedLibrary: generationLibrary), key = "ripple:" + clip.id.uuidString
    change { project in
      guard let index = project.clips.firstIndex(where: { $0.id == clip.id }) else { return }
      project.clips[index].rippleDraft = candidate
      if associate {
        if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
        project.generationPresetSelections?[key] = .init(preset, baseline: baseline)
      }
    }
  }
}
