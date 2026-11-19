import Foundation
import StudioCore

struct ImagePresetUndo {
  var after: GenerationTargetToken
  var draft: DrawThingsImageDraft
  var selection: PresetSelectionState?
}

@MainActor extension StudioStore {
  var imagePresetSelection: PresetSelectionState? {
    guard let key = imageDraft?.storageKey else { return nil }
    return imageWorkspaceLibrary.sessions[key]?.presetSelection
  }
  func setImagePresetSelection(_ selection: PresetSelectionState?) {
    guard let draft = imageDraft else { return }
    objectWillChange.send()
    if imageWorkspaceLibrary.sessions[draft.storageKey] == nil { imageWorkspaceLibrary.record(draft, preview: imagePreviewPath) }
    imageWorkspaceLibrary.sessions[draft.storageKey]?.presetSelection = selection
    persistImageWorkspace()
  }
  func imagePresetTargetToken() throws -> GenerationTargetToken {
    guard let draft = imageDraft else { throw StudioError.invalid("Open an image workspace first.") }
    struct Revision: Encodable { var draft: DrawThingsImageDraft; var selection: PresetSelectionState? }
    return .init(documentSessionID: documentSessionID, workspaceKey: "image:" + draft.storageKey,
      semanticRevision: try GenerationTargetToken.revision(Revision(draft: draft, selection: imagePresetSelection)), leaseID: activeReferenceLease?.id)
  }
  func captureImagePreset() throws -> GenerationPresetSnapshot {
    guard let draft = imageDraft else { throw StudioError.invalid("Open an image workspace first.") }
    return try ImagePresetRecipe.capture(from: draft, resolvedLibrary: generationLibrary)
  }
  func applyImagePreset(_ preset: StudioGenerationPreset, expectedTarget target: GenerationTargetToken, associate: Bool = true) async throws {
    guard try imagePresetTargetToken() == target else { throw StudioError.invalid("The image workspace changed. Select the preset again.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try imagePresetTargetToken() == target, let previous = imageDraft else { throw StudioError.invalid("The image workspace changed while checking models. Its current edits were preserved.") }
    let candidate = try ImagePresetRecipe.candidate(applying: preset.snapshot, to: previous, resolvedLibrary: generationLibrary)
    let baseline = try ImagePresetRecipe.capture(from: candidate, resolvedLibrary: generationLibrary)
    let priorSelection = imagePresetSelection
    if candidate != previous { imageDraft = candidate; imageEstimate = nil }
    if associate { setImagePresetSelection(.init(preset, baseline: baseline)) }
    imagePresetUndo = .init(after: try imagePresetTargetToken(), draft: previous, selection: priorSelection)
  }
  func undoImagePreset() throws {
    guard let undo = imagePresetUndo, try imagePresetTargetToken() == undo.after else { throw StudioError.invalid("The image workspace changed after applying the preset. Its newer edits were preserved.") }
    if imageDraft != undo.draft { imageDraft = undo.draft; imageEstimate = nil }
    setImagePresetSelection(undo.selection); imagePresetUndo = nil
  }
}
