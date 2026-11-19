import AVFoundation
import Foundation
import InferenceContracts
import StudioCore

struct RestorationPresetUndo {
  var after: GenerationTargetToken
  var state: RestorationPresetState
  var modelRoot: String
  var selectedModel: GenerationModelRef?
  var selection: PresetSelectionState?
}

@MainActor extension StudioStore {
  var canUndoRestorationPreset: Bool {
    guard let undo = restorationPresetUndo else { return false }
    return (try? restorationPresetTargetToken()) == undo.after
  }
  func undoRestorationPreset() throws {
    guard let undo = restorationPresetUndo, try restorationPresetTargetToken() == undo.after else { throw StudioError.invalid("The restoration tool, source or settings changed after applying the preset. Its newer edits were preserved.") }
    let changed = restorationPresetState != undo.state
    videoUpscaleSettings = undo.state.flash; h3RegenerateSettings = undo.state.h3
    relayVSRScale = undo.state.relayScale; reCaVSRSettings = undo.state.reCa
    videoUpscaleRuntime.modelRoot = undo.modelRoot
    restorationModelSelections[undo.state.tool.rawValue] = undo.selectedModel
    restorationPresetSelections[undo.state.tool.rawValue] = undo.selection
    if changed { videoUpscalePlan = nil; h3RegeneratePlan = nil }
    restorationPresetUndo = nil; saveVideoUpscalePreferences()
  }
  var restorationPresetSelection: PresetSelectionState? { restorationPresetSelections[videoUpscaleTool.rawValue] }
  func setRestorationPresetSelection(_ selection: PresetSelectionState?) {
    restorationPresetSelections[videoUpscaleTool.rawValue] = selection; saveVideoUpscalePreferences()
  }
  var restorationPresetState: RestorationPresetState {
    var state = RestorationPresetState(tool: RestorationPresetTool(rawValue: videoUpscaleTool.rawValue)!)
    state.flash = videoUpscaleSettings; state.h3 = h3RegenerateSettings; state.relayScale = relayVSRScale; state.reCa = reCaVSRSettings
    if let fps = allAssets.first(where: { $0.id == videoUpscaleSourceAssetID || $0.path == videoUpscaleSourcePath })?.fps, fps > 0 { state.sourceFPS = fps }
    state.model = restorationModel(for: state); return state
  }
  func restorationModel(for state: RestorationPresetState) -> GenerationModelRef? {
    let selected = restorationModelSelections[state.tool.rawValue]
    let selectedEntry = generationLibrary.modelEntries.first { $0.reference == selected }
    var configured: [String: String] = [:]
    if case .restoration(_, let components) = selectedEntry?.binding { configured = components }
    let matching = generationLibrary.models(for: .init(workspace: .restoration, task: state.tool.rawValue)).first { entry in
      guard case .restoration(let kind, let components) = entry.binding, kind == state.tool.rawValue else { return false }
      switch state.tool {
      case .flashVSR:
        return components["modelRoot"] == videoUpscaleRuntime.modelRoot
          && (components["variant"].map { $0 == state.flash.variant.rawValue } ?? true)
          && (components["decoder"].map { $0 == state.flash.decoder.rawValue } ?? true)
      case .h3Regenerate: return components["baseProfilePath"] == state.h3.baseProfilePath
        && (components["modelLibraryRoot"].map { $0 == state.h3.modelLibraryRoot } ?? true)
      case .relayVSR:
        let current = configured["modelDirectory"] ?? runtime.relayVSRModelDirectory ?? ""
        func family(_ path: String) -> String {
          let url = URL(fileURLWithPath: path); return ["x2", "x4"].contains(url.lastPathComponent) ? url.deletingLastPathComponent().path : url.path
        }
        return components["modelDirectory"].map { family($0) == family(current) } == true
          && (components["scale"].map { $0 == String(state.relayScale.rawValue) } ?? true)
      case .reCaVSR: return components["modelDirectory"] == (configured["modelDirectory"] ?? runtime.reCaVSRModelDirectory ?? "")
        && (components["precision"].map { $0 == state.reCa.precision.rawValue } ?? true)
      }
    }
    return matching?.reference ?? selected
  }
  func captureRestorationPreset() throws -> GenerationPresetSnapshot {
    try RestorationPresetRecipe.capture(from: restorationPresetState, resolvedLibrary: generationLibrary)
  }
  func restorationPresetTargetToken() throws -> GenerationTargetToken {
    struct Revision: Encodable {
      var tool, source, sourceIdentity: String
      var flash: FlashVSRSettings; var h3: H3RegenerateSettings
      var scale: Int; var reCa: ReCaVSRSettings; var selection: PresetSelectionState?; var model: GenerationModelRef?
    }
    let value = Revision(tool: videoUpscaleTool.rawValue, source: videoUpscaleSourcePath,
      sourceIdentity: videoUpscaleSourcePath.isEmpty ? "" : GenerationLibraryFileIdentity.signature([videoUpscaleSourcePath]),
      flash: videoUpscaleSettings, h3: h3RegenerateSettings, scale: relayVSRScale.rawValue, reCa: reCaVSRSettings,
      selection: restorationPresetSelection, model: restorationPresetState.model)
    return .init(documentSessionID: documentSessionID, workspaceKey: "restoration:" + videoUpscaleTool.rawValue,
      semanticRevision: try GenerationTargetToken.revision(value))
  }
  func applyRestorationPreset(_ preset: StudioGenerationPreset, expectedTarget target: GenerationTargetToken, associate: Bool = true) async throws {
    guard try restorationPresetTargetToken() == target else { throw StudioError.invalid("The restoration tool, source or settings changed. Select the preset again.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try restorationPresetTargetToken() == target else { throw StudioError.invalid("The restoration target changed while checking models.") }
    var state = restorationPresetState
    let previous = state, previousRoot = videoUpscaleRuntime.modelRoot
    let previousModel = restorationModelSelections[state.tool.rawValue], previousSelection = restorationPresetSelection
    if state.tool == .h3Regenerate, state.sourceFPS == nil, !videoUpscaleSourcePath.isEmpty {
      let asset = AVURLAsset(url: URL(fileURLWithPath: videoUpscaleSourcePath))
      guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw StudioError.invalid("Relink a movie with a supported source frame rate.") }
      state.sourceFPS = Double(try await track.load(.nominalFrameRate))
      guard try restorationPresetTargetToken() == target else { throw StudioError.invalid("The restoration source changed during inspection.") }
    }
    let candidate = try RestorationPresetRecipe.candidate(applying: preset.snapshot, to: state, resolvedLibrary: generationLibrary)
    let baseline = try RestorationPresetRecipe.capture(from: candidate, resolvedLibrary: generationLibrary)
    let components = try RestorationPresetRecipe.binding(candidate.model!, tool: candidate.tool, library: generationLibrary)
    let changed = state != candidate
    videoUpscaleSettings = candidate.flash; h3RegenerateSettings = candidate.h3; relayVSRScale = candidate.relayScale; reCaVSRSettings = candidate.reCa
    if candidate.tool == .flashVSR, let root = components["modelRoot"] { videoUpscaleRuntime.modelRoot = root }
    restorationModelSelections[candidate.tool.rawValue] = candidate.model
    if associate { restorationPresetSelections[candidate.tool.rawValue] = .init(preset, baseline: baseline) }
    if changed { videoUpscalePlan = nil; h3RegeneratePlan = nil }
    saveVideoUpscalePreferences()
    restorationPresetUndo = .init(after: try restorationPresetTargetToken(), state: previous, modelRoot: previousRoot, selectedModel: previousModel, selection: previousSelection)
  }
  func standaloneRestorationRuntime(for tool: RestorationPresetTool) throws -> RuntimeSettings {
    var result = runtime
    guard restorationModelSelections[tool.rawValue] != nil else { return result }
    guard let model = restorationModel(for: restorationPresetState) else { throw StudioError.invalid("Relink the matching restoration model in Models & LoRAs.") }
    let components = try RestorationPresetRecipe.binding(model, tool: tool, library: generationLibrary)
    if tool == .relayVSR { result.relayVSRModelDirectory = components["modelDirectory"] }
    if tool == .reCaVSR { result.reCaVSRModelDirectory = components["modelDirectory"] }
    return result
  }
}
