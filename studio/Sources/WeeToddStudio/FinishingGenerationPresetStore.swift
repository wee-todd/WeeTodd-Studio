import Foundation
import StudioCore

@MainActor extension StudioStore {
  func finishingBindingKey(_ clip: UUID, stage: String) -> String { "finishing:" + clip.uuidString + ":" + stage }
  func finishingModelBindings(for clips: [Clip]) -> [String: GenerationModelRef] {
    let prefixes = clips.map { "finishing:" + $0.id.uuidString + ":" }
    return (project.generationLibraryBindings ?? [:]).filter { item in prefixes.contains { item.key.hasPrefix($0) } }
  }
  func matchesVideoProcessingModelBindings(_ target: VideoProcessingDestination) -> Bool {
    finishingModelBindings(for: target.clips) == target.modelBindings
  }
  var finishingPresetSelection: PresetSelectionState? {
    guard let id = selectedClipID else { return nil }; return project.generationPresetSelections?["finishing:" + id.uuidString]
  }
  func setFinishingPresetSelection(_ selection: PresetSelectionState?) {
    guard let clip = selectedClip else { return }; let key = "finishing:" + clip.id.uuidString
    change { project in
      if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
      project.generationPresetSelections?[key] = selection
      if project.generationPresetSelections?.isEmpty == true { project.generationPresetSelections = nil }
    }
  }
  func finishingModels(for clip: Clip) throws -> [GenerationModelRef] {
    guard let recipe = clip.videoProcessing else { throw StudioError.invalid("Enable the finishing stages before choosing a finishing preset.") }
    return try recipe.steps.map { stage in
      let tool = stage.isNative ? "pixelCQ" : stage.isH3Regenerate ? "h3Regenerate" : "flashVSR"
      if let pinned = project.generationLibraryBindings?[finishingBindingKey(clip.id, stage: stage.id)] {
        _ = try generationLibrary.resolve(pinned, in: .init(workspace: .finishing, task: tool)); return pinned
      }
      let matches = generationLibrary.models(for: .init(workspace: .finishing, task: tool)).filter { entry in
        guard case .restoration(let kind, let components) = entry.binding, kind == tool else { return false }
        if stage.isNative {
          guard let settings = stage.pixelCQ else { return false }
          return (settings.profileID == "auto" || components["profileID"] == settings.profileID)
            && (settings.pixelAdapterPath == nil || components["pixelAdapterPath"] == settings.pixelAdapterPath)
            && (settings.cqAdapterPath == nil || components["cqAdapterPath"] == settings.cqAdapterPath)
        }
        if stage.isH3Regenerate { return components["baseProfilePath"] == stage.h3Regenerate?.baseProfilePath }
        return components["modelRoot"] == videoUpscaleRuntime.modelRoot
          && (components["variant"].map { $0 == stage.settings.variant.rawValue } ?? true)
          && (components["decoder"].map { $0 == stage.settings.decoder.rawValue } ?? true)
      }
      guard matches.count == 1, let model = matches.first else { throw StudioError.invalid("Choose the installed model components for \(stage.label) in Models & LoRAs.") }
      return model.reference
    }
  }
  func captureFinishingPreset() throws -> GenerationPresetSnapshot {
    guard let clip = selectedClip, let recipe = clip.videoProcessing else { throw StudioError.invalid("Enable finishing stages for the selected clip first.") }
    return try FinishingPresetRecipe.capture(from: recipe, models: finishingModels(for: clip), resolvedLibrary: generationLibrary, sourceFPS: finishingSourceFPS(clip))
  }
  func finishingSourceFPS(_ clip: Clip) -> Double {
    allAssets.first(where: { $0.path == clip.sourcePath && ($0.fps ?? 0) > 0 })?.fps ?? clip.settings(in: project).fps
  }
  func finishingPresetTargetToken() throws -> GenerationTargetToken {
    guard let clip = selectedClip else { throw StudioError.invalid("Select a clip first.") }
    let clips = videoProcessingClips(for: clip)
    struct Revision: Encodable {
      var clips: [Clip]; var sources: [String]; var bindings: [String: GenerationModelRef]
      var flashRuntime: FlashVSRRuntime; var runtime: RuntimeSettings; var selection: PresetSelectionState?
    }
    let revision = Revision(clips: clips, sources: clips.map { $0.sourcePath.isEmpty ? "" : GenerationLibraryFileIdentity.signature([$0.sourcePath]) },
      bindings: finishingModelBindings(for: clips), flashRuntime: videoUpscaleRuntime, runtime: runtime, selection: finishingPresetSelection)
    return .init(documentSessionID: documentSessionID, workspaceKey: "finishing:" + project.id.uuidString + ":" + clip.id.uuidString,
      semanticRevision: try GenerationTargetToken.revision(revision), memberIDs: clips.map(\.id))
  }
  func applyFinishingPreset(_ preset: StudioGenerationPreset, expectedTarget target: GenerationTargetToken, associate: Bool = true) async throws {
    guard try finishingPresetTargetToken() == target else { throw StudioError.invalid("The finishing stages or destination changed. Select the preset again.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try finishingPresetTargetToken() == target, let selected = selectedClip else { throw StudioError.invalid("The finishing destination changed while checking models.") }
    let clips = videoProcessingClips(for: selected)
    var applications: [(Clip, FinishingPresetApplication, [GenerationModelRef], GenerationPresetSnapshot)] = []
    for clip in clips {
      guard let recipe = clip.videoProcessing else { throw StudioError.invalid("Enable matching finishing stages for every scene member first.") }
      let previous = (try? finishingModels(for: clip)) ?? []
      let application = try FinishingPresetRecipe.candidate(applying: preset.snapshot, to: recipe, resolvedLibrary: generationLibrary, sourceFPS: finishingSourceFPS(clip))
      try validateFinishingBindings(application.models)
      let baseline = try FinishingPresetRecipe.capture(from: application.recipe, models: application.models, resolvedLibrary: generationLibrary, sourceFPS: finishingSourceFPS(clip))
      applications.append((clip, application, previous, baseline))
    }
    if clips.count > 1, let first = applications.first?.1.recipe, !applications.allSatisfy({ $0.1.recipe == first }) {
      throw StudioError.invalid("Continuous scene members need the same finishing recipe. Their current sizes and stage wiring were preserved.")
    }
    change { project in
      for (clip, application, previous, baseline) in applications {
        guard let index = project.clips.firstIndex(where: { $0.id == clip.id }) else { continue }
        project.clips[index].videoProcessing = application.recipe
        if previous != application.models {
          if project.generationLibraryBindings == nil { project.generationLibraryBindings = [:] }
          for (stage, model) in zip(application.recipe.steps, application.models) {
            project.generationLibraryBindings?[finishingBindingKey(clip.id, stage: stage.id)] = model
          }
        }
        if associate {
          if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
          project.generationPresetSelections?["finishing:" + clip.id.uuidString] = .init(preset, baseline: baseline)
        }
      }
    }
  }
  func validateFinishingBindings(_ models: [GenerationModelRef]) throws {
    let components = models.compactMap { model -> (String, [String: String])? in
      guard let entry = generationLibrary.modelEntries.first(where: { $0.reference == model }), case .restoration(let tool, let fields) = entry.binding else { return nil }
      return (tool, fields)
    }
    let roots = Set(components.filter { $0.0 == "flashVSR" }.compactMap { $0.1["modelRoot"] })
    guard roots.count <= 1 else { throw StudioError.invalid("Use one shared FlashVSR+ component library for this finishing chain.") }
    let pixel = components.filter { $0.0 == "pixelCQ" }.map { [$0.1["profileID"], $0.1["pixelAdapterPath"], $0.1["cqAdapterPath"]] }
    guard pixel.dropFirst().allSatisfy({ $0 == pixel.first }) else { throw StudioError.invalid("This renderer uses shared Pixel Spatial + CQ components across the chain. Choose matching stage models.") }
  }
  func finishingFlashRuntime(for clip: Clip) throws -> FlashVSRRuntime {
    var result = videoUpscaleRuntime
    guard !finishingModelBindings(for: [clip]).isEmpty, let recipe = clip.videoProcessing else { return result }
    let models = try finishingModels(for: clip); try validateFinishingBindings(models)
    for (stage, model) in zip(recipe.steps, models) where stage.isFlashVSR {
      let fields = try RestorationPresetRecipe.binding(model, tool: .flashVSR, workspace: .finishing, library: generationLibrary)
      if let root = fields["modelRoot"] { result.modelRoot = root }
    }
    return result
  }
}
