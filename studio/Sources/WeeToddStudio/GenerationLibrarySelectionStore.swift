import Foundation
import StudioCore

@MainActor extension StudioStore {
  func generationLibraryContext(for target: GenerationPresetTarget, stageIndex: Int? = nil) throws -> GenerationLibraryContext {
    try ensurePresetTarget(target)
    switch target {
    case .image: return ImagePresetRecipe.context(imageDraft!)
    case .clip: return ClipPresetRecipe.context(selectedClip!, project: project)
    case .ripple: return RipplePresetRecipe.context
    case .restoration(let tool): return .init(workspace: .restoration, task: tool.rawValue)
    case .finishing:
      guard let steps = selectedClip?.videoProcessing?.steps, let index = stageIndex, steps.indices.contains(index) else {
        throw StudioError.invalid("Choose a finishing stage first.")
      }
      return .init(workspace: .finishing, task: steps[index].isNative ? "pixelCQ" : steps[index].isH3Regenerate ? "h3Regenerate" : "flashVSR")
    }
  }
  func currentGenerationLibraryModel(for target: GenerationPresetTarget, stageIndex: Int? = nil) -> GenerationModelEntry? {
    if let ref = try? capturePreset(for: target).model {
      return generationLibrary.modelEntries.first { $0.reference == ref }
    }
    switch target {
    case .image:
      guard let draft = imageDraft else { return nil }
      return generationLibrary.modelEntries.first { entry in
        guard entry.reference.backend == draft.executionProvider.rawValue else { return false }
        switch entry.binding {
        case .nativeMOSS(let transformer, _, _): return transformer == draft.mossImage?.transformerDirectory
          && entry.reference.variant == draft.mossImage?.variant && entry.reference.precision == (draft.mossImage?.quantizationBits.map { "q\($0)" } ?? "bf16")
        case .nativeH3Image(let id, _): return id == draft.modelID
        case .nativeQwen(let manifest): return manifest == draft.nativeImage?.manifestPath
        case .drawThings(let connection, let id): return connection == draft.profileID && id == draft.modelID
        default: return false
        }
      }
    case .clip(let id):
      guard let clip = project.clips.first(where: { $0.id == id }) else { return nil }
      return generationLibrary.modelEntries.first { entry in
        switch entry.binding {
        case .nativeVideo(let profile): return entry.reference.backend == clip.engine.rawValue && profile == clip.profileID
        case .drawThings(let connection, let model): return clip.engine == .drawThings && connection == clip.drawThings?.profileID && model == clip.drawThings?.modelID
        default: return false
        }
      }
    case .restoration:
      let ref = restorationPresetState.model; return generationLibrary.modelEntries.first { $0.reference == ref }
    case .ripple(let id):
      let ref = project.clips.first { $0.id == id }?.rippleDraft?.libraryModel ?? legacyRippleLibraryModel()
      return generationLibrary.modelEntries.first { $0.reference == ref }
    case .finishing:
      guard var clip = selectedClip, let recipe = clip.videoProcessing, let index = stageIndex, recipe.steps.indices.contains(index) else { return nil }
      clip.videoProcessing?.steps = [recipe.steps[index]]
      guard let ref = try? finishingModels(for: clip).first else { return nil }
      return generationLibrary.modelEntries.first { $0.reference == ref }
    }
  }
  func useGenerationLibraryModel(_ model: GenerationModelRef, for target: GenerationPresetTarget,
    expectedTarget token: GenerationTargetToken, stageIndex: Int? = nil) async throws {
    try Task.checkCancellation()
    guard try presetTargetToken(for: target) == token else { throw StudioError.invalid("The destination changed. Choose the model again.") }
    if case .finishing = target {
      guard let index = stageIndex else { throw StudioError.invalid("Choose a finishing stage first.") }
      try await useFinishingStageModel(model, stageIndex: index, expectedTarget: token); return
    }
    let context = try generationLibraryContext(for: target)
    let binding = try generationLibrary.resolve(model, in: context)
    var snapshot = GenerationPresetSnapshot(context: presetContext(for: target) ?? .init(workspace: context.workspace, task: context.task), model: model)
    if case .restoration(let tool, let fields) = binding {
      if case .clip = target { snapshot.fields["method"] = .string(tool) }
      if tool == "flashVSR" {
        if let value = fields["variant"] { snapshot.fields["variant"] = .string(value) }
        if let value = fields["decoder"] { snapshot.fields["decoder"] = .string(value) }
      }
      if tool == "reCaVSR", let precision = fields["precision"] { snapshot.fields["precision"] = .string(precision) }
      if tool == "relayVSR", let scale = fields["scale"].flatMap(Int64.init) { snapshot.fields["scale"] = .integer(scale) }
    }
    try await applyLibrarySnapshot(snapshot, for: target, expectedTarget: token)
  }
  func applyLibrarySnapshot(_ snapshot: GenerationPresetSnapshot, for target: GenerationPresetTarget, expectedTarget token: GenerationTargetToken) async throws {
    let recipe = StudioGenerationPreset(name: "Working settings", snapshot: snapshot)
    switch target {
    case .image: try await applyImagePreset(recipe, expectedTarget: token, associate: false)
    case .clip: try await applyClipPreset(recipe, expectedTarget: token, associate: false)
    case .ripple(let id): try await applyRipplePreset(recipe, expectedTarget: token, clipID: id, associate: false)
    case .restoration: try await applyRestorationPreset(recipe, expectedTarget: token, associate: false)
    case .finishing: try await applyFinishingPreset(recipe, expectedTarget: token, associate: false)
    }
  }
  func addGenerationLibraryLoRA(_ reference: GenerationLoRARef, for target: GenerationPresetTarget, expectedTarget token: GenerationTargetToken) async throws {
    try Task.checkCancellation()
    guard try presetTargetToken(for: target) == token else { throw StudioError.invalid("The destination changed. Add the LoRA again.") }
    var snapshot = try capturePreset(for: target)
    guard let model = snapshot.model else { throw StudioError.invalid("Choose a generation model first.") }
    let context = try generationLibraryContext(for: target)
    _ = try generationLibrary.resolve(reference, for: model, in: context)
    guard !(snapshot.adapters ?? []).contains(where: { $0.reference.id == reference.id }) else { return }
    if case .image = target, var preview = imageDraft, preview.executionProvider == .nativeMOSS {
      let member = try generationLibrary.nativePresetMembers([.init(reference: reference)], model: model, context: context)
      try preview.applyMOSSLoRAs(member)
      snapshot = try ImagePresetRecipe.capture(from: preview, resolvedLibrary: generationLibrary)
    } else {
      if snapshot.adapters == nil { snapshot.adapters = [] }
      snapshot.adapters?.append(.init(reference: reference))
    }
    try await applyLibrarySnapshot(snapshot, for: target, expectedTarget: token)
  }
  func useFinishingStageModel(_ model: GenerationModelRef, stageIndex: Int, expectedTarget token: GenerationTargetToken) async throws {
    let target = GenerationPresetTarget.finishing(selectedClipID ?? UUID())
    guard try presetTargetToken(for: target) == token else { throw StudioError.invalid("The finishing target changed.") }
    await refreshGenerationLibrary(force: true); try Task.checkCancellation()
    guard try presetTargetToken(for: target) == token, let selected = selectedClip else { throw StudioError.invalid("The finishing target changed while checking the model.") }
    let context = try generationLibraryContext(for: target, stageIndex: stageIndex)
    guard case .restoration(_, let fields) = try generationLibrary.resolve(model, in: context) else { throw StudioError.invalid("Choose compatible finishing components.") }
    var changes: [(UUID, VideoProcessingRecipe, String)] = []
    for clip in videoProcessingClips(for: selected) {
      guard var recipe = clip.videoProcessing, recipe.steps.indices.contains(stageIndex) else { throw StudioError.invalid("Use matching finishing stages across the scene.") }
      let original = recipe.steps[stageIndex]
      var stage = original; stage.input = "source"
      if stage.isFlashVSR {
        if let variant = fields["variant"].flatMap(FlashVSRSettings.Variant.init(rawValue:)) { stage.settings.variant = variant }
        if let decoder = fields["decoder"].flatMap(FlashVSRSettings.Decoder.init(rawValue:)) { stage.settings.decoder = decoder }
      }
      let one = try VideoProcessingRecipe(steps: [stage])
      let snapshot = try FinishingPresetRecipe.capture(from: one, models: [model], resolvedLibrary: generationLibrary, sourceFPS: finishingSourceFPS(clip))
      let application = try FinishingPresetRecipe.candidate(applying: snapshot, to: one, resolvedLibrary: generationLibrary, sourceFPS: finishingSourceFPS(clip))
      stage = application.recipe.steps[0]; stage.input = original.input; recipe.steps[stageIndex] = stage
      if stage.isFlashVSR && stage.settings.variant == .base { recipe.format = "weetodd-video-processing-v2" }
      try recipe.validate(); changes.append((clip.id, recipe, stage.id))
    }
    change { project in
      if project.generationLibraryBindings == nil { project.generationLibraryBindings = [:] }
      for (id, recipe, stageID) in changes {
        guard let index = project.clips.firstIndex(where: { $0.id == id }) else { continue }
        project.clips[index].videoProcessing = recipe
        project.generationLibraryBindings?[finishingBindingKey(id, stage: stageID)] = model
      }
    }
  }
}
