import Foundation

public struct FinishingPresetApplication {
  public var recipe: VideoProcessingRecipe
  public var models: [GenerationModelRef]
}
public enum FinishingPresetRecipe {
  private static let pixelKeys: Set<String> = ["pixelStrength", "cqStrength", "refinementStrength"]
  public static func capture(from recipe: VideoProcessingRecipe, models: [GenerationModelRef],
    resolvedLibrary library: GenerationLibrarySnapshot, sourceFPS: Double? = nil) throws -> GenerationPresetSnapshot {
    try recipe.validate()
    guard models.count == recipe.steps.count else { throw StudioError.invalid("Choose installed models for every finishing stage.") }
    var stages: [String: GenerationPresetValue] = [:]
    for (index, step) in recipe.steps.enumerated() {
      let snapshot: GenerationPresetSnapshot
      if step.isNative {
        guard let settings = step.pixelCQ else { throw StudioError.invalid("Missing Pixel Spatial + CQ settings.") }
        try settings.validate(); _ = try pixelBinding(models[index], library: library)
        snapshot = .init(context: .init(workspace: .finishing, task: "pixelCQ"), model: models[index], adapters: [],
          fields: try GenerationPresetFields.capture(settings, keys: pixelKeys))
      } else {
        var state = RestorationPresetState(tool: step.isH3Regenerate ? .h3Regenerate : .flashVSR)
        state.model = models[index]; state.sourceFPS = sourceFPS
        if step.isH3Regenerate { guard let settings = step.h3Regenerate else { throw StudioError.invalid("Missing H3 finishing settings.") }; state.h3 = settings }
        else { state.flash = step.settings }
        snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library, workspace: .finishing)
      }
      stages[String(index)] = try .encoded(snapshot)
    }
    let snapshot = GenerationPresetSnapshot(context: .init(workspace: .finishing, task: "finishing"), adapters: [], fields: ["stages": .object(stages)])
    try snapshot.validate(); return snapshot
  }
  public static func candidate(applying snapshot: GenerationPresetSnapshot, to recipe: VideoProcessingRecipe,
    resolvedLibrary library: GenerationLibrarySnapshot, sourceFPS: Double? = nil) throws -> FinishingPresetApplication {
    try snapshot.validate()
    guard snapshot.context == .init(workspace: .finishing, task: "finishing"), snapshot.model == nil, snapshot.adapters?.isEmpty ?? true,
      Set(snapshot.fields.keys) == ["stages"], case .object(let stages) = snapshot.fields["stages"],
      Set(stages.keys) == Set(recipe.steps.indices.map(String.init)) else {
      throw StudioError.invalid("This preset requires a different finishing stage layout. Current stages and their wiring were preserved.")
    }
    var candidate = recipe, models: [GenerationModelRef] = []
    for index in recipe.steps.indices {
      let stage = try stages[String(index)]!.decoded(GenerationPresetSnapshot.self), original = recipe.steps[index]
      try stage.validate()
      guard let model = stage.model else { throw StudioError.invalid("Choose a model for each finishing stage.") }
      models.append(model)
      if original.isNative {
        guard stage.context == .init(workspace: .finishing, task: "pixelCQ"), stage.adapters?.isEmpty ?? true,
          let settings = original.pixelCQ else { throw StudioError.invalid("This preset cannot replace the current finishing stage type.") }
        let binding = try pixelBinding(model, library: library)
        var result = try GenerationPresetFields.applying(stage.fields, to: settings, allowed: pixelKeys)
        if let profile = binding["profileID"] { result.profileID = profile }
        if let adapter = binding["pixelAdapterPath"] { result.pixelAdapterPath = adapter }
        if let adapter = binding["cqAdapterPath"] { result.cqAdapterPath = adapter }
        try result.validate(); candidate.steps[index].pixelCQ = result
      } else {
        var state = RestorationPresetState(tool: original.isH3Regenerate ? .h3Regenerate : .flashVSR)
        state.sourceFPS = sourceFPS
        if original.isH3Regenerate { guard let settings = original.h3Regenerate else { throw StudioError.invalid("Missing H3 finishing settings.") }; state.h3 = settings }
        else { state.flash = original.settings }
        let applied = try RestorationPresetRecipe.candidate(applying: stage, to: state, resolvedLibrary: library, workspace: .finishing)
        if original.isH3Regenerate { candidate.steps[index].h3Regenerate = applied.h3 }
        else { candidate.steps[index].settings = applied.flash }
      }
    }
    if candidate.steps.contains(where: { $0.isH3Regenerate || $0.isFlashVSR && $0.settings.variant == .base }) { candidate.format = "weetodd-video-processing-v2" }
    try candidate.validate(); return .init(recipe: candidate, models: models)
  }
  private static func pixelBinding(_ model: GenerationModelRef, library: GenerationLibrarySnapshot) throws -> [String: String] {
    guard case .restoration(let tool, let components) = try library.resolve(model, in: .init(workspace: .finishing, task: "pixelCQ")), tool == "pixelCQ" else {
      throw StudioError.invalid("Choose installed LTX Pixel Spatial + CQ components.")
    }
    return components
  }
}
