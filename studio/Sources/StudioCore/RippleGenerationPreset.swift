import Foundation

public enum RipplePresetRecipe {
  private static let keys: Set<String> = ["seed", "loraStrength", "diffusionVAE"]
  public static let context = GenerationLibraryContext(workspace: .ripple, task: "ripple")
  public static func binding(_ model: GenerationModelRef, library: GenerationLibrarySnapshot) throws -> ResolvedGenerationModel {
    guard model.backend == "ripple", model.family == "ltx25", model.variant == "distilled",
      case .ripple = try library.resolve(model, in: context) else { throw StudioError.invalid("Ripple requires a plain installed LTX 2.5 distilled model paired with its author adapter.") }
    return try library.resolve(model, in: context)
  }
  public static func capture(from draft: RippleDraft, resolvedLibrary library: GenerationLibrarySnapshot,
    legacyModel: GenerationModelRef? = nil) throws -> GenerationPresetSnapshot {
    guard let model = draft.libraryModel ?? legacyModel else { throw StudioError.invalid("Choose an installed Ripple model and required author adapter in Models & LoRAs.") }
    _ = try binding(model, library: library); try validate(draft)
    let snapshot = GenerationPresetSnapshot(context: .init(workspace: .ripple, task: "ripple"), model: model, adapters: [],
      fields: try GenerationPresetFields.capture(draft, keys: keys))
    try snapshot.validate(); return snapshot
  }
  public static func candidate(applying snapshot: GenerationPresetSnapshot, to draft: RippleDraft,
    resolvedLibrary library: GenerationLibrarySnapshot) throws -> RippleDraft {
    try snapshot.validate()
    guard snapshot.context == .init(workspace: .ripple, task: "ripple"), let model = snapshot.model, snapshot.adapters?.isEmpty ?? true else {
      throw StudioError.invalid("Choose a Ripple preset. Its required author adapter is part of the model binding.")
    }
    _ = try binding(model, library: library)
    var result = try GenerationPresetFields.applying(snapshot.fields, to: draft, allowed: keys)
    result.libraryModel = model; try validate(result); return result
  }
  public static func validate(_ draft: RippleDraft) throws {
    try draft.validate(requireReferences: false)
    if let vae = draft.diffusionVAE {
      guard vae.experimentalEnabled, (1...65_536).contains(vae.queryChunkSize), (1...4096).contains(vae.contextWidthChunks),
        (0...4096).contains(vae.stage4TileWidth), vae.optimization == .stage4WidthTiles ? vae.stage4TileWidth > 0 : vae.stage4TileWidth == 0 else {
        throw StudioError.invalid("Use supported explicitly enabled Ripple decoder controls.")
      }
    }
  }
}
