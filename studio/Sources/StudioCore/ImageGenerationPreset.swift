import Foundation
import ImageLayout

public enum ImagePresetRecipe {
  private static let common: Set<String> = ["steps", "seed"]
  private static let sampling: Set<String> = ["guidance", "sampler", "shift", "negativePrompt"]
  private static let mossKeys: Set<String> = ["sampler", "schedule", "skipLayers", "skipStart", "skipStop"]
  private static let h3Keys: Set<String> = ["decoderMode", "frameSelection"]
  public static func context(_ draft: DrawThingsImageDraft) -> GenerationLibraryContext {
    .init(workspace: .image, task: "image", inputRoles: draft.activeImageInputs.map(\.role))
  }
  public static func model(for draft: DrawThingsImageDraft, library: GenerationLibrarySnapshot) throws -> GenerationModelEntry {
    guard let entry = library.models(for: context(draft)).first(where: { entry in
      guard entry.reference.backend == draft.executionProvider.rawValue else { return false }
      switch entry.binding {
      case .nativeMOSS(let transformer, _, _): return transformer == draft.mossImage?.transformerDirectory
        && entry.reference.variant == (draft.mossImage?.variant ?? "sft")
        && entry.reference.precision == (draft.mossImage?.quantizationBits.map { "q\($0)" } ?? "bf16")
      case .nativeH3Image(let profile, _): return profile == draft.modelID
      case .nativeQwen(let manifest): return manifest == draft.nativeImage?.manifestPath
      case .drawThings(let connection, let id): return connection == draft.profileID && id == draft.modelID
      default: return false
      }
    }) else { throw StudioError.invalid("Choose or link the current model in Models & LoRAs before saving a preset.") }
    _ = try library.resolve(entry.reference, in: context(draft)); return entry
  }
  public static func capture(from draft: DrawThingsImageDraft, resolvedLibrary library: GenerationLibrarySnapshot) throws -> GenerationPresetSnapshot {
    let entry = try model(for: draft, library: library); try validate(draft)
    let provider = draft.executionProvider, context = context(draft)
    var fields = try GenerationPresetFields.capture(draft, keys: common.union(provider == .drawThings || provider == .nativeMOSS ? sampling : []))
    if provider == .nativeMOSS {
      let settings = draft.mossImage ?? .init()
      var owned = try GenerationPresetFields.capture(settings, keys: mossKeys)
      owned["skipLayers"] = try .encoded(settings.resolvedSkipLayers())
      fields["moss"] = .object(owned)
    }
    if provider == .nativeH3 { fields["h3"] = .object(try GenerationPresetFields.capture(draft.h3Image ?? .init(), keys: h3Keys)) }
    let adapters: [GenerationPresetAdapter]
    switch provider {
    case .nativeMOSS: adapters = try library.nativePresetAdapters(draft.mossImage?.loras ?? [], model: entry.reference, context: context)
    case .nativeH3: adapters = try library.nativePresetAdapters(draft.h3Image?.loras ?? [], model: entry.reference, context: context)
    case .nativeMLX: adapters = []
    case .drawThings:
      adapters = try draft.loras.map { lora in
        guard let item = library.loras(for: context, model: entry.reference).first(where: {
          if case .drawThings(let connection, let id) = $0.binding { return connection == draft.profileID && id == lora.modelID }; return false
        }) else { throw StudioError.invalid("Refresh the compatible Draw Things LoRA library before saving.") }
        _ = try library.resolve(item.reference, for: entry.reference, in: context)
        return .init(reference: item.reference, strength: lora.weight, enabled: lora.isEnabled)
      }
    }
    let snapshot = GenerationPresetSnapshot(context: .init(workspace: .image, task: "image"), model: entry.reference, adapters: adapters, fields: fields)
    try snapshot.validate(); return snapshot
  }
  public static func candidate(applying snapshot: GenerationPresetSnapshot, to draft: DrawThingsImageDraft,
    resolvedLibrary library: GenerationLibrarySnapshot) throws -> DrawThingsImageDraft {
    try snapshot.validate()
    guard snapshot.context == .init(workspace: .image, task: "image"), let model = snapshot.model,
      let provider = ImageExecutionProvider(rawValue: model.backend) else { throw StudioError.invalid("Choose an image preset for this workspace.") }
    let context = context(draft), binding = try library.resolve(model, in: context)
    if provider != .nativeMOSS, draft.mossImage?.regionalLayout == true || !(draft.mossLayout?.regions.isEmpty ?? true) || draft.mossLayout?.sourceReconstruction?.enabled == true {
      throw StudioError.invalid("This image uses MOSS layout or source conditioning. Choose a compatible MOSS preset; its boxes are preserved.")
    }
    var candidate = draft; candidate.provider = provider
    switch binding {
    case .nativeMOSS(let transformer, let shared, let root):
      if candidate.mossImage == nil { candidate.mossImage = .init() }
      candidate.mossImage?.transformerDirectory = transformer; candidate.mossImage?.sharedDirectory = shared; candidate.mossImage?.libraryDirectory = root
      candidate.mossImage?.variant = model.variant ?? "sft"
      candidate.mossImage?.quantizationBits = model.precision == "q4" ? 4 : model.precision == "q8" ? 8 : nil
      candidate.modelID = model.variant == "base" ? MOSSImageSettings.baseModelID : MOSSImageSettings.sftModelID
    case .nativeH3Image(let profile, let vae):
      candidate.modelID = profile; if candidate.h3Image == nil { candidate.h3Image = .init() }; candidate.h3Image?.imageVAEPath = vae
    case .nativeQwen(let manifest):
      candidate.modelID = "Qwen/Qwen-Image-2.1"; if candidate.nativeImage == nil { candidate.nativeImage = .init() }; candidate.nativeImage?.manifestPath = manifest
    case .drawThings(let connection, let id): candidate.profileID = connection; candidate.modelID = id
    default: throw StudioError.invalid("The model binding does not support image generation.")
    }
    if provider != draft.executionProvider {
      // Fixed backend recipe values are owned settings; project conditioning strength remains untouched.
      candidate.guidance = 1; candidate.sampler = nil; candidate.shift = nil; candidate.negativePrompt = ""; candidate.loras = []
    }
    if let adapters = snapshot.adapters {
      switch provider {
      case .nativeMOSS:
        candidate.mossImage?.loras = try library.nativePresetMembers(adapters, model: model, context: context, existing: draft.mossImage?.loras ?? [])
        try candidate.reconcileMOSSFastRecipe()
      case .nativeH3: candidate.h3Image?.loras = try library.nativePresetMembers(adapters, model: model, context: context, existing: draft.h3Image?.loras ?? [])
      case .nativeMLX: guard adapters.isEmpty else { throw StudioError.invalid("Qwen Image does not support a native LoRA stack.") }
      case .drawThings:
        candidate.loras = try adapters.map { adapter in
          guard adapter.fields.isEmpty, case .drawThings(let connection, let id) = try library.resolve(adapter.reference, for: model, in: context), connection == candidate.profileID else {
            throw StudioError.invalid("This preset needs a compatible Draw Things LoRA.")
          }
          return DrawThingsLoRA(modelID: id, weight: adapter.strength, enabled: adapter.enabled)
        }
      }
    }
    var fields = snapshot.fields
    let allowed = common.union(provider == .drawThings || provider == .nativeMOSS ? sampling : [])
    if provider == .nativeMOSS, let value = fields.removeValue(forKey: "moss") {
      guard case .object(let owned) = value else { throw StudioError.invalid("Invalid MOSS preset fields.") }
      candidate.mossImage = try GenerationPresetFields.applying(owned, to: candidate.mossImage ?? .init(), allowed: mossKeys)
      if owned["skipLayers"] != nil { candidate.mossImage?.skipLayerExpression = nil }
    }
    if provider == .nativeH3, let value = fields.removeValue(forKey: "h3") {
      guard case .object(let owned) = value else { throw StudioError.invalid("Invalid H3 image preset fields.") }
      candidate.h3Image = try GenerationPresetFields.applying(owned, to: candidate.h3Image ?? .init(), allowed: h3Keys)
    }
    candidate = try GenerationPresetFields.applying(fields, to: candidate, allowed: allowed)
    try validate(candidate)
    return candidate
  }
  public static func validate(_ draft: DrawThingsImageDraft) throws {
    if let issue = draft.imageInputIssue ?? draft.characterSheetRequestIssue ?? draft.managedCharacterPromptIssue { throw StudioError.invalid(issue) }
    guard (1...1000).contains(draft.steps), (-1...4_294_967_295).contains(draft.seed), draft.guidance.isFinite,
      (0...(draft.executionProvider == .drawThings ? 100.0 : 20.0)).contains(draft.guidance), draft.shift.map({ $0.isFinite && (draft.executionProvider == .drawThings ? $0 >= 0 : $0 > 0) }) ?? true else { throw StudioError.invalid("Use valid steps, seed, guidance and shift before saving or applying a preset.") }
    switch draft.executionProvider {
    case .nativeMOSS:
      let settings = draft.mossImage ?? .init()
      try ImageLayoutGeometry.validateCanvas(width: draft.width, height: draft.height)
      guard draft.strength == 1, draft.sampler == nil, !draft.loras.contains(where: \.isEnabled),
        ["euler", "er_sde"].contains(settings.sampler), ["release", "simple"].contains(settings.schedule), settings.skipStart.isFinite,
        settings.skipStop.isFinite, 0 <= settings.skipStart, settings.skipStart <= settings.skipStop, settings.skipStop <= 1 else {
        throw StudioError.invalid("Use supported MOSS sampling and skip settings; project inputs remain unchanged.")
      }
      let layers = try settings.resolvedSkipLayers()
      guard layers.allSatisfy({ (0..<42).contains($0) }), Set(layers).count == layers.count else { throw StudioError.invalid("MOSS skip layers must be distinct indices from 0 through 41.") }
      let adapters = settings.loras ?? []; guard adapters.count <= 8 else { throw StudioError.invalid("MOSS supports eight adapters.") }
      for member in adapters { try member.validateForMOSS() }
      let fast = adapters.filter { $0.isEnabled && $0.strength != 0 && $0.asset.loraProfile == "moss_sft_4step_v2" }
      if !fast.isEmpty {
        guard fast.count == 1, fast[0].strength == 1, settings.variant == "sft", (4...8).contains(draft.steps), draft.guidance == 1,
          (draft.shift ?? 5) == 5, settings.sampler == "euler", settings.schedule == "release" else { throw StudioError.invalid("MOSS Fast requires SFT, strength 1, 4–8 steps, CFG 1, shift 5 and the released Euler schedule.") }
      }
      if settings.regionalLayout == true, draft.mossLayout == nil || draft.mossLayout?.sourceReconstruction?.enabled == true { throw StudioError.invalid("Regional MOSS requires editable boxes and generation from noise.") }
      if draft.mossLayout?.sourceReconstruction?.enabled == true, settings.variant != "sft" || settings.quantizationBits != nil || adapters.contains(where: { $0.isEnabled && $0.strength != 0 }) {
        throw StudioError.invalid("The current legacy source conditioning requires stock BF16 SFT without enabled adapters.")
      }
    case .nativeMLX, .nativeH3:
      guard draft.guidance == 1, draft.strength == 1, draft.sampler == nil, draft.shift == nil, draft.negativePrompt.isEmpty,
        !draft.loras.contains(where: \.isEnabled), draft.width % 32 == 0, draft.height % 32 == 0 else { throw StudioError.invalid("This native image model uses CFG 1, full-strength references and its fixed sampler. The current canvas and inputs were preserved.") }
      if draft.executionProvider == .nativeH3 {
        guard (1...100).contains(draft.steps), [draft.width, draft.height].allSatisfy({ (32...1920).contains($0) }), draft.width * draft.height <= 1920 * 1088,
          draft.activeImageInputs.allSatisfy({ $0.input.strength == 1 }), (draft.h3Image?.loras.count ?? 0) <= 8 else { throw StudioError.invalid("The current canvas, reference strengths or evaluations are unsupported by H3.") }
        for member in draft.h3Image?.loras ?? [] { try member.validate(for: .h3) }
      } else {
        guard [draft.width, draft.height].allSatisfy({ (32...4096).contains($0) }), draft.width * draft.height <= 8_388_608 else { throw StudioError.invalid("The current canvas exceeds the Qwen image limit.") }
      }
    case .drawThings: for adapter in draft.loras { try adapter.validate() }
    }
  }
}
