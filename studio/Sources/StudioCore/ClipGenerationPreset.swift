import Foundation
import InferenceContracts

public struct ClipPresetApplication {
  public var clip: Clip
  public var assets: [MediaAsset]
  public var libraryModel: GenerationModelRef? = nil
}

public enum ClipPresetRecipe {
  public static func presetContext(_ clip: Clip, project: StudioProject) -> GenerationPresetContext {
    .init(workspace: .clip, task: project.isContinuousSceneMember(clip) ? "scene" : clip.inferredTask)
  }
  private static let h3Keys: Set<String> = ["steps", "h3SamplingMethod", "h3VideoDecodePrecision", "h3AttentionPolicy"]
  private static let ltxKeys: Set<String> = ["steps", "refinementSteps", "cfg", "shift", "ltx25Guidance", "ltx25DFR", "ltx25SingleStage", "ltx25DiffusionVAE"]
  private static let remoteKeys: Set<String> = ["steps", "guidanceScale", "sampler", "shift", "audioShift", "hiresFix", "hiresFixStrength"]
  public static func context(_ clip: Clip, project: StudioProject) -> GenerationLibraryContext {
    .init(workspace: .clip, task: clip.inferredTask,
      inputRoles: clip.attachments.filter { $0.role != .lora && $0.isEnabled }.map { $0.role.rawValue },
      continuousScene: project.isContinuousSceneMember(clip), family: clip.inferredTask == "video_upscale" ? (clip.usesReCaVSR ? "reCaVSR" : clip.usesRelayVSR ? "relayVSR" : "ltx25") : nil,
      movieMode: clip.inferredTask == "video_upscale" && !clip.usesNativeMovieUpscaler ? clip.generationSelection?.ltx25MovieUpscale?.mode : nil)
  }
  public static func model(for clip: Clip, project: StudioProject, library: GenerationLibrarySnapshot) throws -> GenerationModelEntry {
    if clip.usesNativeMovieUpscaler, let ref = project.generationLibraryBindings?["clip-model:" + clip.id.uuidString],
      let entry = library.modelEntries.first(where: { $0.reference == ref }) {
      _ = try library.resolve(ref, in: context(clip, project: project)); return entry
    }
    guard let entry = library.models(for: context(clip, project: project)).first(where: { entry in
      guard entry.reference.backend == clip.engine.rawValue else { return false }
      switch entry.binding {
      case .nativeVideo(let id): return id == clip.profileID
      case .drawThings(let connection, let id): return connection == clip.drawThings?.profileID && id == clip.drawThings?.modelID
      default: return false
      }
    }) else { throw StudioError.invalid("Choose an installed named model for this task in Models & LoRAs before saving a preset.") }
    _ = try library.resolve(entry.reference, in: context(clip, project: project)); return entry
  }
  public static func capture(from clip: Clip, project: StudioProject, resolvedLibrary library: GenerationLibrarySnapshot) throws -> GenerationPresetSnapshot {
    let entry = try model(for: clip, project: project, library: library), context = context(clip, project: project)
    if clip.usesNativeMovieUpscaler {
      var state = RestorationPresetState(tool: clip.usesReCaVSR ? .reCaVSR : .relayVSR)
      state.model = entry.reference; state.relayScale = clip.generationSelection?.relayVSRScale ?? .x2
      state.reCa = clip.generationSelection?.reCaVSRSettings ?? (try! ReCaVSRSettings()); state.reCa.seed = UInt64(max(0, clip.seed))
      var result = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library, workspace: .clip)
      result.context.task = "video_upscale"; result.fields["method"] = .string(state.tool.rawValue)
      return result
    }
    try validate(clip, project: project)
    var fields: [String: GenerationPresetValue] = ["seed": .integer(Int64(clip.seed))]
    let adapters: [GenerationPresetAdapter]
    if clip.engine == .drawThings {
      fields["negativePrompt"] = .string(clip.negativePrompt)
      let configuration = clip.drawThings?.configuration ?? [:]
      fields["configuration"] = .object(Dictionary(uniqueKeysWithValues: try remoteKeys.map { key in
        (key, try configuration[key].map { try GenerationPresetValue.encoded($0) } ?? .automatic)
      }))
      adapters = try (clip.drawThings?.loras ?? []).map { lora in
        guard let item = library.loras(for: context, model: entry.reference).first(where: {
          if case .drawThings(let connection, let id) = $0.binding { return connection == clip.drawThings?.profileID && id == lora.modelID }; return false
        }) else { throw StudioError.invalid("Refresh the compatible Draw Things LoRA list before saving.") }
        _ = try library.resolve(item.reference, for: entry.reference, in: context)
        return .init(reference: item.reference, strength: lora.weight, enabled: lora.isEnabled)
      }
    } else {
      fields["selection"] = .object(try GenerationPresetFields.capture(clip.generationSelection ?? .init(task: clip.inferredTask), keys: clip.engine == .h3 ? h3Keys : ltxKeys))
      if clip.engine != .h3 { fields["negativePrompt"] = .string(clip.negativePrompt) }
      if let specialized = clip.generationSelection?.ltx25AdapterBinding {
        guard let linked = library.loraEntries.first(where: {
          if case .nativeFile(let path, let profile, _, _) = $0.binding { return path == specialized.path && profile == specialized.kind.rawValue }; return false
        }) else { throw StudioError.invalid("Link the specialized LTX adapter before saving its preset.") }
        _ = try library.specializedAdapter(linked.reference, kind: specialized.kind)
        fields["specializedAdapter"] = try .encoded(linked.reference)
        fields["specializedKind"] = .string(specialized.kind.rawValue)
        fields["supportsVoiceReferences"] = .bool(specialized.supportsVoiceReferences)
      } else if clip.engine == .ltx25 { fields["specializedAdapter"] = .automatic }
      if clip.inferredTask == "video_upscale", let settings = clip.generationSelection?.ltx25MovieUpscale {
        fields["movie"] = .object(try GenerationPresetFields.capture(settings, keys: ["mode", "refinementStrength", "anchorStrength", "pixelStrength"]))
      }
      let members = try clip.attachments.filter { $0.role == .lora }.map { attachment -> LoRAMember in
        guard let asset = project.assets.first(where: { $0.id == attachment.assetID }) else { throw StudioError.invalid("Relink the clip LoRA before saving this preset.") }
        return .init(asset: asset, strength: attachment.strength, enabled: attachment.enabled, h3Settings: attachment.h3LoRA)
      }
      adapters = try library.nativePresetAdapters(members, model: entry.reference, context: context)
    }
    let snapshot = GenerationPresetSnapshot(context: presetContext(clip, project: project), model: entry.reference, adapters: adapters, fields: fields)
    try snapshot.validate(); return snapshot
  }
  public static func candidate(applying snapshot: GenerationPresetSnapshot, to clip: Clip, project: StudioProject,
    resolvedLibrary library: GenerationLibrarySnapshot) throws -> ClipPresetApplication {
    try snapshot.validate()
    if clip.usesNativeMovieUpscaler { return try movieCandidate(snapshot, clip: clip, library: library) }
    guard snapshot.context == presetContext(clip, project: project), let model = snapshot.model,
      let engine = Engine(rawValue: model.backend), [.h3, .ltx23, .ltx25, .drawThings].contains(engine) else { throw StudioError.invalid("Choose a preset for this video task. Inputs and timing are preserved.") }
    let context = context(clip, project: project), binding = try library.resolve(model, in: context)
    var candidate = clip, added: [MediaAsset] = []
    candidate.engine = engine
    switch binding {
    case .nativeVideo(let profile): candidate.profileID = profile
    case .drawThings(let connection, let id):
      if candidate.drawThings == nil { candidate.drawThings = .init(profileID: connection, modelID: id, modelFamily: model.family) }
      candidate.drawThings?.profileID = connection; candidate.drawThings?.modelID = id; candidate.drawThings?.modelFamily = model.family
    default: throw StudioError.invalid("This model cannot run the current video task.")
    }
    var fields = snapshot.fields
    if let seed = fields.removeValue(forKey: "seed") { candidate.seed = try seed.decoded(Int.self) }
    if let negative = fields.removeValue(forKey: "negativePrompt") { candidate.negativePrompt = try negative.decoded(String.self) }
    if engine == .drawThings {
      if let configuration = fields.removeValue(forKey: "configuration") {
        guard case .object(let values) = configuration, Set(values.keys).isSubset(of: remoteKeys) else { throw StudioError.invalid("Unsupported Draw Things preset configuration.") }
        for (key, value) in values {
          if value == .automatic { candidate.drawThings?.configuration.removeValue(forKey: key) }
          else { candidate.drawThings?.configuration[key] = try value.decoded(JSONValue.self) }
        }
      }
      if let adapters = snapshot.adapters {
        let connectionID = candidate.drawThings?.profileID
        candidate.drawThings?.loras = try adapters.map { adapter in
          guard adapter.fields.isEmpty, case .drawThings(let connection, let id) = try library.resolve(adapter.reference, for: model, in: context), connection == connectionID else {
            throw StudioError.invalid("Choose compatible Draw Things LoRAs.")
          }
          return .init(modelID: id, weight: adapter.strength, enabled: adapter.enabled)
        }
      }
    } else {
      var selection = candidate.generationSelection ?? .init(task: clip.inferredTask)
      if engine != clip.engine {
        // Only backend-owned sampling controls reset. All authored conditioning remains for validation.
        for key in h3Keys.union(ltxKeys) {
          selection = try GenerationPresetFields.applying([key: .automatic], to: selection, allowed: [key])
        }
      }
      if let value = fields.removeValue(forKey: "selection") {
        guard case .object(let values) = value else { throw StudioError.invalid("Invalid native video preset settings.") }
        selection = try GenerationPresetFields.applying(values, to: selection, allowed: engine == .h3 ? h3Keys : ltxKeys)
      }
      if let specialized = fields.removeValue(forKey: "specializedAdapter") {
        guard engine == .ltx25 else { throw StudioError.invalid("This specialized adapter requires LTX 2.5.") }
        if specialized == .automatic { selection.ltx25AdapterBinding = nil }
        else {
          guard let kindValue = fields.removeValue(forKey: "specializedKind"), let voice = fields.removeValue(forKey: "supportsVoiceReferences") else { throw StudioError.invalid("Missing specialized adapter contract.") }
          let kind = try kindValue.decoded(LTXLibraryAdapterKind.self), ref = try specialized.decoded(GenerationLoRARef.self)
          selection.ltx25AdapterBinding = .init(kind: kind, path: try library.specializedAdapter(ref, kind: kind), supportsVoiceReferences: try voice.decoded(Bool.self))
        }
      }
      candidate.generationSelection = selection
      if let movie = fields.removeValue(forKey: "movie") {
        guard clip.inferredTask == "video_upscale", case .object(let values) = movie else { throw StudioError.invalid("This recipe needs an LTX source-movie task.") }
        let current = selection.ltx25MovieUpscale ?? .init()
        let proposed = try GenerationPresetFields.applying(values, to: current, allowed: ["mode", "refinementStrength", "anchorStrength", "pixelStrength"])
        guard proposed.mode == current.mode else { throw StudioError.invalid("Select the desired movie upscale mode explicitly before applying this preset. Source dimensions were preserved.") }
        try proposed.validate(); candidate.generationSelection?.ltx25MovieUpscale = proposed
      }
      if let adapters = snapshot.adapters {
        let existing: [LoRAMember] = clip.attachments.filter { $0.role == .lora }.compactMap { attachment in
          guard let asset = project.assets.first(where: { $0.id == attachment.assetID }) else { return nil }
          return .init(asset: asset, strength: attachment.strength, enabled: attachment.enabled, h3Settings: attachment.h3LoRA)
        }
        let members = try library.nativePresetMembers(adapters, model: model, context: context, existing: existing)
        var attachments: [Attachment] = []
        for member in members {
          try member.validate(for: engine)
          var asset = project.assets.first { $0.kind == .lora && $0.path == member.asset.path } ?? member.asset
          if !project.assets.contains(where: { $0.id == asset.id }) { asset.scope = .project; asset.owner = nil; added.append(asset) }
          var attachment = clip.attachments.first { $0.role == .lora && $0.assetID == asset.id } ?? Attachment(assetID: asset.id, role: .lora)
          attachment.strength = member.strength
          if attachment.isEnabled != member.isEnabled { attachment.enabled = member.isEnabled }
          attachment.h3LoRA = member.h3Settings; attachments.append(attachment)
        }
        candidate.attachments = candidate.attachments.filter { $0.role != .lora } + attachments
      }
    }
    guard fields.isEmpty else { throw StudioError.invalid("This preset contains settings that the selected video backend cannot execute.") }
    func clock(_ value: Clip) -> [Double] {
      let entry = try? Self.model(for: value, project: project, library: library)
      let descriptor = entry?.generation
      let fps = descriptor?.frameRate ?? value.settings(in: project).fps
      let explicit = value.generationSelection?.ltx25DFR
      let enabled = explicit?.enabled ?? (descriptor?.dfrEnabled ?? false)
      let rounds = enabled ? (explicit?.temporalRounds ?? descriptor?.dfrTemporalRounds ?? 0) : 0
      return [fps, fps * pow(2, Double(rounds))]
    }
    if clock(candidate) != clock(clip) {
      throw StudioError.invalid("This preset would change the source clock through DFR temporal scaling. Choose a preset with the same temporal rounds.")
    }
    var proposed = project; proposed.assets += added
    try validate(candidate, project: proposed)
    return .init(clip: candidate, assets: added)
  }
  private static func movieCandidate(_ snapshot: GenerationPresetSnapshot, clip: Clip, library: GenerationLibrarySnapshot) throws -> ClipPresetApplication {
    let tool: RestorationPresetTool = clip.usesReCaVSR ? .reCaVSR : .relayVSR
    guard snapshot.context == .init(workspace: .clip, task: "video_upscale"), snapshot.fields["method"] == .string(tool.rawValue) else {
      throw StudioError.invalid("Choose a preset for the selected source-movie upscaler. Its method and source remain unchanged.")
    }
    var state = RestorationPresetState(tool: tool); state.relayScale = clip.generationSelection?.relayVSRScale ?? .x2
    state.reCa = clip.generationSelection?.reCaVSRSettings ?? (try! ReCaVSRSettings()); state.reCa.seed = UInt64(max(0, clip.seed))
    var recipe = snapshot; recipe.context.task = tool.rawValue; recipe.fields.removeValue(forKey: "method")
    let result = try RestorationPresetRecipe.candidate(applying: recipe, to: state, resolvedLibrary: library, workspace: .clip)
    var candidate = clip
    if tool == .reCaVSR {
      guard result.reCa.seed <= UInt64(Int.max) else { throw StudioError.invalid("This clip seed exceeds the supported integer range.") }
      candidate.seed = Int(result.reCa.seed); candidate.generationSelection?.reCaVSRSettings = result.reCa
    }
    return .init(clip: candidate, assets: [], libraryModel: result.model)
  }
  public static func validate(_ clip: Clip, project: StudioProject) throws {
    guard (-1...4_294_967_295).contains(clip.seed), clip.engine == .drawThings || clip.seed >= 0 else { throw StudioError.invalid("Use a supported seed. Native video uses a fixed nonnegative seed.") }
    let selection = clip.generationSelection
    guard selection?.steps.map({ ((clip.inferredTask == "video_upscale" && selection?.ltx25MovieUpscale?.mode == .latentOnly ? 0 : 1)...1000).contains($0) }) ?? true,
      selection?.refinementSteps.map({ (0...1000).contains($0) }) ?? true,
      [selection?.cfg, selection?.shift].compactMap({ $0 }).allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 100 }) else { throw StudioError.invalid("Invalid native sampling controls.") }
    if project.isContinuousSceneMember(clip) {
      guard clip.engine == .ltx25, selection?.ltx25Guidance == nil, selection?.ltx25DFR == nil,
        selection?.ltx25AutomaticDuration == nil else { throw StudioError.invalid("This preset cannot preserve the current continuous scene.") }
    }
    if clip.engine == .h3 {
      guard clip.settings(in: project).fps == 24, clip.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        selection?.refinementSteps == nil, selection?.cfg == nil, selection?.shift == nil,
        selection?.ltx25Guidance == nil, selection?.ltx25DFR == nil, selection?.ltx25SingleStage == nil,
        selection?.ltx25DiffusionVAE == nil, selection?.ltx25Keyframes == nil, selection?.ltx25AutomaticDuration == nil,
        selection?.ltx25MovieUpscale == nil else { throw StudioError.invalid("H3 requires a 24 FPS task with its supported sampling and conditioning controls. No timing or inputs were changed.") }
    } else if clip.engine == .ltx25, clip.inferredTask == "video_upscale" {
      guard let movie = selection?.ltx25MovieUpscale else { throw StudioError.invalid("Select explicit source-movie settings before saving or applying its preset.") }
      try NativeLTXMoviePreparation.validateEditor(clip: clip, settings: movie, isContinuousScene: project.isContinuousSceneMember(clip))
    } else if clip.engine == .ltx25 {
      _ = try LTX25GenerationConfig(name: "Preset validation", selection: selection ?? .init(task: clip.inferredTask))
    } else if clip.engine == .drawThings {
      for member in clip.drawThings?.loras ?? [] { try member.validate() }
      let config = (clip.drawThings?.configuration ?? [:]).filter { remoteKeys.contains($0.key) }
      let object: [String: Any] = ["configuration": try JSONSerialization.jsonObject(with: JSONEncoder().encode(config))]
      _ = try DrawThingsConfigImport.parse(JSONSerialization.data(withJSONObject: object), operation: "video")
    }
    guard clip.engine == .h3 || (selection?.h3Reference == nil && selection?.h3Joint == nil && selection?.h3MotionFidelity == nil) else {
      throw StudioError.invalid("This model cannot preserve the current H3 conditioning controls.")
    }
  }
}
