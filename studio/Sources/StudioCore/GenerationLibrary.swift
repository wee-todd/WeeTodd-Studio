import Foundation

public enum GenerationWorkspaceKind: String, Codable, CaseIterable {
  case image, clip, ripple, restoration, finishing
}

/// Portable identity. Machine-local paths live in the library binding instead.
public struct GenerationModelRef: Codable, Hashable {
  public var id, backend, family: String
  public var variant, precision, sourceIdentity: String?
  public init(id: String, backend: String, family: String, variant: String? = nil,
    precision: String? = nil, sourceIdentity: String? = nil) {
    self.id = id; self.backend = backend; self.family = family
    self.variant = variant; self.precision = precision; self.sourceIdentity = sourceIdentity
  }
  public func validate() throws {
    guard GenerationLibraryLimits.identifier(id), GenerationLibraryLimits.identifier(backend),
      GenerationLibraryLimits.identifier(family), [variant, precision, sourceIdentity].compactMap({ $0 })
        .allSatisfy(GenerationLibraryLimits.identifier) else {
      throw StudioError.invalid("Invalid model library identity.")
    }
  }
}

public struct GenerationLoRARef: Codable, Hashable {
  public var id, backend, family: String
  public var sourceIdentity: String?
  public init(id: String, backend: String, family: String, sourceIdentity: String? = nil) {
    self.id = id; self.backend = backend; self.family = family; self.sourceIdentity = sourceIdentity
  }
  public func validate() throws {
    try GenerationModelRef(id: id, backend: backend, family: family, sourceIdentity: sourceIdentity).validate()
  }
}

public struct GenerationLibraryContext: Equatable {
  public var workspace: GenerationWorkspaceKind
  public var task: String
  public var backend: String?
  public var family: String?
  public var movieMode: LTX25MovieUpscaleSettings.Mode?
  public var inputRoles: [String]
  public var connectionID: String?
  public var continuousScene: Bool
  public init(workspace: GenerationWorkspaceKind, task: String, backend: String? = nil,
    inputRoles: [String] = [], connectionID: String? = nil, continuousScene: Bool = false, family: String? = nil, movieMode: LTX25MovieUpscaleSettings.Mode? = nil) {
    self.workspace = workspace; self.task = task; self.backend = backend
    self.inputRoles = inputRoles; self.connectionID = connectionID; self.continuousScene = continuousScene
    self.family = family
    self.movieMode = movieMode
  }
}

public enum GenerationLibraryAvailability: Codable, Equatable {
  case installed, available, checking, missingComponents([String]), incompatible(String)
  case driveDisconnected, backendUnavailable(String)
  public var label: String {
    switch self {
    case .installed: return "Installed"
    case .available: return "Available"
    case .checking: return "Checking…"
    case .missingComponents: return "Missing components"
    case .incompatible: return "Incompatible"
    case .driveDisconnected: return "Drive disconnected"
    case .backendUnavailable: return "Connection unavailable"
    }
  }
  public var detail: String? {
    switch self {
    case .missingComponents(let fields): return fields.joined(separator: ", ")
    case .incompatible(let reason), .backendUnavailable(let reason): return reason
    case .driveDisconnected: return "Reconnect the model drive. Your saved selection is retained."
    default: return nil
    }
  }
  public var canResolve: Bool { self == .installed }
}

/// Existing renderer IDs are resolved here; the renderer's request contract is unchanged.
public enum ResolvedGenerationModel: Codable, Equatable {
  case nativeVideo(profileID: String)
  case nativeH3Image(profileID: String, imageVAE: String)
  case nativeMOSS(transformer: String, shared: String, library: String)
  case nativeQwen(manifest: String)
  case drawThings(connectionID: String, modelID: String)
  case restoration(tool: String, components: [String: String])
  case ripple(profileID: String, adapterPath: String)

  public var connectionID: String? {
    if case .drawThings(let id, _) = self { return id }; return nil
  }
  public var localPaths: [String] {
    switch self {
    case .nativeVideo(let id): return id.hasPrefix("/") ? [id] : []
    case .nativeH3Image(let id, let vae): return [id, vae].filter { $0.hasPrefix("/") }
    case .nativeMOSS(let transformer, let shared, let library): return [transformer, shared, library].filter { !$0.isEmpty }
    case .nativeQwen(let manifest): return [manifest]
    case .restoration(_, let fields): return fields.values.filter { $0.hasPrefix("/") }
    case .ripple(let id, let adapter): return [id, adapter].filter { $0.hasPrefix("/") }
    case .drawThings: return []
    }
  }
}

public enum ResolvedGenerationLoRA: Codable, Equatable {
  case nativeFile(path: String, profile: String? = nil, layout: String? = nil, adalnGrid: String? = nil)
  case drawThings(connectionID: String, modelID: String)
}

public struct GenerationModelEntry: Codable, Equatable, Identifiable {
  public var reference: GenerationModelRef
  public var name: String
  public var workspaces: [GenerationWorkspaceKind]
  public var tasks: [String]
  public var inputRoleSets: [[String]]?
  public var supportsLoRAs: Bool
  public var availability: GenerationLibraryAvailability
  public var binding: ResolvedGenerationModel?
  public var detail: String?
  public var identityFiles: [String]?
  public var generation: GenerationDescriptor?
  public var movieModes: [LTX25MovieUpscaleSettings.Mode]?
  public var id: String { reference.id }
  public init(reference: GenerationModelRef, name: String, workspaces: [GenerationWorkspaceKind],
    tasks: [String], inputRoleSets: [[String]]? = nil, supportsLoRAs: Bool = true,
    availability: GenerationLibraryAvailability, binding: ResolvedGenerationModel? = nil, detail: String? = nil, identityFiles: [String]? = nil, generation: GenerationDescriptor? = nil, movieModes: [LTX25MovieUpscaleSettings.Mode]? = nil) {
    self.reference = reference; self.name = name; self.workspaces = workspaces; self.tasks = tasks
    self.inputRoleSets = inputRoleSets; self.supportsLoRAs = supportsLoRAs
    self.availability = availability; self.binding = binding; self.detail = detail; self.identityFiles = identityFiles
    self.generation = generation
    self.movieModes = movieModes
  }
  public func supports(_ context: GenerationLibraryContext) -> Bool {
    workspaces.contains(context.workspace) && (tasks.contains(context.task) || tasks.contains("*"))
      && (context.backend == nil || reference.backend == context.backend)
      && (context.family == nil || reference.family == context.family)
      && (context.movieMode == nil || movieModes == nil || movieModes!.contains(context.movieMode!))
      && (context.connectionID == nil || binding?.connectionID == context.connectionID)
      && (inputRoleSets == nil || inputRoleSets!.contains { $0.sorted() == context.inputRoles.sorted() })
      && (!context.continuousScene || reference.backend == "ltx25" && (generation?.pipelineMode == nil || generation?.pipelineMode == "distilled")
        && generation?.dfrEnabled != true && generation?.singleStageEnabled != true
        && (generation?.referenceFamily == nil || generation?.referenceFamily == "ordinary"))
  }
}

public struct GenerationLoRAEntry: Codable, Equatable, Identifiable {
  public var recipeRequirements: String? {
    guard case .nativeFile(_, "moss_sft_4step_v2", _, _) = binding else { return nil }
    return "Requires MOSS SFT, strength 1, CFG 1, shift 5, Euler and the release schedule. Adding it sets 4 steps; you can edit the preset to use 4–8 steps. Removing it restores the previous standard recipe."
  }
  public var reference: GenerationLoRARef
  public var name: String
  public var compatibleFamilies: [String]
  public var compatibleVariants: [String]?
  public var compatibleModelIDs: [String]?
  public var availability: GenerationLibraryAvailability
  public var binding: ResolvedGenerationLoRA?
  public var detail: String?
  public var id: String { reference.id }
  public init(reference: GenerationLoRARef, name: String, compatibleFamilies: [String],
    compatibleVariants: [String]? = nil, compatibleModelIDs: [String]? = nil,
    availability: GenerationLibraryAvailability, binding: ResolvedGenerationLoRA? = nil, detail: String? = nil) {
    self.reference = reference; self.name = name; self.compatibleFamilies = compatibleFamilies
    self.compatibleVariants = compatibleVariants; self.compatibleModelIDs = compatibleModelIDs
    self.availability = availability; self.binding = binding; self.detail = detail
  }
  public func supports(_ model: GenerationModelEntry) -> Bool {
    if case .nativeFile(_, let profile, _, _) = binding, let profile, LTXLibraryAdapterKind(rawValue: profile) != nil { return false }
    guard model.supportsLoRAs, compatibleFamilies.contains(model.reference.family),
      compatibleVariants == nil || compatibleVariants!.contains(model.reference.variant ?? ""),
      compatibleModelIDs == nil || compatibleModelIDs!.contains(model.reference.id) else { return false }
    switch (binding, model.binding) {
    case (.drawThings(let connection, _), .drawThings(let target, _)): return connection == target
    case (.nativeFile, .drawThings), (.drawThings, _): return false
    case (.nativeFile, _): return true
    default: return false
    }
  }
}

public struct GenerationLibrarySnapshot: Codable, Equatable {
  public var modelEntries: [GenerationModelEntry]
  public var loraEntries: [GenerationLoRAEntry]
  public init(models: [GenerationModelEntry] = [], loras: [GenerationLoRAEntry] = []) {
    modelEntries = models; loraEntries = loras
  }
  public func models(for context: GenerationLibraryContext) -> [GenerationModelEntry] {
    modelEntries.filter { $0.supports(context) }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  public func loras(for context: GenerationLibraryContext, model: GenerationModelRef) -> [GenerationLoRAEntry] {
    guard let entry = modelEntries.first(where: { $0.reference == model }), entry.supports(context) else { return [] }
    return loraEntries.filter { $0.supports(entry) }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  public func resolve(_ reference: GenerationModelRef, in context: GenerationLibraryContext) throws -> ResolvedGenerationModel {
    guard let entry = modelEntries.first(where: { $0.id == reference.id }) else {
      throw StudioError.invalid("This model is not linked. Choose it in Models & LoRAs.")
    }
    guard entry.reference == reference else { throw StudioError.invalid("The model identity or precision changed. Review its current library entry.") }
    guard entry.supports(context) else { throw StudioError.invalid("This model does not support the current task or inputs. Your inputs are preserved.") }
    guard entry.availability.canResolve, let binding = entry.binding else {
      throw StudioError.invalid(entry.availability.detail ?? "This model is not installed. Review Models & LoRAs.")
    }
    return binding
  }
  public func resolve(_ reference: GenerationLoRARef, for model: GenerationModelRef,
    in context: GenerationLibraryContext) throws -> ResolvedGenerationLoRA {
    _ = try resolve(model, in: context)
    guard let entry = loras(for: context, model: model).first(where: { $0.reference == reference }),
      entry.availability.canResolve, let binding = entry.binding else {
      throw StudioError.invalid("The LoRA is missing, changed or incompatible with this model. Review Models & LoRAs.")
    }
    return binding
  }
  public func specializedAdapter(_ reference: GenerationLoRARef, kind: LTXLibraryAdapterKind) throws -> String {
    guard let entry = loraEntries.first(where: { $0.reference == reference }), entry.availability == .installed,
      ["ltx25", "ltx23"].contains(reference.family),
      case .nativeFile(let path, let profile, _, _) = entry.binding, profile == kind.rawValue else {
      throw StudioError.invalid("Relink the selected specialized LTX adapter in Models & LoRAs.")
    }
    return path
  }
}

enum GenerationLibraryLimits {
  static func identifier(_ value: String) -> Bool {
    !value.isEmpty && value.utf8.count <= 512 && !value.hasPrefix("/") && !value.utf8.contains(0)
  }
}
