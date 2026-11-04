import Foundation

/// Reusable execution settings. Installed component paths belong to model setup;
/// authored prompts, media, seeds, canvas and timing belong to the clip.
public struct LTX25GenerationConfig: Codable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var selection: GenerationSelection
  /// Older recipes can carry adapter/VAE policies outside GenerationSelection.
  /// Retain their internal binding when saving; new configs resolve automatically.
  public var componentBindingID: String?
  public var bindingSupportedTasks: [String]?
  public var bindingRequiresIndependent: Bool?
  public var bindingReferenceFamily: String?
  public var isBuiltIn: Bool { id.hasPrefix("builtin-ltx25-") }
  public var isDev: Bool { selection.ltx25Guidance != nil }

  public static let distilled = builtIn("distilled", name: "Distilled · Recommended")
  public static let dev = builtIn("dev", name: "Dev · Recommended", mode: .guided)
  public static let devHQ = builtIn("dev-hq", name: "Dev · HQ", mode: .guidedHQ)
  public static var builtIns: [Self] { [distilled, dev, devHQ] }

  private static func builtIn(_ suffix: String, name: String, mode: LTX25GuidanceMode? = nil) -> Self {
    var selection = GenerationSelection(preset: .balanced)
    selection.ltx25Guidance = mode.map { LTX25GuidanceSettings(mode: $0, experimentalEnabled: true) }
    return Self(id: "builtin-ltx25-" + suffix, name: name, selection: selection)
  }

  private init(id: String, name: String, selection: GenerationSelection) {
    self.id = id; self.name = name; self.selection = selection
  }

  public init(name: String, selection: GenerationSelection, componentBindingID: String? = nil,
    bindingSupportedTasks: [String]? = nil, bindingRequiresIndependent: Bool? = nil,
    bindingReferenceFamily: String? = nil) throws {
    self.init(id: UUID().uuidString, name: name.trimmingCharacters(in: .whitespacesAndNewlines),
      selection: selection)
    self.selection.ltx25ConfigID = nil; self.selection.ltx25ConfigName = nil
    self.componentBindingID = componentBindingID
    self.bindingSupportedTasks = bindingSupportedTasks
    self.bindingRequiresIndependent = bindingRequiresIndependent
    self.bindingReferenceFamily = bindingReferenceFamily
    try validate()
  }

  public func validate() throws {
    guard !id.isEmpty, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      name.count <= 120, selection.h3SamplingMethod == nil, selection.h3Reference == nil,
      selection.h3Joint == nil, selection.h3MotionFidelity == nil,
      selection.h3VideoDecodePrecision == nil, selection.h3AttentionPolicy == nil,
      selection.h3TransformerWeightCacheGB == nil,
      componentBindingID == nil || componentBindingID?.hasPrefix("/") == true else {
      throw StudioError.invalid("Give the LTX 2.5 config a name of 1–120 characters and use LTX settings.")
    }
    if let dfr = selection.ltx25DFR { try dfr.validate() }
    guard selection.shift == nil, selection.projectionBackend == nil, selection.transformerBackend == nil,
      selection.memoryPolicy == nil || selection.memoryPolicy == "recipe",
      selection.ltx25Guidance != nil || (selection.steps == nil && selection.cfg == nil && selection.refinementSteps == nil),
      selection.ltx25MovieUpscale == nil else {
      throw StudioError.invalid("This config includes overrides the Swift LTX 2.5 renderer cannot execute.")
    }
    if selection.ltx25Guidance != nil {
      _ = try NativeLTXPreparation.guidedConfig([:], selection: selection, negativePrompt: "")
    }
    if let single = selection.ltx25SingleStage {
      guard single.experimentalEnabled, single.method == .cfgpp || single.negativeSchedule == .full else {
        throw StudioError.invalid("Single-stage configs need experimental execution and a compatible negative schedule.")
      }
    }
    if let keyframes = selection.ltx25Keyframes {
      guard keyframes.experimentalEnabled, (0...8).contains(keyframes.generatedCount) else {
        throw StudioError.invalid("Keyframe configs need experimental execution and zero to eight generated slots.")
      }
    }
    if componentBindingID != nil {
      guard let tasks = bindingSupportedTasks, !tasks.isEmpty,
        Set(tasks).isSubset(of: ["t2v", "i2v", "fflf", "a2v", "extension", "ref2va", "control"]),
        bindingRequiresIndependent != nil else {
        throw StudioError.invalid("The saved legacy config needs its component binding's compatibility contract.")
      }
    }
    guard !(selection.ltx25Guidance != nil &&
      (selection.ltx25DFR?.enabled == true || selection.ltx25SingleStage != nil)),
      !(selection.ltx25SingleStage != nil && selection.ltx25AutomaticDuration != nil),
      !(selection.ltx25DFR?.enabled == true && (selection.ltx25AutomaticDuration != nil
        || selection.ltx25Keyframes != nil || selection.ltx25SingleStage != nil)),
      [selection.cfg, selection.shift, selection.ltx25Guidance?.audioCFG,
       selection.ltx25Guidance?.stgScale, selection.ltx25Guidance?.videoRescale,
       selection.ltx25Guidance?.audioRescale, selection.ltx25Guidance?.modalityScale]
        .compactMap({ $0 }).allSatisfy(\.isFinite) else {
      throw StudioError.invalid("This config combines incompatible LTX sampling settings.")
    }
  }

  public func appliedSelection(task: String) -> GenerationSelection {
    var value = selection
    value.task = task; value.ltx25ConfigID = id; value.ltx25ConfigName = name
    return value
  }

  public func supports(_ clip: Clip, in project: StudioProject? = nil) -> Bool {
    let independent = clip.continuityMode == "independent" && clip.extensionDirection.isEmpty
      && project?.isContinuousSceneMember(clip) != true
    if let tasks = bindingSupportedTasks,
      !GenerationPresentation.canUseProfile(supportedTasks: tasks, for: clip) { return false }
    if bindingRequiresIndependent == true && !independent { return false }
    if clip.inferredTask == "control", let expected = bindingReferenceFamily,
      let actual = NativeLTXPreparation.controlFamily(for: clip), actual != expected { return false }
    if isDev && (!(["t2v", "i2v", "fflf", "a2v"].contains(clip.inferredTask))
      || !independent) { return false }
    if selection.ltx25DFR?.enabled == true || selection.ltx25AutomaticDuration != nil {
      return ["t2v", "i2v", "fflf"].contains(clip.inferredTask)
        && independent
    }
    if (selection.ltx25SingleStage != nil || selection.ltx25Keyframes != nil) && !independent { return false }
    if selection.ltx25SingleStage?.method == .cfgpp && clip.inferredTask == "a2v" { return false }
    return clip.inferredTask != "video_upscale"
  }

  public func matches(_ value: GenerationSelection?) -> Bool {
    guard var value else { return false }
    var expected = selection
    value.task = "t2v"; expected.task = "t2v"
    value.ltx25ConfigID = nil; value.ltx25ConfigName = nil
    expected.ltx25ConfigID = nil; expected.ltx25ConfigName = nil
    return value == expected
  }
}

extension Clip {
  public mutating func applyLTX25Config(_ config: LTX25GenerationConfig, in project: StudioProject? = nil) throws {
    guard engine == .ltx25 else { throw StudioError.invalid("Select an LTX 2.5 clip to apply this config.") }
    try config.validate()
    guard config.supports(self, in: project) else {
      throw StudioError.invalid("\(config.name) does not support this clip's inputs or continuity. Choose a compatible config; your shot has been preserved.")
    }
    generationSelection = config.appliedSelection(task: inferredTask)
    profileID = config.componentBindingID ?? "auto"
  }
}
