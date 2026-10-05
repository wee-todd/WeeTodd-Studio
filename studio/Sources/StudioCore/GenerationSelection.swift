import Foundation

public enum GenerationProvider: String, CaseIterable, Identifiable {
  case local, drawThings
  public var id: String { rawValue }
  public var label: String { self == .local ? "WeeTodd (local)" : "Draw Things" }
}

extension Clip {
  public var reviewUsesSourceVideo: Bool {
    [.ltx23, .ltx25].contains(engine) && continuityMode == "motion"
  }
  public var reviewUsesContinuityFrame: Bool {
    engine != .drawThings && engine != .movie && continuityMode == "frame"
  }
  public var reviewAttachments: [Attachment] {
    attachments.filter {
      !($0.role == .first && reviewUsesContinuityFrame)
        && ($0.role != .lora || (engine != .drawThings && $0.isEnabled))
    }
  }
  public var reviewMediaCount: Int {
    return reviewAttachments.filter { $0.role != .lora }.count
      + (reviewUsesContinuityFrame || reviewUsesSourceVideo ? 1 : 0)
  }
  public var reviewLoRACount: Int {
    engine == .drawThings ? drawThings?.loras.filter(\.isEnabled).count ?? 0
      : reviewAttachments.filter { $0.role == .lora }.count
  }

  public mutating func selectAutomaticModelComponents() {
    let task = inferredTask
    profileID = "auto"
    if generationSelection == nil { generationSelection = GenerationSelection(task: task) }
  }

  public var generationProvider: GenerationProvider { engine == .drawThings ? .drawThings : .local }

  public mutating func selectGenerationProvider(_ provider: GenerationProvider) {
    guard provider != generationProvider else { return }
    if provider == .drawThings {
      selectGenerationEngine(.drawThings)
    } else {
      selectLocalModel([Engine.h3, .ltx23, .ltx25].contains(lastLocalEngine ?? .movie)
        ? lastLocalEngine! : .ltx25)
    }
  }

  public mutating func selectLocalModel(_ model: Engine) {
    guard [.h3, .ltx23, .ltx25].contains(model), model != engine else { return }
    let hasSavedSettings = savedNativeGenerations?[model.rawValue] != nil
    var media = self
    media.generationSelection = nil
    let task = media.inferredTask
    selectGenerationEngine(model)
    if !hasSavedSettings { generationSelection = GenerationSelection(task: task) }
  }

  public mutating func selectGenerationPreset(_ preset: GenerationPreset) {
    generationSelection = GenerationSelection(task: inferredTask, preset: preset)
  }

  public mutating func selectGenerationTask(_ task: String) {
    if generationSelection == nil { generationSelection = GenerationSelection() }
    generationSelection?.task = task
  }

  public mutating func selectGenerationEngine(_ target: Engine) {
    guard target != engine else { return }
    if [.h3, .ltx23, .ltx25].contains(engine) {
      lastLocalEngine = engine
      if savedNativeGenerations == nil { savedNativeGenerations = [:] }
      savedNativeGenerations?[engine.rawValue] = SavedNativeGenerationSettings(
        profileID: profileID, selection: generationSelection)
    }
    engine = target
    if let saved = savedNativeGenerations?[target.rawValue] {
      profileID = saved.profileID
      generationSelection = saved.selection
    } else {
      profileID = "auto"
      generationSelection = target == .drawThings || target == .movie ? nil : GenerationSelection()
    }
  }
}

/// Inactive native settings survive a backend round trip without affecting the current render.
public struct SavedNativeGenerationSettings: Codable, Equatable {
  public var profileID: String
  public var selection: GenerationSelection?
}

public enum GenerationPreset: String, Codable, CaseIterable, Identifiable {
  case balanced, speed, lowMemory, custom
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .balanced: return "Balanced"
    case .speed: return "Speed"
    case .lowMemory: return "Low memory"
    case .custom: return "Custom"
    }
  }
}

public enum NativeH3SamplingMethod: String, Codable, CaseIterable, Identifiable {
  case euler, resMultistep = "res_multistep"
  public var id: String { rawValue }
  public var label: String { self == .euler ? "Euler" : "Res multistep (experimental)" }
}

public enum LTX25GuidanceMode: String, Codable, CaseIterable, Identifiable {
  case guided, guidedHQ = "guided_hq"
  public var id: String { rawValue }
  public var label: String { self == .guided ? "Production guided (experimental)" : "HQ guided (experimental)" }
}

/// Explicit experimental intent; absent on every existing distilled clip.
public struct LTX25GuidanceSettings: Codable, Equatable {
  public var mode: LTX25GuidanceMode
  public var experimentalEnabled: Bool
  public var audioCFG: Double?
  public var stgScale: Double?
  public var videoRescale: Double?
  public var audioRescale: Double?
  public var modalityScale: Double?
  public var stgBlocks: [Int]?
  /// Nil inherits profile timing; an empty array explicitly selects adaptive timing.
  public var sigmas: [Double]?
  public init(mode: LTX25GuidanceMode, experimentalEnabled: Bool = false) {
    self.mode = mode; self.experimentalEnabled = experimentalEnabled
  }
}

/// Explicit user intent. Nil on older clips preserves their exact recipe behavior.
public struct LTX25KeyframeSettings:Codable,Equatable {
  public var generatedCount:Int
  public var experimentalEnabled:Bool
  public init(generatedCount:Int=0,experimentalEnabled:Bool=false) {
    self.generatedCount=generatedCount;self.experimentalEnabled=experimentalEnabled
  }
}

public enum LTX25SingleStageMethod:String,Codable,CaseIterable,Identifiable {
  case euler,ancestral="euler_ancestral",cfgpp="euler_ancestral_cfg_pp"
  public var id:String { rawValue }
  public var label:String { switch self {
    case .euler:return "Euler"
    case .ancestral:return "Euler ancestral"
    case .cfgpp:return "Euler ancestral CFG++"
  } }
}
public enum LTX25NegativeSchedule:String,Codable,CaseIterable,Identifiable {
  case full,balanced,speed
  public var id:String { rawValue }
  public var label:String { rawValue.capitalized }
  public var evaluationCount:Int { self == .full ? 15 : self == .balanced ? 12 : 10 }
}
public struct LTX25SingleStageSettings:Codable,Equatable {
  public var method:LTX25SingleStageMethod
  public var negativeSchedule:LTX25NegativeSchedule
  public var experimentalEnabled:Bool
  public init(method:LTX25SingleStageMethod = .ancestral,negativeSchedule:LTX25NegativeSchedule = .full,experimentalEnabled:Bool=false) {
    self.method=method;self.negativeSchedule=negativeSchedule;self.experimentalEnabled=experimentalEnabled
  }
}

/// Explicit user intent. Nil on older clips preserves their exact recipe behavior.
public struct GenerationSelection: Codable, Equatable {
  public var task: String
  public var preset: GenerationPreset
  public var steps: Int?
  public var refinementSteps: Int?
  public var cfg: Double?
  public var shift: Double?
  public var memoryPolicy: String?
  public var projectionBackend: String?
  public var transformerBackend: String?
  public var h3SamplingMethod: NativeH3SamplingMethod?
  public var h3Reference:H3ReferenceSettings?
  public var h3Joint:H3JointSettings?
  public var h3MotionFidelity:H3MotionFidelitySettings?
  public var ltx25DiffusionVAE:LTX25DiffusionVAESettings?
  public var ltx25Guidance: LTX25GuidanceSettings?
  public var ltx25AutomaticDuration: LTX25AutomaticDurationSettings?
  public var ltx25Keyframes:LTX25KeyframeSettings?
  public var ltx25SingleStage:LTX25SingleStageSettings?
  public var ltx25MovieUpscale:LTX25MovieUpscaleSettings?
  public init(task: String = "t2v", preset: GenerationPreset = .balanced) {
    self.task = task
    self.preset = preset
  }
  public var isModified: Bool {
    steps != nil || refinementSteps != nil || cfg != nil || shift != nil
      || memoryPolicy != nil || projectionBackend != nil || transformerBackend != nil
      || ltx25DiffusionVAE != nil || h3SamplingMethod != nil || h3Reference != nil || h3Joint != nil || h3MotionFidelity != nil || ltx25Guidance != nil || ltx25AutomaticDuration != nil || ltx25Keyframes != nil || ltx25SingleStage != nil || ltx25MovieUpscale != nil
  }
  public mutating func resetOverrides() {
    steps = nil; refinementSteps = nil; cfg = nil; shift = nil
    memoryPolicy = nil; projectionBackend = nil; transformerBackend = nil
    h3Reference=nil;h3Joint=nil;h3MotionFidelity=nil
    h3SamplingMethod = nil; ltx25Guidance = nil; ltx25AutomaticDuration = nil
    ltx25DiffusionVAE=nil
    ltx25Keyframes = nil;ltx25SingleStage = nil;ltx25MovieUpscale = nil
  }
  public static func taskLabel(_ task: String) -> String {
    switch task {
    case "t2v": return "Text to video"
    case "i2v": return "Image to video"
    case "fflf": return "First and last frames"
    case "ref2va": return "Reference video"
    case "a2v": return "Audio-driven video"
    case "control": return "Controlled video"
    case "extension": return "Video extension"
    case "video_upscale": return "Source movie 2× upscale"
    default: return task
    }
  }
}

public struct GenerationControls: Codable, Equatable {
  public var evaluations: Int?
  public var refinementSteps: Int?
  public var cfg: Double?
  public var shift: Double?
  public var stepsEditable: Bool
  public var refinementStepsEditable: Bool
  public var cfgEditable: Bool
  public var shiftEditable: Bool
  public var stepsExplanation: String
  public var cfgExplanation: String
  public var shiftExplanation: String
}
public struct GenerationDescriptor: Codable, Equatable {
  public struct Preset: Codable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var description: String
  }
  public var supportedTasks: [String]
  public var controls: GenerationControls
  public var presets: [Preset]
  public var pipelineMode: String? = nil
  public var samplingMethod: String? = nil
  public var vdn: Bool? = nil
  public var fasth3: Bool? = nil
}

public struct AccelerationSettings: Codable, Equatable {
  public var h3MemoryPolicy: String = "automatic"
  public var h3ProjectionBackend: String = "auto"
  public init() {}
}

extension GenerationSelection {
  /// Asset records can be relinked or reclassified without changing the clip's attachments.
  public static func assetFingerprint(for clip: Clip, assets: [MediaAsset]) -> String {
    let referenced = Set(clip.attachments.map(\.assetID))
    let records = assets.filter { referenced.contains($0.id) }
      .sorted { $0.id.uuidString < $1.id.uuidString }
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return ((try? encoder.encode(records)) ?? Data()).base64EncodedString()
  }
}

extension AccelerationSettings {
  public static func memoryPolicyLabel(_ policy: String) -> String {
    switch policy {
    case "automatic": return "Automatic"
    case "recipe": return "Recipe default"
    case "paged": return "Paged · lower memory"
    case "pagedNormal": return "Paged · larger workspace"
    case "resident": return "Resident · experimental high RAM"
    default: return policy
    }
  }
}
