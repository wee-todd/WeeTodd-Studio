import Foundation

/// Portable post-generation steps. Runtime paths are supplied by Studio or the frozen job.
public struct VideoProcessingStep: Codable, Equatable, Sendable {
  public var id = "restore"
  public var kind = "flashvsr-turbo"
  public var input = "source"
  public var settings = FlashVSRSettings()
  public var pixelCQ: NativePixelCQSettings?
  public var h3Regenerate: H3RegenerateSettings?
  public var isNative: Bool { kind == "native-pixel-cq" }
  public var isH3Regenerate: Bool { kind == "h3-regenerate" }
  public var isFlashVSR: Bool { kind == "flashvsr-turbo" }
  public var label: String { isH3Regenerate ? "H3 Regenerate" : isNative ? "Native Pixel Spatial + CQ" : "FlashVSR+ " + (settings.variant == .turbo ? "Turbo" : "Base") }
  public init(settings: FlashVSRSettings = .init()) { self.settings = settings }
  public init(pixelCQ: NativePixelCQSettings) { kind = "native-pixel-cq"; self.pixelCQ = pixelCQ }
  public init(h3Regenerate: H3RegenerateSettings) { kind = "h3-regenerate"; self.h3Regenerate = h3Regenerate }
  private enum CodingKeys: String, CodingKey { case id, kind, input, settings, pixelCQ, h3Regenerate }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: ["id", "kind", "input", "settings", "pixelCQ", "h3Regenerate"])
    let fields = try decoder.container(keyedBy: CodingKeys.self)
    id = try fields.decode(String.self, forKey: .id)
    kind = try fields.decode(String.self, forKey: .kind)
    input = try fields.decode(String.self, forKey: .input)
    guard isNative || isH3Regenerate || isFlashVSR else { throw StudioError.invalid("Unknown video processing stage.") }
    if isH3Regenerate {
      guard !fields.contains(.settings), !fields.contains(.pixelCQ) else { throw StudioError.invalid("H3 steps cannot ignore another engine's settings.") }
      h3Regenerate = try fields.decode(H3RegenerateSettings.self, forKey: .h3Regenerate)
    } else if isNative {
      guard !fields.contains(.settings), !fields.contains(.h3Regenerate) else { throw StudioError.invalid("Native Pixel steps cannot ignore another engine's settings.") }
      pixelCQ = try fields.decode(NativePixelCQSettings.self, forKey: .pixelCQ)
    } else {
      guard !fields.contains(.pixelCQ), !fields.contains(.h3Regenerate) else { throw StudioError.invalid("FlashVSR steps cannot ignore another engine's settings.") }
      settings = try fields.decode(FlashVSRSettings.self, forKey: .settings)
    }
  }
  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(id, forKey: .id); try c.encode(kind, forKey: .kind); try c.encode(input, forKey: .input)
    if isH3Regenerate { try c.encode(h3Regenerate, forKey: .h3Regenerate) }
    else if isNative { try c.encode(pixelCQ, forKey: .pixelCQ) }
    else { try c.encode(settings, forKey: .settings) }
  }
}

public struct VideoProcessingRecipe: Codable, Equatable, Sendable {
  public var format = "weetodd-video-processing-v1"
  public var steps: [VideoProcessingStep]
  public init(settings: FlashVSRSettings = .init()) { steps = [.init(settings: settings)] }
  public init(pixelCQ: NativePixelCQSettings) { steps = [.init(pixelCQ: pixelCQ)] }
  public init(h3Regenerate: H3RegenerateSettings) { format = "weetodd-video-processing-v2"; steps = [.init(h3Regenerate: h3Regenerate)] }
  public init(steps: [VideoProcessingStep]) throws {
    format = "weetodd-video-processing-v2"; self.steps = steps; try validate()
  }
  public init(pixelCQ: NativePixelCQSettings, flashVSR: FlashVSRSettings) {
    var flash = VideoProcessingStep(settings: flashVSR); flash.id = "flash"; flash.input = "restore"
    steps = [.init(pixelCQ: pixelCQ), flash]
  }
  public var isNative: Bool { steps.count == 1 && steps[0].isNative }
  public var hasNative: Bool { steps.contains(where: \.isNative) }
  public var hasH3Regenerate: Bool { steps.contains(where: \.isH3Regenerate) }
  public var isSequence: Bool { format == "weetodd-video-processing-v2" }
  public var hasFlashVSR: Bool { steps.contains { $0.kind == "flashvsr-turbo" } }
  public var isChain: Bool { steps.count == 2 && steps[0].isNative && steps[1].kind == "flashvsr-turbo" }
  /// Keep the exact admitted prefix, including its ID, when Flash is appended to a saved native recipe.
  public var nativePrefix: VideoProcessingRecipe? {
    guard steps.first?.isNative == true else { return nil }
    var prefix = self; prefix.steps = [steps[0]]; return prefix
  }
  public var flashSuffix: VideoProcessingRecipe? {
    guard let flash = steps.last, flash.kind == "flashvsr-turbo" else { return nil }
    var suffix = self; var step = flash; step.input = "source"; suffix.steps = [step]; return suffix
  }
  private enum CodingKeys: String, CodingKey { case format, steps }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: ["format", "steps"])
    let fields = try decoder.container(keyedBy: CodingKeys.self)
    format = try fields.decode(String.self, forKey: .format)
    steps = try fields.decode([VideoProcessingStep].self, forKey: .steps)
    try validate()
  }
  public func validate() throws {
    guard (format == "weetodd-video-processing-v1" && (steps.count == 1 || isChain) && !hasH3Regenerate)
      || (isSequence && (1...8).contains(steps.count)) else {
      throw StudioError.invalid("Choose a supported processing recipe or a version 2 linear sequence of one to eight stages.")
    }
    var previous = "source", ids = Set<String>()
    for step in steps {
      guard (["flashvsr-turbo", "native-pixel-cq"].contains(step.kind) || isSequence && step.isH3Regenerate), !step.id.isEmpty, step.id.count <= 64,
        step.id != "source", ids.insert(step.id).inserted, step.input == previous,
        step.id.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45,95].contains($0) }),
        (step.isNative || step.isH3Regenerate || (isSequence || step.settings.variant == .turbo) && step.settings.preserveAudio) else {
        throw StudioError.invalid("Video-processing steps must reference the previous stage and preserve original source audio.")
      }
      if step.isH3Regenerate {
        guard let settings = step.h3Regenerate, step.pixelCQ == nil else { throw StudioError.invalid("H3 processing needs its own explicit settings.") }
        try settings.validate()
      } else if step.isNative {
        guard let settings = step.pixelCQ else { throw StudioError.invalid("Native processing needs explicit Pixel Spatial + CQ settings.") }
        try settings.validate()
      } else {
        guard step.pixelCQ == nil, step.h3Regenerate == nil else { throw StudioError.invalid("Unexpected settings from another engine on a FlashVSR step.") }
        try step.settings.validate()
      }
      previous = step.id
    }
  }
  public var fingerprint: String { get throws {
    try validate()
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    return NativeHeadlessJob.hash(try encoder.encode(self))
  } }
}

/// The original take remains independently selectable and is the source for retries.
public struct VideoProcessingLineage: Codable, Equatable, Sendable {
  public var source: FlashVSRFile
  public var sourceVersionID: UUID?
  public var receiptPath: String
  public var fingerprint: String
  public var recipe: VideoProcessingRecipe
  public var label: String {
    if recipe.isSequence { return recipe.steps.map(\.label).joined(separator: " → ") }
    guard let step = recipe.steps.last else { return "Upscale" }
    if step.isNative {
      guard let settings = step.pixelCQ else { return step.label }
      return String(format: "Native Pixel %.2f · CQ %.2f", settings.pixelStrength, settings.cqStrength)
    }
    return (recipe.isChain ? "Native Pixel + CQ → " : "") + "Turbo " + step.settings.decoder.label
  }
  public init(source: FlashVSRFile, sourceVersionID: UUID?, receiptPath: String,
    fingerprint: String, recipe: VideoProcessingRecipe) {
    self.source = source; self.sourceVersionID = sourceVersionID; self.receiptPath = receiptPath
    self.fingerprint = fingerprint; self.recipe = recipe
  }
}
