import Foundation

public struct MOSSImageSettings: Codable, Equatable {
  public static let sftModelID = "OpenMOSS-Team/MOSS-Image-SFT"
  public static let baseModelID = "OpenMOSS-Team/MOSS-Image-Base"
  public var transformerDirectory = ""
  public var sharedDirectory = ""
  public var libraryDirectory = ""
  public var variant = "sft"
  public var weightCacheGB = 0
  public var quantizationBits: Int?
  public var loras: [LoRAMember]?
  public var stockRecipe: MOSSRecipeSnapshot?
  public var regionalLayout: Bool?
  public var sampler = "euler"
  public var schedule = "release"
  public var skipLayers = [22]
  public var skipLayerExpression: String?
  public var skipStart = 0.0, skipStop = 1.0
  public init() {}
  public mutating func selectVariant(_ value: String) throws {
    guard ["base", "sft"].contains(value) else { throw StudioError.invalid("Choose MOSS Base or SFT.") }
    variant = value; transformerDirectory = ""
  }

  public func resolvedSkipLayers() throws -> [Int] {
    guard let expression = skipLayerExpression else { return skipLayers }
    if expression.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [] }
    let fields = expression.split(separator: ",", omittingEmptySubsequences: false)
    let values = fields.compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    guard values.count == fields.count, values.allSatisfy({ (0..<42).contains($0) }), Set(values).count == values.count else {
      throw StudioError.invalid("Negative-branch skip layers must be distinct layer numbers from 0 through 41, separated by commas.")
    }
    return values
  }
}

public struct MOSSRecipeSnapshot: Codable, Equatable {
  public var steps: Int
  public var guidance, shift, skipStart, skipStop: Double
  public var sampler, schedule: String
  public var skipLayers: [Int]
  public var skipLayerExpression: String?
  public var negativePrompt: String?
  public init(steps: Int = 50, guidance: Double = 3, shift: Double = 5, sampler: String = "euler", schedule: String = "release",
    skipLayers: [Int] = [22], skipStart: Double = 0, skipStop: Double = 1, negativePrompt: String = "") {
    self.steps = steps; self.guidance = guidance; self.shift = shift; self.sampler = sampler; self.schedule = schedule
    self.skipLayers = skipLayers; self.skipStart = skipStart; self.skipStop = skipStop; self.negativePrompt = negativePrompt
  }
  public init(draft: DrawThingsImageDraft) {
    let settings = draft.mossImage ?? .init()
    steps = draft.steps; guidance = draft.guidance; shift = draft.shift ?? 5
    sampler = settings.sampler; schedule = settings.schedule; skipLayers = settings.skipLayers
    skipStart = settings.skipStart; skipStop = settings.skipStop; skipLayerExpression = settings.skipLayerExpression
    negativePrompt = draft.negativePrompt
  }
  public func apply(to draft: inout DrawThingsImageDraft) {
    draft.steps = steps; draft.guidance = guidance; draft.shift = shift
    if let negativePrompt { draft.negativePrompt = negativePrompt }
    draft.mossImage?.sampler = sampler; draft.mossImage?.schedule = schedule
    draft.mossImage?.skipLayers = skipLayers; draft.mossImage?.skipLayerExpression = skipLayerExpression
    draft.mossImage?.skipStart = skipStart; draft.mossImage?.skipStop = skipStop
  }
}

public extension DrawThingsImageDraft {
  mutating func applyFaithfulMOSSReconstructionRecipe() {
    guard executionProvider == .nativeMOSS else { return }
    var settings = mossImage ?? .init()
    if settings.variant != "sft" || settings.quantizationBits != nil { settings.transformerDirectory = "" }
    settings.variant = "sft"; settings.quantizationBits = nil; settings.stockRecipe = nil
    if settings.loras != nil { for i in settings.loras!.indices { settings.loras![i].enabled = false } }
    settings.sampler = "euler"; settings.schedule = "release"
    settings.skipLayers = [22]; settings.skipLayerExpression = nil; settings.skipStart = 0; settings.skipStop = 1
    mossImage = settings; modelID = MOSSImageSettings.sftModelID
    steps = 8; guidance = 1; shift = 5; negativePrompt = ""
  }
}
