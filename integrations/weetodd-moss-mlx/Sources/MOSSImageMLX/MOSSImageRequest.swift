import Foundation
import ImageLayout

public struct MOSSError: Error, LocalizedError, Sendable {
  public let message: String
  public init(_ message: String) { self.message = message }
  public var errorDescription: String? { message }
}

public struct MOSSGenerationSettings: Codable, Equatable, Sendable {
  public var width = 1024, height = 1024, steps = 50
  public var seed: UInt64 = 42
  public var guidance: Float = 3, shift: Float = 5
  public var negativePrompt = ""
  public var skipLayers = [22]
  public var skipStart: Float = 0, skipStop: Float = 1
  public var sampler = "euler", schedule = "release"
  public var weightCacheGB = 0
  public var quantizationBits: Int?
  public var profile: Bool?
  public var regionalLayout: Bool?
  public init() {}
  public func validate() throws {
    try ImageLayoutGeometry.validateCanvas(width: width, height: height)
    guard guidance.isFinite, guidance >= 0, guidance <= 20, shift.isFinite, shift > 0,
      skipStart.isFinite, skipStop.isFinite, 0 <= skipStart, skipStart <= skipStop, skipStop <= 1,
      Set(skipLayers).count == skipLayers.count, skipLayers.allSatisfy({ (0..<42).contains($0) }),
      [0, 2, 4, 8, 16].contains(weightCacheGB), quantizationBits == nil || [4, 8].contains(quantizationBits!),
      ["euler", "er_sde"].contains(sampler), ["release", "simple"].contains(schedule) else {
      throw MOSSError("Unsupported or nonfinite MOSS sampling, layer, precision or memory setting.")
    }
    _ = try schedule == "simple" ? MOSSSchedule.simple(steps: steps, shift: shift) : MOSSSchedule.sigmas(steps: steps, shift: shift)
  }
}

public struct MOSSLoRADescriptor: Codable, Equatable, Sendable {
  public var path: String
  public var strength: Float
  public var enabled: Bool
  public var profile: String
  public init(path: String, strength: Float = 1, enabled: Bool = true, profile: String = "ordinary") {
    self.path = path; self.strength = strength; self.enabled = enabled; self.profile = profile
  }
}

public struct MOSSImageRequest: Codable, Sendable {
  public var schema = "weetodd-moss-image-request-v1"
  public var requestID = UUID().uuidString
  public var modelVariant = "sft"
  public var transformerDirectory: String
  public var sharedDirectory: String
  public var prompt: String
  public var layout: ImageLayoutDraft?
  public var sourceImage: MOSSReferenceImage?
  public var configuration: MOSSGenerationSettings
  public var loras: [MOSSLoRADescriptor] = []
  public var expectedFingerprint: String?
  public var outputPath: String?
  public init(transformerDirectory: String, sharedDirectory: String, prompt: String,
    configuration: MOSSGenerationSettings = .init(), layout: ImageLayoutDraft? = nil) {
    self.transformerDirectory = transformerDirectory; self.sharedDirectory = sharedDirectory
    self.prompt = prompt; self.configuration = configuration; self.layout = layout
  }
  public func effectivePrompt() throws -> String {
    guard schema == "weetodd-moss-image-request-v1", UUID(uuidString: requestID) != nil,
      ["base", "sft"].contains(modelVariant), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= 1024 * 1024 else { throw MOSSError("Invalid MOSS request or prompt.") }
    try configuration.validate()
    if configuration.regionalLayout == true {
      guard layout != nil, sourceImage == nil else { throw MOSSError("Subject-box control requires an editable layout and generation from noise.") }
    }
    if let sourceImage {
      try sourceImage.validate()
      guard modelVariant == "sft", configuration.quantizationBits == nil,
        !loras.contains(where: { $0.enabled && $0.strength != 0 }) else {
        throw MOSSError("Source reconstruction currently requires stock SFT, official BF16 and no enabled adapters. Apply the faithful reconstruction recipe or adjust these settings explicitly.")
      }
    }
    if let layout {
      guard layout.width == configuration.width, layout.height == configuration.height,
        layout.summary.caption == prompt else { throw MOSSError("Layout dimensions or scene description differ from the generation request.") }
      return try MOSSLayoutCompiler.compile(layout).effectivePrompt
    }
    return prompt
  }
}
