import Foundation

public enum ImageExecutionProvider: String, Codable, CaseIterable, Identifiable {
  case drawThings, nativeMLX
  public var id: String { rawValue }
  public var label: String { self == .nativeMLX ? "Local MLX" : "Draw Things" }
}

public struct NativeImageSettings: Codable, Equatable {
  public var manifestPath = ""
  public var referenceResolution = 1024
  public var memoryMode = "automatic"
  public var livePreview = true
  public init() {}
}

/// Only sampling settings are restored on backend switches. Media IDs and prompts remain shared.
public struct ImageProviderSettings: Codable, Equatable {
  public var modelID: String
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidance: Double
  public var strength: Double
  public var sampler: Int?
  public var shift: Double?
  public var negativePrompt: String
  public var loras: [DrawThingsLoRA]
}

public struct ActiveImageInput {
  public let input: ImageWorkspaceInput
  public let role: String
  public let index: Int
}

extension DrawThingsImageDraft {
  public var executionProvider: ImageExecutionProvider { provider ?? .drawThings }
  public var activeImageInputs: [ActiveImageInput] {
    var values: [(ImageWorkspaceInput, String)] = []
    if let canvas, canvas.enabled { values.append((canvas, "canvas")) }
    values += moodboard.filter { $0.enabled && $0.strength > 0 }.map { ($0, "moodboard") }
    return values.enumerated().map { ActiveImageInput(input: $0.element.0, role: $0.element.1, index: $0.offset + 1) }
  }
  public var imageInputIssue: String? {
    let inputs = activeImageInputs
    if executionProvider == .nativeMLX {
      return inputs.count > 10 ? "Qwen supports 10 active images including the canvas. Disable an input to continue." : nil
    }
    return inputs.filter { $0.role == "moodboard" }.count > 8
      ? "This Draw Things transport supports eight active mood-board images. Your other images are preserved." : nil
  }
  public mutating func selectProvider(_ value: ImageExecutionProvider) {
    guard executionProvider != value else { return }
    var saved = providerSettings ?? [:]
    saved[executionProvider.rawValue] = ImageProviderSettings(modelID: modelID, width: width,
      height: height, steps: steps, guidance: guidance, strength: strength, sampler: sampler,
      shift: shift, negativePrompt: negativePrompt, loras: loras)
    if let settings = saved[value.rawValue] {
      modelID = settings.modelID; width = settings.width; height = settings.height
      steps = settings.steps; guidance = settings.guidance; strength = settings.strength
      sampler = settings.sampler; shift = settings.shift; negativePrompt = settings.negativePrompt
      loras = settings.loras
    } else {
      modelID = value == .nativeMLX ? "Qwen/Qwen-Image-2.1" : ""
      width = value == .nativeMLX ? 1024 : 512; height = width
      steps = value == .nativeMLX ? 40 : 4; guidance = 1; strength = 1
      sampler = nil; shift = nil; negativePrompt = ""; loras = []
    }
    providerSettings = saved; provider = value
    if value == .nativeMLX, nativeImage == nil { nativeImage = NativeImageSettings() }
  }
  /// File hashing belongs on a background worker operating on a captured value-type draft.
  public func nativeRequest(id: String) throws -> [String: Any] {
    guard executionProvider == .nativeMLX else { throw StudioError.invalid("Choose Local MLX first.") }
    if let issue = imageInputIssue { throw StudioError.invalid(issue) }
    let settings = nativeImage ?? NativeImageSettings()
    guard !settings.manifestPath.isEmpty else { throw StudioError.invalid("Choose the local Qwen model manifest.") }
    guard guidance == 1, sampler == nil, shift == nil, negativePrompt.isEmpty,
      !loras.contains(where: \.isEnabled), strength == 1 else {
      throw StudioError.invalid("Qwen uses Euler with automatic shift and CFG 1. Unsupported overrides must be cleared.")
    }
    let inputs = try activeImageInputs.map { value -> [String: Any] in
      let canonical = try value.input.canonical(role: value.role)
      return ["id": value.input.id.uuidString, "role": value.role,
        "imageIndex": value.index, "path": canonical["path"]!, "sha256": canonical["sha256"]!]
    }
    return ["schema": "weetodd-native-image-request-v1", "requestID": id,
      "engine": "qwen_image21", "modelID": modelID, "modelManifestPath": settings.manifestPath,
      "prompt": prompt, "inputs": inputs, "configuration": [
        "width": width, "height": height, "steps": steps, "seed": seed, "guidance": guidance,
        "scheduler": "flow_euler_dynamic", "referenceResolution": settings.referenceResolution,
        "memoryMode": settings.memoryMode, "livePreview": settings.livePreview]]
  }
}
