import CryptoKit
import Foundation

public struct ImageWorkspaceInput: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var path: String
  public var enabled = true
  public var strength = 1.0
  public var fit = "fit"
  public init(path: String) { self.path = path }
  public func canonical(role: String) throws -> [String: Any] {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    guard strength.isFinite, (0...1).contains(strength), ["fit", "fill"].contains(fit),
      let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber,
      size.intValue > 0, size.intValue <= 64 * 1024 * 1024 else {
      throw StudioError.invalid("Relink a valid image (up to 64 MiB) and use a reference strength from 0–100%.")
    }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return ["role": role, "path": url.path, "sha256": hash.finalize().map { String(format: "%02x", $0) }.joined(),
      "strength": role == "canvas" ? 1 : strength, "fit": fit]
  }
}

public struct DrawThingsImageDraft: Codable, Equatable {
  public var provider: ImageExecutionProvider?
  public var nativeImage: NativeImageSettings?
  public var providerSettings: [String: ImageProviderSettings]?
  public var destination: ImageAssetDestination
  public var name = "Generated image"
  public var profileID = ""
  public var modelID = ""
  public var prompt = ""
  public var negativePrompt = ""
  public var width = 512
  public var height = 512
  public var steps = 4
  public var seed = -1
  public var randomSeedEachGeneration: Bool {
    get { seed == -1 }
    set {
      if newValue { seed = -1 }
      else if seed == -1 { seed = 0 }
    }
  }
  public var guidance = 1.0
  public var strength = 1.0
  public var sampler: Int?
  public var shift: Double?
  public var canvas: ImageWorkspaceInput?
  public var moodboard: [ImageWorkspaceInput] = []
  public var loras: [DrawThingsLoRA] = []
  public var referenceSheet: ReferenceSheetContext?
  public var characterPanel: CharacterPanelPromptContext?
  public var characterSheetLoRAID: String?
  public var characterSheetModelPending: Bool?
  public var rippleReference: RippleImageContext?
  public var storageKey: String {
    destination.storageKey + (referenceSheet.map { ":reference:" + $0.subjectKey } ?? "")
      + (referenceSheet?.template == .characterSheet ? ":four-view" : "")
      + (rippleReference.map { ":ripple:" + $0.referenceID.uuidString } ?? "")
  }
  public init(destination: ImageAssetDestination) { self.destination = destination }
  /// Call only for an explicit picker edit, never while restoring/importing a draft.
  public mutating func selectConnection(_ id: String) {
    guard profileID != id else { return }
    profileID = id; modelID = ""
  }
  public mutating func selectModel(_ id: String, compatibleLoRAIDs: Set<String>?) {
    guard modelID != id else { return }
    modelID = id
    // Keep assignments for review; request preflight validates the new model's compatibility.
  }
  public var configuration: [String: Any] {
    var value: [String: Any] = ["width": width, "height": height, "steps": steps, "seed": seed,
      "guidanceScale": guidance, "strength": canvas?.enabled == true ? strength : 1]
    if let sampler { value["sampler"] = sampler }
    if let shift { value["shift"] = shift }
    return value
  }
  public var characterSheetRequestIssue: String? {
    guard referenceSheet?.template == .characterSheet else { return nil }
    guard referenceSheet?.characterDefinition != nil else {
      return "Map and review this legacy character description in Character Director before generating a new sheet."
    }
    guard executionProvider == .drawThings else {
      return "Four-panel character sheets require the Draw Things LoRA. Choose Draw Things or another reference template."
    }
    guard prompt.hasPrefix(ReferenceSheetTemplate.characterSheetPrefix) else {
      return "The character-sheet prompt must begin with the four-view turnaround instruction. Apply the template to restore it."
    }
    guard let id = characterSheetLoRAID,
      loras.contains(where: { $0.modelID == id && $0.isEnabled && $0.weight > 0 }) else {
      return "Choose and enable the installed four-panel LoRA for this character sheet."
    }
    return nil
  }
  public func request(id: String) throws -> [String: Any] {
    if let issue = managedCharacterPromptIssue ?? characterSheetRequestIssue { throw StudioError.invalid(issue) }
    guard executionProvider == .drawThings else { throw StudioError.invalid("Use the native image request for Local MLX.") }
    guard imageInputIssue == nil, strength.isFinite, (0...1).contains(strength) else {
      throw StudioError.invalid("Use up to eight mood-board images and a generation strength from 0–100%.")
    }
    var inputs: [[String: Any]] = []
    if let canvas, canvas.enabled { inputs.append(try canvas.canonical(role: "canvas")) }
    for item in moodboard where item.enabled {
      guard item.strength.isFinite, (0...1).contains(item.strength) else {
        throw StudioError.invalid("Use a reference strength from 0–100%.")
      }
      if item.strength > 0 { inputs.append(try item.canonical(role: "moodboard")) }
    }
    let activeLoRAs = loras.filter(\.isEnabled)
    for lora in activeLoRAs { try lora.validate() }
    return ["schema": "weetodd-drawthings-request-v1", "requestID": id, "operation": "image",
      "profileID": profileID, "modelID": modelID, "prompt": prompt, "negativePrompt": negativePrompt,
      "configuration": configuration, "inputs": inputs,
      "loras": activeLoRAs.map { ["modelID": $0.modelID, "weight": $0.weight] }, "billingPolicy": "freeOnly"]
  }
}
