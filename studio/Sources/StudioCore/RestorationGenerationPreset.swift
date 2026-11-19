import Foundation
import InferenceContracts

public enum RestorationPresetTool: String, Codable, CaseIterable { case flashVSR, h3Regenerate, relayVSR, reCaVSR }
public struct RestorationPresetState: Equatable {
  public var tool: RestorationPresetTool
  public var model: GenerationModelRef?
  public var flash = FlashVSRSettings()
  public var h3 = H3RegenerateSettings()
  public var relayScale: NativeRelayVSRScale = .x2
  public var reCa = try! ReCaVSRSettings()
  public var sourceFPS: Double?
  public init(tool: RestorationPresetTool) { self.tool = tool }
}

public enum RestorationPresetRecipe {
  public static let flashKeys: Set<String> = ["backend", "variant", "decoder", "seed", "temporalCompression", "temporalSigma", "colorFix", "waveletLevels", "continuity"]
  public static let h3Keys: Set<String> = ["decoder", "encoderProfile", "schedule", "seed", "posteriorSeed", "latentUpscaleMode"]
  public static let reCaKeys: Set<String> = ["scale", "seed", "precision", "colorFix"]
  public static func binding(_ model: GenerationModelRef, tool: RestorationPresetTool, workspace: GenerationWorkspaceKind = .restoration,
    library: GenerationLibrarySnapshot) throws -> [String: String] {
    guard case .restoration(let kind, let components) = try library.resolve(model, in: .init(workspace: workspace, task: workspace == .clip ? "video_upscale" : tool.rawValue)), kind == tool.rawValue else {
      throw StudioError.invalid("Choose the installed model components for \(tool.rawValue).")
    }
    return components
  }
  public static func capture(from state: RestorationPresetState, resolvedLibrary library: GenerationLibrarySnapshot,
    workspace: GenerationWorkspaceKind = .restoration) throws -> GenerationPresetSnapshot {
    guard let model = state.model else { throw StudioError.invalid("Choose an installed restoration model in Models & LoRAs.") }
    _ = try binding(model, tool: state.tool, workspace: workspace, library: library)
    try validate(state)
    let fields: [String: GenerationPresetValue]
    switch state.tool {
    case .flashVSR: fields = try GenerationPresetFields.capture(state.flash, keys: flashKeys)
    case .h3Regenerate: fields = try GenerationPresetFields.capture(state.h3, keys: h3Keys)
    case .relayVSR: fields = ["scale": .integer(Int64(state.relayScale.rawValue))]
    case .reCaVSR: fields = try GenerationPresetFields.capture(state.reCa, keys: reCaKeys)
    }
    let snapshot = GenerationPresetSnapshot(context: .init(workspace: workspace, task: state.tool.rawValue), model: model, adapters: [], fields: fields)
    try snapshot.validate()
    _ = try candidate(applying: snapshot, to: state, resolvedLibrary: library, workspace: workspace)
    return snapshot
  }
  public static func candidate(applying snapshot: GenerationPresetSnapshot, to state: RestorationPresetState,
    resolvedLibrary library: GenerationLibrarySnapshot, workspace: GenerationWorkspaceKind = .restoration) throws -> RestorationPresetState {
    try snapshot.validate()
    guard snapshot.context == .init(workspace: workspace, task: state.tool.rawValue), let model = snapshot.model,
      snapshot.adapters?.isEmpty ?? true else { throw StudioError.invalid("Choose a preset for the current restoration tool.") }
    let components = try binding(model, tool: state.tool, workspace: workspace, library: library)
    var result = state; result.model = model
    switch state.tool {
    case .flashVSR:
      result.flash = try GenerationPresetFields.applying(snapshot.fields, to: state.flash, allowed: flashKeys)
      guard components["variant"].map({ $0 == result.flash.variant.rawValue }) ?? true,
        components["decoder"].map({ $0 == result.flash.decoder.rawValue }) ?? true else { throw StudioError.invalid("The FlashVSR model does not match this variant and decoder recipe.") }
    case .h3Regenerate:
      result.h3 = try GenerationPresetFields.applying(snapshot.fields, to: state.h3, allowed: h3Keys)
      if let value = components["baseProfilePath"] { result.h3.baseProfilePath = value }
      if let value = components["modelLibraryRoot"] { result.h3.modelLibraryRoot = value }
      if let value = components["publisherEncoderPath"] { result.h3.publisherEncoderPath = value }
      if let value = components["fastDecoderPath"] { result.h3.fastDecoderPath = value }
    case .relayVSR:
      guard Set(snapshot.fields.keys).isSubset(of: ["scale"]) else { throw StudioError.invalid("Unsupported RelayVSR recipe fields.") }
      if let scale = snapshot.fields["scale"] { result.relayScale = try scale.decoded(NativeRelayVSRScale.self) }
      guard result.relayScale == state.relayScale else { throw StudioError.invalid("This preset changes the RelayVSR output size. Select the desired scale explicitly before applying its matching preset.") }
      guard components["scale"].map({ $0 == String(result.relayScale.rawValue) }) ?? true else { throw StudioError.invalid("Choose the matching RelayVSR scale checkpoint pair.") }
    case .reCaVSR:
      result.reCa = try GenerationPresetFields.applying(snapshot.fields, to: state.reCa, allowed: reCaKeys)
      guard result.reCa.scale == state.reCa.scale else { throw StudioError.invalid("This preset changes the ReCaVSR output size. Select the desired scale explicitly first.") }
      guard components["precision"].map({ $0 == result.reCa.precision.rawValue }) ?? true else { throw StudioError.invalid("Select the ReCaVSR checkpoint matching this precision.") }
    }
    try validate(result); return result
  }
  public static func validate(_ state: RestorationPresetState) throws {
    switch state.tool {
    case .flashVSR: try state.flash.validateExecution()
    case .h3Regenerate:
      try state.h3.validate()
      if let fps = state.sourceFPS, !fps.isFinite || abs(fps - 24) > 0.000001 { throw StudioError.invalid("H3 Regenerate requires a 24 FPS source. Its frame rate was preserved.") }
    case .relayVSR: break // The typed scale admits only the released x2/x4 pairs.
    case .reCaVSR: try state.reCa.validate()
    }
  }
}
