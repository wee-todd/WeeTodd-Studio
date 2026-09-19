import Foundation

/// A user-facing purpose and its exact renderer input. Prepared media remains a visible asset.
public struct ReferenceAction: Identifiable, Equatable {
  public let id: String
  public let label: String
  public let detail: String
  public let role: MediaRole
  public var controlType: String = "canny_edges"
  public var preparation: String? = nil
}

extension Clip {
  public var usesDrawThingsImageReferences: Bool {
    engine == .drawThings && drawThings?.modelFamily.lowercased() == "minimaxh3"
      && drawThings?.modelModifier == "ref2va"
  }

  public func referenceActions(for asset: MediaAsset) -> [ReferenceAction] {
    guard engine != .movie else { return [] }
    if engine == .drawThings {
      return asset.kind == .image && usesDrawThingsImageReferences ? [ReferenceAction(
        id: "imageAppearance", label: "Image reference · H3", detail: "H3 Ref2VA uses up to nine still images. Movie and audio references require WeeTodd (local).",
        role: .reference)] : []
    }
    let ingredients = ReferenceAction(id: "ingredients", label: "Appearance · Ingredients sheet",
      detail: "Requires an Ingredients IC-LoRA. Describe the subjects in one sheet. Minimum 5 seconds at 24 fps."
        + (engine == .ltx23 ? " LTX 2.3 uses 768 × 448." : ""),
      role: engine == .ltx23 ? .reference : .control, controlType: "ingredients_reference_sheet")
    switch asset.kind {
    case .image:
      if engine == .ltx23 { return [ingredients] }
      let appearance = ReferenceAction(id: "imageAppearance",
        label: engine == .ltx25 ? "Appearance · MSR image" : "Appearance · image reference",
        detail: engine == .ltx25
          ? "Requires the dedicated MSR adapter. Use 1–5 images and describe each subject, object, clothing or background."
          : "H3 Ref2VA uses this image as a visual reference; it is not a fixed first frame.", role: .reference)
      return engine == .ltx25 ? [appearance, ingredients] : [appearance]
    case .video, .sequence:
      // Imported sequences are movie files in the current Studio media contract.
      return [
        ReferenceAction(id: "movieAppearance", label: engine == .h3
          ? "Appearance / story · movie reference" : "Appearance / story · make reference sheet",
          detail: engine == .h3 ? "H3 Ref2VA receives the movie as reference context. Describe what to retain in the prompt."
            : "Samples six frames into a visible Ingredients sheet. This preserves visual context, not the movie's timing or soundtrack. Describe the story in the prompt.",
          role: engine == .ltx25 ? .control : .reference,
          controlType: "ingredients_reference_sheet", preparation: engine == .h3 ? nil : "sheet"),
        ReferenceAction(id: "movieMotion", label: "Motion / composition · make edge guide",
          detail: engine == .h3 ? "Creates a Canny edge movie for the H3 Fun ControlNet. Requires a control model."
            : "Creates a Canny edge movie for the Union IC-LoRA. Keeps motion and composition; appearance comes from the prompt.",
          role: .control, preparation: "canny"),
        ReferenceAction(id: "preprocessedControl", label: "Already prepared control guide…",
          detail: "Use only an existing edge, depth, pose or motion guide. Select the matching type and dedicated adapter.", role: .control)
      ]
    case .audio:
      let driver = ReferenceAction(id: "audioDriver", label: "Audio · drive video",
        detail: "Uses this audio as the timed driver. LTX preserves the supplied audio; this is not a voice/style reference.", role: .audioDriver)
      return engine == .h3 ? [driver, ReferenceAction(id: "audioReference", label: "Audio · reference sound / voice",
        detail: "H3 Ref2VA also needs an image or movie reference. Describe which sound or voice to retain.", role: .reference)] : [driver]
    default: return []
    }
  }

  public func canAssignMedia(_ asset: MediaAsset, role: MediaRole) -> Bool {
    if engine == .drawThings { return canAssignDrawThingsInput(asset, role: role) }
    guard engine != .movie else { return false }
    if role == .lora { return asset.kind == .lora }
    if [.first, .last, .keyframe].contains(role) { return asset.kind == .image }
    return referenceActions(for: asset).contains { $0.role == role && $0.preparation == nil }
  }

  public mutating func attachReference(_ asset: MediaAsset, action: ReferenceAction) throws {
    var source = asset
    if let preparation = action.preparation {
      guard asset.kind == (preparation == "sheet" ? .image : .video) else {
        throw StudioError.invalid("Prepare the reference media before attaching it.")
      }
      source.kind = .video
    }
    guard referenceActions(for: source).contains(action) else {
      throw StudioError.invalid("This reference purpose is not supported by the selected model.")
    }
    var attachment = Attachment(assetID: asset.id, role: action.role)
    attachment.controlType = action.controlType
    attachment.description = asset.name
    if action.role == .audioDriver { attachments.removeAll { $0.role == .audioDriver } }
    attachments.append(attachment)
    // Selecting a purpose selects its task in one undoable change. Other inputs stay visible.
    let task = action.role == .audioDriver ? "a2v" : action.role == .control ? "control" : "ref2va"
    if generationSelection == nil { generationSelection = GenerationSelection(task: task,
      preset: engine == .drawThings ? .custom : .balanced) }
    else { generationSelection?.task = task }
    if engine == .drawThings { generationSelection = GenerationSelection(task: task, preset: .custom) }
  }
}
