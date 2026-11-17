import Foundation
import ImageLayout

public enum NativeMOSSImagePreparation {
  public static func layoutForSavedTake(admitted: ImageLayoutDraft, draft: DrawThingsImageDraft, admittedSource: MOSSLayoutSource? = nil) throws -> ImageLayoutDraft {
    guard var authored = draft.mossLayout else { throw StudioError.invalid("The generated layout has no matching editor draft.") }
    let guide = authored.guideImagePath, opacity = authored.guideOpacity
    let source = authored.sourceReconstruction
    if let source { try source.validate() }
    let enabledSource = source?.enabled == true ? source : nil
    let sameSource: Bool
    if let expected = enabledSource, let admittedSource {
      sameSource = admittedSource.enabled && expected.path == admittedSource.path && expected.sha256 == admittedSource.sha256
        && Float(expected.noiseLevel) == Float(admittedSource.noiseLevel)
    } else { sameSource = enabledSource == nil && admittedSource == nil }
    guard sameSource else { throw StudioError.invalid("The generated source differs from the prepared editor source.") }
    authored.width = draft.width; authored.height = draft.height; authored.summary.caption = draft.prompt
    authored.guideImagePath = nil; authored.guideOpacity = nil
    authored.sourceReconstruction = nil
    guard authored == admitted else { throw StudioError.invalid("The generated layout differs from the prepared editor draft.") }
    var saved = admitted; saved.guideImagePath = guide; saved.guideOpacity = opacity
    saved.sourceReconstruction = admittedSource ?? source
    return saved
  }
  public static func request(draft: DrawThingsImageDraft, requestID: UUID, seed: UInt32) throws -> Data {
    guard draft.executionProvider == .nativeMOSS else { throw StudioError.invalid("Choose Local MOSS first.") }
    if let issue = draft.managedCharacterPromptIssue ?? draft.characterSheetRequestIssue ?? draft.imageInputIssue { throw StudioError.invalid(issue) }
    let settings = draft.mossImage ?? MOSSImageSettings()
    let adapters = settings.loras ?? []
    guard adapters.count <= 8, Set(adapters.map(\.fileKey)).count == adapters.count else { throw StudioError.invalid("Use at most eight distinct MOSS adapters.") }
    for adapter in adapters { try adapter.validateForMOSS() }
    let modelID = settings.variant == "sft" ? MOSSImageSettings.sftModelID : MOSSImageSettings.baseModelID
    guard ["sft", "base"].contains(settings.variant), draft.modelID == modelID,
      !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      draft.guidance.isFinite, (0...20).contains(draft.guidance), draft.strength == 1,
      !draft.loras.contains(where: \.isEnabled), draft.sampler == nil,
      settings.transformerDirectory.hasPrefix("/"), settings.sharedDirectory.hasPrefix("/"),
      !settings.transformerDirectory.utf8.contains(0), !settings.sharedDirectory.utf8.contains(0),
      (1...1000).contains(draft.steps), [0, 2, 4, 8, 16].contains(settings.weightCacheGB) else {
      throw StudioError.invalid("Link MOSS Base/SFT components and use supported native sampling settings. Clear incompatible backend overrides.")
    }
    try ImageLayoutGeometry.validateCanvas(width: draft.width, height: draft.height)
    var layout = draft.mossLayout
    let reference = layout?.sourceReconstruction
    if settings.regionalLayout == true, layout == nil || reference?.enabled == true {
      throw StudioError.invalid("Subject-box control requires a layout and generation from noise. Disable legacy source reconstruction first.")
    }
    if let reference { try reference.validate() }
    if reference?.enabled == true {
      guard settings.variant == "sft", settings.quantizationBits == nil,
        !adapters.contains(where: { $0.isEnabled && $0.strength != 0 }) else {
        throw StudioError.invalid("Source reconstruction requires stock SFT, official BF16 and no enabled adapters. Apply Faithful reconstruction recipe or adjust the settings explicitly.")
      }
    }
    if layout != nil {
      layout!.width = draft.width; layout!.height = draft.height; layout!.summary.caption = draft.prompt
      layout!.guideImagePath = nil; layout!.guideOpacity = nil
      layout!.sourceReconstruction = nil
      _ = try MOSSLayoutCompiler.compile(layout!)
    }
    var root: [String: Any] = ["schema": "weetodd-moss-image-request-v1", "requestID": requestID.uuidString,
      "modelVariant": settings.variant, "transformerDirectory": settings.transformerDirectory,
      "sharedDirectory": settings.sharedDirectory, "prompt": draft.prompt,
      "loras": adapters.map { ["path": $0.asset.path, "strength": $0.strength, "enabled": $0.isEnabled,
        "profile": $0.asset.loraProfile == "moss_sft_4step_v2" ? "sft_4step_v2" : "ordinary"] as [String: Any] },
      "configuration": ["width": draft.width, "height": draft.height, "steps": draft.steps, "seed": seed,
        "guidance": draft.guidance, "shift": draft.shift ?? 5, "negativePrompt": draft.negativePrompt,
        "skipLayers": try settings.resolvedSkipLayers(), "skipStart": settings.skipStart, "skipStop": settings.skipStop,
        "sampler": settings.sampler, "schedule": settings.schedule, "weightCacheGB": settings.weightCacheGB]]
    if let layout { root["layout"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(layout)) }
    if let reference, reference.enabled {
      root["sourceImage"] = ["path": reference.path, "sha256": reference.sha256, "noiseLevel": reference.noiseLevel]
    }
    if let bits = settings.quantizationBits, var configuration = root["configuration"] as? [String: Any] {
      configuration["quantizationBits"] = bits; root["configuration"] = configuration
    }
    if let regional = settings.regionalLayout, var configuration = root["configuration"] as? [String: Any] {
      configuration["regionalLayout"] = regional; root["configuration"] = configuration
    }
    return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes])
  }
}
