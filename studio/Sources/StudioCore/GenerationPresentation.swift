import Foundation

/// Presentation detail is independent of the saved generation settings.
public enum GenerationSettingsDetail: String, CaseIterable, Identifiable {
  case basic, advanced, all
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .basic: return "Basic"
    case .advanced: return "Advanced"
    case .all: return "All"
    }
  }
  public var showsBasic: Bool { self != .advanced }
  public var showsAdvanced: Bool { self != .basic }
}

/// Names creative purposes without changing their renderer task or transport role.
public enum GenerationPresentation {
  public static func methodLabel(for clip: Clip, referenceFamily: String? = nil) -> String {
    if clip.engine == .movie { return "Imported media" }
    if clip.reviewUsesSourceVideo { return "Video extension" }
    let task = clip.inferredTask
    if [.ltx23, .ltx25].contains(clip.engine), ["control", "ref2va"].contains(task),
      referenceFamily == "ingredients" || clip.attachments.contains(where: {
        [.control, .reference].contains($0.role) && $0.controlType == "ingredients_reference_sheet"
      }) {
      return "Reference sheet · Ingredients"
    }
    switch task {
    case "t2v", "t2va": return "Prompt video"
    case "i2v", "fflf":
      if clip.attachments.contains(where: { $0.role == .last }) { return "First and last frames" }
      if clip.attachments.contains(where: { $0.role == .keyframe }) { return "Timed keyframes" }
      if task == "i2v" || clip.attachments.contains(where: { $0.role == .first }) { return "Animate image" }
      return "First and last frames"
    case "ref2va":
      return clip.engine == .ltx25 || referenceFamily == "msr"
        ? "Appearance references · MSR" : "Appearance references"
    case "a2v": return "Audio-driven video"
    case "control": return "Motion / scene guide"
    case "extension": return "Video extension"
    case "video_upscale": return "Source movie 2× upscale"
    default: return GenerationSelection.taskLabel(task)
    }
  }

  public static func inputLabel(for attachment: Attachment, engine: Engine) -> String {
    switch attachment.role {
    case .first: return "First frame"
    case .last: return "Last frame"
    case .keyframe:
      let seconds = String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), attachment.time)
      return "Keyframe · \(seconds) s"
    case .audioDriver: return "Audio driver"
    case .lora: return "LoRA"
    case .control:
      switch attachment.controlType {
      case "ingredients_reference_sheet": return "Reference sheet"
      case "crossview_warp":
        return attachment.referenceRole == "source" ? "Original source and audio" : "CrossView guide"
      case "motion_track": return "Motion guide"
      default: return "Control guide"
      }
    case .reference:
      if [.ltx23, .ltx25].contains(engine), attachment.controlType == "ingredients_reference_sheet" {
        return "Reference sheet"
      }
      if engine == .ltx25 {
        if attachment.msrAudioReferenceID != nil || attachment.audioSourceStart != nil
          || attachment.audioSourceDuration != nil { return "Voice reference" }
        switch attachment.referenceRole {
        case "background": return "Background reference"
        case "object": return "Object reference"
        case "clothing": return "Clothing reference"
        default: break
        }
      }
      return "Appearance reference"
    }
  }

  public static func canUseProfile(supportedTasks: [String], for clip: Clip) -> Bool {
    let task = clip.inferredTask
    if supportedTasks.contains(task) { return true }
    // Native first/last adapters also execute first-frame-only shots. Keep remote
    // task capability exact and never admit a last frame to an image-only profile.
    guard [.h3, .ltx23, .ltx25].contains(clip.engine), clip.extensionDirection.isEmpty,
      clip.attachments.contains(where: { $0.role == .first }),
      !clip.attachments.contains(where: { [.last, .keyframe].contains($0.role) }) else { return false }
    return task == "i2v" && supportedTasks.contains("fflf")
      || task == "fflf" && supportedTasks.contains("i2v")
  }
}

extension Clip {
  /// Selecting an installed recipe keeps the shot's authored overrides and inputs.
  public mutating func selectGenerationProfile(_ id: String) {
    profileID = id
  }

  /// An explicit defaults action restores inheritance from the selected recipe.
  public mutating func useSelectedModelDefaults() {
    generationSelection = GenerationSelection(task: inferredTask, preset: .custom)
    h3PagingCacheGB = nil
  }
}
