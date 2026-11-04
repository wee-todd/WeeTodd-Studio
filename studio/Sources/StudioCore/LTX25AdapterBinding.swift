import Foundation

/// A selected specialized component travels with the clip/config, independently
/// of shared model bindings. The native worker retains its full admission checks.
public struct LTX25AdapterBinding: Codable, Equatable {
  public var kind: LTXLibraryAdapterKind
  public var path: String
  public var supportsVoiceReferences: Bool
  public init(kind: LTXLibraryAdapterKind, path: String, supportsVoiceReferences: Bool = false) {
    self.kind = kind; self.path = path; self.supportsVoiceReferences = supportsVoiceReferences
  }
  public func validate() throws {
    guard path.hasPrefix("/"), !path.utf8.contains(0), path.utf8.count <= 4096,
      URL(fileURLWithPath: path).pathExtension.lowercased() == "safetensors",
      !supportsVoiceReferences || kind == .msr else {
      throw StudioError.invalid("Choose a compatible local LTX adapter file.")
    }
  }
  public var task: String? { switch kind {
    case .msr: "ref2va"
    case .ingredients, .union, .motionTrack, .crossview: "control"
    case .dfr, .distilledRefinement: nil
  } }
  public var referenceFamily: String { switch kind {
    case .msr: "msr"
    case .ingredients: "ingredients"
    case .union: "union"
    case .motionTrack: "motion_track"
    case .crossview: "crossview_warp"
    case .dfr, .distilledRefinement: "ordinary"
  } }
  public var supportedTasks: [String] {
    if let task { return [task] }
    return kind == .dfr ? ["t2v", "i2v", "fflf"] : ["t2v", "i2v", "fflf", "a2v"]
  }
  public func supports(_ action: ReferenceAction) -> Bool {
    switch action.role {
    case .reference:
      return kind == .msr && (action.id == "imageAppearance"
        || action.id == "msrVoice" && supportsVoiceReferences)
    case .control:
      if kind == .ingredients { return action.controlType == "ingredients_reference_sheet" }
      if kind == .union && action.id == "movieMotion" { return true }
      if kind == .union { return ["canny_edges", "depth_map", "pose_skeleton"].contains(action.controlType) }
      if kind == .motionTrack { return action.controlType == "motion_track" && action.preparation == nil }
      return kind == .crossview && action.controlType == "crossview_warp"
        && ["warp", "source"].contains(action.referenceRole ?? "") && action.preparation == nil
    case .audioDriver: return supportedTasks.contains("a2v")
    case .first, .last, .keyframe: return supportedTasks.contains("fflf")
    case .lora: return false
    }
  }
  public var preparedGuideActions: [ReferenceAction] {
    func action(_ id: String, _ label: String, _ type: String, _ detail: String,
      role: String? = nil) -> ReferenceAction {
      ReferenceAction(id: id, label: label, detail: detail, role: .control, controlType: type, referenceRole: role)
    }
    switch kind {
    case .union:
      return [("canny_edges", "Canny edges"), ("depth_map", "depth"), ("pose_skeleton", "pose")].map { type, label in
        action("prepared_" + type, "Prepared " + label + " guide", type, "Use an already prepared \(label) control movie.")
      }
    case .motionTrack:
      return [action("prepared_motion_track", "Prepared Motion Track guide", "motion_track",
        "Use a movie with drawn or tracked trajectories.")]
    case .crossview:
      return [action("prepared_crossview_warp", "Prepared CrossView warp", "crossview_warp",
        "Use the prepared depth-warped movie.", role: "warp"),
        action("prepared_crossview_source", "Original CrossView source", "crossview_warp",
          "Use the matching original source movie and audio.", role: "source")]
    default: return []
    }
  }
  public struct InputRequirement: Identifiable {
    public var id: String
    public var label: String
    public var detail: String
    public var satisfied: Bool
  }
  public func inputRequirements(for clip: Clip, assets: [MediaAsset]) -> [InputRequirement] {
    func matches(_ attachment: Attachment, kinds: [AssetKind]) -> Bool {
      assets.contains { $0.id == attachment.assetID && kinds.contains($0.kind) }
    }
    func guide(_ type: String, role: String? = nil) -> Bool {
      clip.attachments.contains { $0.role == .control && $0.controlType == type
        && (role == nil || $0.referenceRole == role) && matches($0, kinds: [.video, .sequence]) }
    }
    switch kind {
    case .msr:
      return [InputRequirement(id: "appearance", label: "Appearance references",
        detail: supportsVoiceReferences ? "Add described images. Voice references can also be bound to image 1 or 2."
          : "Add described reference images. This adapter supports images.",
        satisfied: clip.attachments.contains { $0.role == .reference
          && !$0.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          && matches($0, kinds: [.image]) })]
    case .ingredients:
      return [InputRequirement(id: "ingredients", label: "Described reference sheet",
        detail: "Add a sheet and describe its subjects and roles.", satisfied: clip.attachments.contains {
          $0.role == .control && $0.controlType == "ingredients_reference_sheet"
            && matches($0, kinds: [.image]) && !$0.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })]
    case .union:
      return [InputRequirement(id: "union", label: "Canny, depth or pose guide",
        detail: "Add one prepared control movie.", satisfied: ["canny_edges", "depth_map", "pose_skeleton"].contains { guide($0) })]
    case .motionTrack:
      return [InputRequirement(id: "motion_track", label: "Motion Track guide",
        detail: "Add a movie with drawn or tracked trajectories.", satisfied: guide("motion_track"))]
    case .crossview:
      return [InputRequirement(id: "crossview_warp", label: "Prepared CrossView warp",
        detail: "Add the depth-warped guide movie.", satisfied: guide("crossview_warp", role: "warp")),
        InputRequirement(id: "crossview_source", label: "Original source movie",
        detail: "Add the matching original movie and its audio.", satisfied: guide("crossview_warp", role: "source"))]
    case .dfr, .distilledRefinement: return []
    }
  }
}

extension Clip {
  public var selectedLTX25AdapterInputTasks: [String]? {
    guard engine == .ltx25, let binding = generationSelection?.ltx25AdapterBinding else { return nil }
    var tasks = binding.supportedTasks
    if [.union, .motionTrack, .crossview].contains(binding.kind),
      generationSelection?.ltx25SingleStage?.experimentalEnabled == true { tasks.append("fflf") }
    return tasks
  }
}
