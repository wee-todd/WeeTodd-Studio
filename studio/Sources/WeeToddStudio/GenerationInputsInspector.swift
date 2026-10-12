import Foundation
import StudioCore
import SwiftUI

/// Named input purposes retain the renderer's exact reference contract.
enum GenerationInputPurpose: String, CaseIterable, Identifiable {
  case appearance, ingredients, audioDriver, motionGuide
  var id: String { rawValue }
  var label: String {
    switch self {
    case .appearance: return "Appearance references"
    case .ingredients: return "Reference sheet / Ingredients"
    case .audioDriver: return "Audio driver"
    case .motionGuide: return "Motion / scene guide"
    }
  }
  var importHint: String {
    switch self {
    case .appearance: return "Import reference images, movies or audio in Media & Assets."
    case .ingredients: return "Import a reference sheet or a movie to prepare a sheet."
    case .audioDriver: return "Import the audio that will drive this shot."
    case .motionGuide: return "Import a movie or an already prepared control guide."
    }
  }
  func actions(for asset: MediaAsset, clip: Clip, supportedTasks: [String]? = nil,
               referenceFamily: String? = nil) -> [ReferenceAction] {
    clip.referenceActions(for: asset).filter { action in
      let matches: Bool
      switch self {
      case .appearance:
        matches = ["imageAppearance", "audioReference", "msrVoice"].contains(action.id)
          || (action.id == "movieAppearance" && action.preparation == nil)
      case .ingredients:
        matches = action.id == "ingredients" || (action.id == "movieAppearance" && action.preparation == "sheet")
      case .audioDriver: matches = action.id == "audioDriver"
      case .motionGuide: matches = ["movieMotion", "preprocessedControl"].contains(action.id)
      }
      let task = action.role == .audioDriver ? "a2v" : action.role == .control ? "control" : "ref2va"
      return matches && (supportedTasks == nil || supportedTasks!.contains(task))
        && familySupports(action, engine: clip.engine, referenceFamily: referenceFamily)
    }
  }

  private func familySupports(_ action: ReferenceAction, engine: Engine, referenceFamily: String?) -> Bool {
    guard [.ltx23, .ltx25].contains(engine), let family = Self.knownFamily(referenceFamily) else { return true }
    if action.role == .audioDriver { return family == "ordinary" }
    if action.role == .reference {
      return engine == .ltx23 ? family == "ingredients" && action.controlType == "ingredients_reference_sheet"
        : family == "msr" && ["imageAppearance", "msrVoice"].contains(action.id)
    }
    return Self.controlTypeSupported(action.controlType, engine: engine, referenceFamily: family)
  }

  static func knownFamily(_ family: String?) -> String? {
    guard let family = family?.lowercased(),
      ["ordinary", "ingredients", "msr", "union", "motion_track", "crossview_warp", "crossview_ingredients"].contains(family) else { return nil }
    return family
  }

  static func controlTypeSupported(_ type: String, engine: Engine, referenceFamily: String?) -> Bool {
    let allowed: [String]
    switch engine {
    case .h3: allowed = ["canny_edges", "depth_map", "pose_skeleton", "hed_edges", "mlsd_lines"]
    case .ltx23: allowed = ["canny_edges", "depth_map", "pose_skeleton", "motion_track", "ingredients_reference_sheet"]
    case .ltx25: allowed = ["canny_edges", "depth_map", "pose_skeleton", "motion_track", "ingredients_reference_sheet", "crossview_warp"]
    case .drawThings, .movie: return false
    }
    guard allowed.contains(type) else { return false }
    guard [.ltx23, .ltx25].contains(engine), let family = knownFamily(referenceFamily) else { return true }
    switch family {
    case "ingredients": return type == "ingredients_reference_sheet"
    case "union": return ["canny_edges", "depth_map", "pose_skeleton"].contains(type)
    case "motion_track": return type == "motion_track"
    case "crossview_warp": return engine == .ltx25 && type == "crossview_warp"
    case "crossview_ingredients": return engine == .ltx25 && ["crossview_warp", "ingredients_reference_sheet"].contains(type)
    default: return false
    }
  }

  static func canReplace(_ attachment: Attachment, original: MediaAsset?, with asset: MediaAsset,
                         clip: Clip, supportedTasks: [String]? = nil, referenceFamily: String? = nil) -> Bool {
    if [.first, .last].contains(attachment.role) {
      return asset.kind == .image && clip.supportsEndpoint(attachment.role, supportedTasks: supportedTasks)
    }
    if attachment.role == .keyframe {
      return clip.engine != .drawThings && asset.kind == .image
        && (supportedTasks == nil || supportedTasks!.contains("fflf"))
    }
    if attachment.role == .control && attachment.controlType != "ingredients_reference_sheet" {
      return [.video, .sequence].contains(asset.kind)
        && (supportedTasks == nil || supportedTasks!.contains("control"))
        && controlTypeSupported(attachment.controlType, engine: clip.engine, referenceFamily: referenceFamily)
        && (attachment.controlType != "crossview_warp" || ["warp", "source"].contains(attachment.referenceRole ?? ""))
    }
    if let original, original.kind != asset.kind { return false }
    return allCases.flatMap { $0.actions(for: asset, clip: clip, supportedTasks: supportedTasks, referenceFamily: referenceFamily) }
      .contains { $0.preparation == nil && $0.role == attachment.role && $0.controlType == attachment.controlType }
  }
}

enum GenerationControlPurpose: String, CaseIterable, Identifiable {
  case canny, depth, pose, hed, mlsd, motionTracks, crossViewWarp, crossViewSource
  var id: String { rawValue }
  var label: String {
    switch self {
    case .canny: return "Prepared Canny edges"
    case .depth: return "Prepared depth guide"
    case .pose: return "Prepared pose guide"
    case .hed: return "Prepared HED edges"
    case .mlsd: return "Prepared MLSD lines"
    case .motionTracks: return "Motion tracks"
    case .crossViewWarp: return "CrossView warped guide"
    case .crossViewSource: return "CrossView original movie + audio"
    }
  }
  var controlType: String {
    switch self {
    case .canny: return "canny_edges"
    case .depth: return "depth_map"
    case .pose: return "pose_skeleton"
    case .hed: return "hed_edges"
    case .mlsd: return "mlsd_lines"
    case .motionTracks: return "motion_track"
    case .crossViewWarp, .crossViewSource: return "crossview_warp"
    }
  }
  var referenceRole: String? {
    switch self {
    case .crossViewWarp: return "warp"
    case .crossViewSource: return "source"
    default: return nil
    }
  }
  var detail: String {
    switch self {
    case .motionTracks: return "Use an already prepared motion-track movie with the matching MotionTrack adapter."
    case .crossViewWarp: return "Use the warped camera guide, then add the original source movie and its audio."
    case .crossViewSource: return "Use the original source movie and its audio, together with a warped camera guide."
    default: return "Use an already prepared guide of this type with its compatible control model."
    }
  }
  static func available(engine: Engine, referenceFamily: String?, supportedTasks: [String]? = nil) -> [Self] {
    guard supportedTasks == nil || supportedTasks!.contains("control") else { return [] }
    // A known H3 control engine has its fixed guide set. Unknown LTX recipes keep
    // the existing explicit type editor until preparation identifies the adapter.
    guard engine == .h3 || GenerationInputPurpose.knownFamily(referenceFamily) != nil else { return [] }
    return allCases.filter {
      GenerationInputPurpose.controlTypeSupported($0.controlType, engine: engine, referenceFamily: referenceFamily)
    }
  }
}

extension StudioStore {
  func generationInputReferenceFamily(for clip: Clip) -> String? {
    if let resolved = generationDescriptions[clip.id],
      resolved["studioEngine"] as? String == clip.engine.rawValue,
      resolved["studioTask"] as? String == clip.inferredTask,
      resolved["studioProfile"] as? String == clip.profileID,
      resolved["studioInput"] as? String == generationRequestKey(for: clip),
      let generation = resolved["generation"] as? [String: Any],
      let family = generation["referenceFamily"] as? String { return family }
    guard clip.profileID != "auto" else { return nil }
    return profiles.first { $0.id == clip.profileID && $0.engine == clip.engine.rawValue }?.generation?.referenceFamily
  }

  func generationInputSupportedTasks(for clip: Clip) -> [String]? {
    guard clip.profileID != "auto" else { return nil }
    return profiles.first { $0.id == clip.profileID && $0.engine == clip.engine.rawValue }?.generation?.supportedTasks
  }

  func generationInputSupportsKeyframes(for clip: Clip) -> Bool {
    guard [.h3, .ltx23, .ltx25].contains(clip.engine), clip.extensionDirection.isEmpty else { return false }
    let tasks = endpointTasks(for: clip)
    return tasks == nil || tasks!.contains("fflf")
  }

  @discardableResult func useGenerationKeyframe(_ asset: MediaAsset) -> Bool {
    guard !operationBusy, let clip = selectedClip else { return false }
    guard asset.kind == .image, allAssets.contains(asset), clip.canAssignMedia(asset, role: .keyframe),
      generationInputSupportsKeyframes(for: clip) else {
      error = "Choose an imported image and a model or recipe that supports timed keyframes."
      return false
    }
    let time = selectedClipPlayhead ?? 0
    let ordinaryTask = ["t2v", "i2v", "fflf"].contains(clip.inferredTask)
    editClip { current in
      current.attachments.append(Attachment(assetID: asset.id, role: .keyframe, time: time))
      if ordinaryTask {
        if current.generationSelection == nil {
          current.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
        } else { current.generationSelection?.task = "fflf" }
      }
    }
    selectedAssetID = asset.id
    notice = "Timed keyframe: " + asset.name
    return true
  }

  @discardableResult func usePreparedGenerationGuide(_ asset: MediaAsset, purpose: GenerationControlPurpose) -> Bool {
    guard !operationBusy, let clip = selectedClip else { return false }
    guard [.video, .sequence].contains(asset.kind), allAssets.contains(asset) else {
      error = "Choose an imported movie for this prepared guide."
      return false
    }
    let family = generationInputReferenceFamily(for: clip)
    guard GenerationControlPurpose.available(engine: clip.engine, referenceFamily: family,
      supportedTasks: generationInputSupportedTasks(for: clip)).contains(purpose) else {
      error = "The selected model or recipe does not support \(purpose.label.lowercased()). Prepare the clip to verify its adapter."
      return false
    }
    var attachment = Attachment(assetID: asset.id, role: .control)
    attachment.controlType = purpose.controlType
    attachment.referenceRole = purpose.referenceRole
    attachment.description = asset.name
    editClip { current in
      current.attachments.append(attachment)
      if current.generationSelection == nil {
        current.generationSelection = GenerationSelection(task: "control", preset: .balanced)
      } else { current.generationSelection?.task = "control" }
    }
    selectedAssetID = asset.id
    notice = purpose.label + ": " + asset.name
    return true
  }
}

struct GenerationInputsInspector: View {
  var clip: Clip
  var showAdvanced: Bool = false
  @EnvironmentObject var store: StudioStore

  private var supportedTasks: [String]? { store.generationInputSupportedTasks(for: clip) }
  private var referenceFamily: String? { store.generationInputReferenceFamily(for: clip) }
  private var preparedGuides: [GenerationControlPurpose] {
    GenerationControlPurpose.available(engine: clip.engine, referenceFamily: referenceFamily,
      supportedTasks: supportedTasks)
  }
  private var supportsTimedKeyframe: Bool {
    store.generationInputSupportsKeyframes(for: clip)
  }
  private var inputs: [Attachment] { clip.attachments.filter { $0.role != .lora } }
  private var images: [MediaAsset] { store.allAssets.filter { $0.kind == .image } }

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        SmallLabel(text: "Inputs")
        Spacer()
        Menu("Add input") {
          endpointMenu(.first)
          endpointMenu(.last)
          if supportsTimedKeyframe { timedKeyframeMenu }
          ForEach(GenerationInputPurpose.allCases) { purpose in purposeMenu(purpose) }
          Divider()
          Button("Import media…") { store.chooseImports(scope: .clip) }
        }.controlSize(.small).disabled(store.operationBusy)
      }
      if inputs.isEmpty {
        Text("Start with a prompt, or add a first frame, appearance references, audio or a motion guide.")
          .font(.caption2).foregroundStyle(.secondary)
        if store.allAssets.isEmpty {
          Button("Import media…") { store.chooseImports(scope: .clip) }
            .font(.caption).disabled(store.operationBusy)
        }
      }
      ForEach(inputs) { attachment in inputCard(attachment) }
    }
  }

  @ViewBuilder private func endpointMenu(_ role: MediaRole) -> some View {
    Menu(role.label) {
      if !store.supportsEndpoint(role, for: clip) {
        Text("The selected model or recipe does not support \(role.label.lowercased()).")
        if clip.engine == .drawThings && role == .last {
          Text("Last frame requires H3 FL2VA.")
        }
      } else if images.isEmpty {
        Text("Import an image in Media & Assets.")
      } else {
        ForEach(images) { asset in
          Button(asset.name + (fileAvailable(asset) ? "" : " · Missing file")) {
            store.useAsset(asset, role: role)
          }.disabled(!fileAvailable(asset))
        }
      }
    }
  }

  @ViewBuilder private func purposeMenu(_ purpose: GenerationInputPurpose) -> some View {
    let choices = actionChoices(purpose)
    Menu(purpose.label) {
      if choices.isEmpty {
        if !purposeAvailable(purpose) {
          Text("The selected model or recipe does not support this input purpose.")
        } else { Text(purpose.importHint) }
      } else {
        ForEach(choices) { choice in
          Button(choice.asset.name + " · " + (choice.controlPurpose?.label ?? choice.action.label)
            + (fileAvailable(choice.asset) ? "" : " · Missing file")) {
            guard store.selectedClipID == clip.id else { return }
            if let purpose = choice.controlPurpose {
              store.usePreparedGenerationGuide(choice.asset, purpose: purpose)
            } else { Task { await store.useReference(choice.asset, action: choice.action) } }
          }.help(choice.controlPurpose?.detail ?? choice.action.detail).disabled(!fileAvailable(choice.asset))
        }
      }
    }
  }

  private func purposeAvailable(_ purpose: GenerationInputPurpose) -> Bool {
    if purpose == .motionGuide && !preparedGuides.isEmpty { return true }
    return [AssetKind.image, .video, .audio].contains { kind in
      !purpose.actions(for: MediaAsset(name: "", kind: kind), clip: clip, supportedTasks: supportedTasks,
        referenceFamily: referenceFamily).isEmpty
    }
  }

  private func actionChoices(_ purpose: GenerationInputPurpose) -> [InputActionChoice] {
    store.allAssets.flatMap { asset in
      var choices = purpose.actions(for: asset, clip: clip, supportedTasks: supportedTasks,
        referenceFamily: referenceFamily).filter {
          $0.id != "preprocessedControl" || preparedGuides.isEmpty
        }.map { InputActionChoice(asset: asset, action: $0) }
      if purpose == .motionGuide, [.video, .sequence].contains(asset.kind),
        let action = clip.referenceActions(for: asset).first(where: { $0.id == "preprocessedControl" }) {
        choices += preparedGuides.map { InputActionChoice(asset: asset, action: action, controlPurpose: $0) }
      }
      return choices
    }
  }

  @ViewBuilder private var timedKeyframeMenu: some View {
    Menu("Timed keyframe") {
      Text(store.selectedClipPlayhead == nil ? "Add at 0 seconds; set time in Input details." : "Add at the playhead; set time in Input details.")
      if images.isEmpty { Text("Import an image in Media & Assets.") }
      ForEach(images) { asset in
        Button(asset.name + (fileAvailable(asset) ? "" : " · Missing file")) {
          guard store.selectedClipID == clip.id else { return }
          store.useGenerationKeyframe(asset)
        }.disabled(!fileAvailable(asset))
      }
    }
  }

  @ViewBuilder private func inputCard(_ attachment: Attachment) -> some View {
    let asset = store.allAssets.first { $0.id == attachment.assetID }
    let supported = inputSupported(attachment, asset: asset)
    VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .top, spacing: 9) {
        inputThumbnail(asset)
        VStack(alignment: .leading, spacing: 3) {
          Text(inputLabel(attachment, asset: asset))
            .font(.caption.weight(.semibold))
          Text(asset?.name ?? "Missing asset").font(.caption2).foregroundStyle(.secondary).lineLimit(2)
          if let asset, !fileAvailable(asset) {
            Text("Missing file · relink or replace this input.").font(.caption2).foregroundStyle(.orange)
          } else if asset == nil {
            Text("The linked asset is unavailable. Replace or remove this input.").font(.caption2).foregroundStyle(.orange)
          }
        }
        Spacer(minLength: 0)
        Button { remove(attachment) } label: { Image(systemName: "xmark") }
          .buttonStyle(.plain).help("Remove input").accessibilityLabel("Remove \(attachment.role.label)")
      }
      if !supported {
        Text("This input is unsupported by the selected model or recipe. Choose a compatible model, or replace or remove the input.")
          .font(.caption2).foregroundStyle(.orange)
      }
      if [.ltx23, .ltx25].contains(clip.engine), [.reference, .control].contains(attachment.role),
        GenerationInputPurpose.knownFamily(referenceFamily) == nil {
        Text("Prepare the clip to verify its adapter. Set prepared guide types in Input details.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      if let asset, let action = clip.referenceActions(for: asset).first(where: {
        $0.role == attachment.role && $0.controlType == attachment.controlType
      }) {
        Text(action.detail).font(.caption2).foregroundStyle(.secondary)
      }
      if attachment.role == .reference || attachment.controlType == "ingredients_reference_sheet" {
        TextField("Describe this reference", text: Binding(get: {
          store.selectedClip?.attachments.first { $0.id == attachment.id }?.description ?? attachment.description
        }, set: { value in edit(attachment) { $0.description = value } }))
          .font(.caption).textFieldStyle(.roundedBorder)
      }
      HStack {
        Menu("Replace") {
          let candidates = replacementCandidates(attachment, original: asset)
          if candidates.isEmpty { Text("Import compatible media in Media & Assets.") }
          ForEach(candidates) { candidate in
            Button(candidate.name) { replace(attachment, with: candidate) }
          }
        }.disabled(store.operationBusy)
        if let asset, !fileAvailable(asset) {
          Button("Relink…") { store.relink(asset) }.disabled(store.operationBusy)
        }
      }.font(.caption).controlSize(.small)
      if showAdvanced {
        AttachmentRow(attachment: attachment)
      } else {
        DisclosureGroup("Input details") { AttachmentRow(attachment: attachment) }
          .font(.caption2)
      }
    }.padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 7))
  }

  @ViewBuilder private func inputThumbnail(_ asset: MediaAsset?) -> some View {
    Group {
      if let asset, asset.kind == .image, fileAvailable(asset) {
        CachedImageThumbnail(path: asset.path, maximumPixelSize: 160, contentMode: .fill)
      } else {
        Image(systemName: asset?.kind == .audio ? "waveform" :
          [.video, .sequence].contains(asset?.kind ?? .text) ? "film" : "photo")
          .foregroundStyle(.secondary)
      }
    }.frame(width: 54, height: 44).background(.black.opacity(0.15))
      .clipShape(RoundedRectangle(cornerRadius: 4))
  }

  private func inputSupported(_ attachment: Attachment, asset: MediaAsset?) -> Bool {
    guard let asset else { return false }
    if [.first, .last].contains(attachment.role) {
      return asset.kind == .image && store.supportsEndpoint(attachment.role, for: clip)
    }
    if attachment.role == .keyframe { return asset.kind == .image && supportsTimedKeyframe }
    return GenerationInputPurpose.canReplace(attachment, original: asset, with: asset,
      clip: clip, supportedTasks: supportedTasks, referenceFamily: referenceFamily)
  }

  private func inputLabel(_ attachment: Attachment, asset: MediaAsset?) -> String {
    if attachment.role == .reference && asset?.kind == .audio {
      return clip.engine == .ltx25 ? "Voice reference" : "Sound / voice reference"
    }
    return GenerationPresentation.inputLabel(for: attachment, engine: clip.engine)
  }

  private func replacementCandidates(_ attachment: Attachment, original: MediaAsset?) -> [MediaAsset] {
    store.allAssets.filter { candidate in
      guard candidate.id != attachment.assetID, fileAvailable(candidate) else { return false }
      if [.first, .last].contains(attachment.role) {
        return candidate.kind == .image && store.supportsEndpoint(attachment.role, for: clip)
      }
      if attachment.role == .keyframe { return candidate.kind == .image && supportsTimedKeyframe }
      return GenerationInputPurpose.canReplace(attachment, original: original, with: candidate,
        clip: clip, supportedTasks: supportedTasks, referenceFamily: referenceFamily)
    }
  }

  private func fileAvailable(_ asset: MediaAsset) -> Bool {
    !asset.path.isEmpty && FileManager.default.fileExists(atPath: asset.path)
  }
  private func edit(_ attachment: Attachment, _ mutation: @escaping (inout Attachment) -> Void) {
    guard store.selectedClipID == clip.id else { return }
    store.editClip { current in
      guard let index = current.attachments.firstIndex(where: { $0.id == attachment.id }) else { return }
      mutation(&current.attachments[index])
    }
  }
  private func replace(_ attachment: Attachment, with asset: MediaAsset) {
    // Keep stable attachment identity, purpose, timing, strength and voice bindings.
    edit(attachment) { $0.assetID = asset.id }
  }
  private func remove(_ attachment: Attachment) {
    guard store.selectedClipID == clip.id else { return }
    if [.first, .last].contains(attachment.role) {
      store.removeEndpoint(from: clip.id, role: attachment.role)
    } else {
      store.editClip { $0.attachments.removeAll { $0.id == attachment.id } }
    }
  }
}

private struct InputActionChoice: Identifiable {
  let asset: MediaAsset
  let action: ReferenceAction
  var controlPurpose: GenerationControlPurpose? = nil
  var id: String { asset.id.uuidString + ":" + action.id + ":" + (controlPurpose?.rawValue ?? "") }
}
