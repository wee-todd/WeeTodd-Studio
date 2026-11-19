import Foundation
import StudioCore

@MainActor extension StudioStore {
  func clipPresetSelectionKey(_ id: UUID) -> String { "clip:" + id.uuidString }
  var clipPresetSelection: PresetSelectionState? {
    guard let id = selectedClipID else { return nil }; return project.generationPresetSelections?[clipPresetSelectionKey(id)]
  }
  func setClipPresetSelection(_ selection: PresetSelectionState?) {
    guard let clip = selectedClip else { return }
    let key = clipPresetSelectionKey(clip.id)
    change { project in
      if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
      project.generationPresetSelections?[key] = selection
      if project.generationPresetSelections?.isEmpty == true { project.generationPresetSelections = nil }
    }
  }
  func clipPresetTargetToken() throws -> GenerationTargetToken {
    guard let clip = selectedClip else { throw StudioError.invalid("Select a video clip first.") }
    let members = project.isContinuousSceneMember(clip) ? try project.continuousSceneMembers(for: clip) : [clip]
    let ids = Set(members.flatMap { $0.attachments.map(\.assetID) })
    struct Revision: Encodable {
      var clips: [Clip]; var assets: [MediaAsset]; var settings: MovieSettings; var selection: PresetSelectionState?
      var bindings: [String: GenerationModelRef]?
    }
    let revision = Revision(clips: members, assets: allAssets.filter { ids.contains($0.id) }, settings: clip.settings(in: project), selection: clipPresetSelection,
      bindings: project.generationLibraryBindings?.filter { item in members.contains { item.key == "clip-model:" + $0.id.uuidString } })
    return .init(documentSessionID: documentSessionID, workspaceKey: "clip:" + project.id.uuidString + ":" + clip.id.uuidString,
      semanticRevision: try GenerationTargetToken.revision(revision), memberIDs: members.map(\.id))
  }
  func captureClipPreset() throws -> GenerationPresetSnapshot {
    guard let clip = selectedClip else { throw StudioError.invalid("Select a video clip first.") }
    var input = project; input.assets += globalAssets
    if clip.usesNativeMovieUpscaler, input.generationLibraryBindings?["clip-model:" + clip.id.uuidString] == nil, let model = inferredMovieLibraryModel(for: clip) {
      if input.generationLibraryBindings == nil { input.generationLibraryBindings = [:] }
      input.generationLibraryBindings?["clip-model:" + clip.id.uuidString] = model
    }
    var effective = clip
    if clip.profileID == "auto", !clip.usesNativeMovieUpscaler, clip.engine != .drawThings,
      !generationLibrary.modelEntries.contains(where: { if case .nativeVideo("auto") = $0.binding { return $0.reference.backend == clip.engine.rawValue }; return false }) {
      if let description = generationDescriptions[clip.id], description["studioInput"] as? String == generationRequestKey(for: clip),
        let resolved = description["profileID"] as? String { effective.profileID = resolved }
      else {
        let candidates = generationLibrary.models(for: ClipPresetRecipe.context(clip, project: project)).filter { entry in
          if case .nativeVideo = entry.binding { return entry.reference.backend == clip.engine.rawValue && entry.availability == .installed }; return false
        }
        if candidates.count == 1, case .nativeVideo(let profile) = candidates[0].binding { effective.profileID = profile }
      }
    }
    return try ClipPresetRecipe.capture(from: effective, project: input, resolvedLibrary: generationLibrary)
  }
  func applyClipPreset(_ preset: StudioGenerationPreset, expectedTarget target: GenerationTargetToken, associate: Bool = true) async throws {
    guard try clipPresetTargetToken() == target else { throw StudioError.invalid("The video target changed. Select the preset again.") }
    await refreshGenerationLibrary(force: true)
    try Task.checkCancellation()
    guard generationLibraryError == nil else { throw StudioError.invalid(generationLibraryError!) }
    guard try clipPresetTargetToken() == target, let clip = selectedClip else { throw StudioError.invalid("The video target changed while checking models. Its current edits were preserved.") }
    var input = project; input.assets += globalAssets
    let members = project.isContinuousSceneMember(clip) ? try project.continuousSceneMembers(for: clip) : [clip]
    var applications: [(ClipPresetApplication, GenerationPresetSnapshot)] = []
    var bindingUpdates: [UUID: GenerationModelRef] = [:]
    for member in members {
      let application = try ClipPresetRecipe.candidate(applying: preset.snapshot, to: member, project: input, resolvedLibrary: generationLibrary)
      input.assets += application.assets
      if let model = application.libraryModel {
        if project.generationLibraryBindings?["clip-model:" + member.id.uuidString] != nil || inferredMovieLibraryModel(for: member) != model {
          bindingUpdates[member.id] = model
        }
        if input.generationLibraryBindings == nil { input.generationLibraryBindings = [:] }
        input.generationLibraryBindings?["clip-model:" + member.id.uuidString] = model
      }
      let baseline = try ClipPresetRecipe.capture(from: application.clip, project: input, resolvedLibrary: generationLibrary)
      applications.append((application, baseline))
    }
    try await admitClipPresetApplications(applications.map { $0.0 }, project: input)
    guard try clipPresetTargetToken() == target else { throw StudioError.invalid("The video destination changed during native model admission. Its current edits were preserved.") }
    change { project in
      for (application, baseline) in applications {
        guard let index = project.clips.firstIndex(where: { $0.id == application.clip.id }) else { continue }
        project.assets += application.assets; project.clips[index] = application.clip
        if let model = bindingUpdates[application.clip.id] {
          if project.generationLibraryBindings == nil { project.generationLibraryBindings = [:] }
          project.generationLibraryBindings?["clip-model:" + application.clip.id.uuidString] = model
        }
        if associate {
          if project.generationPresetSelections == nil { project.generationPresetSelections = [:] }
          project.generationPresetSelections?[clipPresetSelectionKey(application.clip.id)] = .init(preset, baseline: baseline)
        }
      }
    }
  }
  func inferredMovieLibraryModel(for clip: Clip) -> GenerationModelRef? {
    let tool = clip.usesReCaVSR ? "reCaVSR" : "relayVSR"
    return generationLibrary.modelEntries.first { entry in
      guard case .restoration(let kind, let fields) = entry.binding, kind == tool else { return false }
      if clip.usesReCaVSR { return fields["modelDirectory"] == runtime.reCaVSRModelDirectory && fields["precision"] == (clip.generationSelection?.reCaVSRSettings?.precision.rawValue ?? "bf16") }
      guard let directory = fields["modelDirectory"], let root = runtime.relayVSRModelDirectory else { return false }
      func family(_ path: String) -> String { let url = URL(fileURLWithPath: path); return ["x2", "x4"].contains(url.lastPathComponent) ? url.deletingLastPathComponent().path : url.path }
      return family(directory) == family(root) && fields["scale"] == String((clip.generationSelection?.relayVSRScale ?? .x2).rawValue)
    }?.reference
  }
  func clipLibraryRuntime(for clip: Clip) throws -> RuntimeSettings {
    var result = runtime
    guard clip.usesNativeMovieUpscaler, let model = project.generationLibraryBindings?["clip-model:" + clip.id.uuidString] else { return result }
    let tool = clip.usesReCaVSR ? RestorationPresetTool.reCaVSR : .relayVSR
    let fields = try RestorationPresetRecipe.binding(model, tool: tool, workspace: .clip, library: generationLibrary)
    if tool == .relayVSR {
      guard fields["scale"] == String((clip.generationSelection?.relayVSRScale ?? .x2).rawValue) else { throw StudioError.invalid("Select the matching RelayVSR scale in Models & LoRAs.") }
      result.relayVSRModelDirectory = fields["modelDirectory"]
    } else {
      guard fields["precision"] == (clip.generationSelection?.reCaVSRSettings?.precision.rawValue ?? "bf16") else { throw StudioError.invalid("Select the matching ReCaVSR precision in Models & LoRAs.") }
      result.reCaVSRModelDirectory = fields["modelDirectory"]
    }
    return result
  }
}
