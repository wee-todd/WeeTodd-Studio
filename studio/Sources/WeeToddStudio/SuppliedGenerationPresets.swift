import Foundation
import InferenceContracts
import StudioCore

enum SuppliedGenerationPresets {
  static func records(library source: GenerationLibrarySnapshot) -> [StudioGenerationPreset] {
    // Encoding known recipes is metadata-only. Keep them visible offline; application uses the real availability.
    var library = source
    for i in library.modelEntries.indices where library.modelEntries[i].binding != nil { library.modelEntries[i].availability = .installed }
    for i in library.loraEntries.indices where library.loraEntries[i].binding != nil { library.loraEntries[i].availability = .installed }
    var records: [StudioGenerationPreset] = []
    func add(_ key: String, name: String, snapshot: GenerationPresetSnapshot) {
      guard (try? snapshot.validate()) != nil else { return }
      let id = "studio.default." + GenerationLibraryFileIdentity.id(key)
      var label = String(name.prefix(78))
      if records.contains(where: { $0.name.caseInsensitiveCompare(label) == .orderedSame }) { label = String(name.prefix(68)) + " · " + id.suffix(6) }
      records.append(.init(id: id, name: label, origin: .supplied, snapshot: snapshot))
    }
    for entry in library.modelEntries.sorted(by: { $0.id < $1.id }) where entry.availability == .installed {
      if entry.workspaces.contains(.image), let provider = ImageExecutionProvider(rawValue: entry.reference.backend) {
        var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
        draft.selectProvider(provider); draft.width = 1024; draft.height = 1024; draft.seed = 0
        switch entry.binding {
        case .nativeMOSS(let transformer, let shared, let root):
          draft.mossImage?.variant = entry.reference.variant ?? "sft"; draft.mossImage?.transformerDirectory = transformer
          draft.mossImage?.sharedDirectory = shared; draft.mossImage?.libraryDirectory = root
          draft.mossImage?.quantizationBits = entry.reference.precision == "q4" ? 4 : entry.reference.precision == "q8" ? 8 : nil
        case .nativeQwen(let manifest): draft.nativeImage?.manifestPath = manifest
        case .nativeH3Image(let profile, let vae): draft.modelID = profile; draft.h3Image?.imageVAEPath = vae
        case .drawThings(let connection, let model): draft.profileID = connection; draft.modelID = model
        default: continue
        }
        if let snapshot = try? ImagePresetRecipe.capture(from: draft, resolvedLibrary: library) { add(entry.id + "image", name: entry.name + " · Standard", snapshot: snapshot) }
        if provider == .nativeMOSS, entry.reference.variant == "sft" {
          let fast = library.loras(for: ImagePresetRecipe.context(draft), model: entry.reference).filter {
            if case .nativeFile(_, let profile, _, _) = $0.binding { return profile == "moss_sft_4step_v2" && $0.availability == .installed }; return false
          }
          if fast.count == 1, let members = try? library.nativePresetMembers([.init(reference: fast[0].reference)], model: entry.reference, context: ImagePresetRecipe.context(draft)) {
            try? draft.applyMOSSLoRAs(members)
            for steps in [4, 6, 8] {
              draft.steps = steps
              if let snapshot = try? ImagePresetRecipe.capture(from: draft, resolvedLibrary: library) {
                add(entry.id + "fast\(steps)", name: entry.name + " · Fast \(steps) steps", snapshot: snapshot)
              }
            }
          }
        }
      }
      if case .nativeVideo(let profile) = entry.binding, let engine = Engine(rawValue: entry.reference.backend) {
        if engine == .ltx25, entry.tasks.contains("video_upscale") {
          for mode in entry.movieModes ?? [.refine] {
            var project = StudioProject(), clip = Clip(engine: .ltx25), movie = LTX25MovieUpscaleSettings()
            movie.experimentalEnabled = true; movie.mode = mode; if mode == .latentOnly { movie.anchors = .none }
            clip.profileID = profile; clip.generationSelection = .init(task: "video_upscale", preset: .custom); clip.generationSelection?.ltx25MovieUpscale = movie
            project.clips = [clip]
            if let snapshot = try? ClipPresetRecipe.capture(from: clip, project: project, resolvedLibrary: library) {
              add(entry.id + "movie-" + mode.rawValue, name: entry.name + " · " + mode.label, snapshot: snapshot)
            }
          }
        }
        for task in entry.tasks where task != "video_upscale" {
          var project = StudioProject(), clip = Clip(engine: engine); clip.profileID = profile; clip.generationSelection = .init(task: task)
          if engine == .ltx25, let mode = entry.generation?.pipelineMode.flatMap(LTX25GuidanceMode.init(rawValue:)) {
            clip.generationSelection?.ltx25Guidance = .init(mode: mode, experimentalEnabled: true)
          }
          project.clips = [clip]
          if let snapshot = try? ClipPresetRecipe.capture(from: clip, project: project, resolvedLibrary: library) {
            add(entry.id + task, name: entry.name + " · " + task.uppercased(), snapshot: snapshot)
          }
        }
        if engine == .ltx25, entry.tasks.contains("t2v"), entry.supports(.init(workspace: .clip, task: "t2v", continuousScene: true)) {
          var scene = StudioProject(), first = Clip(engine: .ltx25), second = Clip(engine: .ltx25)
          first.profileID = profile; second.profileID = profile
          first.generationSelection = .init(task: "t2v"); second.generationSelection = .init(task: "t2v")
          second.continuity = .init(mode: "scene", sourceClipID: first.id); scene.clips = [first, second]
          if let snapshot = try? ClipPresetRecipe.capture(from: first, project: scene, resolvedLibrary: library) {
            add(entry.id + "scene", name: entry.name + " · Scene", snapshot: snapshot)
          }
        }
      }
      if case .drawThings(let connection, let model) = entry.binding, entry.workspaces.contains(.clip) {
        for task in entry.tasks {
          var project = StudioProject(), clip = Clip(engine: .drawThings); clip.generationSelection = .init(task: task)
          clip.drawThings = .init(profileID: connection, modelID: model, modelFamily: entry.reference.family); project.clips = [clip]
          // Endpoint-role admission is checked against the destination when selected.
          var matching = library
          if let index = matching.modelEntries.firstIndex(where: { $0.id == entry.id }) { matching.modelEntries[index].inputRoleSets = nil }
          if let snapshot = try? ClipPresetRecipe.capture(from: clip, project: project, resolvedLibrary: matching) {
            add(entry.id + task, name: entry.name + " · " + task.uppercased(), snapshot: snapshot)
          }
        }
      }
      if case .ripple = entry.binding {
        var source = Clip(engine: .movie); source.sourcePath = "/source.mov"
        var draft = RippleDraft(clip: source, frameRate: 24); draft.libraryModel = entry.reference
        if let snapshot = try? RipplePresetRecipe.capture(from: draft, resolvedLibrary: library) { add(entry.id + "ripple", name: entry.name + " · Standard", snapshot: snapshot) }
      }
      if case .restoration(let kind, let fields) = entry.binding, let tool = RestorationPresetTool(rawValue: kind) {
        var state = RestorationPresetState(tool: tool); state.model = entry.reference
        if let variant = fields["variant"].flatMap(FlashVSRSettings.Variant.init(rawValue:)) { state.flash.variant = variant }
        if let decoder = fields["decoder"].flatMap(FlashVSRSettings.Decoder.init(rawValue:)) { state.flash.decoder = decoder }
        if let precision = fields["precision"].flatMap(ReCaPrecision.init(rawValue:)) { state.reCa.precision = precision }
        if let scale = fields["scale"].flatMap(Int.init).flatMap(NativeRelayVSRScale.init(rawValue:)) { state.relayScale = scale }
        for workspace in entry.workspaces where workspace == .restoration || workspace == .finishing {
          if workspace == .restoration, let snapshot = try? RestorationPresetRecipe.capture(from: state, resolvedLibrary: library) {
            add(entry.id + "restore", name: entry.name + " · Restore", snapshot: snapshot)
          } else if workspace == .finishing {
            let recipe: VideoProcessingRecipe
            if tool == .h3Regenerate { recipe = .init(h3Regenerate: state.h3) }
            else { guard let value = try? VideoProcessingRecipe(steps: [.init(settings: state.flash)]) else { continue }; recipe = value }
            if let snapshot = try? FinishingPresetRecipe.capture(from: recipe, models: [entry.reference], resolvedLibrary: library, sourceFPS: 24) {
              add(entry.id + "finish", name: entry.name + " · Finishing", snapshot: snapshot)
            }
          }
        }
        if entry.workspaces.contains(.clip), tool == .relayVSR || tool == .reCaVSR,
          var snapshot = try? RestorationPresetRecipe.capture(from: state, resolvedLibrary: library, workspace: .clip) {
          snapshot.context.task = "video_upscale"; snapshot.fields["method"] = .string(tool.rawValue)
          add(entry.id + "movie", name: entry.name + " · Source movie", snapshot: snapshot)
        }
      }
      if case .restoration("pixelCQ", let fields) = entry.binding {
        var pixel = NativePixelCQSettings(); pixel.profileID = fields["profileID"] ?? "auto"; pixel.pixelAdapterPath = fields["pixelAdapterPath"]; pixel.cqAdapterPath = fields["cqAdapterPath"]
        if let snapshot = try? FinishingPresetRecipe.capture(from: .init(pixelCQ: pixel), models: [entry.reference], resolvedLibrary: library) {
          add(entry.id + "pixel", name: entry.name + " · Finishing", snapshot: snapshot)
        }
      }
    }
    let singleStages = records.compactMap { record -> (StudioGenerationPreset, GenerationPresetValue, String)? in
      guard record.snapshot.context.workspace == .finishing, case .object(let stages) = record.snapshot.fields["stages"], stages.count == 1,
        let value = stages["0"], let stage = try? value.decoded(GenerationPresetSnapshot.self) else { return nil }
      return (record, value, stage.context.task)
    }
    for (first, firstStage, tool) in singleStages where tool == "h3Regenerate" || tool == "pixelCQ" {
      for (flash, flashStage, secondTool) in singleStages where secondTool == "flashVSR" {
        let label = tool == "h3Regenerate" ? "H3 → Flash" : "Pixel + CQ → Flash"
        add(first.id + flash.id + "chain", name: label + " · " + first.name.replacingOccurrences(of: " · Finishing", with: "") + " · " + flash.name.replacingOccurrences(of: " · Finishing", with: ""),
          snapshot: .init(context: .init(workspace: .finishing, task: "finishing"), adapters: [], fields: ["stages": .object(["0": firstStage, "1": flashStage])]))
      }
    }
    return records
  }
}

@MainActor extension StudioStore {
  func refreshSuppliedGenerationPresets() {
    do { var candidate = generationPresets; try candidate.replaceDefaults(SuppliedGenerationPresets.records(library: generationLibrary)); generationPresets = candidate }
    catch { generationPresetError = error.localizedDescription }
  }
}
