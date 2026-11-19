import AppKit
import Foundation
import StudioCore

@MainActor extension StudioStore {
  var generationLinkedLoRAAssets: [MediaAsset] {
    let drafts = [imageDraft].compactMap { $0 } + imageWorkspaceLibrary.sessions.values.map(\.draft)
    var assets = drafts.flatMap { ($0.mossImage?.loras ?? []) + ($0.h3Image?.loras ?? []) }.map(\.asset)
    let bindings = savedLTX25Configs.compactMap { $0.selection.ltx25AdapterBinding } + project.clips.compactMap { $0.generationSelection?.ltx25AdapterBinding }
    assets += bindings.map { binding in
      var asset = MediaAsset(name: binding.kind.label, kind: .lora, path: binding.path, scope: .global); asset.loraModel = .ltx25; return asset
    }
    return assets
  }
  var generationLibraryRootDirectory: String { generationLibraryIndex.rootDirectory ?? "" }
  var generationLibraryFile: URL { dataDirectory.appendingPathComponent("generation-library.json") }
  func loadGenerationLibrary() {
    if FileManager.default.fileExists(atPath: generationLibraryFile.path) {
      do { generationLibraryIndex = try .read(from: generationLibraryFile); generationLibrary = generationLibraryIndex.snapshot }
      catch {
        do { try FileManager.default.moveItem(at: generationLibraryFile, to: dataDirectory.appendingPathComponent("generation-library-unreadable-\(UUID().uuidString).json")) }
        catch { generationLibraryWritable = false }
        generationLibraryError = "The model library could not be read. Its original file was preserved."
      }
    }
    if generationLibraryIndex.rootDirectory == nil, dataDirectory == Self.supportDirectory {
      generationLibraryIndex.rootDirectory = MOSSModelDiscovery.mountedLibraries().first {
        FileManager.default.fileExists(atPath: $0.path)
      }?.path
    }
  }
  func setGenerationLibraryRoot(_ directory: String) throws {
    guard directory.hasPrefix("/"), directory.utf8.count <= 4096, !directory.utf8.contains(0) else {
      throw StudioError.invalid("Choose an absolute model-library folder.")
    }
    var candidate = generationLibraryIndex; candidate.rootDirectory = URL(fileURLWithPath: directory).standardizedFileURL.path
    try persistGenerationLibrary(candidate)
  }
  func persistGenerationLibrary(_ candidate: GenerationLibraryIndex) throws {
    guard generationLibraryWritable else { throw StudioError.invalid("The original model library could not be preserved. Resolve its file permissions before saving.") }
    try candidate.write(to: generationLibraryFile)
    generationLibraryIndex = candidate; generationLibrary = candidate.snapshot
  }
  func refreshGenerationLibrary(force: Bool = false) async {
    if let active = generationLibraryRefreshTask {
      if force { generationLibraryRefreshRequested = true }
      await active.value; return
    }
    let task = Task { @MainActor in
      repeat { generationLibraryRefreshRequested = false; await performGenerationLibraryRefresh() } while generationLibraryRefreshRequested
    }
    generationLibraryRefreshTask = task
    await task.value; generationLibraryRefreshTask = nil
  }
  private func performGenerationLibraryRefresh() async {
    let attempt = UUID(); generationLibraryAttempt = attempt; generationLibraryBusy = true
    defer { if generationLibraryAttempt == attempt { generationLibraryBusy = false } }
    let sourceIndex = generationLibraryIndex
    var initial = sourceIndex
    initial.refreshAvailability(models: { $0.reference.backend == "drawThings" ? .backendUnavailable("Refresh the Draw Things connection to relink this model.") : $0.availability },
      loras: { $0.reference.backend == "drawThings" ? .backendUnavailable("Refresh the Draw Things connection to relink this LoRA.") : $0.availability })
    // Cached Draw Things capability mapping is authoritative. Never infer support from checkpoint filenames.
    for connection in drawThingsConnections {
      for (workspace, operation, task) in [(GenerationWorkspaceKind.image, "image", "image"), (.clip, "video", "t2v"), (.clip, "video", "i2v"), (.clip, "video", "fflf"), (.clip, "video", "ref2va")] {
        for model in drawThingsModels(connection.id, operation: operation, task: task == "image" ? nil : task) {
          let alias = "dt|\(connection.id)|\(model.id)|\(task)"
          let family = drawThingsModelFamily(profileID: connection.id, modelID: model.id) ?? "unknown"
          let ref = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: "drawThings", family: family)
          let roles: [[String]]? = task == "t2v" ? [[]] : task == "i2v" ? [["first"]] : task == "fflf" ? [["first", "last"]] : task == "ref2va" ? (1...9).map { Array(repeating: "reference", count: $0) } : nil
          let status: GenerationLibraryAvailability = drawThingsDiscovery.errors[connection.id].map { .backendUnavailable($0) } ?? .installed
          try? initial.registerModel(.init(reference: ref, name: model.name, workspaces: [workspace], tasks: [task], inputRoleSets: roles,
            availability: status, binding: .drawThings(connectionID: connection.id, modelID: model.id)), alias: alias)
          for lora in drawThingsLoRAs(profileID: connection.id, modelID: model.id) {
            let alias = "dt|\(connection.id)|\(lora.id)"
            let ref = GenerationLoRARef(id: GenerationLibraryFileIdentity.id(alias), backend: "drawThings", family: lora.family)
            let compatible = initial.snapshot.modelEntries.filter { entry in
              if case .drawThings(let linked, let id) = entry.binding {
                return linked == connection.id && lora.compatibleModelIDs.contains(id)
              }
              return false
            }.map(\.id)
            try? initial.registerLoRA(.init(reference: ref, name: lora.name, compatibleFamilies: [lora.family], compatibleModelIDs: compatible,
              availability: status, binding: .drawThings(connectionID: connection.id, modelID: lora.id)), alias: alias)
          }
        }
      }
    }
    let index = initial, settings = runtime, linked = mossModelLinks, models = profiles, vae = h3ImageVAEPath
    let assets = allAssets + generationLinkedLoRAAssets, groups = loraGroups, folders = loraFolders, automatic = automaticLoRADiscoveryEnabled
    let manifests = Set(([imageDraft?.nativeImage?.manifestPath] + imageWorkspaceLibrary.sessions.values.map { $0.draft.nativeImage?.manifestPath }).compactMap { $0 }.filter { !$0.isEmpty })
    let flash = videoUpscaleRuntime, h3 = h3RegenerateSettings
    do {
      let found = try await Task.detached(priority: .utility) {
        try GenerationLibraryInventory.refresh(index: index, settings: settings, moss: linked, profiles: models,
          imageVAE: vae, assets: assets, groups: groups, folders: folders, automatic: automatic, manifests: Array(manifests),
          flash: flash, h3: h3)
      }.value
      guard !Task.isCancelled, generationLibraryAttempt == attempt else { return }
      guard generationLibraryIndex == sourceIndex else { generationLibraryRefreshRequested = true; return }
      try persistGenerationLibrary(found); generationLibraryError = nil; refreshSuppliedGenerationPresets()
      do { try migrateLegacyRestorationPresets(); try migrateLegacyLTXPresets() } catch { generationPresetError = error.localizedDescription }
    } catch { if generationLibraryAttempt == attempt { generationLibraryError = error.localizedDescription } }
  }
}

/// Metadata-only projection over existing catalogs. Tensor loading remains in the established workers.
enum GenerationLibraryInventory {
  static func refresh(index: GenerationLibraryIndex, settings: RuntimeSettings, moss: MOSSModelLinks,
    profiles: [ModelProfile], imageVAE: String, assets: [MediaAsset], groups: [LoRAGroup], folders: [LoRAFolder],
    automatic: Bool, manifests: [String], flash: FlashVSRRuntime? = nil, h3: H3RegenerateSettings? = nil) throws -> GenerationLibraryIndex {
    var result = index
    let fm = FileManager.default
    var localProfiles = profiles
    var checkedCatalogs = Set<String>(), admittedProfiles = Set<String>()
    let profileRoot = URL(fileURLWithPath: settings.profilesDirectory).standardizedFileURL.resolvingSymlinksInPath().path
    if settings.profilesDirectory.hasPrefix("/"), (try? URL(fileURLWithPath: profileRoot).resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
      for engine in ["h3", "ltx25"] where engine == "h3" ? settings.usesNativeH3 : settings.usesNativeLTX25 {
        let rows = engine == "h3" ? try NativeH3Preparation.catalog(directory: profileRoot) : try NativeLTXPreparation.catalog(directory: profileRoot)
        checkedCatalogs.insert(engine)
        localProfiles.removeAll { $0.engine == engine && $0.id.hasPrefix(profileRoot + "/") }
        for row in rows {
          guard let id = row["id"] as? String, let name = row["name"] as? String, let task = row["task"] as? String else { continue }
          admittedProfiles.insert(id)
          let generation = (row["generation"] as? [String: Any]).flatMap { try? JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: $0)) }
          localProfiles.append(.init(id: id, name: name, engine: engine, task: task, generation: generation))
        }
      }
    }
    func availability(_ paths: [String]) -> GenerationLibraryAvailability {
      if paths.contains(where: { value in
        let parts = URL(fileURLWithPath: value).pathComponents
        return parts.count > 2 && parts[1] == "Volumes" && !fm.fileExists(atPath: "/Volumes/" + parts[2])
      }) { return .driveDisconnected }
      let missing = paths.filter { !fm.fileExists(atPath: $0) }
      return missing.isEmpty ? .installed : .missingComponents(missing.map { URL(fileURLWithPath: $0).lastPathComponent })
    }
    result.refreshAvailability(models: { entry in
      if entry.reference.backend == "drawThings" { return entry.availability }
      let profileID: String?
      switch entry.binding { case .nativeVideo(let id), .nativeH3Image(let id, _): profileID = id; default: profileID = nil }
      let engine = entry.reference.backend == "nativeH3" ? "h3" : entry.reference.backend
      if let profileID, checkedCatalogs.contains(engine) {
        if profileID == "auto", !localProfiles.contains(where: { $0.engine == engine }) { return .incompatible("Link compatible native model components in Advanced setup.") }
        if profileID.hasPrefix(profileRoot + "/"), !admittedProfiles.contains(profileID), fm.fileExists(atPath: profileID) {
          return .incompatible("The native model recipe no longer passes component admission. Reimport or repair it in Advanced setup.")
        }
      }
      let state = availability(entry.identityFiles ?? entry.binding?.localPaths ?? ["unlinked"])
      guard state == .installed else { return state }
      if case .nativeQwen(let manifest) = entry.binding, (try? GenerationLibraryPreparedModels.qwenFiles(URL(fileURLWithPath: manifest))) == nil {
        return .incompatible("The prepared Qwen package no longer passes manifest and component admission. Relink or repair it.")
      }
      return state
    }, loras: { entry in
      if case .nativeFile(let path, _, _, _) = entry.binding {
        if folders.contains(where: { !$0.enabled && (path == $0.path || path.hasPrefix($0.path + "/")) }) { return .incompatible("This LoRA folder is excluded from discovery.") }
        return availability([path])
      }
      return entry.availability
    })
    func register(_ alias: String, name: String, backend: String, family: String, variant: String? = nil,
      precision: String? = nil, workspaces: [GenerationWorkspaceKind], tasks: [String], binding: ResolvedGenerationModel,
      files: [String], supportsLoRAs: Bool = true, generation: GenerationDescriptor? = nil, movieModes: [LTX25MovieUpscaleSettings.Mode]? = nil) throws {
      let status = availability(files)
      let old = result.modelReference(alias: alias)
      let reference = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: backend, family: family,
        variant: variant, precision: precision, sourceIdentity: status == .installed ? GenerationLibraryFileIdentity.signature(files) : old?.sourceIdentity)
      try result.registerModel(.init(reference: reference, name: name, workspaces: workspaces, tasks: tasks,
        supportsLoRAs: supportsLoRAs, availability: status, binding: binding, identityFiles: files, generation: generation, movieModes: movieModes), alias: alias)
    }
    func supportedMovieModes(_ profileID: String) -> [LTX25MovieUpscaleSettings.Mode] {
      var project = StudioProject(), clip = Clip(engine: .ltx25)
      clip.profileID = profileID; clip.generationWidth = 768; clip.generationHeight = 448
      clip.generationSelection = .init(task: "video_upscale", preset: .custom)
      return LTX25MovieUpscaleSettings.Mode.allCases.filter { mode in
        var options = LTX25MovieUpscaleSettings(); options.experimentalEnabled = true; options.mode = mode
        if mode == .latentOnly { options.anchors = .none }
        clip.generationSelection?.ltx25MovieUpscale = options; project.clips = [clip]
        guard let object = try? project.object(), let runtime = try? JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)),
          let binding = try? NativeLTXPreparation.movieModelBinding(request: ["project": object, "clipID": clip.id.uuidString, "globalAssets": [], "runtime": runtime], settings: options) else { return false }
        return binding.components.values.allSatisfy { fm.fileExists(atPath: $0) }
      }
    }
    let roots = index.rootDirectory.map { [URL(fileURLWithPath: $0)] } ?? []
    for variant in ["sft", "base"] {
      var settings = MOSSImageSettings(); settings.variant = variant
      let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(settings), remembered: moss, libraries: roots)
      for bits in [nil, 4, 8] as [Int?] {
        settings.quantizationBits = bits
        var query = links.applying(to: MOSSModelQuery(settings))
        if query.transformerDirectory.isEmpty, let bits {
          for root in roots {
            let candidate = root.appendingPathComponent("WeeTodd/MOSS-Image/Native-Q\(bits)/\(variant.uppercased())/transformer")
            if let marker = try? GenerationLibraryModelImport.json(candidate.appendingPathComponent("moss_quantization.json")),
              marker["format"] as? String == "weetodd-moss-affine-v1", marker["bits"] as? Int == bits,
              marker["groupSize"] as? Int == 64, marker["component"] as? String == "transformer" {
              query.transformerDirectory = candidate.path; break
            }
          }
        }
        guard !query.transformerDirectory.isEmpty else { continue }
        let transformer = URL(fileURLWithPath: query.transformerDirectory), shared = URL(fileURLWithPath: query.sharedDirectory)
        var files = [transformer.appendingPathComponent("config.json").path,
          shared.appendingPathComponent("text_encoder/config.json").path, shared.appendingPathComponent("vae/config.json").path,
          shared.appendingPathComponent("tokenizer/tokenizer.json").path, shared.appendingPathComponent("tokenizer/tokenizer_config.json").path]
        files += GenerationLibraryFileIdentity.componentFiles(transformer)
        files += GenerationLibraryFileIdentity.componentFiles(shared.appendingPathComponent("text_encoder"))
        files += GenerationLibraryFileIdentity.componentFiles(shared.appendingPathComponent("vae"))
        for component in ["text_encoder", "vae"] {
          let folder = shared.appendingPathComponent(component)
          if !GenerationLibraryFileIdentity.componentFiles(folder).contains(where: { $0.hasSuffix(".safetensors") }) { files.append(folder.appendingPathComponent("model.safetensors").path) }
        }
        if bits != nil { files.append(transformer.appendingPathComponent("moss_quantization.json").path) }
        try register("moss|" + query.transformerDirectory, name: "MOSS Image " + variant.uppercased() + " · " + (bits.map { "Q\($0)" } ?? "BF16"),
          backend: "nativeMOSS", family: "moss_image", variant: variant, precision: bits.map { "q\($0)" } ?? "bf16",
          workspaces: [.image], tasks: ["image"], binding: .nativeMOSS(transformer: query.transformerDirectory, shared: query.sharedDirectory, library: query.libraryDirectory), files: files)
      }
    }
    for profile in localProfiles where ["h3", "ltx23", "ltx25"].contains(profile.engine) {
      let files = GenerationLibraryModelImport.profileFiles(profile.id)
      let movieModes = profile.engine == "ltx25" ? supportedMovieModes(profile.id) : []
      try register("profile|" + profile.id, name: profile.name, backend: profile.engine, family: profile.engine,
        workspaces: [.clip], tasks: (profile.generation?.supportedTasks ?? [profile.task]) + (movieModes.isEmpty ? [] : ["video_upscale"]), binding: .nativeVideo(profileID: profile.id), files: files, generation: profile.generation, movieModes: movieModes)
      if profile.engine == "h3", ["t2v", "fflf", "ref2va"].contains(profile.task), profile.generation?.vdn != true,
        profile.generation?.fasth3 != true, profile.generation?.fasth3Omni != true {
        try register("h3-image|" + profile.id, name: profile.name + " · Image", backend: "nativeH3", family: "h3", workspaces: [.image], tasks: ["image"],
          binding: .nativeH3Image(profileID: profile.id, imageVAE: imageVAE), files: files + (imageVAE.isEmpty ? [] : [imageVAE]))
      }
    }
    for engine in ["h3", "ltx23", "ltx25"] {
      let compatible = localProfiles.filter { $0.engine == engine }
      guard !compatible.isEmpty else { continue }
      let files = Array(Set(compatible.flatMap { GenerationLibraryModelImport.profileFiles($0.id) })).sorted()
      let tasks = Array(Set(compatible.flatMap { $0.generation?.supportedTasks ?? [$0.task] })).sorted()
      let movieModes = engine == "ltx25" ? supportedMovieModes("auto") : []
      try register("automatic|" + engine, name: "Automatic · " + engine.uppercased(), backend: engine, family: engine,
        workspaces: [.clip], tasks: tasks + (movieModes.isEmpty ? [] : ["video_upscale"]), binding: .nativeVideo(profileID: "auto"), files: files, movieModes: movieModes)
      if engine == "h3", compatible.contains(where: { ["t2v", "fflf", "ref2va"].contains($0.task) && $0.generation?.vdn != true && $0.generation?.fasth3 != true && $0.generation?.fasth3Omni != true }) {
        try register("automatic|h3-image", name: "Automatic · H3 Image", backend: "nativeH3", family: "h3", workspaces: [.image], tasks: ["image"],
          binding: .nativeH3Image(profileID: "auto", imageVAE: imageVAE), files: files + (imageVAE.isEmpty ? [] : [imageVAE]))
      }
    }
    let retainedManifests = index.snapshot.modelEntries.compactMap { entry -> String? in if case .nativeQwen(let manifest) = entry.binding { return manifest }; return nil }
    for path in Set(manifests + retainedManifests) where !path.isEmpty {
      guard let files = try? GenerationLibraryPreparedModels.qwenFiles(URL(fileURLWithPath: path)) else { continue }
      try register("qwen|" + path, name: "Qwen Image 2.1 · 8-bit", backend: "nativeMLX", family: "qwen_image", precision: "q8", workspaces: [.image], tasks: ["image"],
        binding: .nativeQwen(manifest: path), files: files, supportsLoRAs: false)
    }
    let discovery = try LoRAFolderDiscovery.discover(folders: folders, profilesDirectory: settings.profilesDirectory,
      assets: assets, groups: groups, additionalPaths: [settings.rippleAdapterPath].compactMap { $0 }, enabled: automatic, libraryRoot: index.rootDirectory)
    let scan = try NativeLoRAFolderScan.scan(discovery.folders, modelHints: discovery.modelHints)
    let entries = try JSONDecoder().decode([LoRAFolderEntry].self, from: JSONSerialization.data(withJSONObject: scan["entries"] ?? []))
    for entry in entries {
      guard let family = entry.loraModel?.rawValue else { continue }
      let alias = "native-lora|" + entry.path
      let ref = GenerationLoRARef(id: GenerationLibraryFileIdentity.id(alias), backend: "native", family: family, sourceIdentity: GenerationLibraryFileIdentity.signature([entry.path]))
      try result.registerLoRA(.init(reference: ref, name: entry.name, compatibleFamilies: family == "ltx23" ? ["ltx23", "ltx25"] : [family],
        compatibleVariants: entry.loraProfile == "moss_sft_4step_v2" ? ["sft"] : nil,
        availability: entry.status == "ready" || entry.adapterKind != nil ? .installed : .incompatible(entry.detail ?? "Use the specialized model recipe for this adapter."),
        binding: .nativeFile(path: entry.path, profile: entry.adapterKind?.rawValue ?? entry.loraProfile, layout: entry.loraLayout, adalnGrid: entry.loraAdalnInputGrid), detail: entry.detail), alias: alias)
    }
    let scanned = Dictionary(entries.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
    result.refreshAvailability(models: { $0.availability }, loras: { entry in
      guard case .nativeFile(let path, _, _, _) = entry.binding, let scan = scanned[path], scan.loraModel == nil || ["unsupported", "needsModel"].contains(scan.status) else { return entry.availability }
      return .incompatible(scan.detail ?? "This file could not be admitted as a compatible LoRA.")
    })
    try refreshSpecialized(index: &result, settings: settings, profiles: localProfiles, flash: flash, h3: h3)
    result.refreshAvailability(models: { entry in
      guard entry.availability == .installed, entry.identityFiles != nil, case .nativeMOSS(let transformer, let shared, let root) = entry.binding else { return entry.availability }
      do {
        let checked = try GenerationLibraryModelImport.inspect(URL(fileURLWithPath: transformer), profilesDirectory: settings.profilesDirectory,
          preferredVariant: entry.reference.variant ?? "sft", shared: shared, libraryRoot: root, explicitVariant: true)
        guard checked.entry.reference.precision == entry.reference.precision, checked.entry.reference.variant == entry.reference.variant else { return .incompatible("The MOSS variant or precision changed. Relink its current package.") }
        return entry.availability
      } catch { return .incompatible(error.localizedDescription) }
    }, loras: { $0.availability })
    result.refreshLocalModelIdentities()
    return result
  }
}
