import Foundation
import InferenceContracts
import StudioCore

enum GenerationLibraryImportIntent {
  case libraryRoot
  case model(target: GenerationPresetTarget?, expectedTarget: GenerationTargetToken?, addAndUse: Bool, stageIndex: Int? = nil)
  case lora(model: GenerationModelRef, target: GenerationPresetTarget?, expectedTarget: GenerationTargetToken?, addAndUse: Bool)
}
struct GenerationLibraryImportResult {
  var model: GenerationModelRef?
  var lora: GenerationLoRARef?
  var applied = false
  var detail: String?
}

@MainActor extension StudioStore {
  @discardableResult func linkGenerationLibraryItem(_ source: URL, intent: GenerationLibraryImportIntent) async throws -> GenerationLibraryImportResult {
    guard source.isFileURL else { throw StudioError.invalid("Choose an installed local model or LoRA.") }
    let source = source.standardizedFileURL.resolvingSymlinksInPath()
    try Task.checkCancellation()
    switch intent {
    case .libraryRoot:
      guard (try? source.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { throw StudioError.invalid("Choose the existing shared model-library folder.") }
      try setGenerationLibraryRoot(source.path); await refreshGenerationLibrary(force: true)
      if let error = generationLibraryError { throw StudioError.invalid(error) }
      return .init(detail: "Linked the model library. Installed files remain in place.")
    case .lora(let model, let target, let token, let use):
      guard model.backend != "drawThings", let hint = LoRAModel(rawValue: model.family) else {
        throw StudioError.invalid("Install Draw Things LoRAs through Draw Things, then refresh its connection. Native imports require a supported SafeTensors family.")
      }
      let before = GenerationLibraryFileIdentity.signature([source.path])
      let info = try await NativeLoRAImport.inspect(source, modelHint: hint)
      try Task.checkCancellation()
      guard before == GenerationLibraryFileIdentity.signature([source.path]),
        let family = info["loraModel"] as? String, LoRAModel(rawValue: family) != nil else {
        throw StudioError.invalid("The LoRA changed during inspection or its model family could not be verified. Import it again.")
      }
      let alias = "native-lora|" + source.path
      let reference = GenerationLoRARef(id: GenerationLibraryFileIdentity.id(alias), backend: "native", family: family, sourceIdentity: before)
      let profile = info["loraProfile"] as? String
      let entry = GenerationLoRAEntry(reference: reference, name: source.deletingPathExtension().lastPathComponent,
        compatibleFamilies: family == "ltx23" ? ["ltx23", "ltx25"] : [family], compatibleVariants: profile == "moss_sft_4step_v2" ? ["sft"] : nil,
        availability: .installed, binding: .nativeFile(path: source.path, profile: profile, layout: info["loraLayout"] as? String,
          adalnGrid: info["loraAdalnInputGrid"] as? String), detail: info["detail"] as? String)
      var index = generationLibraryIndex; let linked = try index.registerLoRA(entry, alias: alias); try persistGenerationLibrary(index)
      var result = GenerationLibraryImportResult(lora: linked)
      if use, let target, let token {
        guard (try? presetTargetToken(for: target)) == token, currentGenerationLibraryModel(for: target)?.reference == model else {
          result.detail = "Added to the library. The editing destination changed, so its current values were preserved."; notice = result.detail!; return result
        }
        do { try await addGenerationLibraryLoRA(linked, for: target, expectedTarget: token); result.applied = true }
        catch { result.detail = "Added to the library. " + error.localizedDescription }
      }
      return result
    case .model(let target, let token, let use, let stageIndex):
      let settings = runtime, shared = mossModelLinks.sharedDirectory, root = generationLibraryRootDirectory
      let preferred = imageDraft?.mossImage?.variant ?? "sft"
      let imageVAE = h3ImageVAEPath
      let imageVAEIdentity = imageVAE.isEmpty ? nil : GenerationLibraryFileIdentity.signature([imageVAE])
      let inspected = try await Task.detached(priority: .utility) {
        try GenerationLibraryModelImport.inspect(source, profilesDirectory: settings.profilesDirectory,
          preferredVariant: preferred, shared: shared, libraryRoot: root)
      }.value
      try Task.checkCancellation()
      if let profile = inspected.profile, !profiles.contains(where: { $0.id == profile.id }) { profiles.append(profile) }
      var index = generationLibraryIndex
      let linked = try index.registerModel(inspected.entry, alias: inspected.alias); try persistGenerationLibrary(index)
      await refreshGenerationLibrary(force: true); try Task.checkCancellation()
      let selected: GenerationModelRef
      do { selected = try GenerationLibraryModelImport.selection(inspected: inspected, linked: linked, library: generationLibrary, image: target == .image, imageVAE: imageVAE, imageVAEIdentity: imageVAEIdentity) }
      catch { return .init(model: linked, detail: "Added to the library. The model components changed during inspection; select the current library entry again before using it.") }
      var result = GenerationLibraryImportResult(model: selected)
      if use, let target, let token {
        guard (try? presetTargetToken(for: target)) == token else {
          result.detail = "Added to the library. The editing destination changed, so its current values were preserved."; notice = result.detail!; return result
        }
        do { try await useGenerationLibraryModel(selected, for: target, expectedTarget: token, stageIndex: stageIndex); result.applied = true }
        catch { result.detail = "Added to the library. " + error.localizedDescription }
      }
      return result
    }
  }
}

enum GenerationLibraryModelImport {
  struct Result { var entry: GenerationModelEntry; var alias: String; var profile: ModelProfile? }
  static func selection(inspected: Result, linked: GenerationModelRef, library: GenerationLibrarySnapshot, image: Bool, imageVAE: String = "", imageVAEIdentity: String? = nil) throws -> GenerationModelRef {
    guard let admitted = library.modelEntries.first(where: { $0.id == linked.id }), admitted.reference == linked,
      admitted.availability == .installed, let files = inspected.entry.identityFiles, !files.isEmpty,
      GenerationLibraryFileIdentity.signature(files) == inspected.entry.reference.sourceIdentity else { throw StudioError.invalid("The inspected model changed. Select it again.") }
    var selected = admitted.reference
    if image, let profile = inspected.profile, profile.engine == "h3",
      let entry = library.modelEntries.first(where: { if case .nativeH3Image(let id, _) = $0.binding { return id == profile.id }; return false }) {
      guard entry.availability == .installed, imageVAE.isEmpty || imageVAEIdentity == GenerationLibraryFileIdentity.signature([imageVAE]),
        entry.reference.sourceIdentity == GenerationLibraryFileIdentity.signature(files + (imageVAE.isEmpty ? [] : [imageVAE])) else { throw StudioError.invalid("The H3 image components changed during inspection.") }
      selected = entry.reference
    }
    return selected
  }
  static func profileFiles(_ path: String) -> [String] {
    guard path.hasPrefix("/") else { return [] }
    var files = [path]
    func collect(_ value: Any) {
      if let string = value as? String, string.hasPrefix("/") {
        let url = URL(fileURLWithPath: string)
        if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true { files += GenerationLibraryFileIdentity.componentFiles(url) }
        else { files.append(string) }
      } else if let object = value as? [String: Any] { for child in object.values { collect(child) } }
      else if let array = value as? [Any] { for child in array { collect(child) } }
    }
    if let raw = try? json(URL(fileURLWithPath: path)) { collect(raw["components"] ?? [:]); collect(raw["loras"] ?? [:]); collect(raw["config"] ?? [:]) }
    return Array(Set(files)).sorted()
  }
  static func json(_ file: URL) throws -> [String: Any] {
    let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, let size = values.fileSize, (1...1024 * 1024).contains(size) else { throw StudioError.invalid("Choose a regular model manifest or recipe under 1 MiB.") }
    guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] else { throw StudioError.invalid("Invalid model manifest.") }
    return object
  }
  static func inspect(_ source: URL, profilesDirectory: String, preferredVariant: String, shared: String, libraryRoot: String, explicitVariant: Bool = false) throws -> Result {
    if (try source.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
      if let restored = try restoration(source) { return restored }
      let manager = FileManager.default
      let direct = manager.fileExists(atPath: source.appendingPathComponent("config.json").path)
      let inferred = source.pathComponents.reversed().map { $0.lowercased() }.first { ["sft", "base", "moss-image-sft", "moss-image-base"].contains($0) }
        .map { $0.hasSuffix("sft") ? "sft" : "base" }
      guard !direct || inferred != nil || explicitVariant else { throw StudioError.invalid("Choose a MOSS family folder containing Base/SFT, or link an explicitly selected variant in Advanced model setup.") }
      for variant in inferred.map({ [$0] }) ?? (explicitVariant ? [preferredVariant] : [preferredVariant, preferredVariant == "sft" ? "base" : "sft"]) {
        var settings = MOSSImageSettings(); settings.variant = variant
        if manager.fileExists(atPath: source.appendingPathComponent("config.json").path) { settings.transformerDirectory = source.path }
        if manager.fileExists(atPath: source.appendingPathComponent("moss_quantization.json").path) {
          let marker = try json(source.appendingPathComponent("moss_quantization.json"))
          guard marker["format"] as? String == "weetodd-moss-affine-v1", let bits = marker["bits"] as? Int, [4, 8].contains(bits), marker["groupSize"] as? Int == 64,
            marker["component"] as? String == "transformer" else {
            throw StudioError.invalid("Choose a validated native MOSS affine Q4/Q8 conversion.")
          }
          settings.quantizationBits = bits
        }
        var remembered = MOSSModelLinks(); remembered.sharedDirectory = shared
        let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(settings), remembered: remembered,
          libraries: [source] + (libraryRoot.isEmpty ? [] : [URL(fileURLWithPath: libraryRoot)]))
        let query = links.applying(to: MOSSModelQuery(settings))
        guard !query.transformerDirectory.isEmpty, !query.sharedDirectory.isEmpty else { continue }
        let transformer = URL(fileURLWithPath: query.transformerDirectory), common = URL(fileURLWithPath: query.sharedDirectory)
        let files = [transformer.appendingPathComponent("config.json"), common.appendingPathComponent("text_encoder/config.json"), common.appendingPathComponent("vae/config.json"), common.appendingPathComponent("tokenizer/tokenizer.json"), common.appendingPathComponent("tokenizer/tokenizer_config.json")]
        guard files.allSatisfy({ manager.isReadableFile(atPath: $0.path) }),
          let weights = try? manager.contentsOfDirectory(at: transformer, includingPropertiesForKeys: nil).filter({ $0.pathExtension == "safetensors" }), !weights.isEmpty else { continue }
        let architecture = try json(transformer.appendingPathComponent("config.json"))
        let expected = ["dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42,
          "text_dim": 4096, "latent_channels": 128, "freq_dim": 256, "rope_base": 10000]
        guard expected.allSatisfy({ architecture[$0.key] as? Int == $0.value }) else {
          throw StudioError.invalid("This folder does not contain the supported MOSS Image transformer architecture.")
        }
        let componentFiles = GenerationLibraryFileIdentity.componentFiles(transformer)
          + GenerationLibraryFileIdentity.componentFiles(common.appendingPathComponent("text_encoder"))
          + GenerationLibraryFileIdentity.componentFiles(common.appendingPathComponent("vae"))
        let alias = "moss|" + transformer.path, precision = settings.quantizationBits.map { "q\($0)" } ?? "bf16"
        let ref = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: "nativeMOSS", family: "moss_image", variant: variant, precision: precision,
          sourceIdentity: GenerationLibraryFileIdentity.signature(files.map(\.path) + componentFiles))
        return .init(entry: .init(reference: ref, name: "MOSS Image " + variant.uppercased() + " · " + precision.uppercased(), workspaces: [.image], tasks: ["image"], availability: .installed,
          binding: .nativeMOSS(transformer: transformer.path, shared: common.path, library: query.libraryDirectory), identityFiles: files.map(\.path) + componentFiles), alias: alias)
      }
      throw StudioError.invalid("No complete supported model package was found. Link the shared library or use Advanced component setup for native video checkpoints.")
    }
    let content = try json(source)
    if content["format"] as? String == "weetodd-qwen-image21-v1" {
      let files = try GenerationLibraryPreparedModels.qwenFiles(source)
      let alias = "qwen|" + source.path
      return .init(entry: .init(reference: .init(id: GenerationLibraryFileIdentity.id(alias), backend: "nativeMLX", family: "qwen_image", precision: "q8", sourceIdentity: GenerationLibraryFileIdentity.signature(files)),
        name: "Qwen Image 2.1 · 8-bit", workspaces: [.image], tasks: ["image"], supportsLoRAs: false, availability: .installed, binding: .nativeQwen(manifest: source.path), identityFiles: files), alias: alias)
    }
    guard content["format"] as? String == "weetodd-headless-v2", let engine = content["engine"] as? String, ["h3", "ltx25"].contains(engine) else {
      throw StudioError.invalid("Choose a supported native H3/LTX model recipe, prepared Qwen manifest or complete MOSS model folder.")
    }
    let catalog = engine == "h3" ? try NativeH3Preparation.catalog(directory: source.deletingLastPathComponent().path) : try NativeLTXPreparation.catalog(directory: source.deletingLastPathComponent().path)
    guard let admitted = catalog.first(where: { $0["id"] as? String == source.path }) else { throw StudioError.invalid("This native recipe is incomplete or incompatible. Use Advanced component setup to link its supported components.") }
    let destinationRoot = URL(fileURLWithPath: profilesDirectory).standardizedFileURL
    var destination = source
    if source.deletingLastPathComponent() != destinationRoot {
      try FileManager.default.createDirectory(at: destinationRoot, withIntermediateDirectories: true)
      destination = destinationRoot.appendingPathComponent(source.deletingPathExtension().lastPathComponent + "-" + UUID().uuidString + ".json")
      try JSONSerialization.data(withJSONObject: content, options: [.sortedKeys]).write(to: destination, options: .atomic)
    }
    let descriptor = (admitted["generation"] as? [String: Any]).flatMap { try? JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: $0)) }
    let profile = ModelProfile(id: destination.path, name: source.deletingPathExtension().lastPathComponent, engine: engine, task: admitted["task"] as? String ?? "t2v", generation: descriptor)
    let alias = "profile|" + destination.path
    let files = profileFiles(destination.path)
    let ref = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: engine, family: engine, sourceIdentity: GenerationLibraryFileIdentity.signature(files))
    return .init(entry: .init(reference: ref, name: profile.name, workspaces: [.clip], tasks: descriptor?.supportedTasks ?? [profile.task], availability: .installed, binding: .nativeVideo(profileID: destination.path), identityFiles: files), alias: alias, profile: profile)
  }
  private static func restoration(_ source: URL) throws -> Result? {
    func result(_ alias: String, name: String, tool: String, variant: String? = nil, precision: String? = nil, components: [String: String], files: [String]) -> Result {
      let movie = ["relayVSR", "reCaVSR"].contains(tool)
      let ref = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: "restoration", family: tool, variant: variant, precision: precision,
        sourceIdentity: GenerationLibraryFileIdentity.signature(files))
      return .init(entry: .init(reference: ref, name: name, workspaces: [.restoration, .finishing] + (movie ? [.clip] : []), tasks: [tool] + (movie ? ["video_upscale"] : []), supportsLoRAs: false,
        availability: .installed, binding: .restoration(tool: tool, components: components), identityFiles: files), alias: alias)
    }
    let roots = [source] + (["turbo", "base", "x2", "x4", "official", "int8"].contains(source.lastPathComponent) ? [source.deletingLastPathComponent()] : [])
    for root in roots {
      for variant in [NativeFlashVSRRecipe.Variant.turbo, .base] { for decoder in [NativeFlashVSRRecipe.Decoder.tiny, .wan] {
        guard let files = try? NativeFlashVSRAssets.resolve(root: root, variant: variant, decoder: decoder) else { continue }
        let label = decoder == .tiny ? "tiny" : "vae"
        return result("flash|\(root.path)|\(variant.rawValue)|\(label)", name: "FlashVSR+ \(variant.rawValue.capitalized) · \(label)", tool: "flashVSR", variant: variant.rawValue,
          components: ["modelRoot": root.path, "variant": variant.rawValue, "decoder": label], files: files.files.map(\.path))
      } }
      for scale in NativeRelayVSRScale.allCases {
        guard let folder = try? scale.resolveDirectory(root) else { continue }
        let files = scale.checkpointNames.map { folder.appendingPathComponent($0) }
        guard let wan = try? GenerationLibraryInventory.metadata(files[0]), let decoder = try? GenerationLibraryInventory.metadata(files[1]),
          (try? scale.validateMetadata(wan: wan, flash: decoder)) != nil else { continue }
        return result("relay|\(folder.path)|\(scale.rawValue)", name: "RelayVSR \(scale.label)", tool: "relayVSR", variant: "x\(scale.rawValue)",
          components: ["modelDirectory": folder.path, "scale": String(scale.rawValue)], files: files.map(\.path))
      }
      for precision in [ReCaPrecision.bf16, .int8ConvRot] {
        guard let assets = try? ReCaReleasedAssets.resolve(directory: root, precision: precision) else { continue }
        return result("reca|\(root.path)|\(precision.rawValue)", name: "ReCaVSR · \(precision.rawValue.uppercased())", tool: "reCaVSR", precision: precision.rawValue,
          components: ["modelDirectory": root.path, "precision": precision.rawValue], files: assets.files.map { $0.file.path })
      }
    }
    return nil
  }
}
