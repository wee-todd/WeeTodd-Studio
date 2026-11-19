import Foundation
import InferenceContracts
import StudioCore

extension GenerationLibraryInventory {
  static func metadata(_ url: URL) throws -> [String: String] {
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true else { throw StudioError.invalid("Choose regular checkpoint files.") }
    let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
    guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else { throw StudioError.invalid("Invalid checkpoint header.") }
    let size = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
    guard size >= 2, size <= 4 * 1024 * 1024, let data = try file.read(upToCount: Int(size)), data.count == Int(size),
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], let metadata = object["__metadata__"] as? [String: String] else {
      throw StudioError.invalid("Missing checkpoint metadata.")
    }
    return metadata
  }
  static func refreshSpecialized(index: inout GenerationLibraryIndex, settings: RuntimeSettings,
    profiles: [ModelProfile], flash: FlashVSRRuntime?, h3: H3RegenerateSettings?) throws {
    let fm = FileManager.default
    // Managed rows must be readmitted, rather than promoted merely because their old files still exist.
    index.refreshAvailability(models: { entry in
      guard entry.identityFiles != nil else { return entry.availability }
      switch entry.binding {
      case .restoration, .ripple:
        return entry.availability == .installed ? .incompatible("The installed components no longer pass model admission. Relink or repair them in Advanced setup.") : entry.availability
      default: return entry.availability
      }
    }, loras: { $0.availability })
    func register(_ alias: String, name: String, family: String, variant: String? = nil, precision: String? = nil,
      workspaces: [GenerationWorkspaceKind] = [.restoration, .finishing], task: String, binding: ResolvedGenerationModel, files: [String]) throws {
      let ref = GenerationModelRef(id: GenerationLibraryFileIdentity.id(alias), backend: family == "ltx25" ? "ripple" : "restoration",
        family: family, variant: variant, precision: precision, sourceIdentity: GenerationLibraryFileIdentity.signature(files))
      try index.registerModel(.init(reference: ref, name: name, workspaces: workspaces + (["relayVSR", "reCaVSR"].contains(task) ? [.clip] : []), tasks: [task] + (["relayVSR", "reCaVSR"].contains(task) ? ["video_upscale"] : []), supportsLoRAs: false,
        availability: .installed, binding: binding, identityFiles: files), alias: alias)
    }
    let savedFlashRoots = index.snapshot.modelEntries.compactMap { entry -> String? in if case .restoration("flashVSR", let fields) = entry.binding { return fields["modelRoot"] }; return nil }
    let knownFlashRoots = index.rootDirectory.map { root in ["FlashVSR-Plus", "FlashVSR-Plus/upstream/FlashVSR_plus", "WeeTodd/FlashVSR-Plus"].map { URL(fileURLWithPath: root).appendingPathComponent($0).path } } ?? []
    for root in Set(([flash?.modelRoot].compactMap { $0 } + savedFlashRoots + knownFlashRoots).filter { !$0.isEmpty }).sorted() {
      for variant in [NativeFlashVSRRecipe.Variant.turbo, .base] {
        for decoder in [NativeFlashVSRRecipe.Decoder.tiny, .wan] {
          guard let assets = try? NativeFlashVSRAssets.resolve(root: URL(fileURLWithPath: root), variant: variant, decoder: decoder) else { continue }
          let label = decoder == .tiny ? "tiny" : "vae"
          try register("flash|\(root)|\(variant.rawValue)|\(label)", name: "FlashVSR+ \(variant.rawValue.capitalized) · \(label)", family: "flashVSR", variant: variant.rawValue,
            task: "flashVSR", binding: .restoration(tool: "flashVSR", components: ["modelRoot": root, "variant": variant.rawValue, "decoder": label]), files: assets.files.map(\.path))
        }
      }
    }
    let knownRoot = index.rootDirectory.map { URL(fileURLWithPath: $0).appendingPathComponent("WeeTodd") }
    let savedRelay = index.snapshot.modelEntries.compactMap { entry -> String? in if case .restoration("relayVSR", let fields) = entry.binding { return fields["modelDirectory"] }; return nil }
    let relayRoots = Set(([settings.relayVSRModelDirectory, knownRoot?.appendingPathComponent("RelayVSR").path].compactMap { $0 } + savedRelay).filter { !$0.isEmpty })
    for root in relayRoots.sorted() {
      for scale in NativeRelayVSRScale.allCases {
        guard let directory = try? scale.resolveDirectory(URL(fileURLWithPath: root)) else { continue }
        let files = scale.checkpointNames.map { directory.appendingPathComponent($0) }
        guard let wan = try? metadata(files[0]),
          let decoder = try? metadata(files[1]),
          (try? scale.validateMetadata(wan: wan, flash: decoder)) != nil else { continue }
        try register("relay|\(directory.path)|\(scale.rawValue)", name: "RelayVSR \(scale.label)", family: "relayVSR", variant: "x\(scale.rawValue)",
          task: "relayVSR", binding: .restoration(tool: "relayVSR", components: ["modelDirectory": directory.path, "scale": String(scale.rawValue)]), files: files.map(\.path))
      }
    }
    let savedReCa = index.snapshot.modelEntries.compactMap { entry -> String? in if case .restoration("reCaVSR", let fields) = entry.binding { return fields["modelDirectory"] }; return nil }
    let recaRoots = Set(([settings.reCaVSRModelDirectory, knownRoot?.appendingPathComponent("ReCaVSR").path].compactMap { $0 } + savedReCa).filter { !$0.isEmpty })
    for root in recaRoots.sorted() {
      for precision in [ReCaPrecision.bf16, .int8ConvRot] {
        guard let assets = try? ReCaReleasedAssets.resolve(directory: URL(fileURLWithPath: root), precision: precision) else { continue }
        try register("reca|\(root)|\(precision.rawValue)", name: "ReCaVSR · \(precision.rawValue.uppercased())", family: "reCaVSR", precision: precision.rawValue,
          task: "reCaVSR", binding: .restoration(tool: "reCaVSR", components: ["modelDirectory": root, "precision": precision.rawValue]), files: assets.files.map { $0.file.path })
      }
    }
    let savedH3 = index.snapshot.modelEntries.compactMap { entry -> H3RegenerateSettings? in
      guard case .restoration("h3Regenerate", let fields) = entry.binding else { return nil }
      var value = H3RegenerateSettings(); value.baseProfilePath = fields["baseProfilePath"] ?? ""; value.modelLibraryRoot = fields["modelLibraryRoot"] ?? ""
      value.publisherEncoderPath = fields["publisherEncoderPath"] ?? ""; value.fastDecoderPath = fields["fastDecoderPath"] ?? ""; return value
    }
    for h3 in savedH3 + [h3].compactMap({ $0 }) where !h3.baseProfilePath.isEmpty {
      let components = ["baseProfilePath": h3.baseProfilePath, "modelLibraryRoot": h3.modelLibraryRoot,
        "publisherEncoderPath": h3.publisherEncoderPath, "fastDecoderPath": h3.fastDecoderPath]
      if let files = try? H3RegeneratePreparation.libraryModelFiles(settings: h3) {
        try register("h3-restore|" + h3.baseProfilePath, name: "H3 Regenerate", family: "h3Regenerate", task: "h3Regenerate",
          binding: .restoration(tool: "h3Regenerate", components: components), files: files)
      }
    }
    // A specialized adapter and its plain distilled backbone form one selectable recipe.
    let adapters = index.snapshot.loraEntries
    let savedRipple = index.snapshot.modelEntries.compactMap { entry -> String? in if case .ripple(_, let adapter) = entry.binding { return adapter }; return nil }
    let ripplePaths = Set(([settings.rippleAdapterPath].compactMap { $0 }) + savedRipple + adapters.compactMap { entry -> String? in
      guard case .nativeFile(let path, let profile, _, _) = entry.binding, profile?.localizedCaseInsensitiveContains("ripple") == true else { return nil }; return path
    })
    for profile in profiles where profile.engine == "ltx25" && profile.generation?.pipelineMode == "distilled" {
      guard let raw = try? GenerationLibraryModelImport.json(URL(fileURLWithPath: profile.id)),
        let components = raw["components"] as? [String: Any],
        (components["loras"] as? [Any] ?? []).isEmpty, (components["ic_loras"] as? [Any] ?? []).isEmpty,
        ["msr_lora_path", "distilled_lora_path"].allSatisfy({ (components[$0] as? String ?? "").isEmpty }) else { continue }
      let paths = ["transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path"].compactMap { components[$0] as? String }
      guard paths.count == 4, paths.allSatisfy({ fm.fileExists(atPath: $0) }) else { continue }
      for adapter in ripplePaths.sorted() where fm.isReadableFile(atPath: adapter) {
        guard let files = try? NativeRipplePreparation.libraryModelFiles(profileID: profile.id, adapterPath: adapter, profilesDirectory: settings.profilesDirectory) else { continue }
        try register("ripple|" + profile.id + "|" + adapter, name: "Ripple · " + profile.name, family: "ltx25", variant: "distilled",
          workspaces: [.ripple], task: "ripple", binding: .ripple(profileID: profile.id, adapterPath: adapter), files: files)
      }
      let pathsInLibrary = adapters.compactMap { entry -> String? in if case .nativeFile(let path, _, _, _) = entry.binding { return path }; return nil }
      let pixel = pathsInLibrary.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased() == "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" }
      let cq = pathsInLibrary.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased() == "ltx2.5-cq-enhancer-lora-v2.safetensors" }
      for enhancer in cq.prefix(8) {
        var options = NativePixelCQSettings(); options.profileID = profile.id; options.cqAdapterPath = enhancer
        guard let admitted = try? NativePixelCQRuntime.resolve(settings: options, profilesDirectory: settings.profilesDirectory,
          workerPath: "", ffmpegPath: "", ffprobePath: "", cqCandidates: [enhancer], pixelCandidates: pixel),
          let adapter = admitted.components["pixel_spatial_adapter"] else { continue }
        try register("pixel|" + profile.id + "|" + adapter + "|" + enhancer, name: "Pixel Spatial + CQ · " + profile.name, family: "pixelCQ",
          workspaces: [.finishing], task: "pixelCQ", binding: .restoration(tool: "pixelCQ", components: ["profileID": profile.id, "pixelAdapterPath": adapter, "cqAdapterPath": enhancer]), files: [profile.id, adapter, enhancer] + paths)
      }
    }
  }
}
