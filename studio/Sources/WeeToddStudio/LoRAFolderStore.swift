import AppKit
import StudioCore

@MainActor extension StudioStore {
  var defaultLoRAFolder: URL { dataDirectory.appendingPathComponent("Models/LoRAs") }
  var loraFolders: [LoRAFolder] {
    runtime.loraFolders ?? [LoRAFolder(path: defaultLoRAFolder.path)]
  }
  var automaticLoRADiscoveryEnabled: Bool {
    runtime.automaticallyDiscoverLoRAs ?? (runtime.loraFolders?.isEmpty != true)
  }
  var loraFolderConfigurationKey: String {
    let manual = loraFolders.map { "\($0.path)|\($0.enabled)|\($0.recursive)|\($0.modelHint?.rawValue ?? "")" }
      .joined(separator: "\n")
    guard automaticLoRADiscoveryEnabled else { return manual + "|automatic:false" }
    let links = (allAssets.filter { $0.kind == .lora } + loraGroups.flatMap { $0.members.map(\.asset) })
      .map { $0.path + "|" + ($0.loraModel?.rawValue ?? "") }.sorted().joined(separator: "\n")
    return manual + "|automatic:true|" + runtime.profilesDirectory + "|" + links
      + "|" + profiles.map(\.id).sorted().joined(separator: "\n") + "|" + (runtime.rippleAdapterPath ?? "")
  }
  var folderLoRAEntries: [LoRAFolderEntry] {
    loraFolderScanKey == loraFolderConfigurationKey ? scannedLoRAEntries : []
  }
  func refreshLoRAFolders() async {
    guard !loraFolderScanBusy else { loraFolderRefreshRequested = true; return }
    loraFolderScanBusy = true
    defer { loraFolderScanBusy = false }
    repeat {
      loraFolderRefreshRequested = false
      let key = loraFolderConfigurationKey, settings = runtime, folders = loraFolders
      let assets = allAssets, groups = loraGroups, automatic = automaticLoRADiscoveryEnabled
      do {
        if folders.contains(where: { $0.enabled && $0.path == defaultLoRAFolder.path }) {
          try FileManager.default.createDirectory(at: defaultLoRAFolder, withIntermediateDirectories: true)
        }
        let discovery = try await Task.detached(priority: .utility) {
          try LoRAFolderDiscovery.discover(folders: folders, profilesDirectory: settings.profilesDirectory,
            assets: assets, groups: groups, additionalPaths: [settings.rippleAdapterPath].compactMap { $0 }, enabled: automatic)
        }.value
        let result: [String: Any]
        if !discovery.folders.contains(where: \.enabled) {
          result = ["entries": [], "warnings": []]
        } else if NativeLoRAImport.usesNativeScan(runtime: settings) {
          result = try await NativeLoRAImport.scan(discovery.folders, modelHints: discovery.modelHints)
        } else {
          let payload = try discovery.folders.map { try $0.object() }
          result = try await descriptionBridge.independent().invoke("lora-scan", runtime: settings,
            payload: ["folders": payload, "fileModelHints": discovery.modelHints.mapValues(\.rawValue)],
            output: dataDirectory.appendingPathComponent("Cache/lora-index.json"))
        }
        if key == loraFolderConfigurationKey {
          scannedLoRAEntries = try JSONDecoder().decode([LoRAFolderEntry].self,
            from: JSONSerialization.data(withJSONObject: result["entries"] ?? []))
          automaticallyDiscoveredLoRAFolders = discovery.automaticFolders
          loraFolderScanKey = key
          loraFolderWarnings = discovery.warnings + (result["warnings"] as? [String] ?? [])
        } else { loraFolderRefreshRequested = true }
      } catch {
        if key == loraFolderConfigurationKey {
          loraFolderWarnings = ["Could not refresh LoRA folders: \(error.localizedDescription)"]
        } else { loraFolderRefreshRequested = true }
      }
    } while loraFolderRefreshRequested
  }
  func chooseLoRAFolders() {
    let panel = NSOpenPanel()
    panel.title = "Add LoRA folders"
    panel.message = "Studio indexes compatible SafeTensors adapters in place. Originals stay in these folders."
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = true
    guard panel.runModal() == .OK else { return }
    var folders = loraFolders
    var paths = Set(folders.map { URL(fileURLWithPath: $0.path).standardizedFileURL.resolvingSymlinksInPath().path })
    for url in panel.urls {
      let path = url.standardizedFileURL.resolvingSymlinksInPath().path
      if paths.insert(path).inserted { folders.append(LoRAFolder(path: path)) }
    }
    guard folders.count <= 64 else { error = "Choose up to 64 LoRA folders."; return }
    runtime.loraFolders = folders
    saveRuntime(reloadProfiles: false)
    Task { await refreshLoRAFolders() }
  }
  func openDefaultLoRAFolder() {
    do {
      try FileManager.default.createDirectory(at: defaultLoRAFolder, withIntermediateDirectories: true)
      if !loraFolders.contains(where: { $0.path == defaultLoRAFolder.path }) {
        runtime.loraFolders = loraFolders + [LoRAFolder(path: defaultLoRAFolder.path)]
        saveRuntime(reloadProfiles: false)
      }
      NSWorkspace.shared.open(defaultLoRAFolder)
    } catch { self.error = error.localizedDescription }
  }
  func drawThingsLoRASelection(imageWorkspace: Bool = false) -> DrawThingsSelection? {
    if imageWorkspace {
      guard let draft = imageDraft else { return nil }
      return DrawThingsSelection(profileID: draft.profileID, modelID: draft.modelID,
        modelFamily: drawThingsModelFamily(profileID: draft.profileID, modelID: draft.modelID), loras: draft.loras)
    }
    guard selectedClip?.engine == .drawThings else { return nil }
    return selectedClip?.drawThings
  }
  func addDrawThingsLoRA(profileID: String, modelID: String, imageWorkspace: Bool = false) {
    guard let selection = drawThingsLoRASelection(imageWorkspace: imageWorkspace),
      selection.profileID == profileID,
      drawThingsLoRAs(profileID: profileID, modelID: selection.modelID).contains(where: { $0.id == modelID })
    else { error = "Select a compatible Draw Things model and refresh its connection first."; return }
    guard !selection.loras.contains(where: { $0.modelID == modelID }) else { return }
    guard selection.loras.count < 16 else { error = "Use up to 16 Draw Things LoRAs per stack."; return }
    var member = DrawThingsLoRA(modelID: modelID)
    let metadata = drawThingsPromptMetadata(profileID: profileID, modelID: modelID)
    member.promptMetadata = metadata
    if var draft = imageWorkspace ? imageDraft : nil {
      draft.loras.append(member)
      do {
        try insertLoRAPromptGuidance([metadata], into: &draft)
        imageDraft = draft; imageEstimate = nil
      } catch { self.error = error.localizedDescription }
    } else if !imageWorkspace {
      editClip { clip in
        clip.drawThings?.loras.append(member)
        clip.prompt = LoRAPromptInsertion.applying([metadata], to: clip.prompt)
      }
    }
  }
}
