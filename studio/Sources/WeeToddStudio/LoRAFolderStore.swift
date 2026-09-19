import AppKit
import StudioCore

@MainActor extension StudioStore {
  var defaultLoRAFolder: URL { dataDirectory.appendingPathComponent("Models/LoRAs") }
  var loraFolders: [LoRAFolder] {
    runtime.loraFolders ?? [LoRAFolder(path: defaultLoRAFolder.path)]
  }
  var loraFolderConfigurationKey: String {
    loraFolders.map { "\($0.path)|\($0.enabled)|\($0.recursive)|\($0.modelHint?.rawValue ?? "")" }
      .joined(separator: "\n")
  }
  var folderLoRAEntries: [LoRAFolderEntry] {
    guard let scanned = loraFolderScanKey,
      scanned.map({ [$0.path, String($0.enabled), String($0.recursive), $0.modelHint?.rawValue ?? ""] })
        == loraFolders.map({ [$0.path, String($0.enabled), String($0.recursive), $0.modelHint?.rawValue ?? ""] })
    else { return [] }
    return scannedLoRAEntries
  }
  func refreshLoRAFolders() async {
    guard !loraFolderScanBusy else { return }
    let folders = loraFolders
    let key = loraFolderConfigurationKey
    let settings = runtime
    loraFolderScanBusy = true
    defer { loraFolderScanBusy = false }
    do {
      if folders.contains(where: { $0.enabled && $0.path == defaultLoRAFolder.path }) {
        try FileManager.default.createDirectory(at: defaultLoRAFolder, withIntermediateDirectories: true)
      }
      let payload = try folders.map { try $0.object() }
      let result = try await descriptionBridge.independent().invoke("lora-scan", runtime: settings,
        payload: ["folders": payload], output: dataDirectory.appendingPathComponent("Cache/lora-index.json"))
      guard key == loraFolderConfigurationKey else { return }
      let entries = try JSONDecoder().decode([LoRAFolderEntry].self,
        from: JSONSerialization.data(withJSONObject: result["entries"] ?? []))
      scannedLoRAEntries = entries
      loraFolderScanKey = folders
      loraFolderWarnings = result["warnings"] as? [String] ?? []
    } catch {
      guard key == loraFolderConfigurationKey else { return }
      loraFolderWarnings = ["Could not refresh LoRA folders: \(error.localizedDescription)"]
    }
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
    if imageWorkspace {
      imageDraft?.loras.append(DrawThingsLoRA(modelID: modelID))
      imageEstimate = nil
    } else {
      editClip { $0.drawThings?.loras.append(DrawThingsLoRA(modelID: modelID)) }
    }
  }
}
