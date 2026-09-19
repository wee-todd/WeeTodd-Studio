import AppKit
import StudioCore

@MainActor extension StudioStore {
  func compatibleLoRAs(for engine: Engine) -> [MediaAsset] {
    localLoRALibraryAssets.filter { $0.loraModel?.supports(engine) == true }
  }
  var localLoRALibraryAssets: [MediaAsset] {
    var seen = Set<String>()
    // Reimported library metadata wins; existing clip-owned snapshots remain untouched.
    let candidates = Array(globalAssets.reversed()) + project.assets
      + folderLoRAEntries.filter { ["ready", "needsModel"].contains($0.status) }.map(\.asset)
    return candidates.filter {
      $0.kind == .lora
        && ($0.scope != .clip || $0.owner == selectedClipID)
        && seen.insert(LoRAMember(asset: $0).fileKey).inserted
    }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  func loadLoRAGroups() {
    let file = dataDirectory.appendingPathComponent("lora-groups.json")
    guard FileManager.default.fileExists(atPath: file.path) else { return }
    do {
      loraGroups = try JSONDecoder().decode([LoRAGroup].self, from: Data(contentsOf: file))
    } catch { self.error = "Could not read the LoRA group library: \(error.localizedDescription)" }
  }
  @discardableResult func saveLoRAGroup(_ group: LoRAGroup) -> Bool {
    do {
      try group.validate()
      var updated = loraGroups.filter { $0.id != group.id }
      updated.append(group)
      try writeLoRAGroups(updated)
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
  func deleteLoRAGroup(_ id: UUID) {
    do { try writeLoRAGroups(loraGroups.filter { $0.id != id }) } catch {
      self.error = error.localizedDescription
    }
  }
  private func writeLoRAGroups(_ groups: [LoRAGroup]) throws {
    try JSONEncoder().encode(groups).write(
      to: dataDirectory.appendingPathComponent("lora-groups.json"), options: .atomic)
    loraGroups = groups
  }
  func applyLoRAGroup(_ group: LoRAGroup, mode: LoRAGroupApplicationMode = .add) {
    do {
      try group.validate()
      guard let clip = selectedClip, group.supports(clip.engine) else {
        throw StudioError.invalid("Select a clip matching this group's model.")
      }
      applyLoRAMembers(group.members, groupName: group.name, mode: mode)
    } catch { self.error = error.localizedDescription }
  }
  func applyLoRAMembers(
    _ members: [LoRAMember], groupName: String? = nil, mode: LoRAGroupApplicationMode = .add
  ) {
    guard let clip = selectedClip else {
      error = "Select a clip first."
      return
    }
    do {
      let existing = Set(
        clip.attachments.filter { $0.role == .lora }.compactMap { a in
          allAssets.first { $0.id == a.assetID }.map { LoRAMember(asset: $0).fileKey }
        })
      guard mode == .replace || existing.isDisjoint(with: members.map(\.fileKey)) else {
        throw StudioError.invalid(
          "This clip already uses a LoRA in this selection. Choose Replace current stack or remove its existing entry."
        )
      }
      var updated = project
      try updated.applyLoRAs(members, to: clip.id, groupName: groupName, mode: mode)
      change { $0 = updated }
      notice = "Applied \(groupName ?? members.first?.asset.name ?? "LoRAs") to \(clip.name)."
    } catch { self.error = error.localizedDescription }
  }
  func currentLoRAGroup() -> LoRAGroup? {
    guard let clip = selectedClip else { return nil }
    do {
      return try project.loraGroupSnapshot(for: clip.id, name: clip.name + " LoRAs", additionalAssets: globalAssets)
    } catch { self.error = error.localizedDescription; return nil }
  }
  func setLoRAModel(_ model: LoRAModel, for asset: MediaAsset) {
    if let index = globalAssets.firstIndex(where: { $0.id == asset.id }) {
      globalAssets[index].loraModel = model
      saveGlobals()
    } else if project.assets.contains(where: { $0.id == asset.id }) {
      change { p in
        if let index = p.assets.firstIndex(where: { $0.id == asset.id }) {
          p.assets[index].loraModel = model
        }
      }
    } else {
      // A manual classification becomes an explicit library entry. Applied clips
      // still own their snapshots and generation rechecks the file's metadata.
      var linked = asset
      linked.loraModel = model
      globalAssets.append(linked)
      saveGlobals()
    }
  }
  func chooseLoRAImports(model: LoRAModel, profile: String = "standard", layout: String = "auto", adalnInputGrid: String? = nil) {
    let panel = NSOpenPanel()
    panel.title = "Link \(model.label) LoRAs"
    panel.message =
      "Files remain in place. Checkpoint metadata takes precedence over the selected training model."
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    panel.allowedContentTypes = [.init(filenameExtension: "safetensors") ?? .data]
    guard panel.runModal() == .OK else { return }
    Task { await importURLs(panel.urls, scope: .global, loraModel: model,
      loraProfile: model == .h3 ? profile : nil,
      loraLayout: model == .h3 ? layout : nil,
      loraAdalnInputGrid: model == .h3 ? adalnInputGrid : nil) }
  }
}
