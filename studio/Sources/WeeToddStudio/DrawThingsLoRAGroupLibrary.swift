import StudioCore
import SwiftUI

struct DrawThingsLoRAGroupLibrary: View {
  @EnvironmentObject var store: StudioStore
  var imageWorkspace = false
  @State private var name = ""
  var selection: DrawThingsSelection? { store.drawThingsLoRASelection(imageWorkspace: imageWorkspace) }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Groups · Draw Things").font(.headline)
      Text("Save the current stack, including strengths and disabled LoRAs. Groups stay associated with their connection and compatible models.")
        .font(.caption).foregroundStyle(.secondary)
      TextField("Group name", text: $name).textFieldStyle(.roundedBorder)
      Button("Save current stack as group") {
        if store.saveCurrentDrawThingsLoRAGroup(name: name, imageWorkspace: imageWorkspace) { name = "" }
      }.disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selection?.loras.isEmpty != false)
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 12) {
          ForEach(store.drawThingsLoRAGroups.filter { group in
            guard let selection else { return false }
            return group.profileID == selection.profileID && group.family == selection.modelFamily
              && group.compatibleModelIDs.contains(selection.modelID)
          }) { group in
            VStack(alignment: .leading, spacing: 6) {
              Text(group.name).font(.subheadline.bold())
              Text(group.members.map {
                "\($0.modelID) · \(String(format: "%.2f", $0.weight))\($0.isEnabled ? "" : " · disabled")"
              }.joined(separator: "\n")).font(.caption).foregroundStyle(.secondary)
              HStack {
                Menu("Apply group") {
                  ForEach(LoRAGroupApplicationMode.allCases) { mode in
                    Button(mode.label) {
                      guard var selected = selection else { return }
                      do {
                        try selected.apply(group, mode: mode)
                        if imageWorkspace {
                          store.imageDraft?.loras = selected.loras; store.imageEstimate = nil
                        } else { store.editClip { $0.drawThings = selected } }
                      } catch { store.error = error.localizedDescription }
                    }
                  }
                }
                Spacer()
                Button("Delete") { store.deleteDrawThingsLoRAGroup(group.id) }
                  .help("Delete the group template; applied clips retain their settings")
              }.controlSize(.small)
            }.padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
          }
        }
      }
    }
  }
}

@MainActor extension StudioStore {
  @discardableResult func saveCurrentDrawThingsLoRAGroup(name: String, imageWorkspace: Bool = false) -> Bool {
    guard let selection = drawThingsLoRASelection(imageWorkspace: imageWorkspace) else { return false }
    let discovered = drawThingsLoRAs(profileID: selection.profileID, modelID: selection.modelID)
    guard !selection.loras.isEmpty,
      selection.loras.allSatisfy({ member in
        discovered.contains { $0.id == member.modelID && $0.family == selection.modelFamily }
      }) else { error = "Refresh the connection and use compatible LoRAs before saving a group."; return false }
    let members = discovered.filter { candidate in selection.loras.contains { $0.modelID == candidate.id } }
    let compatible = members.reduce(Set(members.first?.compatibleModelIDs ?? [])) {
      $0.intersection(Set($1.compatibleModelIDs))
    }
    let group = DrawThingsLoRAGroup(name: name, profileID: selection.profileID,
      family: selection.modelFamily, compatibleModelIDs: Array(compatible).sorted(), members: selection.loras)
    do {
      try group.validate(profileID: selection.profileID, family: selection.modelFamily, modelID: selection.modelID)
      try persistDrawThingsLoRAGroups(drawThingsLoRAGroups + [group])
      return true
    } catch { self.error = error.localizedDescription; return false }
  }
  func deleteDrawThingsLoRAGroup(_ id: UUID) {
    do { try persistDrawThingsLoRAGroups(drawThingsLoRAGroups.filter { $0.id != id }) }
    catch { self.error = error.localizedDescription }
  }
  private func persistDrawThingsLoRAGroups(_ groups: [DrawThingsLoRAGroup]) throws {
    try JSONEncoder().encode(groups).write(
      to: dataDirectory.appendingPathComponent("drawthings-lora-groups.json"), options: .atomic)
    drawThingsLoRAGroups = groups
  }
}
