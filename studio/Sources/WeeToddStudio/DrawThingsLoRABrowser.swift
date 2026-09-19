import StudioCore
import SwiftUI

struct DrawThingsLoRABrowser: View {
  @EnvironmentObject var store: StudioStore
  var search: String
  var compatibleOnly: Bool
  var imageWorkspace = false
  var body: some View {
    if store.drawThingsConnections.isEmpty {
      Text("Add a Draw Things connection to browse its installed LoRAs.")
        .font(.caption).foregroundStyle(.secondary)
    }
    ForEach(store.drawThingsConnections) { connection in
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Text("Draw Things · \(connection.name)").font(.headline)
          Spacer()
          Button(store.drawThingsDiscovery.loading.contains(connection.id) ? "Refreshing…" : "Refresh") {
            Task { await store.discoverDrawThings(connection, force: true) }
          }.disabled(store.drawThingsDiscovery.loading.contains(connection.id)).controlSize(.small)
        }
        let entries = entries(connection.id)
        ForEach(entries, id: \.id) { entry in
          let selected = store.drawThingsLoRASelection(imageWorkspace: imageWorkspace)
          let compatible = selected?.profileID == connection.id
            && entry.compatibleModelIDs.contains(selected?.modelID ?? "")
          let used = compatible && selected?.loras.contains { $0.modelID == entry.id } == true
          VStack(alignment: .leading, spacing: 6) {
            Text(entry.name).font(.subheadline.bold()).lineLimit(2)
            Text("Draw Things · \(entry.family)").font(.caption).foregroundStyle(.secondary)
            if !compatible {
              Text("Requires a compatible model on this Draw Things connection")
                .font(.caption2).foregroundStyle(.secondary)
            }
            Button(used ? "In current stack" : "Add to current stack") {
              store.addDrawThingsLoRA(profileID: connection.id, modelID: entry.id, imageWorkspace: imageWorkspace)
            }.disabled(!compatible || used).controlSize(.small)
          }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8)).help(entry.id)
        }
        if entries.isEmpty {
          Text(store.drawThingsCatalogs[connection.id] == nil
            ? "Refresh to load installed LoRAs. No generation is submitted."
            : "No matching LoRAs. Check the compatibility filter and search. For a local server, enable Model Browsing in Draw Things and refresh.")
            .font(.caption).foregroundStyle(.secondary)
        }
        if let error = store.drawThingsDiscovery.errors[connection.id] {
          Text(error).font(.caption).foregroundStyle(.orange)
        }
      }
    }
  }
  func entries(_ profileID: String) -> [StudioStore.DiscoveredDrawThingsLoRA] {
    let selected = store.drawThingsLoRASelection(imageWorkspace: imageWorkspace)
    var seen = Set<String>()
    return (store.drawThingsCatalogs[profileID]?["loras"] as? [[String: Any]] ?? []).compactMap { item in
      guard let id = item["id"] as? String, seen.insert(id).inserted,
        let name = item["name"] as? String, let family = item["family"] as? String,
        let models = item["compatibleModelIDs"] as? [String],
        search.isEmpty || name.localizedCaseInsensitiveContains(search) || id.localizedCaseInsensitiveContains(search)
      else { return nil }
      if compatibleOnly, imageWorkspace || store.selectedClip != nil {
        guard let selected, selected.profileID == profileID,
          models.contains(selected.modelID) else { return nil }
      }
      return StudioStore.DiscoveredDrawThingsLoRA(id: id, name: name, family: family, compatibleModelIDs: models)
    }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
}
