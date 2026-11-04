import StudioCore
import SwiftUI

struct LoRAFolderSettings: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("LoRA model folders").font(.headline)
      Toggle("Discover linked model libraries automatically", isOn: Binding(get: {
        store.automaticLoRADiscoveryEnabled
      }, set: { enabled in
        store.runtime.automaticallyDiscoverLoRAs = enabled
        store.saveRuntime(reloadProfiles: false)
        Task { await store.refreshLoRAFolders() }
      }))
      Text("Studio finds libraries from linked model setups and imported LoRAs. You can add other folders below. Files stay in place; clips and groups keep their saved links.")
        .font(.caption).foregroundStyle(.secondary)
      if store.automaticLoRADiscoveryEnabled && !store.automaticallyDiscoveredLoRAFolders.isEmpty {
        DisclosureGroup("Automatically discovered · \(store.automaticallyDiscoveredLoRAFolders.count)") {
          ForEach(store.automaticallyDiscoveredLoRAFolders) { folder in
            HStack {
              Text(folder.path).font(.caption2).textSelection(.enabled)
              Spacer()
              Button("Exclude") {
                guard store.loraFolders.count < 64 else {
                  store.error = "Choose up to 64 LoRA folders. Remove a folder before adding another exclusion."; return
                }
                var excluded = folder; excluded.enabled = false
                store.runtime.loraFolders = store.loraFolders + [excluded]
                store.saveRuntime(reloadProfiles: false)
                Task { await store.refreshLoRAFolders() }
              }.controlSize(.small)
            }
          }
        }
      }
      ForEach(store.loraFolders) { folder in
        VStack(alignment: .leading, spacing: 6) {
          HStack {
            Toggle(URL(fileURLWithPath: folder.path).lastPathComponent, isOn: binding(folder, \.enabled))
            Spacer()
            Button("Remove") {
              store.runtime.loraFolders = store.loraFolders.filter { $0.path != folder.path }
              store.saveRuntime(reloadProfiles: false)
            }.help("Remove this folder from discovery; keep all files")
          }
          Text(folder.path).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
          HStack {
            Toggle("Include subfolders", isOn: binding(folder, \.recursive))
            Picker("Missing model metadata", selection: binding(folder, \.modelHint)) {
              Text("Ask per LoRA").tag(Optional<LoRAModel>.none)
              ForEach(LoRAModel.allCases) { Text($0.label).tag(Optional($0)) }
            }
          }.font(.caption).controlSize(.small)
        }.padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
      }
      HStack {
        Button("Add folders…") { store.chooseLoRAFolders() }
        Button("Open default folder") { store.openDefaultLoRAFolder() }
        Button(store.loraFolderScanBusy ? "Scanning…" : "Refresh folders") {
          Task { await store.refreshLoRAFolders() }
        }.disabled(store.loraFolderScanBusy)
      }
      Text("Draw Things LoRAs are discovered through its connection. Its .ckpt stores are not native SafeTensors adapters.")
        .font(.caption).foregroundStyle(.secondary)
      ForEach(store.loraFolderWarnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
    }
  }
  func binding<T>(_ folder: LoRAFolder, _ key: WritableKeyPath<LoRAFolder, T>) -> Binding<T> {
    Binding(get: { store.loraFolders.first { $0.path == folder.path }?[keyPath: key] ?? folder[keyPath: key] },
      set: { value in
        var folders = store.loraFolders
        if let index = folders.firstIndex(where: { $0.path == folder.path }) { folders[index][keyPath: key] = value }
        store.runtime.loraFolders = folders
        store.saveRuntime(reloadProfiles: false)
      })
  }
}
