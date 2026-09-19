import StudioCore
import SwiftUI

struct LoRAFolderSettings: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("LoRA model folders").font(.headline)
      Text("Add existing libraries, including folders on external drives. Files stay in place. Removing a folder removes it from discovery; clips and groups keep their saved links.")
        .font(.caption).foregroundStyle(.secondary)
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
