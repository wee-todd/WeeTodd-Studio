import StudioCore
import SwiftUI

struct LTX25ConfigInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  @State private var saving = false
  @State private var configName = ""

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Picker("Config", selection: Binding(get: {
        clip.generationSelection?.ltx25ConfigID ?? "current"
      }, set: { id in
        if let config = store.ltx25Configs.first(where: { $0.id == id }) { store.applyLTX25Config(config) }
      })) {
        if store.selectedLTX25Config == nil {
          Text(clip.generationSelection?.ltx25ConfigName ?? "Current settings")
            .tag(clip.generationSelection?.ltx25ConfigID ?? "current")
        }
        Section("Built-in") {
          ForEach(LTX25GenerationConfig.builtIns) { config in
            Text(config.name).tag(config.id).disabled(!config.supports(clip, in: store.project))
          }
        }
        if !store.savedLTX25Configs.isEmpty {
          Section("My configs") {
            ForEach(store.savedLTX25Configs) { config in
              Text(config.name).tag(config.id).disabled(!config.supports(clip, in: store.project))
            }
          }
        }
      }
      .accessibilityIdentifier("ltx25-config-selection")
      HStack {
        Button("Try recommended settings") {
          store.applyLTX25Config(clip.generationSelection?.ltx25Guidance == nil ? .distilled : .dev)
        }
        Spacer()
        Menu {
          Button("Save as…") {
            configName = (clip.generationSelection?.ltx25ConfigName ?? "LTX 2.5") + " copy"
            saving = true
          }
          Button("Reset config") { store.resetLTX25Config() }
          if let config = store.selectedLTX25Config, !config.isBuiltIn {
            Button("Delete saved config", role: .destructive) { store.deleteLTX25Config(config.id) }
          }
        } label: { Image(systemName: "ellipsis.circle") }
        .accessibilityLabel("Manage LTX 2.5 configs")
      }.font(.caption2)
      if store.ltx25ConfigIsModified {
        Text("Modified · Save as a new config or reset to its saved settings.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      if clip.profileID == "auto", clip.generationSelection?.ltx25Guidance == nil,
        clip.generationSelection?.ltx25SingleStage?.method != .cfgpp, !clip.negativePrompt.isEmpty {
        Text("This distilled config retains your negative prompt for Dev or CFG++; it does not evaluate it.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      if clip.generationSelection?.ltx25Guidance != nil {
        Text(clip.profileID == "auto"
          ? "Dev config · experimental. Required model components and the refinement adapter resolve automatically."
          : "Dev config · experimental. This saved setup retains its linked component binding.")
          .font(.caption2).foregroundStyle(.secondary)
      }
    }
    .alert("Save LTX 2.5 config", isPresented: $saving) {
      TextField("Config name", text: $configName)
      Button("Cancel", role: .cancel) { }
      Button("Save") { Task { _ = await store.saveLTX25Config(name: configName) } }
        .disabled(configName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    } message: {
      Text("Save sampling, guidance and advanced execution settings. Prompts, media, LoRAs, seed, canvas and shot timing stay with each clip.")
    }
  }
}
