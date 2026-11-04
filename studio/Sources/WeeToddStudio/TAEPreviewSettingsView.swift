import StudioCore
import SwiftUI

struct TAEPreviewSettingsView: View {
  @EnvironmentObject var store: StudioStore

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Picker("Swift LTX 2.5 live previews", selection: Binding(
        get: { store.runtime.ltx25PreviewDecoder ?? "off" },
        set: { store.runtime.ltx25PreviewDecoder = $0 == "off" ? nil : $0 })) {
        Text("Off · existing decoded previews").tag("off")
        Text("TAELTX2.3 · approximate").tag("taeltx2_3")
        Text("TAELTX2.3 Wide · approximate").tag("taeltx2_3_wide")
      }
      if store.runtime.ltx25PreviewDecoder == "taeltx2_3" {
        PathPicker(label: "TAELTX2.3 preview weights", value: Binding(
          get: { store.runtime.ltx25PreviewCheckpoint ?? "" },
          set: { store.runtime.ltx25PreviewCheckpoint = $0 }))
      } else if store.runtime.ltx25PreviewDecoder == "taeltx2_3_wide" {
        PathPicker(label: "TAELTX2.3 Wide preview weights", value: Binding(
          get: { store.runtime.ltx25PreviewWideCheckpoint ?? "" },
          set: { store.runtime.ltx25PreviewWideCheckpoint = $0 }))
      }
      Text("TAE previews are approximate sampling snapshots. The final clip always uses its existing full VAE. Link compatible local preview weights; generation does not download them. Ordinary two-stage Distilled and Dev Euler/HQ jobs support TAE previews. Specialized jobs retain their existing decoded previews.")
        .font(.caption).foregroundStyle(.secondary)
      if !store.runtime.usesNativeLTX25 {
        Text("Legacy Python is selected for LTX 2.5. This preview preference applies only when Swift MLX is selected.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}
