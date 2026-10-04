import StudioCore
import SwiftUI

struct DiffusionVAEOptionsControl:View {
  @Binding var settings:LTX25DiffusionVAESettings?
  private func update(_ body:(inout LTX25DiffusionVAESettings)->Void) {
    var value=settings ?? LTX25DiffusionVAESettings();body(&value);settings=value
  }
  var body:some View {
    VStack(alignment:.leading,spacing:8) {
      Toggle("Advanced Diffusion VAE controls",isOn:Binding(get:{ settings?.experimentalEnabled == true },set:{ enabled in
        if enabled { update { $0.experimentalEnabled=true } } else { settings=nil }
      }))
      Text("The selected VAE checkpoint determines the decoder architecture. Turning advanced controls off restores combined defaults; it does not switch checkpoints.").font(.caption2).foregroundStyle(.secondary)
      if settings != nil {
        Picker("Decoder execution",selection:Binding(get:{ settings?.optimization ?? .combined },set:{ mode in update {
          $0.optimization=mode;$0.stage4TileWidth=mode == .stage4WidthTiles ? max(1,$0.stage4TileWidth):0
        } })) {
          ForEach(LTX25DiffusionVAEOptimization.allCases,id:\.rawValue) { Text($0.rawValue).tag($0) }
        }
        Stepper("Query chunk: \(settings?.queryChunkSize ?? 512)",value:Binding(get:{ settings?.queryChunkSize ?? 512 },set:{ n in update { $0.queryChunkSize=n } }),in:1...65_536)
        Stepper("Context width chunks: \(settings?.contextWidthChunks ?? 4)",value:Binding(get:{ settings?.contextWidthChunks ?? 4 },set:{ n in update { $0.contextWidthChunks=n } }),in:1...4096)
        if settings?.optimization == .stage4WidthTiles {
          Stepper("Stage-four width cells: \(settings?.stage4TileWidth ?? 1)",value:Binding(get:{ settings?.stage4TileWidth ?? 1 },set:{ n in update { $0.stage4TileWidth=n } }),in:1...4096)
        }
        Text("Select the official one-step Diffusion VAE checkpoint in Model Setup. Bounds depend on full decode geometry and this Mac’s stage allowance. Metal modes and width stripes need separate numerical qualification; no automatic fallback.").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
}
struct DiffusionVAEInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  var body:some View {
    DiffusionVAEOptionsControl(settings:Binding(get:{ store.selectedClip?.generationSelection?.ltx25DiffusionVAE },set:{ value in
      store.editClip { edited in
        var selection=edited.generationSelection ?? GenerationSelection(task:edited.inferredTask,preset:.custom)
        selection.ltx25DiffusionVAE=value;edited.generationSelection=selection
      }
    }))
  }
}
