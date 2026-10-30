import StudioCore
import SwiftUI

struct NativePixelCQInspector: View {
  @EnvironmentObject var store: StudioStore
  let settings: NativePixelCQSettings
  private func binding<T>(_ key: WritableKeyPath<NativePixelCQSettings, T>) -> Binding<T> {
    Binding(get: { store.selectedClip?.videoProcessing?.steps.first(where: \.isNative)?.pixelCQ?[keyPath: key] ?? settings[keyPath: key] },
      set: { value in store.editPixelCQProcessing { $0[keyPath: key] = value } })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Native Swift/MLX · 2× source dimensions · three refinement evaluations")
        .font(.caption).foregroundStyle(.secondary)
      LabeledContent("Pixel Spatial strength", value: settings.pixelStrength.formatted(.number.precision(.fractionLength(2))))
      Slider(value: binding(\.pixelStrength), in: 0.05...2, step: 0.05)
        .accessibilityLabel("Pixel Spatial strength").accessibilityIdentifier("pixel-cq-pixel-strength")
      LabeledContent("CQ strength", value: settings.cqStrength.formatted(.number.precision(.fractionLength(2))))
      Slider(value: binding(\.cqStrength), in: 0...2, step: 0.05)
        .accessibilityLabel("CQ strength").accessibilityIdentifier("pixel-cq-cq-strength")
      Text("Pixel guides the 2× upscale. CQ adds detail in the same pass; 0 disables CQ. Source audio and FPS are preserved.")
        .font(.caption).foregroundStyle(.secondary)
      DisclosureGroup("Native upscale settings") {
        LabeledContent("Refinement strength", value: settings.refinementStrength.formatted(.number.precision(.fractionLength(2))))
        Slider(value: binding(\.refinementStrength), in: 0.05...0.9, step: 0.05)
          .accessibilityLabel("Refinement strength")
        Text("Refinement controls how much the source changes. It is separate from both LoRA strengths. Tested: Pixel 1.0, CQ 0.5, refinement 0.75.")
          .font(.caption).foregroundStyle(.secondary)
        Picker("Native LTX components", selection: binding(\.profileID)) {
          Text("Automatic · installed distilled models").tag("auto")
          ForEach(store.profiles.filter { $0.engine == "ltx25" && ($0.generation?.pipelineMode ?? "distilled") == "distilled" }) {
            Text($0.name).tag($0.id)
          }
        }
        Text(settings.pixelAdapterPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Pixel Spatial: use native model profile")
          .font(.caption).lineLimit(2)
        Button("Choose Pixel Spatial…") { store.choosePixelCQAdapter(cq: false) }
        Text(settings.cqAdapterPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "CQ V2: use installed LoRA library")
          .font(.caption).lineLimit(2)
        Button("Choose CQ V2…") { store.choosePixelCQAdapter(cq: true) }
        Text("Requires a source on the 32-pixel grid at 1–60 constant FPS. Unsupported sources stop before inference; FPS is never converted automatically.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}
