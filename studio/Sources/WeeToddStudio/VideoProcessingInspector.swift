import StudioCore
import SwiftUI

struct VideoProcessingInspector: View {
  @EnvironmentObject var store: StudioStore
  let clip: Clip
  private func binding<T>(_ key: WritableKeyPath<FlashVSRSettings, T>, fallback: T) -> Binding<T> {
    Binding(get: { store.selectedClip?.videoProcessing?.steps.first(where: \.isFlashVSR)?.settings[keyPath: key] ?? fallback },
      set: { value in store.editVideoProcessing { $0[keyPath: key] = value } })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Toggle("Upscale after generation", isOn: Binding(get: { clip.videoProcessing != nil },
        set: { store.setVideoProcessingEnabled($0) }))
        .accessibilityIdentifier("video-processing-enabled")
      if let recipe = clip.videoProcessing {
        Picker("Upscaler", selection: Binding(get: { recipe.hasH3Regenerate ? (recipe.hasFlashVSR ? "h3-regenerate-flashvsr" : "h3-regenerate") : recipe.isChain ? "native-pixel-cq-flashvsr" : recipe.steps[0].kind }, set: { store.setVideoProcessingEngine($0) })) {
          Text("H3 Regenerate · native · 24 FPS").tag("h3-regenerate")
          Text("H3 Regenerate → FlashVSR+").tag("h3-regenerate-flashvsr")
          Text("Native Pixel + CQ → FlashVSR+ Turbo").tag("native-pixel-cq-flashvsr")
          Text("Native Pixel Spatial + CQ · 2×").tag("native-pixel-cq")
          Text("FlashVSR+ · native Swift/MLX").tag("flashvsr-turbo")
        }.accessibilityIdentifier("video-processing-engine")
      }
      if let recipe = clip.videoProcessing, recipe.hasH3Regenerate {
        if recipe.hasFlashVSR { Text("1. H3 Regenerate").font(.caption.bold()) }
        H3RegenerateInspector(processing: true)
      }
      if let recipe = clip.videoProcessing, let settings = recipe.steps.first(where: \.isNative)?.pixelCQ {
        if recipe.isChain { Text("1. Native Pixel Spatial + CQ").font(.caption.bold()) }
        NativePixelCQInspector(settings: settings)
      }
      if let recipe = clip.videoProcessing, let settings = recipe.steps.first(where: \.isFlashVSR)?.settings {
        if recipe.steps.count > 1 { Text("2. FlashVSR+").font(.caption.bold()) }
        Picker("Model", selection: binding(\.variant, fallback: .turbo)) {
          Text("Turbo").tag(FlashVSRSettings.Variant.turbo)
          Text("Base").tag(FlashVSRSettings.Variant.base)
        }
        Picker("Decoder", selection: binding(\.decoder, fallback: .tiny)) {
          Text("Tiny · faster").tag(FlashVSRSettings.Decoder.tiny)
          Text("Full · Wan VAE").tag(FlashVSRSettings.Decoder.vae)
        }.accessibilityIdentifier("video-processing-decoder")
        HStack {
          Text("Target")
          TextField("Width", value: binding(\.width, fallback: settings.width), format: .number)
          Text("×")
          TextField("Height", value: binding(\.height, fallback: settings.height), format: .number)
        }.textFieldStyle(.roundedBorder)
        Picker("Aspect", selection: binding(\.fit, fallback: .fit)) {
          Text("Fit entire image").tag(FlashVSRSettings.Fit.fit)
          Text("Fill and crop").tag(FlashVSRSettings.Fit.fill)
        }
        Text("Original audio and frame rate preserved. Original take stays in Versions.")
          .font(.caption).foregroundStyle(.secondary)
        DisclosureGroup("Restoration settings") {
          Picker("Motion", selection: binding(\.temporalCompression, fallback: 4)) {
            Text("Normal · 4").tag(4); Text("High motion · 2").tag(2); Text("Maximum samples · 1").tag(1)
          }
          Picker("Color", selection: binding(\.colorFix, fallback: .wavelet)) {
            Text("Wavelet").tag(FlashVSRSettings.ColorFix.wavelet)
            Text("AdaIN").tag(FlashVSRSettings.ColorFix.adain); Text("Off").tag(FlashVSRSettings.ColorFix.off)
          }
          TextField("Seed", value: binding(\.seed, fallback: UInt32(0)), format: .number)
          Button("Configure native models…") { store.openVideoUpscale() }
        }
      }
      if clip.videoProcessing != nil {
        if store.videoUpscaleBusy {
          Text(store.videoUpscaleStatus).font(.caption).lineLimit(3)
          Button("Cancel upscale") { store.cancelVideoProcessing() }
        } else if !clip.sourcePath.isEmpty {
          Button("Upscale accepted take") { store.retryClipVideoProcessing() }
            .disabled(store.operationBusy).accessibilityIdentifier("retry-video-processing")
          Text("Reuses completed stages when their settings match. Source generation is not repeated.").font(.caption).foregroundStyle(.secondary)
        }
      }
    }.font(.caption)
  }
}
