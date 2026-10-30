import InferenceContracts
import StudioCore
import SwiftUI

/// Shared controls for the standalone tool and a generation finishing stage.
struct H3RegenerateInspector: View {
  @EnvironmentObject var store: StudioStore
  var processing = false
  private var settings: H3RegenerateSettings {
    processing ? store.selectedClip?.videoProcessing?.steps.first(where: \.isH3Regenerate)?.h3Regenerate ?? store.h3RegenerateSettings
      : store.h3RegenerateSettings
  }
  private func binding<T>(_ key: WritableKeyPath<H3RegenerateSettings, T>) -> Binding<T> {
    Binding(get: { settings[keyPath: key] }, set: { value in
      if processing { store.editH3VideoProcessing { $0[keyPath: key] = value } }
      else { store.editH3Regenerate { $0[keyPath: key] = value } }
    })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("H3 Regenerate · native Swift/MLX · three fixed refinement steps")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        Text("Output")
        TextField("Width", value: binding(\.targetWidth), format: .number).frame(maxWidth: 100)
        Text("×")
        TextField("Height", value: binding(\.targetHeight), format: .number).frame(maxWidth: 100)
        Menu("Presets") {
          Button("1080p") { target(1920, 1080) }
          Button("1440p · publisher") { target(2560, 1440) }
        }
      }.textFieldStyle(.roundedBorder)
      Picker("Decoder", selection: binding(\.decoder)) {
        Text("Released H3 decoder").tag(H3RegenerateJob.Decoder.releasedH3)
        Text("Fast H3 · whole context").tag(H3RegenerateJob.Decoder.fastH3)
      }.accessibilityIdentifier("h3-regenerate-decoder")
      Picker("Publisher source preparation", selection: binding(\.fit)) {
        Text("Padded source → target grid").tag(FlashVSRSettings.Fit.fit)
        Text("Fill source grid · center crop → target grid").tag(FlashVSRSettings.Fit.fill)
      }
      Text("Publisher preprocessing resizes each latent axis to the target grid. Different grid proportions can change picture aspect; these choices describe H3 preprocessing.")
        .font(.caption).foregroundStyle(.secondary)
      Text("Preserves the selected source interval, 24 FPS and original audio. Direct upscaling supports up to 2× per axis. Other source rates stop before inference.")
        .font(.caption).foregroundStyle(.secondary)
      TextField("Preservation prompt", text: binding(\.prompt), axis: .vertical).lineLimit(2...5)
      if !processing, let plan = store.h3RegeneratePlan {
        Text("Source grid \(plan.sourceWidth) × \(plan.sourceHeight) → model grid \((settings.targetWidth + 31) / 32 * 32) × \((settings.targetHeight + 31) / 32 * 32) · exact delivery \(settings.targetWidth) × \(settings.targetHeight)")
          .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
      }
      DisclosureGroup("Model profile and reproducibility") {
        VStack(alignment: .leading, spacing: 8) {
          Picker("Ordinary Ref2VA components", selection: binding(\.baseProfilePath)) {
            Text("Choose installed components").tag("")
            if !settings.baseProfilePath.isEmpty && !store.profiles.contains(where: { $0.id == settings.baseProfilePath }) {
              Text(URL(fileURLWithPath: settings.baseProfilePath).lastPathComponent).tag(settings.baseProfilePath)
            }
            ForEach(store.profiles.filter { $0.engine == "h3" && $0.task == "ref2va" }) { Text($0.name).tag($0.id) }
          }
          TextField("Model library", text: binding(\.modelLibraryRoot))
          Text("Original Ref2VA base + released Refiner/Turbo/Sharp stack. Strength and sampling schedule are fixed by the publisher profile.")
            .font(.caption).foregroundStyle(.secondary)
          HStack {
            TextField("Seed", value: binding(\.seed), format: .number)
            Button("Randomize") { binding(\.seed).wrappedValue = .random(in: 0...UInt64.max) }
          }
          Picker("Source encoder", selection: binding(\.encoderProfile)) {
            Text("Publisher F32 sampled posterior").tag(H3RegenerateJob.EncoderProfile.publisherPosteriorFP32)
            Text("Native F16 posterior mean").tag(H3RegenerateJob.EncoderProfile.nativePosteriorMeanF16)
          }.onChange(of: settings.encoderProfile) { _, value in
            if processing { store.editH3VideoProcessing { $0.posteriorSeed = value == .publisherPosteriorFP32 ? 0 : nil } }
            else { store.editH3Regenerate { $0.posteriorSeed = value == .publisherPosteriorFP32 ? 0 : nil } }
          }
          if settings.encoderProfile == .publisherPosteriorFP32 {
            TextField("Posterior noise seed", value: Binding(get: { settings.posteriorSeed ?? 0 }, set: { value in
              if processing { store.editH3VideoProcessing { $0.posteriorSeed = value } }
              else { store.editH3Regenerate { $0.posteriorSeed = value } }
            }), format: .number)
          }
          Picker("Latent upscale", selection: binding(\.latentUpscaleMode)) {
            Text("Publisher direct · up to 2×").tag(H3RegenerateJob.UpscaleMode.directPublisher)
            Text("Native multihop · explicit larger scale").tag(H3RegenerateJob.UpscaleMode.multiHopNative)
          }
          Text("Multihop changes preprocessing. The native encoder profile differs from publisher sampling; each choice is recorded in the result.")
            .font(.caption).foregroundStyle(.secondary)
        }.padding(.top, 6)
      }
    }
  }
  private func target(_ width: Int, _ height: Int) {
    binding(\.targetWidth).wrappedValue = width; binding(\.targetHeight).wrappedValue = height
  }
}
