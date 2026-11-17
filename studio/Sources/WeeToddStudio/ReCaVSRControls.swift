import AppKit
import InferenceContracts
import StudioCore
import SwiftUI

struct ReCaVSRControls: View {
  @EnvironmentObject var store: StudioStore
  @Binding var settings: ReCaVSRSettings
  private func binding<T>(_ key: WritableKeyPath<ReCaVSRSettings, T>) -> Binding<T> {
    Binding(get: { settings[keyPath: key] }, set: { settings[keyPath: key] = $0 })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Experimental ReCaVSR · source frame timing and audio retained")
        .font(.caption).foregroundStyle(.secondary)
      Picker("Scale", selection: binding(\.scale)) {
        Text("4×").tag(4.0); Text("2×").tag(2.0); Text("1.5×").tag(1.5)
      }.accessibilityIdentifier("recavsr-scale")
      Picker("Precision", selection: binding(\.precision)) {
        Text("BF16 reference").tag(ReCaPrecision.bf16)
        Text("INT8 ConvRot").tag(ReCaPrecision.int8ConvRot)
      }
      Picker("Color correction", selection: binding(\.colorFix)) {
        Text("None").tag(ReCaColorFix.none); Text("AdaIN").tag(ReCaColorFix.adain); Text("Wavelet").tag(ReCaColorFix.wavelet)
      }
      TextField("Seed", value: binding(\.seed), format: .number)
      Toggle("Keep both model stages in memory", isOn: Binding(get: { settings.residency == .resident },
        set: { settings.residency = $0 ? .resident : .staged }))
      Text("The default releases each model stage between blocks. Keeping stages loaded requires more memory.")
        .font(.caption2).foregroundStyle(.secondary)
      HStack {
        Button("Choose model folder…") {
          let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
          if let folder = store.runtime.reCaVSRModelDirectory { panel.directoryURL = URL(fileURLWithPath: folder) }
          if panel.runModal() == .OK, let url = panel.url { store.selectReCaVSRModelDirectory(url) }
        }
        Button("Download selected model files") { store.downloadReCaVSRModels(precision: settings.precision) }
      }
      Text(store.runtime.reCaVSRModelDirectory ?? "Choose the shared ReCaVSR model folder.")
        .font(.caption2).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
      Text("Image fidelity is still being qualified. ReCaVSR treats a movie as a continuous shot.")
        .font(.caption2).foregroundStyle(.secondary)
    }
  }
}

@MainActor extension StudioStore {
  func selectReCaVSRModelDirectory(_ directory: URL) {
    runtime.reCaVSRModelDirectory = directory.path
    saveRuntime(reloadProfiles: false)
  }

  func downloadReCaVSRModels(precision: ReCaPrecision) {
    guard !videoUpscaleBusy else { return }
    if nativeUpscalerModelLibrary.isEmpty { chooseNativeUpscalerLibrary() }
    guard !nativeUpscalerModelLibrary.isEmpty else { return }
    let library = nativeUpscalerModelLibrary
    videoUpscaleBusy = true
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let result = try await NativeUpscalerSetup.download(packageID: precision == .bf16 ? "recavsr-native-bf16" : "recavsr-native-int8",
          catalog: self.nativeUpscalerCatalogURL(), modelLibrary: URL(fileURLWithPath: library),
          progress: { [weak self] message, _ in Task { @MainActor in self?.videoUpscaleStatus = message } })
        self.selectReCaVSRModelDirectory(result)
        self.videoUpscaleStatus = "ReCaVSR model files verified."
      } catch { self.videoUpscaleStatus = error.localizedDescription }
    }
  }
}
