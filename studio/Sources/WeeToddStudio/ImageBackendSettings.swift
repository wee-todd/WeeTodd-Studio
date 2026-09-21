import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

extension ImageGenerationEditor {
  var isNative: Bool { draft?.executionProvider == .nativeMLX }
  func nativeBinding<T>(_ key: WritableKeyPath<NativeImageSettings, T>, fallback: T) -> Binding<T> {
    Binding(get: { draft?.nativeImage?[keyPath: key] ?? fallback }, set: { value in
      if store.imageDraft?.nativeImage == nil { store.imageDraft?.nativeImage = NativeImageSettings() }
      store.imageDraft?.nativeImage?[keyPath: key] = value
      store.imageEstimate = nil
    })
  }
  var backendPicker: some View {
    Section("Backend") {
      Picker("Run with", selection: Binding(get: { draft?.executionProvider ?? .drawThings }, set: {
        store.imageDraft?.selectProvider($0); store.imageEstimate = nil
      })) {
        ForEach(ImageExecutionProvider.allCases) { Text($0.label).tag($0) }
      }.disabled(store.bridge.busy)
    }
  }
  var nativeSettings: some View {
    Section("Local model") {
      Text("Qwen-Image-2.1").font(.headline)
      Text("Image generation and editing · up to 10 inputs").font(.caption)
      TextField("Model manifest", text: nativeBinding(\.manifestPath, fallback: ""))
        .textFieldStyle(.roundedBorder)
      Button("Choose prepared model…") {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url {
          nativeBinding(\.manifestPath, fallback: "").wrappedValue = url.path
        }
      }.disabled(store.bridge.busy)
      Button("Download & prepare 8-bit model…") { Task { await store.prepareNativeImageModel() } }.disabled(store.operationBusy)
      Text("Inference runs in Studio’s local MLX runtime. Model setup preserves the vision encoder and prepares live previews. Weights use Qwen’s noncommercial research license.")
        .font(.caption).foregroundStyle(.secondary)
      Picker("Memory", selection: nativeBinding(\.memoryMode, fallback: "automatic")) {
        Text("Automatic").tag("automatic")
        Text("Lower memory").tag("lower_memory")
      }
      Picker("Reference resolution", selection: nativeBinding(\.referenceResolution, fallback: 1024)) {
        Text("512 · lower memory").tag(512)
        Text("1024").tag(1024)
      }
      Toggle("Live previews", isOn: nativeBinding(\.livePreview, fallback: true))
      Text("Approximate previews update while sampling; final decode preserves RGBA.")
        .font(.caption2).foregroundStyle(.secondary)
      Link("Qwen model license", destination: URL(string: "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE")!)
    }
  }
  func referenceLabel(_ item: ImageWorkspaceInput) -> String {
    guard let slot = draft?.activeImageInputs.first(where: { $0.input.id == item.id }) else { return "Inactive reference" }
    return isNative ? "image\(slot.index)" : "Reference \(slot.index)"
  }
}
