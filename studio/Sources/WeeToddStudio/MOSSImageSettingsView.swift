import AppKit
import StudioCore
import SwiftUI

struct MOSSImageSettingsView: View {
  @EnvironmentObject var store: StudioStore
  @State private var setupTask: Task<Void, Never>?
  @State private var setupMessage = ""
  @State private var setupFraction = 0.0
  private var settings: MOSSImageSettings { store.imageDraft?.mossImage ?? .init() }
  private func binding<T>(_ key: WritableKeyPath<MOSSImageSettings, T>, _ fallback: T) -> Binding<T> {
    Binding(get: { store.imageDraft?.mossImage?[keyPath: key] ?? fallback }, set: {
      if store.imageDraft?.mossImage == nil { store.imageDraft?.mossImage = .init() }
      store.imageDraft?.mossImage?[keyPath: key] = $0; store.imageEstimate = nil
    })
  }
  var body: some View {
    Section("MOSS-Image") {
      Picker("Model", selection: Binding(get: { settings.variant }, set: { value in
        var changed = settings
        do { try changed.selectVariant(value) } catch { setupMessage = error.localizedDescription; return }
        store.imageDraft?.mossImage = changed; store.imageEstimate = nil
        store.imageDraft?.modelID = value == "sft" ? MOSSImageSettings.sftModelID : MOSSImageSettings.baseModelID
        if !settings.libraryDirectory.isEmpty { scan(URL(fileURLWithPath: settings.libraryDirectory)) }
      })) { Text("SFT").tag("sft"); Text("Base").tag("base") }.disabled(setupTask != nil || store.operationBusy)
      Text("Native Swift / MLX · text and layout generation").font(.caption).foregroundStyle(.secondary)
      HStack {
        Button("Link model library…") { chooseLibrary(download: false) }
        Button("Download…") { chooseLibrary(download: true) }
      }.disabled(setupTask != nil || store.operationBusy)
      if setupTask != nil {
        ProgressView(value: setupFraction)
        Button("Cancel setup") { setupTask?.cancel() }
      }
      if !setupMessage.isEmpty { Text(setupMessage).font(.caption).textSelection(.enabled) }
      DisclosureGroup("Component folders") {
        TextField("Transformer folder", text: binding(\.transformerDirectory, ""))
        Button("Choose transformer…") { chooseComponent(\.transformerDirectory) }
        TextField("Shared folder", text: binding(\.sharedDirectory, ""))
        Button("Choose shared components…") { chooseComponent(\.sharedDirectory) }
        Text("Shared contains text_encoder, tokenizer and vae. Base and SFT reuse these files. Check Settings validates every component.").font(.caption2)
      }
      Picker("Weight cache", selection: binding(\.weightCacheGB, 0)) {
        Text("Stream weights").tag(0)
        ForEach([2, 4, 8, 16], id: \.self) { Text("Up to \($0) GiB").tag($0) }
      }
      Picker("Transformer precision", selection: binding(\.quantizationBits, nil)) {
        Text("Official BF16").tag(nil as Int?); Text("Native affine Q8").tag(Optional(8)); Text("Native affine Q4").tag(Optional(4))
      }
      if let bits = settings.quantizationBits {
        Button("Convert official transformer to Q\(bits)…") { convert(bits: bits) }.disabled(store.operationBusy)
        Text("Choose a new folder in your model library. Conversion preserves source weights; link the resulting folder after it completes. Q4/Q8 are explicit quality tradeoffs.").font(.caption2)
      }
      Text("Weighted stages unload between conditioning, sampling and decoding. A larger cache reduces repeated transformer reads.").font(.caption2).foregroundStyle(.secondary)
      Toggle("Enforce subject boxes", isOn: Binding(get: { settings.regionalLayout == true }, set: { value in
        binding(\.regionalLayout, nil).wrappedValue = value; store.imageEstimate = nil
      })).disabled(store.imageDraft?.mossLayout == nil)
      if settings.regionalLayout == true {
        Text("Subjects use their own descriptions and boxes. Shared style comes from recognized medium, lighting and palette; edit each region for precise content.").font(.caption2).foregroundStyle(.secondary)
      }
      DisclosureGroup("Sampling and negative guidance") {
        if store.imageDraft?.mossLayout?.sourceReconstruction?.enabled == true {
        Button("Faithful reconstruction recipe") {
          store.imageDraft?.applyFaithfulMOSSReconstructionRecipe()
          let library = settings.libraryDirectory.isEmpty ? URL(fileURLWithPath: settings.sharedDirectory).deletingLastPathComponent().path : settings.libraryDirectory
          if settings.transformerDirectory.isEmpty, library.hasPrefix("/"), let found = MOSSModelSetup.discover(root: URL(fileURLWithPath: library), variant: "sft") {
            binding(\.transformerDirectory, "").wrappedValue = found.transformerDirectory
          }
          store.imageEstimate = nil
        }
        Text("Stock SFT / BF16, adapters disabled, 8 Euler steps, CFG 1. Source mode and change strength are in the layout editor.").font(.caption2)
        }
        Picker("Sampler", selection: binding(\.sampler, "euler")) { Text("Euler").tag("euler"); Text("ER-SDE").tag("er_sde") }
        Picker("Schedule", selection: binding(\.schedule, "release")) { Text("Publisher release").tag("release"); Text("Comfy simple").tag("simple") }
        Button("Apply attached layout recipe") {
          store.imageDraft?.steps = 25; store.imageDraft?.guidance = 2.5; store.imageDraft?.shift = 5
          store.imageDraft?.negativePrompt = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
          binding(\.sampler, "euler").wrappedValue = "er_sde"; binding(\.schedule, "release").wrappedValue = "simple"
          binding(\.skipLayers, [22]).wrappedValue = [22]; binding(\.skipLayerExpression, nil).wrappedValue = nil
          binding(\.skipStart, 0).wrappedValue = 0; binding(\.skipStop, 1).wrappedValue = 0.5
        }.disabled(settings.stockRecipe != nil)
        TextField("Shift", value: Binding(get: { store.imageDraft?.shift ?? 5 }, set: { store.imageDraft?.shift = $0; store.imageEstimate = nil }), format: .number)
        TextField("Negative prompt", text: Binding(get: { store.imageDraft?.negativePrompt ?? "" }, set: { store.imageDraft?.negativePrompt = $0; store.imageEstimate = nil }), axis: .vertical)
        TextField("Skip layers", text: Binding(get: { settings.skipLayerExpression ?? settings.skipLayers.map { String($0) }.joined(separator: ",") }, set: { binding(\.skipLayerExpression, nil).wrappedValue = $0 }))
        TextField("Skip start", value: binding(\.skipStart, 0), format: .number)
        TextField("Skip stop", value: binding(\.skipStop, 1), format: .number)
        Text("Skip applies to the negative CFG branch only. Progress runs from 0 to 1; CFG 1 uses no negative branch.").font(.caption2)
      }
      Link("MOSS model license", destination: URL(string: "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE")!)
    }
  }
  private func folderPanel() -> URL? {
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
    return panel.runModal() == .OK ? panel.url : nil
  }
  private func chooseComponent(_ key: WritableKeyPath<MOSSImageSettings, String>) {
    if let url = folderPanel() { binding(key, "").wrappedValue = url.path }
  }
  private func scan(_ root: URL) {
    guard let found = MOSSModelSetup.discover(root: root, variant: settings.variant) else {
      setupMessage = "No complete component folders found. Choose the MOSS family folder or link the two component folders directly."; return
    }
    binding(\.libraryDirectory, "").wrappedValue = found.libraryDirectory
    binding(\.transformerDirectory, "").wrappedValue = found.transformerDirectory
    binding(\.sharedDirectory, "").wrappedValue = found.sharedDirectory
    setupMessage = "Linked \(settings.variant.uppercased()) and shared components. Check Settings before generating."
  }
  private func chooseLibrary(download: Bool) {
    guard let root = folderPanel() else { return }
    if !download { scan(root); return }
    guard let catalog = NativeModelDownloads.bundledCatalog else { setupMessage = "Bundled model catalog is unavailable."; return }
    let variant = settings.variant, session = store.documentSessionID, key = store.imageDraft?.storageKey
    setupMessage = "Verifying installed files before downloading…"; setupFraction = 0
    setupTask = Task {
      defer { setupTask = nil }
      do {
        let found = try await MOSSModelSetup.prepare(root: root, variant: variant, catalog: catalog) { message, fraction in
          Task { @MainActor in setupMessage = message; setupFraction = fraction }
        }
        guard store.documentSessionID == session, store.imageDraft?.storageKey == key, store.imageDraft?.mossImage?.variant == variant else { return }
        scan(URL(fileURLWithPath: found.libraryDirectory))
      } catch { setupMessage = error.localizedDescription }
    }
  }
  private func convert(bits: Int) {
    let panel = NSSavePanel(); panel.title = "Choose a new model-library folder"; panel.nameFieldStringValue = "MOSS-\(settings.variant.uppercased())-Q\(bits)"
    guard panel.runModal() == .OK, let output = panel.url else { return }
    let source = settings.transformerDirectory, session = store.documentSessionID, key = store.imageDraft?.storageKey
    Task {
      do {
        let result = try await store.bridge.invoke("moss-native-convert", runtime: store.runtime,
          payload: ["source": source, "bits": bits, "component": "transformer"], output: output)
        guard store.documentSessionID == session, store.imageDraft?.storageKey == key,
          store.imageDraft?.mossImage?.transformerDirectory == source else { return }
        if let folder = result["componentDirectory"] as? String { binding(\.transformerDirectory, "").wrappedValue = folder }
        store.imageEstimate = nil
      } catch is CancellationError {
        store.notice = "MOSS model conversion cancelled."
      } catch { store.error = error.localizedDescription }
    }
  }
}
