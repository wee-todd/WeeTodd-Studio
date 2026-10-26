import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct H3ImageModelSettings: View {
  @EnvironmentObject var store: StudioStore
  var settings: H3ImageSettings { store.imageDraft?.h3Image ?? H3ImageSettings() }
  func binding<T>(_ key: WritableKeyPath<H3ImageSettings,T>, _ fallback: T) -> Binding<T> {
    Binding(get: { store.imageDraft?.h3Image?[keyPath:key] ?? fallback }, set: { value in
      if store.imageDraft?.h3Image == nil { store.imageDraft?.h3Image = H3ImageSettings() }
      store.imageDraft?.h3Image?[keyPath:key] = value; store.imageEstimate = nil
    })
  }
  var body: some View {
    Section("H3 image model") {
      Picker("Model profile", selection: Binding(get: { store.imageDraft?.modelID ?? "" }, set: store.selectH3ImageProfile)) {
        Text("Choose an installed model").tag("")
        ForEach(store.h3ImageProfiles) { Text($0.name).tag($0.id) }
        if let id = store.imageDraft?.modelID, !id.isEmpty, !store.h3ImageProfiles.contains(where: { $0.id == id }) {
          Text("Saved profile · unavailable").tag(id).disabled(true)
        }
      }.disabled(store.bridge.busy)
      Button("Model setup…") { store.showRuntime = true }.disabled(store.bridge.busy)
      Text("Text-to-image uses the selected checkpoint. Canvas and mood-board editing need a Ref2VA model and up to nine ordered images.")
        .font(.caption).foregroundStyle(.secondary)
      Picker("Decoder", selection: binding(\.decoderMode,.qualityFiveFrames)) {
        ForEach(H3ImageDecoderMode.allCases) { Text($0.label).tag($0) }
      }.disabled(store.bridge.busy)
      if settings.decoderMode == .qualityFiveFrames {
        Picker("Selected still", selection: binding(\.frameSelection,.middle)) {
          Text("First").tag(H3ImageCandidate.first); Text("Middle").tag(H3ImageCandidate.middle); Text("Last").tag(H3ImageCandidate.last)
        }
        Text("Samples five frames and decodes the chosen still with the original VAE. Saves one PNG; audio is not decoded.")
          .font(.caption2).foregroundStyle(.secondary)
      } else {
        TextField("Prepared image VAE", text: binding(\.imageVAEPath,"")).textFieldStyle(.roundedBorder)
        Button("Choose prepared image VAE…") {
          let panel = NSOpenPanel(); panel.allowedContentTypes = [UTType(filenameExtension:"safetensors") ?? .data]
          if panel.runModal() == .OK, let url = panel.url {
            do { try store.saveH3ImageVAEPath(url.path,applyToDraft:true) } catch { store.error = error.localizedDescription }
          }
        }.disabled(store.bridge.busy)
        Button("Download & prepare image VAE…") { Task { await store.prepareH3ImageVAE() } }.disabled(store.operationBusy)
        Text("Experimental single-latent decoding can soften text and fine detail. Requires the prepared image-only VAE; the original encoder remains in use for references.")
          .font(.caption2).foregroundStyle(.secondary)
        Link("Image VAE source and limitations", destination:URL(string:"https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE")!).font(.caption2)
      }
      if store.imageDraft?.activeImageInputs.contains(where: { $0.input.strength != 1 }) == true {
        Button("Set active references to full strength") {
          if store.imageDraft?.canvas?.enabled == true { store.imageDraft?.canvas?.strength = 1 }
          for index in (store.imageDraft?.moodboard.indices ?? (0..<0)) {
            if store.imageDraft?.moodboard[index].enabled == true, (store.imageDraft?.moodboard[index].strength ?? 0) > 0 {
              store.imageDraft?.moodboard[index].strength = 1
            }
          }
          store.imageEstimate = nil
        }
      }
    }
  }
}

struct H3ImageLoRAControls: View {
  @EnvironmentObject var store: StudioStore
  @State private var libraryOpen = false
  var body: some View {
    VStack(alignment:.leading,spacing:10) {
      Button("Add / Groups…") { libraryOpen = true }
      Text("Native H3 adapters · maximum eight. Turbo requires four Euler evaluations and a matching model task.")
        .font(.caption2).foregroundStyle(.secondary)
      ForEach(store.imageDraft?.h3Image?.loras ?? []) { member in
        VStack(alignment:.leading,spacing:5) {
          HStack {
            Toggle(member.asset.name,isOn:Binding(get: {
              store.imageDraft?.h3Image?.loras.first { $0.id == member.id }?.isEnabled ?? false
            },set: { store.setH3ImageLoRAEnabled(member.id,enabled:$0) }))
            Button { store.imageDraft?.h3Image?.loras.removeAll { $0.id == member.id }; store.imageEstimate = nil }
              label: { Image(systemName:"xmark") }
          }
          LoRAStrength(value:Binding(get: {
            store.imageDraft?.h3Image?.loras.first { $0.id == member.id }?.strength ?? 1
          },set: { value in
            if let index=store.imageDraft?.h3Image?.loras.firstIndex(where: { $0.id == member.id }) {
              store.imageDraft?.h3Image?.loras[index].strength=value;store.imageEstimate=nil
            }
          }),range:-10...10)
        }.font(.caption)
      }
    }.sheet(isPresented:$libraryOpen) { LoRALibrary(imageWorkspace:true).environmentObject(store) }
  }
}
