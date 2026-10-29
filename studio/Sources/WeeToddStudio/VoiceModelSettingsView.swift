import AppKit
import StudioCore
import SwiftUI

struct VoiceModelSettingsView: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Speech models").font(.headline)
      Text("Configure model locations once for all movies. The Voice workspace selects a family and an installed variant.")
        .font(.caption).foregroundStyle(.secondary)
      ForEach(VoiceEngine.allCases, id: \.self) { engine in
        Text(engine.label).font(.subheadline.bold())
        let models = store.installedVoiceModels.filter { $0.engine == engine }
        if models.isEmpty { Text("No installed models configured").font(.caption).foregroundStyle(.secondary) }
        ForEach(models) { model in
          VStack(alignment: .leading, spacing: 4) {
            HStack {
              Text(model.variantName)
              if !store.voiceModelAvailable(model) { Text("Unavailable").foregroundStyle(.orange) }
              Spacer()
              Button("Check") { Task { await store.inspectVoiceModel(model) } }
              if model.engine == .fishDrama3 {
                Button("Complete scene controls…") { store.completeNativeDramaSceneControls(model) }
              }
              Button("Remove") { store.removeVoiceModel(model) }
            }
            Text(model.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          }.disabled(store.operationBusy).padding(8).background(Theme.raised, in: RoundedRectangle(cornerRadius: 6))
        }
      }
      HStack {
        Button("Add installed model…") { store.chooseVoiceModel() }
        Menu("Download speech model…") {
          Button("Drama 3 · Balanced 4-bit · recommended") { store.downloadVoiceModel("drama3-4bit") }
          Button("Drama 3 · Compact 3-bit") { store.downloadVoiceModel("drama3-3bit") }
          Button("Drama 3 · Quality 8-bit") { store.downloadVoiceModel("drama3-8bit") }
          Button("Drama 3 · BF16") { store.downloadVoiceModel("drama3-bf16") }
          Divider()
          Button("Fish S2 Pro · 8-bit") { store.downloadVoiceModel("fish-s2-pro-8bit") }
          Button("Fish S2 Pro · BF16") { store.downloadVoiceModel("fish-s2-pro-bf16") }
          Divider()
          Button("Qwen3-TTS Base · 1.7B · 8-bit") { store.downloadVoiceModel("qwen3-tts-1.7b-base-8bit") }
          Button("Qwen3-TTS Base · 0.6B · 8-bit") { store.downloadVoiceModel("qwen3-tts-0.6b-base-8bit") }
          Button("Qwen3-TTS CustomVoice · 1.7B · 8-bit · emotion instructions") { store.downloadVoiceModel("qwen3-tts-1.7b-customvoice-8bit") }
        }
      }
      if let message=store.nativeVoiceSetupMessage {HStack{ProgressView().controlSize(.small);Text(message).font(.caption);Button("Cancel setup"){store.cancelNativeDramaSetup()}.disabled(store.nativeVoiceSetupTask==nil)}}
      Text("Complete scene controls reuses a verified original shard from shared model storage. If none is available, it downloads 4,638,423,240 bytes beside the installed model. The source is removed after success; failed setup can be retried.").font(.caption).foregroundStyle(.secondary)
      Text("Built with Fish Audio. Drama 3 runs natively in Swift/MLX and uses the Fish Audio Community License 1.0.").font(.caption).foregroundStyle(.secondary)
      Link("Drama 3 license and terms",destination:URL(string:"https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE")!).font(.caption)
      Text("Fish S2 Pro weights use the Fish Audio Research License; commercial use requires separate permission. Qwen3-TTS weights use Apache 2.0.")
        .font(.caption).foregroundStyle(.secondary)
      Link("Fish model license", destination: URL(string: "https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md")!)
        .font(.caption)
    }
  }
}
