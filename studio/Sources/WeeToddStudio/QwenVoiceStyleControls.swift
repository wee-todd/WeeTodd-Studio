import StudioCore
import SwiftUI

struct QwenVoiceStyleControls: View {
  @Binding var speaker: String?
  @Binding var instructions: String?
  var showSpeaker = true
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if showSpeaker {
        Picker("Preset voice", selection: Binding(get: { speaker ?? "ryan" }, set: { speaker = $0 })) {
          ForEach(QwenVoiceStyle.speakers, id: \.self) { Text(QwenVoiceStyle.label($0)).tag($0) }
        }
        Text("CustomVoice uses built-in voices. Choose a Base model for audio-reference cloning.").font(.caption).foregroundStyle(.secondary)
      }
      HStack {
        Text("Delivery").font(.headline)
        Menu("Emotion preset") {
          ForEach(QwenVoiceStyle.emotions, id: \.0) { emotion in Button(emotion.0) { instructions = emotion.1 } }
        }
      }
      TextField("e.g. Start softly, then sound increasingly excited", text: Binding(get: { instructions ?? "" }, set: { instructions = $0 }), axis: .vertical)
        .lineLimit(2...4).textFieldStyle(.roundedBorder)
      Text("Instructions guide the performance; their strength varies by voice and wording. Leave blank for neutral delivery.").font(.caption).foregroundStyle(.secondary)
    }
  }
}
