import StudioCore
import SwiftUI

struct FishVoiceDirectionControls: View {
  @Binding var direction: FishVoiceDirection?
  var showsTitle = true
  var resetKeepsOverride = false
  private var value: FishVoiceDirection { direction ?? FishVoiceDirection() }
  private func field(_ key: WritableKeyPath<FishVoiceDirection, String>) -> Binding<String> {
    Binding(get: { value[keyPath: key] }, set: { text in
      var next = value; next[keyPath: key] = text; direction = next
    })
  }
  private func label(_ tag: String) -> String {
    switch tag {
    case "": return "Unspecified"
    case "low voice": return "Low"
    case "pitch up": return "Higher"
    case "slow delivery": return "Slow"
    case "fast delivery": return "Fast"
    case "warm voice": return "Warm"
    case "breathy voice": return "Breathy"
    case "raspy voice": return "Raspy"
    case "clear resonant voice": return "Clear and resonant"
    default: return tag
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if showsTitle { Text("Voice direction").font(.headline) }
      Picker("Pitch", selection: field(\.pitch)) {
        ForEach(FishVoiceDirection.pitchOptions, id: \.self) { Text(label($0)).tag($0) }
      }
      Picker("Pace", selection: field(\.pace)) {
        ForEach(FishVoiceDirection.paceOptions, id: \.self) { Text(label($0)).tag($0) }
      }
      Picker("Timbre", selection: field(\.timbre)) {
        ForEach(FishVoiceDirection.timbreOptions, id: \.self) { Text(label($0)).tag($0) }
      }
      VStack(alignment: .leading, spacing: 4) {
        Text("Accent")
        TextField("e.g. British or American Southern", text: field(\.accent))
          .accessibilityLabel("Voice accent")
      }
      VStack(alignment: .leading, spacing: 4) {
        Text("Voice description")
        TextField("e.g. A mature male narrator with a deep, warm voice", text: field(\.description), axis: .vertical)
          .lineLimit(2...4).accessibilityLabel("Voice description")
        Text("Describe age, vocal character or other qualities in your own words (up to 120 characters).")
          .font(.caption).foregroundStyle(.secondary)
      }
      Text("These instructions guide Fish with or without a sample. Results vary by wording and reference; they do not lock speaker identity.")
        .font(.caption).foregroundStyle(.secondary)
      if let tags = try? value.tags() {
        if !tags.isEmpty {
          DisclosureGroup("Applied Fish instructions") {
            Text(tags.map { "[\($0)]" }.joined(separator: " "))
              .font(.caption).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
          }
        }
      } else {
        Text("Use a short description and accent without brackets, chat markers or line breaks.")
          .font(.caption).foregroundStyle(.orange)
      }
      if direction != nil {
        Button(resetKeepsOverride ? "Clear line direction" : "Reset voice direction") {
          direction = resetKeepsOverride ? FishVoiceDirection() : nil
        }.font(.caption)
      }
    }
  }
}
