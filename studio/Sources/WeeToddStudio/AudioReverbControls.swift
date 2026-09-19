import SwiftUI
import StudioCore

struct AudioReverbControls: View {
  @Binding var effect: AudioReverb?
  private var value: AudioReverb { effect ?? AudioReverb() }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Toggle("Reverb", isOn: Binding(get: { effect?.enabled ?? false }, set: { enabled in
        var next = value; next.enabled = enabled; effect = next
      }))
      if effect != nil {
        VStack(alignment: .leading, spacing: 10) {
          Picker("Space", selection: Binding(get: { value.preset }, set: { preset in
            var next = value; next.applyPreset(preset); effect = next
          })) {
            ForEach(AudioReverbPreset.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
          }
          HStack {
            Text("Amount"); Slider(value: binding(\.mix), in: 0...1)
              .accessibilityLabel("Reverb amount")
            Text("\(Int((value.mix * 100).rounded()))%").monospacedDigit().frame(width: 38)
          }
          HStack {
            Text("Decay"); Slider(value: binding(\.decay), in: 0.2...6)
              .accessibilityLabel("Reverb decay")
            Text("\(value.decay, specifier: "%.1f") s").monospacedDigit().frame(width: 38)
          }
          DisclosureGroup("Advanced reverb") {
            VStack(alignment: .leading, spacing: 8) {
              HStack {
                Text("Dark"); Slider(value: binding(\.tone), in: 0...1).accessibilityLabel("Reverb tone"); Text("Bright")
              }
              HStack {
                Text("Pre-delay")
                Slider(value: binding(\.preDelay), in: 0...0.1).accessibilityLabel("Reverb pre-delay")
                Text("\(Int((value.preDelay * 1000).rounded())) ms").monospacedDigit().frame(width: 48)
              }
            }.padding(.top, 6)
          }
        }.disabled(!value.enabled)
        Text("Applies to the whole track in preview, export and audio drivers. Tails continue through gaps; leave space at the end of the movie for them to finish.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
  private func binding(_ key: WritableKeyPath<AudioReverb, Double>) -> Binding<Double> {
    Binding(get: { value[keyPath: key] }, set: { v in
      var next = value; next[keyPath: key] = v; effect = next
    })
  }
}
