import SwiftUI

struct H3SecondPassStrengthControl: View {
  @Binding var strength: Double

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text("Second-pass strength")
        Spacer()
        TextField("Second-pass strength", value: $strength, format: .number)
          .multilineTextAlignment(.trailing)
          .frame(width: 70)
          .accessibilityLabel("Second-pass strength")
          .accessibilityIdentifier("h3-second-pass-strength")
      }
      Slider(value: Binding(get: {
        strength.isFinite ? min(1, max(0.01, strength)) : 0.35
      }, set: { strength = $0 }), in: 0.01...1)
        .accessibilityLabel("Second-pass strength slider")
      Text("Default 0.35. Lower values retain more of the draft; higher values allow larger visual changes.")
        .font(.caption2).foregroundStyle(.secondary)
      if !strength.isFinite || strength <= 0 || strength > 1 {
        Text("Enter a strength greater than 0 and at most 1.")
          .font(.caption2).foregroundStyle(.orange)
      }
    }
  }
}
