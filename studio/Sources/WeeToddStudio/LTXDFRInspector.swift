import StudioCore
import SwiftUI

struct LTXDFRInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var generation: GenerationDescriptor?

  var mode: String {
    guard let settings = clip.generationSelection?.ltx25DFR else { return "profile" }
    return settings.enabled ? String(settings.temporalRounds) : "off"
  }
  func edit(_ body: @escaping (inout GenerationSelection) -> Void) {
    store.editClip {
      var value = $0.generationSelection ?? GenerationSelection(task: $0.inferredTask, preset: .custom)
      body(&value); $0.generationSelection = value
    }
  }
  var body: some View {
    DisclosureGroup("DFR spatial and temporal refinement") {
      Picker("DFR mode", selection: Binding(get: { mode }, set: { value in
        edit {
          if value == "profile" { $0.ltx25DFR = nil }
          else {
            $0.ltx25DFR = LTX25DFRSettings(enabled: value != "off",
              temporalRounds: Int(value) ?? 0,
              detailingStrength: $0.ltx25DFR?.detailingStrength ?? generation?.dfrDetailingStrength ?? 0.5,
              experimentalEnabled: value != "off")
          }
        }
      })) {
        Text("Use model profile").tag("profile")
        Text("Off").tag("off")
        Text("Spatial refinement · experimental").tag("0")
        Text("Spatial + temporal 2× · experimental").tag("1")
        Text("Spatial + temporal 4× · experimental").tag("2")
      }
      if clip.generationSelection?.ltx25DFR?.enabled == true {
        HStack {
          Text("Detailing strength")
          TextField("DFR detailing strength", value: Binding(get: {
            clip.generationSelection?.ltx25DFR?.detailingStrength ?? 0.5
          }, set: { value in edit { $0.ltx25DFR?.detailingStrength = value } }), format: .number)
        }
      }
      if mode == "profile", generation?.dfrEnabled == true {
        Text("Profile DFR: \(generation?.dfrTemporalRounds ?? 0) temporal rounds; detailing strength \(generation?.dfrDetailingStrength ?? 0.5, specifier: "%.2f").")
          .font(.caption2).foregroundStyle(.secondary)
      }
      Text("DFR refines a half-size base to the requested canvas. Temporal rounds double frame rate each time and preserve duration. Requires Pixel-Spatial components and a temporal upscaler for temporal modes. Supports independent text, first-image and first/last-frame shots.")
        .font(.caption2).foregroundStyle(.secondary)
    }
  }
}
