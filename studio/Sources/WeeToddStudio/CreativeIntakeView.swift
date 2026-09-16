import SwiftUI

/// Plain-language inputs for the guided workflow; the exported job keeps ordinary typed inputs.
struct CreativeIntakeView: View {
  @Binding var fields: [String: String]
  private func value(_ key: String) -> Binding<String> {
    Binding(get: { fields[key] ?? "" }, set: { fields[key] = $0 })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Tell us your story").font(.headline)
      TextEditor(text: value("brief")).frame(minHeight: 110)
        .accessibilityLabel("Movie idea")
      Text("The agent will ask about unclear identities and missing story decisions before designing assets.")
        .font(.caption).foregroundStyle(.secondary)
      VStack(alignment: .leading) {
        Text("How long should the finished movie be?").font(.subheadline.bold())
        HStack {
          TextField("Seconds", text: value("duration_seconds")).frame(width: 75)
          Text("seconds")
          Button("30 s") { fields["duration_seconds"] = "30" }
          Button("60 s") { fields["duration_seconds"] = "60" }
        }
      }
      choice("What should it look like?", key: "visual_style", options: ["Let the director decide", "Realistic cinematic", "Stylized 3D animation", "Illustrated / anime", "Follow reference images"])
      choice("Where will you watch or share it?", key: "presentation", options: ["Widescreen", "Vertical", "Square"])
      choice("How should the camera feel?", key: "camera_style", options: ["Let the director decide", "Smooth and cinematic", "Energetic action", "Mostly steady, letting the comedy play out"])
      choice("What should we hear?", key: "audio_style", options: ["Effects only", "Effects and music", "Include dialogue"])
      choice("May the agent fill in missing details?", key: "design_policy", options: ["Propose missing details for approval", "Ask before adding details"])
      Text("Anything that must appear or be avoided?").font(.subheadline.bold())
      TextEditor(text: value("constraints")).frame(minHeight: 55)
        .accessibilityLabel("Creative requirements and exclusions")
      DisclosureGroup("Advanced timing") {
        HStack { Text("Preferred clip length"); TextField("Seconds", text: value("target_clip_seconds")); Text("s") }
        HStack { Text("Movie frame rate"); TextField("FPS", text: value("frame_rate")) }
        Text("Frame rate starts from the movie settings. The shot list is divided into clips; this does not submit a render.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
  private func choice(_ title: String, key: String, options: [String]) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.subheadline.bold())
      Menu {
        ForEach(options, id: \.self) { option in Button(option) { fields[key] = option } }
      } label: { Text(fields[key]?.isEmpty == false ? fields[key]! : "Choose a preference") }
      TextField("Or describe your preference", text: value(key)).font(.caption)
        .accessibilityLabel(title)
    }
  }
}
