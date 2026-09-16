import AppKit
import StudioCore
import SwiftUI

struct H3PromptReviewView: View {
  let preview: H3PromptPreview
  @State private var copiedClipID: String?

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 14) {
        Text("H3 prompt preview").font(.title2.bold())
        Label("Draft prompts · review before rendering", systemImage: "doc.text.magnifyingglass")
          .foregroundStyle(.orange)
        Text("These prompts describe the planned video and sound. This preview has not generated media.")
          .font(.callout).foregroundStyle(.secondary)
        ForEach(preview.warnings.indices, id: \.self) { index in
          Label(preview.warnings[index], systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
        }
        ForEach(preview.prompts) { prompt in
          VStack(alignment: .leading, spacing: 10) {
            HStack {
              Text(prompt.clipID).font(.headline)
              Text("\(prompt.durationSeconds, specifier: "%.1f") seconds").font(.caption).foregroundStyle(.secondary)
              Spacer()
              Button(copiedClipID == prompt.clipID ? "Copied" : "Copy prompt", systemImage: "doc.on.doc") {
                NSPasteboard.general.clearContents()
                if NSPasteboard.general.setString(prompt.prompt, forType: .string) { copiedClipID = prompt.clipID }
              }
            }
            Text(prompt.prompt).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
            DisclosureGroup("Video and sound details") {
              VStack(alignment: .leading, spacing: 8) {
                component("Integrated video and sound", prompt.integratedMultimodalDescription)
                component("Overall soundscape", prompt.overallSoundscape)
                component("Non-diegetic music", prompt.nonDiegeticMusic)
                component("Subjects", prompt.subjectIDs.joined(separator: ", "))
                component("Reference assets", prompt.referenceAssets.joined(separator: ", "))
              }.frame(maxWidth: .infinity, alignment: .leading)
            }
          }.padding(12).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
        }
        if preview.prompts.isEmpty { Text("No prompt drafts are available.").foregroundStyle(.secondary) }
      }.padding(12)
    }.onChange(of: preview) { _, _ in copiedClipID = nil }
  }

  private func component(_ title: String, _ text: String) -> some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(title).font(.caption.bold())
      Text(text.isEmpty ? "None" : text).font(.caption).textSelection(.enabled)
    }
  }
}
