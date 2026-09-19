import AppKit
import StudioCore
import SwiftUI

/// Raw workflow data can contain entire song analyses. Never give all of it to Text.
struct WorkflowTextPreviewView: View {
  let text: String

  var body: some View {
    let preview = WorkflowTextPreview(text)
    VStack(alignment: .leading, spacing: 8) {
      Text(preview.text).textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
      if preview.isTruncated {
        HStack {
          Text("Preview shortened. Full text is retained.").font(.caption).foregroundStyle(.secondary)
          Button("Copy full text") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(preview.fullText, forType: .string)
          }.font(.caption)
        }
      }
    }
  }
}
