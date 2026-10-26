import StudioCore
import SwiftUI

struct LoRAPromptMetadataControl: View {
  @EnvironmentObject var store: StudioStore
  let key: String
  let name: String
  var fallback: LoRAPromptMetadata?
  @State private var editing = false
  private var metadata: LoRAPromptMetadata { store.loRAPromptMetadata(key: key, fallback: fallback) }
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      if !metadata.triggers.isEmpty {
        Text("Triggers: " + metadata.triggers.joined(separator: ", "))
          .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
      }
      if !metadata.notes.isEmpty {
        DisclosureGroup("Prompt notes") {
          Text(metadata.notes).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }.font(.caption)
      }
      Button("Edit triggers and notes…") { editing = true }.controlSize(.small)
    }
    .sheet(isPresented: $editing) {
      LoRAPromptMetadataEditor(name: name, initial: metadata) { value in
        store.saveLoRAPromptMetadata(value, key: key)
      }
    }
  }
}

struct LoRAPromptMetadataEditor: View {
  let name: String
  let initial: LoRAPromptMetadata
  let save: (LoRAPromptMetadata) -> Bool
  @Environment(\.dismiss) private var dismiss
  @State private var triggers: String
  @State private var notes: String
  init(name: String, initial: LoRAPromptMetadata, save: @escaping (LoRAPromptMetadata) -> Bool) {
    self.name = name; self.initial = initial; self.save = save
    _triggers = State(initialValue: initial.triggers.joined(separator: "\n"))
    _notes = State(initialValue: initial.notes)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("LoRA triggers and notes").font(.title2)
      Text(name).font(.headline).textSelection(.enabled)
      Text("Trigger words or phrases · one per line").font(.subheadline)
      TextEditor(text: $triggers).font(.body).frame(minHeight: 95)
        .accessibilityLabel("LoRA trigger words")
        .overlay(RoundedRectangle(cornerRadius: 5).stroke(.quaternary))
      Text("Missing triggers are inserted into the prompt when you add or enable this LoRA. Leave empty when no trigger is needed.")
        .font(.caption).foregroundStyle(.secondary)
      Text("Prompt notes").font(.subheadline)
      TextEditor(text: $notes).font(.body).frame(minHeight: 125)
        .accessibilityLabel("LoRA prompt notes")
        .overlay(RoundedRectangle(cornerRadius: 5).stroke(.quaternary))
      Text("Notes are saved for reference. They are never inserted into the prompt. Changes apply to future selections; saved groups and existing clips keep their snapshots.")
        .font(.caption).foregroundStyle(.secondary)
      if let source = initial.sourceURL.flatMap(URL.init(string:)), ["https", "http"].contains(source.scheme ?? "") {
        Link("Source and author guidance", destination: source).font(.caption)
      }
      HStack {
        Spacer()
        Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
        Button("Save") {
          let value = LoRAPromptMetadata(
            triggers: triggers.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty },
            notes: notes, sourceURL: initial.sourceURL)
          if save(value) { dismiss() }
        }.keyboardShortcut(.defaultAction)
      }
    }.padding(24).frame(width: 610, height: 540)
  }
}
