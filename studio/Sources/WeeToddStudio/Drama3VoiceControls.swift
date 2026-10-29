import AVFoundation
import AppKit
import StudioCore
import SwiftUI

struct Drama3VoiceControls: View {
  @EnvironmentObject var store: StudioStore
  var draft: VoiceDraft { store.project.voiceDraft ?? VoiceDraft() }
  func text(_ key: WritableKeyPath<VoiceDraft, String?>) -> Binding<String> {
    Binding(
      get: { draft[keyPath: key] ?? (key == \.dramaEditOperation ? "replace" : "") },
      set: { value in store.change { $0.voiceDraft?[keyPath: key] = value.isEmpty ? nil : value } })
  }
  var body: some View {
    GroupBox("Drama 3") {
      VStack(alignment: .leading, spacing: 10) {
        Picker(
          "Task",
          selection: Binding(
            get: { draft.dramaMode ?? "speak" },
            set: { value in store.change { $0.voiceDraft?.dramaMode = value } })
        ) {
          Text("Speak").tag("speak")
          Text("Conversation · line continuation").tag("lineContinuation")
          Text("Joint scene (experimental)").tag("scene")
          Text("Separate each script line").tag("lineSeparated")
          Text("Direct a previous take").tag("direct")
          Text("Edit a speech range").tag("infill")
          Text("Stream text").tag("streamingText")
        }
        if ["direct", "infill"].contains(draft.dramaMode ?? "") {
          Picker(
            "Previous take",
            selection: Binding<UUID?>(
              get: { draft.dramaParentTakeID },
              set: { value in store.change { $0.voiceDraft?.dramaParentTakeID = value } })
          ) {
            Text("Choose a take").tag(UUID?.none)
            ForEach(store.voiceTakes.filter { $0.voiceGeneration?.request.engine == .fishDrama3 }) {
              Text($0.name).tag(Optional($0.id))
            }
          }
        }
        TextField("Director note / freeform delivery", text: text(\.dramaDirection))
        TextField("Scene context (optional)", text: text(\.dramaScene))
        HStack {
          TextField("Language code, e.g. en", text: text(\.dramaLanguage))
          TextField(
            "Duration hint (seconds)",
            text: Binding(
              get: { draft.dramaDurationHint.map { String($0) } ?? "" },
              set: { value in store.change { $0.voiceDraft?.dramaDurationHint = Double(value) } }))
        }
        if draft.dramaMode == "infill" {
          Picker("Operation", selection: text(\.dramaEditOperation)) {
            Text("Replace").tag("replace")
            Text("Insert").tag("insert")
            Text("Delete range").tag("delete")
          }
          HStack {
            Text("Replace from")
            TextField(
              "Start seconds",
              value: Binding(
                get: { draft.dramaEditStart ?? 0 },
                set: { v in store.change { $0.voiceDraft?.dramaEditStart = v } }), format: .number)
            Text("to")
            TextField(
              "End seconds",
              value: Binding(
                get: { draft.dramaEditEnd ?? 1 },
                set: { v in store.change { $0.voiceDraft?.dramaEditEnd = v } }), format: .number)
          }
          Button("Import speech to edit…") {
            let panel = NSOpenPanel()
            panel.title = "Choose speech to edit"
            if panel.runModal() == .OK, let url = panel.url {
              store.change {
                $0.voiceDraft?.dramaSourceAudio = url.path
                $0.voiceDraft?.dramaParentTakeID = nil
              }
            }
          }
          if let source = store.voiceTakes.first(where: { $0.id == draft.dramaParentTakeID })?.path
            ?? draft.dramaSourceAudio
          {
            Text(URL(fileURLWithPath: source).lastPathComponent).font(.caption)
            Drama3EditWaveform(source: source)
          }
          Text(
            "Enter the full updated transcript in the script, including words on both sides of the edit. Prefix and suffix codes remain intact; edit boundaries snap to codec frames (46 ms)."
          ).font(.caption).foregroundStyle(.secondary)
        }
        if draft.dramaMode == "streamingText" { Drama3LiveTextControls() }
        Toggle(
          "Play audio as it generates",
          isOn: Binding(
            get: { draft.dramaStream ?? false },
            set: { v in store.change { $0.voiceDraft?.dramaStream = v } }))
        DisclosureGroup("Native budgets") {
          VStack(alignment: .leading) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
              GridRow {
                Text("Prompt rows")
                TextField(
                  "Prompt rows",
                  value: Binding(
                    get: { draft.dramaPromptRows ?? 8192 },
                    set: { v in store.change { $0.voiceDraft?.dramaPromptRows = v } }),
                  format: .number
                )
                .accessibilityLabel("Prompt rows")
              }
              GridRow {
                Text("Memory (GiB)")
                TextField(
                  "Memory GiB (0 = automatic)",
                  value: Binding(
                    get: { draft.dramaMemoryGiB ?? 0 },
                    set: { v in store.change { $0.voiceDraft?.dramaMemoryGiB = v == 0 ? nil : v } }),
                  format: .number
                )
                .accessibilityLabel("Memory GiB (0 = automatic)")
              }
              GridRow {
                Text("First audio chunk")
                TextField(
                  "First audio chunk (codec frames)",
                  value: Binding(
                    get: { draft.dramaFirstChunkFrames ?? 4 },
                    set: { v in store.change { $0.voiceDraft?.dramaFirstChunkFrames = v } }),
                  format: .number
                )
                .accessibilityLabel("First audio chunk (codec frames)")
              }
              GridRow {
                Text("Following chunks")
                TextField(
                  "Following audio chunks (codec frames)",
                  value: Binding(
                    get: { draft.dramaStreamIntervalFrames ?? 7 },
                    set: { v in store.change { $0.voiceDraft?.dramaStreamIntervalFrames = v } }),
                  format: .number
                )
                .accessibilityLabel("Following audio chunks (codec frames)")
              }
            }
            Text(
              "Memory: 0 = automatic. Audio chunks use codec frames (1–256). Prompt rows: 1–32,768. Larger context increases cache memory. Automatic uses up to 85% of physical memory; other apps also need memory."
            ).font(.caption).foregroundStyle(.secondary)
          }
        }
        Text(
          "Inline [delivery cues], <whispering>scoped cues</whispering>, multilingual text and speaker direction are supported. Duration is guidance. The worker releases model memory when it exits."
        ).font(.caption).foregroundStyle(.secondary)
      }.frame(maxWidth: .infinity, alignment: .leading)
    }
  }
}
struct Drama3AdditionalSamples: View {
  @EnvironmentObject var store: StudioStore
  var samples: [VoiceReference] { store.project.voiceDraft?.dramaReferences ?? [] }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(Array(samples.enumerated()), id: \.element.id) { index, sample in
        DisclosureGroup(
          "Additional sample \(index+1) · \(URL(fileURLWithPath:sample.path).lastPathComponent)"
        ) {
          VoiceSampleControls(
            reference: Binding(
              get: { store.project.voiceDraft?.dramaReferences?[safeDrama: index] },
              set: { value in
                store.change { p in
                  if let value, p.voiceDraft?.dramaReferences?.indices.contains(index) == true {
                    p.voiceDraft?.dramaReferences?[index] = value
                  }
                }
              }), mode: .audioAndTranscript)
          Button("Remove sample", role: .destructive) {
            store.change { $0.voiceDraft?.dramaReferences?.remove(at: index) }
          }
        }
      }
      Button("Add another reference sample") {
        store.change {
          if $0.voiceDraft?.dramaReferences == nil { $0.voiceDraft?.dramaReferences = [] }
          $0.voiceDraft?.dramaReferences?.append(.init(path: "", duration: 5))
        }
      }
    }
  }
}
extension Array {
  fileprivate subscript(safeDrama i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
struct Drama3SpeakerVoiceControls: View {
  @EnvironmentObject var store: StudioStore
  let speakerID: UUID
  func apply(_ voice: VoiceProfileSnapshot, characterID: UUID? = nil) {
    store.change { p in
      guard let i = p.voiceDraft?.dialogue?.speakers.firstIndex(where: { $0.id == speakerID })
      else { return }
      p.voiceDraft?.dialogue?.speakers[i].voiceProfile = voice
      p.voiceDraft?.dialogue?.speakers[i].characterID = characterID
      p.voiceDraft?.dialogue?.speakers[i].reference = voice.samples.first
      p.voiceDraft?.dialogue?.speakers[i].dramaReferences = Array(voice.samples.dropFirst())
      p.voiceDraft?.dialogue?.speakers[i].referenceMode = .audioAndTranscript
    }
  }
  var body: some View {
    HStack {
      Menu("Saved voice") {
        ForEach(store.voiceLibraryProfiles, id: \.reference) { voice in
          Button(voice.name) { apply(voice) }
        }
      }
      Menu("Character voice") {
        ForEach(store.planning.subjects.filter { $0.kind == .character && $0.voiceBinding != nil })
        { character in
          Button(character.name) {
            if let voice = character.voiceBinding?.profile {
              apply(voice, characterID: character.id)
            }
          }
        }
      }
      if let voice = store.project.voiceDraft?.dialogue?.speakers.first(where: {
        $0.id == speakerID
      })?.voiceProfile {
        Text("\(voice.name) · v\(voice.reference.version)").font(.caption)
        if let speaker = store.project.voiceDraft?.dialogue?.speakers.first(where: {
          $0.id == speakerID
        }),
          speaker.reference != voice.samples.first
        {
          Text("Edited first sample").font(.caption).foregroundStyle(.secondary)
        }
      }
    }.task { await store.refreshSavedVoices() }
  }
}

struct Drama3LiveTextControls: View {
  @EnvironmentObject var store: StudioStore
  @State private var text = ""
  @State private var submitted = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(
        "Generate starts with the script as the first chunk. Append more text here, then Finish input. Each segment waits for one chunk of lookahead."
      ).font(.caption).foregroundStyle(.secondary)
      TextEditor(text: $text).frame(height: 70).border(Color.secondary.opacity(0.25))
      HStack {
        Button("Append text") {
          do {
            try store.bridge.appendSpeechText(text)
            submitted = text
          } catch { store.error = error.localizedDescription }
        }.disabled(
          !store.bridge.acceptsSpeechText || store.bridge.speechTextPending || text.isEmpty)
        Button("Finish input") { store.bridge.finishSpeechText() }.disabled(
          !store.bridge.acceptsSpeechText)
      }
    }.onChange(of: store.bridge.speechTextAcceptedRevision) { _ in
      if text == submitted { text = "" }
    }
  }
}
