import AppKit
import AVFoundation
import StudioCore
import SwiftUI

struct VoiceDialogueEditor: View {
  @EnvironmentObject var store: StudioStore
  @State private var selected: UUID?
  @State private var showSpeakers = false
  @State private var showPerformance = false
  var draft: VoiceDraft { store.project.voiceDraft ?? VoiceDraft() }
  var dialogue: VoiceDialogue { draft.dialogue ?? VoiceDialogue(speakers: [], turns: []) }
  var current: VoiceTurn? { dialogue.turns.first { $0.id == selected } ?? dialogue.turns.first }
  private func edit(_ action: (inout VoiceDialogue) -> Void) {
    store.change { if $0.voiceDraft?.dialogue != nil { action(&$0.voiceDraft!.dialogue!) } }
  }
  private func turn<T>(_ key: WritableKeyPath<VoiceTurn, T>, default fallback: T) -> Binding<T> {
    let id = current?.id, session = store.documentSessionID, draftID = draft.id
    return Binding(get: { store.project.voiceDraft?.dialogue?.turns.first { $0.id == id }?[keyPath: key] ?? fallback }, set: { value in
      guard store.documentSessionID == session, store.project.voiceDraft?.id == draftID else { return }
      store.change { project in
        if let index = project.voiceDraft?.dialogue?.turns.firstIndex(where: { $0.id == id }) {
          project.voiceDraft?.dialogue?.turns[index][keyPath: key] = value
        }
      }
    })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("Conversation").font(.headline)
        Spacer(); Button("Speakers…") { showSpeakers = true }
        Button("Add line", systemImage: "plus") { addLine() }.disabled(dialogue.turns.count >= 64 || dialogue.speakers.isEmpty)
      }
      ScrollView(.horizontal) {
        HStack {
          ForEach(Array(dialogue.turns.enumerated()), id: \.element.id) { index, line in
            Button { selected = line.id } label: {
              VStack(alignment: .leading, spacing: 2) {
                Text("\(index + 1) · \(dialogue.speakers.first { $0.id == line.speakerID }?.name ?? "Missing speaker")").font(.caption.bold())
                Text(line.text.isEmpty ? "Empty line" : line.text).lineLimit(1).font(.caption).frame(width: 145, alignment: .leading)
              }.padding(7).background(current?.id == line.id ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
            }.buttonStyle(.plain)
          }
        }
      }
      if let current {
        HStack {
          Picker("Speaker", selection: Binding(get: { current.speakerID }, set: { id in edit { $0.assignSpeaker(id, to: current.id) } })) {
            ForEach(dialogue.speakers) { Text($0.name).tag($0.id) }
          }
          Text("Pause after")
          TextField("Seconds", value: turn(\.gapAfter, default: 0), format: .number).frame(width: 55)
          Text("s").foregroundStyle(.secondary)
          Button { move(-1) } label: { Image(systemName: "arrow.left") }.help("Move line earlier").disabled(dialogue.turns.first?.id == current.id)
          Button { move(1) } label: { Image(systemName: "arrow.right") }.help("Move line later").disabled(dialogue.turns.last?.id == current.id)
          Button(role: .destructive) { edit { $0.turns.removeAll { $0.id == current.id } }; selected = nil } label: { Image(systemName: "trash") }.help("Remove line").disabled(dialogue.turns.count <= 1)
        }
        VoiceScriptEditor(text: turn(\.text, default: ""), engine: draft.engine, contextID: current.id.uuidString, customVoice: store.selectedVoiceModel?.supportsInstructions == true)
          .id(current.id)
        if store.selectedVoiceModel?.supportsInstructions == true {
          Toggle("Override speaker delivery for this line", isOn: Binding(get: { current.instructions != nil }, set: { value in turn(\.instructions, default: nil).wrappedValue = value ? "" : nil }))
          if current.instructions != nil {
            QwenVoiceStyleControls(speaker: .constant(nil), instructions: turn(\.instructions, default: nil), showSpeaker: false)
          }
        } else {
        HStack {
          Button(current.reference == nil ? "Use a different performance sample…" : "Edit line’s performance sample…") { showPerformance = true }
            .disabled(dialogue.speakers.first { $0.id == current.speakerID }?.referenceMode == .synthetic)
          if current.reference != nil { Button("Use speaker default") { turn(\.reference, default: nil).wrappedValue = nil } }
        }.font(.caption)
        }
        Text("Lines render in order using each speaker’s voice, then join into one dialogue take. Set the last pause to 0 for a tight ending.").font(.caption).foregroundStyle(.secondary)
      }
    }
    .sheet(isPresented: $showSpeakers) { VoiceSpeakersView().environmentObject(store) }
    .sheet(isPresented: $showPerformance) {
      if let current {
        VoicePerformanceSheet(reference: turn(\.reference, default: nil),
          mode: dialogue.speakers.first { $0.id == current.speakerID }?.referenceMode ?? .audioAndTranscript,
          title: "This line’s performance sample").environmentObject(store)
      }
    }
  }
  private func addLine() {
    guard let speaker = dialogue.speakers.first else { return }
    let nextSpeaker = dialogue.speakers.first { $0.id != current?.speakerID } ?? speaker
    let line = VoiceTurn(speakerID: nextSpeaker.id)
    edit { $0.turns.append(line) }; selected = line.id
  }
  private func move(_ offset: Int) {
    guard let id = current?.id, let index = dialogue.turns.firstIndex(where: { $0.id == id }), dialogue.turns.indices.contains(index + offset) else { return }
    edit { $0.turns.swapAt(index, index + offset) }; selected = id
  }
}

struct VoiceSpeakersView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @State private var selected: UUID?
  var speakers: [VoiceSpeaker] { store.project.voiceDraft?.dialogue?.speakers ?? [] }
  var current: VoiceSpeaker? { speakers.first { $0.id == selected } ?? speakers.first }
  private func field<T>(_ key: WritableKeyPath<VoiceSpeaker, T>, default fallback: T) -> Binding<T> {
    let id = current?.id, session = store.documentSessionID, draftID = store.project.voiceDraft?.id
    return Binding(get: { store.project.voiceDraft?.dialogue?.speakers.first { $0.id == id }?[keyPath: key] ?? fallback }, set: { value in
      guard store.documentSessionID == session, store.project.voiceDraft?.id == draftID else { return }
      store.change { project in
        if let index = project.voiceDraft?.dialogue?.speakers.firstIndex(where: { $0.id == id }) {
          project.voiceDraft?.dialogue?.speakers[index][keyPath: key] = value
        }
      }
    })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Conversation speakers").font(.title2); Spacer()
        Button("Add speaker") {
          var speaker = VoiceSpeaker(name: "Speaker \(speakers.count + 1)")
          if store.selectedVoiceModel?.supportsInstructions == true { speaker.referenceMode = .customVoice }
          store.change { $0.voiceDraft?.dialogue?.speakers.append(speaker) }; selected = speaker.id
        }.disabled(speakers.count >= 64)
        Button("Done") { dismiss() }
      }
      if let current {
        Picker("Edit speaker", selection: Binding(get: { current.id }, set: { selected = $0 })) {
          ForEach(speakers) { Text($0.name).tag($0.id) }
        }
        HStack {
          TextField("Character name", text: field(\.name, default: ""))
          Button("Remove speaker", role: .destructive) {
            store.change { $0.voiceDraft?.dialogue?.speakers.removeAll { $0.id == current.id } }; selected = nil
          }.disabled(speakers.count <= 1 || store.project.voiceDraft?.dialogue?.turns.contains { $0.speakerID == current.id } == true)
        }
        Text("Names identify dialogue lines and are never spoken. Reassign a speaker’s lines before removing them.").font(.caption).foregroundStyle(.secondary)
        if store.selectedVoiceModel?.supportsInstructions == true {
          QwenVoiceStyleControls(speaker: field(\.presetVoice, default: nil), instructions: field(\.instructions, default: nil))
        } else {
        Picker("Voice source", selection: field(\.referenceMode, default: .audioAndTranscript)) {
          Text("Audio + transcript").tag(VoiceReferenceMode.audioAndTranscript)
          if store.project.voiceDraft?.engine == .fishS2Pro { Text("Synthetic voice").tag(VoiceReferenceMode.synthetic) }
          else { Text("Speaker identity only").tag(VoiceReferenceMode.speakerIdentityOnly) }
        }
        if current.referenceMode != .synthetic {
          VoiceSampleControls(reference: field(\.reference, default: nil), mode: current.referenceMode).id(current.id)
        } else { Text("Use a reference sample for a consistent character voice across lines.").font(.caption).foregroundStyle(.secondary) }
        }
      }
    }.padding(24).frame(width: 630).textFieldStyle(.roundedBorder)
      .onDisappear { store.musicPlayer.pause() }
  }
}

struct VoicePerformanceSheet: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @Binding var reference: VoiceReference?
  let mode: VoiceReferenceMode
  let title: String
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack { Text(title).font(.title2); Spacer(); Button("Done") { dismiss() } }
      Text("Choose the same character performing the delivery you want. This sample overrides the speaker default for this line only.").foregroundStyle(.secondary)
      VoiceSampleControls(reference: $reference, mode: mode)
    }.padding(24).frame(width: 630).textFieldStyle(.roundedBorder)
      .onDisappear { store.musicPlayer.pause() }
  }
}

struct VoiceSampleControls: View {
  @EnvironmentObject var store: StudioStore
  @Binding var reference: VoiceReference?
  let mode: VoiceReferenceMode
  @State private var loading = false
  @State private var error: String?
  private func field<T>(_ key: WritableKeyPath<VoiceReference, T>, _ fallback: T) -> Binding<T> {
    Binding(get: { reference?[keyPath: key] ?? fallback }, set: { reference?[keyPath: key] = $0 })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Button("Choose sample…") { choose() }.disabled(loading)
        Menu("Use existing audio") {
          ForEach(store.project.voicePresets ?? []) { preset in Button("Preset · \(preset.name)") { reference = preset.reference } }
          ForEach(store.project.assets.filter { $0.kind == .audio || $0.kind == .video }) { asset in
            Button(asset.name) { reference = VoiceReference(assetID: asset.id, path: asset.path, duration: min(10, max(0.1, asset.duration))) }
          }
        }
        if loading { ProgressView().controlSize(.small) }
      }
      if let sample = reference {
        Text(URL(fileURLWithPath: sample.path).lastPathComponent).font(.caption)
        HStack {
          Text("Start"); TextField("Seconds", value: field(\.start, 0), format: .number)
          Text("Length"); TextField("Seconds", value: field(\.duration, 5), format: .number)
          Text("seconds")
        }
        Picker("Sample channel", selection: field(\.channel, "mix")) {
          Text("Downmix").tag("mix"); Text("Left").tag("left"); Text("Right").tag("right")
        }
        if mode == .audioAndTranscript {
          Text("Words spoken in the selected sample range").font(.headline)
          TextEditor(text: field(\.transcript, "")).frame(height: 100).border(Color.secondary.opacity(0.3))
          Text("Select an expressive performance and enter the exact words spoken in the chosen range.").font(.caption).foregroundStyle(.secondary)
        } else { Text("Identity-only mode captures the speaker’s identity; use Audio + transcript for a stronger performance reference.").font(.caption).foregroundStyle(.secondary) }
        HStack {
          Button("Audition sample range") {
            store.musicPlayer.pause()
            let item = AVPlayerItem(url: URL(fileURLWithPath: sample.path))
            item.forwardPlaybackEndTime = CMTime(seconds: sample.start + sample.duration, preferredTimescale: 48000)
            store.musicPlayer.replaceCurrentItem(with: item)
            store.musicPlayer.seek(to: CMTime(seconds: sample.start, preferredTimescale: 48000)); store.musicPlayer.play()
          }
          Button("Stop") { store.musicPlayer.pause() }
        }
      } else { Text("Add a reference sample for this speaker.").foregroundStyle(.secondary) }
      if let error { Text(error).foregroundStyle(.red) }
    }
  }
  private func choose() {
    let panel = NSOpenPanel(); panel.title = "Choose a voice performance sample"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let original = reference
    let binding = $reference
    loading = true
    Task { @MainActor in
      defer { loading = false }
      do {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard let _ = try await asset.loadTracks(withMediaType: .audio).first, duration.isFinite, duration > 0 else {
          throw StudioError.invalid("Choose a file with an audio track.")
        }
        guard binding.wrappedValue == original else { return }
        binding.wrappedValue = VoiceReference(path: url.path, duration: min(10, duration)); error = nil
      } catch { self.error = error.localizedDescription }
    }
  }
}
