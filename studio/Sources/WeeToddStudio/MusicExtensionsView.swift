import AppKit
import StudioCore
import SwiftUI

struct MusicModelSettingsView: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Music models").font(.headline)
      Text("Local YuE2 and SheetSage2 inference. These weights are CC BY-NC 4.0 (noncommercial). Model locations are shared across movies.")
        .font(.caption).foregroundStyle(.secondary)
      ForEach(store.installedMusicModels) { model in
        VStack(alignment: .leading, spacing: 4) {
          HStack {
            VStack(alignment: .leading) { Text(model.name); Text(model.kind.label).font(.caption).foregroundStyle(.secondary) }
            Spacer()
            Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)]) }
            Button("Relink…") { store.addMusicModel(model.kind, separateDecoder: model.decoderPath != nil, replacing: model.id) }.disabled(store.operationBusy)
            Button("Remove") { store.removeMusicModel(model) }.disabled(store.operationBusy)
          }
          Text(model.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          if let decoder = model.decoderPath { Text("Decoder: \(decoder)").font(.caption).textSelection(.enabled) }
        }
      }
      HStack {
        Menu("Add installed…") {
          ForEach(MusicModelKind.allCases, id: \.self) { kind in Button(kind.label) { store.addMusicModel(kind) } }
          Button("YuE2 with separate decoder…") { store.addMusicModel(.generator, separateDecoder: true) }
        }
        Menu("Download verified…") {
          Button("YuE2 3B · 8-bit · 4.53 GB") { store.downloadMusicModel() }
          Button("Instrumental LoRA · 140 MB") { store.downloadMusicExtension(.instrumental) }
          Button("SheetSage2 · 1.39 GB") { store.downloadMusicExtension(.sheetsage) }
        }
      }.disabled(store.operationBusy)
      if store.bridge.busy { Text(store.bridge.message).font(.caption); Button("Cancel setup") { store.bridge.cancel() } }
    }
  }
}

struct MusicArrangementEditor: View {
  @EnvironmentObject var store: StudioStore
  var arrangement: MusicArrangement { store.project.musicDraft?.arrangement ?? MusicArrangement() }
  func update(_ body: (inout MusicArrangement) -> Void) {
    var next = arrangement; body(&next); store.change { $0.musicDraft?.arrangement = next }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Picker("Section plan", selection: Binding(get: { arrangement.mode }, set: { mode in
        update { $0.mode = mode; if mode != .automatic && $0.sections.isEmpty {
          $0.sections = [MusicSection(name: "intro", start: 0, end: 15), MusicSection(name: "verse", start: 15, end: 45), MusicSection(name: "outro", start: 45, end: 60)]
        } }
      })) { Text("Automatic").tag(MusicArrangement.Mode.automatic); Text("Ordered sections").tag(MusicArrangement.Mode.ordered); Text("Timed sections").tag(MusicArrangement.Mode.timed) }
      if arrangement.mode != .automatic {
        ScrollView {
          VStack(spacing: 6) {
            ForEach(arrangement.sections) { section in
              HStack {
                Picker("Section", selection: Binding(get: { section.name }, set: { value in
                  update { plan in if let i = plan.sections.firstIndex(where: { $0.id == section.id }) { plan.sections[i].name = value } }
                })) { ForEach(MusicSection.names, id: \.self) { Text($0.capitalized).tag($0) } }.labelsHidden()
                if arrangement.mode == .timed {
                  timeField("Start", section, \.start); timeField("End", section, \.end)
                }
                Button { update { plan in if let i = plan.sections.firstIndex(where: { $0.id == section.id }), i > 0 { plan.sections.swapAt(i, i-1) } } }
                  label: { Image(systemName: "arrow.up") }.help("Move section earlier")
                Button { update { $0.sections.removeAll { $0.id == section.id } } }
                  label: { Image(systemName: "trash") }.help("Delete section")
              }
            }
          }
        }.frame(maxHeight: 190)
        Button("Add section") { update { plan in
          let start = plan.sections.last?.end ?? 0
          plan.sections.append(MusicSection(name: "chorus", start: start, end: start+30))
        } }.disabled(arrangement.sections.count >= 64)
      }
      Text("Section times guide proportions, not exact duration. This LoRA favors longer songs; use the timeline to trim a take and fade its ending.")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
  func timeField(_ title: String, _ section: MusicSection, _ key: WritableKeyPath<MusicSection, Int?>) -> some View {
    HStack(spacing: 3) {
      Text(title).font(.caption)
      TextField("seconds", value: Binding(get: { section[keyPath: key] ?? 0 }, set: { value in
        update { plan in if let i = plan.sections.firstIndex(where: { $0.id == section.id }) { plan.sections[i][keyPath: key] = value } }
      }), format: .number).frame(width: 45)
      Text("s").font(.caption)
    }
  }
}

struct MusicReferenceView: View {
  @EnvironmentObject var store: StudioStore
  @State private var expanded = false
  var reference: MusicReference? { store.project.musicDraft?.reference }
  func field<T>(_ key: WritableKeyPath<MusicReference, T>, default value: T) -> Binding<T> {
    Binding(get: { reference?[keyPath: key] ?? value }, set: { newValue in
      store.change { $0.musicDraft?.reference?[keyPath: key] = newValue; $0.musicDraft?.reference?.analysis = nil }
    })
  }
  var body: some View {
    DisclosureGroup("Reference audio → editable score", isExpanded: $expanded) {
      VStack(alignment: .leading, spacing: 9) {
        HStack {
          Button("Choose reference…") { store.chooseMusicReference() }
          if reference != nil { Button("Remove") { store.change { $0.musicDraft?.reference = nil } } }
        }.disabled(store.operationBusy)
        if let reference {
          Text(URL(fileURLWithPath: reference.path).lastPathComponent).font(.caption).textSelection(.enabled)
          Picker("Reference encoder", selection: field(\.encoderID, default: nil)) {
            Text("Choose installed encoder").tag(Optional<UUID>.none)
            ForEach(store.installedMusicModels.filter { $0.kind == .sheetsage }) { Text($0.name).tag(Optional($0.id)) }
          }
          HStack {
            Text("Start (s)"); TextField("Start", value: field(\.sourceIn, default: 0), format: .number)
            Text("Length (s)"); TextField("Length", value: field(\.duration, default: 30), format: .number)
          }
          Toggle("Melody only · reimagine the harmony", isOn: field(\.melodyOnly, default: true))
          Text("Analyze 1–300 seconds locally. Melody-only keeps the notes; turning it off includes the extracted chords. Transcription may need corrections.")
            .font(.caption).foregroundStyle(.secondary)
          Button("Analyze reference") { Task { await store.analyzeMusicReference() } }.disabled(store.operationBusy)
          if let analysis = reference.analysis {
            Text("Extracted score · \(analysis.eventCount) musical events").font(.caption.bold())
            TextEditor(text: Binding(get: { store.project.musicDraft?.reference?.analysis?.abc ?? "" }, set: { value in
              store.change { $0.musicDraft?.reference?.analysis?.abc = value }
            })).font(.system(.caption, design: .monospaced)).frame(height: 130)
            HStack {
              Button("Use score") { store.useMusicReferenceScore() }
              Button("Reveal analysis") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: analysis.artifacts)]) }
            }
          }
        } else { Text("Use a recording to guide the melody and arrangement of new music.").font(.caption).foregroundStyle(.secondary) }
      }.padding(.top, 8)
    }
  }
}
