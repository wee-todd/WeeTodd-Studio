import AppKit
import AVFoundation
import CryptoKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

/// Uses the Director's persisted input dictionary so leaving the sheet preserves the draft.
struct MusicVideoIntakeView: View {
  @EnvironmentObject var store: StudioStore
  @Binding var fields: [String: String]
  var currentRunID: () -> UUID
  var onAnalysisBusy: (Bool) -> Void = { _ in }
  @State private var loading = false
  @State private var error: String?
  private func value(_ key: String) -> Binding<String> {
    Binding(get: { fields[key] ?? "" }, set: { fields[key] = $0 })
  }
  private func numericValue(_ key: String) -> Binding<Double> {
    Binding(get: { Double(fields[key] ?? "") ?? 0 }, set: { fields[key] = String($0) })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Create music video").font(.headline)
      Text("Choose your song, describe the visual direction, and let Director plan the timing. Missing creative decisions are requested in the brief review.")
        .font(.callout).foregroundStyle(.secondary)
      HStack {
        Button("Choose audio…") { chooseAudio() }.disabled(loading)
        Menu("Use movie audio") {
          ForEach(store.project.assets.filter { $0.kind == .audio }) { asset in
            Button(asset.name) { Task { await selectAudio(asset.path, lyrics: asset.musicGeneration?.draft.lyrics,
              instrumental: asset.musicGeneration?.draft.instrumental ?? false) } }
          }
        }.disabled(loading)
        if loading { ProgressView().controlSize(.small) }
      }
      Text(fields["audio_path"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0).lastPathComponent } ?? "Select an existing song, or generate one using Movie → Generate Music.")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        Text("Start in song"); TextField("Seconds", value: numericValue("source_start_seconds"), format: .number.precision(.fractionLength(0...3))).frame(width: 70)
        Text("Use"); TextField("Seconds", value: numericValue("duration_seconds"), format: .number.precision(.fractionLength(0...3))).frame(width: 70); Text("seconds")
      }
      Picker("Lyrics", selection: value("lyrics_status")) {
        Text("I have lyrics").tag("supplied")
        Text("Lyrics unknown / ask if needed").tag("unknown")
        Text("Instrumental").tag("instrumental")
      }
      if fields["lyrics_status"] == "supplied" {
        TextEditor(text: value("lyrics")).frame(minHeight: 100).accessibilityLabel("Supplied lyrics")
        Button("Import lyrics…") { importLyrics() }
        Text("Your exact text is preserved. Supplied lyrics are not treated as verified word timestamps.").font(.caption).foregroundStyle(.secondary)
      }
      AudioTimingReviewView(fields: $fields, currentRunID: currentRunID, onBusyChanged: onAnalysisBusy)
      Text("Visual idea and requirements").font(.subheadline.bold())
      TextEditor(text: value("brief")).frame(minHeight: 90).accessibilityLabel("Music video idea")
      HStack {
        Picker("Visual approach", selection: value("visual_style")) {
          ForEach(["Let the director decide", "Performance", "Narrative", "Abstract", "Performance and narrative", "Follow reference images"], id: \.self) { Text($0).tag($0) }
        }
        Picker("Pacing", selection: value("pacing")) {
          ForEach(["Let the director decide", "Restrained", "Balanced", "Energetic"], id: \.self) { Text($0).tag($0) }
        }
      }
      Picker("Model", selection: value("generation_engine")) {
        ForEach([Engine.ltx25, .ltx23, .h3, .drawThings]) { Text($0.label).tag($0.rawValue) }
      }.onChange(of: fields["generation_engine"]) { _, engine in
        if engine == "drawThings" { fields["generation_task"] = "t2v" }
      }
      Picker("Generation", selection: value("generation_task")) {
        if fields["generation_engine"] != "drawThings" { Text("Audio-driven video").tag("a2v") }
        Text("Visuals with music soundtrack").tag("t2v")
      }
      Text("The original song stays on the Music track. Available reference and continuity controls depend on the selected model.")
        .font(.caption).foregroundStyle(.secondary)
      DisclosureGroup("Timing and creative controls") {
        HStack {
          Text("Minimum seconds (0 = model minimum)"); TextField("0", text: value("min_clip_seconds")).frame(width: 65)
          Text("Maximum"); TextField("15", text: value("max_clip_seconds")).frame(width: 65)
        }
        Toggle("Prefer continuous local scenes where compatible", isOn: Binding(get: { fields["continuity_enabled"] != "false" }, set: { fields["continuity_enabled"] = $0 ? "true" : "false" }))
        HStack { Text("Movie frame rate"); TextField("FPS", text: value("frame_rate")).frame(width: 65) }
        if fields["generation_engine"] == "drawThings" {
          HStack {
            Text("Selected model minimum"); TextField("Seconds", text: value("backend_min_seconds")).frame(width: 65)
            Text("Maximum"); TextField("Seconds", text: value("backend_max_seconds")).frame(width: 65)
          }
          Text("Enter the selected Draw Things model's supported duration range. Generation validates its actual capabilities.").font(.caption)
        }
        TextField("Camera direction", text: value("camera_style"))
        TextField("Must include / avoid", text: value("constraints"))
        Picker("Missing details", selection: value("design_policy")) {
          Text("Ask before adding details").tag("Ask before adding details")
          Text("Propose missing details for approval").tag("Propose missing details for approval")
        }
      }
      if let error { Text(error).font(.caption).foregroundStyle(.red) }
    }
  }
  private func chooseAudio() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.audio]; panel.title = "Choose music for your video"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await selectAudio(url.path) }
  }
  private func selectAudio(_ path: String, lyrics: String? = nil, instrumental: Bool = false) async {
    loading = true; error = nil
    let document = store.documentSessionID, projectID = store.project.id
    let run = currentRunID(), draft = fields
    defer { loading = false }
    do {
      let duration = try await AVURLAsset(url: URL(fileURLWithPath: path)).load(.duration).seconds
      guard duration.isFinite, duration > 0 else { throw StudioError.invalid("Choose audio with a finite, positive duration.") }
      let hash = try await Task.detached(priority: .utility) {
        let stream = try FileHandle(forReadingFrom: URL(fileURLWithPath: path)); defer { try? stream.close() }
        var digest = SHA256()
        while let data = try stream.read(upToCount: 1024 * 1024), !data.isEmpty { digest.update(data: data) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
      }.value
      guard store.documentSessionID == document, store.project.id == projectID,
        currentRunID() == run, fields == draft else { return }
      fields["audio_path"] = path; fields["audio_sha256"] = hash
      fields["source_start_seconds"] = "0"; fields["duration_seconds"] = String(min(duration, 3600))
      if instrumental { fields["lyrics_status"] = "instrumental"; fields["lyrics"] = "" }
      else if let lyrics, !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        fields["lyrics"] = lyrics; fields["lyrics_status"] = "supplied"
      } else { fields["lyrics"] = ""; fields["lyrics_status"] = "unknown" }
      if !store.project.assets.contains(where: { $0.path == path && $0.kind == .audio }) {
        var asset = MediaAsset(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent, kind: .audio, path: path)
        asset.duration = duration
        store.change { $0.assets.append(asset) }
      }
    } catch { self.error = error.localizedDescription }
  }
  private func importLyrics() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.plainText]; panel.title = "Import lyrics"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do {
      guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= 1_048_576 else { throw StudioError.invalid("Choose a lyric text file smaller than 1 MiB.") }
      fields["lyrics"] = try String(contentsOf: url, encoding: .utf8); fields["lyrics_status"] = "supplied"
    } catch { self.error = error.localizedDescription }
  }
}
