import AppKit
import StudioCore
import SwiftUI

struct AudioTimingReviewView: View {
  @EnvironmentObject var store: StudioStore
  @Binding var fields: [String: String]
  var currentRunID: () -> UUID
  var onBusyChanged: (Bool) -> Void = { _ in }
  @State private var busy = false
  @State private var error: String?
  @State private var review: AudioAnalysisReview?
  @State private var markers: [AudioTimingMarker] = []
  @State private var selectedKind = "downbeat"
  @State private var showWords = false
  @State private var reviewedWord: AudioWordTiming?
  @State private var reviewedSeconds = ""
  @State private var showingReviewedCut = false
  private var inputIdentity: [String] { ["audio_sha256", "lyrics", "lyrics_status", "analysis_mode", "analysis_model_directory", "analysis_vocal_mode", "source_start_seconds", "duration_seconds", "frame_rate"].map { fields[$0] ?? "" } }
  private func value(_ key: String) -> Binding<String> {
    Binding(get: { fields[key] ?? "" }, set: { fields[key] = $0 })
  }
  private var selectedCues: [AudioEditingCue] { review?.cues.filter { $0.kind == selectedKind } ?? [] }
  private var fps: Double { Double(fields["frame_rate"] ?? "") ?? 24 }
  private var sourceStart: Double { Double(fields["source_start_seconds"] ?? "") ?? 0 }
  private var duration: Double { Double(fields["duration_seconds"] ?? "") ?? 0 }
  var body: some View {
    DisclosureGroup("Audio analysis and timing markers") {
      VStack(alignment: .leading, spacing: 10) {
        Picker("Analysis", selection: value("analysis_mode")) {
          Text("Learned beats + English word evidence").tag("neural")
          Text("Quick onset and dynamics preview").tag("fast")
        }
        if fields["analysis_mode"] == "neural" {
          if fields["lyrics_status"] != "instrumental" {
            Picker("Word timing audio", selection: Binding(
              get: { fields["analysis_vocal_mode"] ?? "mixed" },
              set: { fields["analysis_vocal_mode"] = $0 }
            )) {
              Text("Original mix").tag("mixed")
              Text("Isolate vocals first (optional)").tag("isolated")
            }.disabled(busy)
            if fields["analysis_vocal_mode"] == "isolated" {
              Text("Adds a local vocal-isolation pass and about 36 MB of model files. Cached vocals are used for English word evidence; beats and the movie soundtrack use your original song.")
                .font(.caption).foregroundStyle(.secondary)
            }
          }
          HStack {
            Button("Choose model folder…") { chooseFolder(download: false) }
            Button("Set up analysis models…") { chooseFolder(download: true) }
          }.disabled(busy || store.operationBusy)
          Text(fields["analysis_model_directory"].flatMap { $0.isEmpty ? nil : $0 } ?? "Set up the compact beat and English acoustic models (about 386 MB).")
            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          Text("Sung words may remain unaligned. Scores show acoustic support, not a guarantee; review omissions, repeats and section suggestions.")
            .font(.caption).foregroundStyle(.secondary)
        }
        HStack {
          Button("Analyze and review timing") { Task { await analyze() } }
            .disabled(busy || store.operationBusy || fields["audio_sha256"]?.count != 64)
          if busy { ProgressView().controlSize(.small); Button("Cancel") { store.bridge.cancel() } }
        }
        if let review, review.sourceSHA256 == fields["audio_sha256"] {
          if review.vocalMode == "isolated" { Text("Word evidence uses estimated isolated vocals; sung timing still requires review.").font(.caption).foregroundStyle(.secondary) }
          Text("\(review.beatCount) beats · \(review.downbeatCount) downbeats · \(review.words.filter { $0.startSeconds != nil }.count)/\(review.words.count) words timed")
            .font(.caption)
          DisclosureGroup("Words and lyric lines — review required") {
            if fields["lyrics_status"] == "supplied", let raw = review.recognizedText, !raw.isEmpty {
              DisclosureGroup("Compare recognition with supplied lyrics") {
                VStack(alignment: .leading, spacing: 6) {
                  Text("Original recognition").fontWeight(.semibold)
                  Text(raw).textSelection(.enabled)
                  if let assisted = review.lyricAssistedText, assisted != raw {
                    Text("Lyric-assisted wording — review required").fontWeight(.semibold)
                    Text(assisted).textSelection(.enabled)
                  }
                  Text("Corrections require acoustic support. Unresolved lyrics are not inserted into the transcription.")
                    .foregroundStyle(.secondary)
                }.font(.caption)
              }
            }
            Toggle("Show individual words", isOn: $showWords)
            Text(review.warnings.joined(separator: "\n")).font(.caption).foregroundStyle(.secondary)
            ScrollView {
              LazyVStack(alignment: .leading, spacing: 5) {
                ForEach(Array((showWords || review.lines.isEmpty ? review.words + review.extraWords : review.lines).enumerated()), id: \.offset) { _, word in
                  HStack(alignment: .top) {
                    Text(word.startSeconds.map { String(format: "%.3f–%.3f", $0, word.endSeconds ?? $0) } ?? "Unaligned")
                      .monospacedDigit().frame(width: 110, alignment: .leading)
                    VStack(alignment: .leading) {
                      Text(word.text)
                      if let observed = word.observedText, observed.uppercased() != word.text.uppercased() {
                        Text("Recognition: \(observed)").foregroundStyle(.secondary)
                      }
                      if let status = word.verificationLabel { Text(status).foregroundStyle(.secondary) }
                      if word.startSeconds != nil { Text(String(format: "Acoustic support %.2f", word.confidence)).foregroundStyle(.secondary) }
                      if !word.flags.isEmpty { Text(word.flags.joined(separator: ", ")).foregroundStyle(.orange) }
                    }
                    Spacer()
                    Button("Reviewed cut…") {
                      reviewedWord = word
                      reviewedSeconds = word.endSeconds.map { String(format: "%.3f", $0) } ?? ""
                      showingReviewedCut = true
                    }.disabled(busy || duration <= 0 || markers.count >= 1000)
                  }.font(.caption)
                }
              }
            }.frame(maxHeight: 180)
          }
          HStack {
            Picker("Suggestions", selection: $selectedKind) {
              Text("Downbeats").tag("downbeat"); Text("Beats").tag("beat")
              Text("Onsets").tag("onset"); Text("Dynamics").tag("energy_change")
              Text("Sections").tag("section"); Text("Line ends").tag("line_end")
              Text("Word ends").tag("word_end"); Text("Pauses").tag("pause")
            }
            Button("Add suggestions") { addSuggestions() }.disabled(selectedCues.isEmpty)
          }
          Text("Add suggested cuts, then adjust source seconds or lock required cuts. The planner snaps cuts to movie frames.")
            .font(.caption).foregroundStyle(.secondary)
        }
        HStack {
          Text("Reviewed markers").font(.subheadline.bold())
          Button("Add marker") {
            markers.append(AudioTimingMarker(timeSeconds: sourceStart + max(0.01, duration / 2)))
          }.disabled(duration <= 0 || markers.count >= 1000)
        }
        ScrollView {
          LazyVStack(alignment: .leading) {
            ForEach($markers) { $marker in
              HStack {
                TextField("Label", text: $marker.label)
                TextField("Source seconds", value: $marker.timeSeconds, format: .number.precision(.fractionLength(0...3))).frame(width: 90)
                Text((try? marker.frame(fps: fps, sourceStart: sourceStart, duration: duration)).map { "Frame \($0)" } ?? "Outside range")
                  .font(.caption).frame(width: 95)
                Toggle("Lock", isOn: $marker.locked).toggleStyle(.checkbox)
                Button { markers.removeAll { $0.id == marker.id } } label: { Image(systemName: "minus.circle") }
              }
            }
          }
        }.frame(maxHeight: markers.isEmpty ? 0 : 210)
        if let error { Text(error).foregroundStyle(.red).font(.caption) }
      }.padding(.top, 8)
    }
    .onAppear { restore(); if fields["analysis_mode"] == "fast" { selectedKind = "onset" } }
    .onChange(of: fields["analysis_mode"]) { _, mode in
      selectedKind = mode == "fast" ? "onset" : "downbeat"
      if mode == "fast" { fields["analysis_vocal_mode"] = "mixed" }
    }
    .onChange(of: markers) { _, _ in saveMarkers() }
    .onChange(of: busy) { _, value in onBusyChanged(value) }
    .onChange(of: inputIdentity) { _, _ in
      review = nil; fields["audio_analysis_preview"] = nil
      showingReviewedCut = false; reviewedWord = nil
    }
    .onChange(of: fields["audio_sha256"]) { _, _ in review = nil; markers = []; fields["audio_analysis_preview"] = nil }
    .sheet(isPresented: $showingReviewedCut) {
      VStack(alignment: .leading, spacing: 12) {
        Text("Reviewed lyric cut").font(.headline)
        Text(reviewedWord?.text ?? "")
        Text("Enter the boundary you reviewed in the original song. This creates an editable cut marker and keeps the acoustic timing evidence unchanged.")
          .font(.caption).foregroundStyle(.secondary)
        TextField("Source seconds", text: $reviewedSeconds)
        HStack {
          Button("Cancel") { showingReviewedCut = false }
          Spacer()
          Button("Add reviewed marker") { addReviewedCut() }
            .disabled(Double(reviewedSeconds).flatMap { seconds in
              try? reviewedWord?.reviewedCutMarker(at: seconds, fps: fps, sourceStart: sourceStart, duration: duration)
            } == nil)
        }
      }.padding(20).frame(width: 420)
    }
  }
  private func addReviewedCut() {
    guard let word = reviewedWord, let seconds = Double(reviewedSeconds) else { return }
    do {
      markers.append(try word.reviewedCutMarker(at: seconds, fps: fps, sourceStart: sourceStart, duration: duration))
      markers.sort { $0.timeSeconds < $1.timeSeconds }
      showingReviewedCut = false
    } catch { self.error = error.localizedDescription }
  }
  private func restore() {
    if let text = fields["timing_markers"], let data = text.data(using: .utf8) {
      markers = (try? JSONDecoder().decode([AudioTimingMarker].self, from: data)) ?? []
    }
    if let text = fields["audio_analysis_preview"], let data = text.data(using: .utf8) {
      review = try? JSONDecoder().decode(AudioAnalysisReview.self, from: data)
    }
  }
  private func saveMarkers() {
    do { fields["timing_markers"] = String(decoding: try JSONEncoder().encode(markers), as: UTF8.self) }
    catch { self.error = "Marker times must be finite numbers." }
  }
  private func addSuggestions() {
    for cue in selectedCues where markers.count < 1000 {
      if !markers.contains(where: { abs($0.timeSeconds-cue.timeSeconds) < 0.001 }) {
        markers.append(AudioTimingMarker(timeSeconds: cue.timeSeconds, label: cue.label))
      }
    }
    markers.sort { $0.timeSeconds < $1.timeSeconds }
  }
  private func chooseFolder(download: Bool) {
    let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
    panel.canCreateDirectories = download
    panel.title = download ? "Choose a library for audio analysis models (Apache-2.0 / MIT)" : "Choose the WeeTodd-Analysis model folder"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    if !download { fields["analysis_model_directory"] = url.path; return }
    Task {
      busy = true; error = nil
      let session = store.documentSessionID, run = currentRunID(), draft = fields
      defer { busy = false }
      do {
        let result = try await store.bridge.invoke("music-analysis-setup", runtime: store.runtime, payload: ["directory": url.path, "include_vocals": fields["analysis_vocal_mode"] == "isolated" && fields["lyrics_status"] != "instrumental"])
        guard session == store.documentSessionID, run == currentRunID(), fields == draft else { return }
        guard let path = result["model_directory"] as? String else { throw StudioError.invalid("Model setup did not return its folder.") }
        fields["analysis_model_directory"] = path
      } catch {
        guard session == store.documentSessionID, run == currentRunID(), fields == draft else { return }
        self.error = error.localizedDescription
      }
    }
  }
  private func analyze() async {
    busy = true; error = nil
    let session = store.documentSessionID, run = currentRunID(), draft = fields
    defer { busy = false }
    do {
      var request: [String: Any] = [:]
      for key in ["audio_path", "audio_sha256", "lyrics", "lyrics_status", "analysis_mode", "analysis_model_directory"] { request[key] = fields[key] ?? "" }
      request["analysis_vocal_mode"] = fields["analysis_vocal_mode"] ?? "mixed"
      request["frame_rate"] = fps; request["source_start_seconds"] = sourceStart; request["duration_seconds"] = duration
      let output = store.dataDirectory.appendingPathComponent("AudioAnalysis/\(UUID().uuidString)")
      let result = try await store.bridge.invoke("music-analyze", runtime: store.runtime, payload: request, output: output)
      guard session == store.documentSessionID, run == currentRunID(), fields == draft else { return }
      let value = try AudioAnalysisReview(result: result)
      review = value
      fields["audio_analysis_preview"] = String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
    } catch {
      guard session == store.documentSessionID, run == currentRunID(), fields == draft else { return }
      self.error = error.localizedDescription
    }
  }
}
