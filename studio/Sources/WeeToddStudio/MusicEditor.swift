import AppKit
import AVFoundation
import StudioCore
import SwiftUI

struct MusicEditor: View {
  @EnvironmentObject var store: StudioStore
  @State private var advanced = false
  @State private var sourceIn: Double = 0
  @State private var muteClipAudio = false
  @State private var auditionTime: Double = 0
  @State private var auditionPlaying = false
  private let clock = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()
  var draft: MusicDraft { store.project.musicDraft ?? MusicDraft() }
  func field<T>(_ key: WritableKeyPath<MusicDraft, T>) -> Binding<T> {
    Binding(get: { draft[keyPath: key] }, set: { value in store.change { $0.musicDraft?[keyPath: key] = value } })
  }
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Button { store.musicPlayer.pause(); store.showMusic = false } label: { Label("Back to movie", systemImage: "arrow.left") }
        TextField("Take name", text: field(\.name)).frame(maxWidth: 320)
        Spacer()
        Text("YuE2 · WeeTodd native MLX").font(.caption).foregroundStyle(.secondary)
        Button("Export job…") { store.exportMusicRequest() }
      }.padding(16)
      Divider()
      HSplitView {
        ScrollView {
          VStack(alignment: .leading, spacing: 14) {
            Text("Native music model").font(.headline)
            HStack {
              TextField("Local YuE2 MLX model folder", text: field(\.modelPath))
              Button("Choose…") { store.chooseMusicFolder() }
            }
            Text("Choose a compatible merged MLX folder with its tokenizer and stereo decoder. Weights: CC BY-NC 4.0.")
              .font(.caption).foregroundStyle(.secondary)
            Link("YuE2 MLX weights", destination: URL(string: "https://huggingface.co/npario/YuE2-3B-MLX")!).font(.caption)
            Button("Download verified 8-bit model…") { store.downloadMusicModel() }.disabled(store.operationBusy)
            Divider()
            Text("Musical direction").font(.headline)
            Toggle("Instrumental", isOn: field(\.instrumental))
            choices("Genre", field(\.genre), ["Indie pop", "Pop", "Rock", "Electronic", "Hip hop", "Jazz", "Soul", "Folk", "Cinematic", "Ambient"])
            choices("Mood", field(\.mood), ["Hopeful", "Joyful", "Melancholic", "Dreamy", "Tense", "Romantic", "Playful", "Dark"])
            choices("Energy", field(\.energy), ["Low", "Medium", "High", "Building"])
            TextField("Instruments · e.g. piano, synth bass", text: field(\.instruments))
            if !draft.instrumental {
              TextField("Vocal character", text: field(\.vocal))
              choices("Language", field(\.language), ["English", "Mandarin", "Spanish", "Japanese", "Korean", "French"])
            }
            HStack { Text("Requested BPM"); TextField("0 = unspecified", value: field(\.bpm), format: .number).frame(width: 90) }
            Text("These choices guide the model. Audition each take to confirm the tempo, vocals and feel.").font(.caption).foregroundStyle(.secondary)
            TextField("Additional musical direction", text: field(\.direction), axis: .vertical).lineLimit(3...5)
            Picker("Acoustic preset", selection: field(\.steps)) {
              Text("Quality · 32 steps").tag(32); Text("Fast · 8 steps").tag(8)
              if ![8, 32].contains(draft.steps) { Text("Custom · \(draft.steps) steps").tag(draft.steps) }
            }
            HStack {
              Text("Length budget (seconds)")
              TextField("Maximum seconds", value: Binding(get: { draft.maximumSeconds }, set: { seconds in
                store.change { $0.musicDraft?.setMaximumSeconds(seconds) }
              }), format: .number).frame(width: 90)
            }
            Text("Songs may end earlier. A take that reaches its token budget is marked truncated.").font(.caption).foregroundStyle(.secondary)
            HStack {
              Text("Seed"); TextField("Seed", value: field(\.seed), format: .number)
              Button { store.change { $0.musicDraft?.seed = Int.random(in: 0...Int(Int32.max)) } } label: { Image(systemName: "dice") }
            }
            DisclosureGroup("Advanced generation", isExpanded: $advanced) { advancedSettings }
          }.textFieldStyle(.roundedBorder).padding(18)
        }.frame(minWidth: 325, idealWidth: 365, maxWidth: 420)
        VStack(alignment: .leading, spacing: 12) {
          Text("Style sent to the model").font(.headline)
          Text(draft.style).font(.callout).textSelection(.enabled).padding(12)
            .frame(maxWidth: .infinity, alignment: .leading).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
          HStack {
            Text(draft.instrumental ? "Instrumental arrangement" : "Lyrics").font(.headline); Spacer()
            Menu("Add section") {
              ForEach(["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Interlude", "Outro"], id: \.self) { section in
                Button(section) { store.change { $0.musicDraft?.lyrics += "\n[\(section)]\n" } }
              }
            }.disabled(draft.instrumental)
          }
          if draft.instrumental {
            Text("Describe arrangement and musical changes in Additional musical direction.").foregroundStyle(.secondary)
            Spacer(minLength: 20)
          } else {
            TextEditor(text: field(\.lyrics)).font(.body).scrollContentBackground(.hidden)
              .padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8)).frame(minHeight: 150)
          }
          audition
          takes
        }.padding(18).frame(minWidth: 440, maxWidth: .infinity)
      }
      Divider()
      HStack {
        if store.bridge.busy {
          ProgressView().controlSize(.small); Text(store.bridge.message).font(.caption)
          Button("Cancel") { store.bridge.cancel() }
        } else { Text(store.musicModelStatus ?? "Generate, audition, then choose where to use your take.").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        Button("Check model") { Task { await store.inspectMusicModel() } }.disabled(store.operationBusy)
        Button("Generate music") { Task { await store.generateMusic() } }.buttonStyle(.borderedProminent)
          .disabled(store.operationBusy || draft.modelPath.isEmpty)
      }.padding(16)
    }.background(Theme.background)
      .onChange(of: draft.modelPath) { _ in store.musicModelStatus = nil }
      .onChange(of: draft.vaePath) { _ in store.musicModelStatus = nil }
      .onChange(of: draft.precision) { _ in store.musicModelStatus = nil }
      .onReceive(clock) { _ in
        let seconds = store.musicPlayer.currentTime().seconds
        auditionTime = seconds.isFinite ? seconds : 0; auditionPlaying = store.musicPlayer.rate > 0
      }.onDisappear { store.musicPlayer.pause() }
  }
  func choices(_ label: String, _ selection: Binding<String>, _ values: [String]) -> some View {
    HStack {
      Text(label).frame(width: 64, alignment: .leading); TextField(label, text: selection)
      Menu { ForEach(values, id: \.self) { value in Button(value) { selection.wrappedValue = value } } }
        label: { Image(systemName: "chevron.down") }.menuStyle(.borderlessButton).frame(width: 20)
    }
  }
  var advancedSettings: some View {
    VStack(alignment: .leading, spacing: 12) {
      Picker("Composition", selection: field(\.cot)) {
        Text("Melody + chords").tag("full"); Text("Melody only").tag("melody"); Text("Direct · no score").tag("off")
      }
      Picker("Precision", selection: field(\.precision)) {
        Text("From checkpoint").tag("auto"); Text("BF16").tag("bf16"); Text("8-bit").tag("8bit"); Text("4-bit").tag("4bit")
      }
      Text("Precision must match the files; this menu does not convert weights.").font(.caption).foregroundStyle(.secondary)
      HStack {
        TextField("Optional separate decoder folder", text: field(\.vaePath))
        Button("Choose…") { store.chooseMusicFolder(decoder: true) }
      }
      Toggle("Checkpoint guidance default", isOn: field(\.useDefaultGuidance))
      if !draft.useDefaultGuidance { HStack { Text("Guidance"); TextField("Guidance", value: field(\.guidance), format: .number) } }
      HStack { Text("Acoustic midpoint steps"); TextField("Steps", value: field(\.steps), format: .number) }
      Picker("Memory", selection: field(\.memoryMode)) {
        Text("Unload between stages").tag("staged"); Text("Retain within this job").tag("resident")
      }
      MusicSamplingControls(title: "Score sampling", sampling: field(\.scoreSampling))
      MusicSamplingControls(title: "Music sampling", sampling: field(\.musicSampling))
      if draft.cot != "off" {
        Text("ABC score · leave empty to compose").font(.caption.bold())
        TextEditor(text: field(\.abc)).font(.system(.caption, design: .monospaced)).frame(height: 110)
        Button("Import score…") { store.loadMusicScore() }
        Button("Compose score only") { Task { await store.planMusic() } }
          .disabled(store.operationBusy || draft.modelPath.isEmpty)
      }
    }.padding(.top, 8)
  }
  var audition: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let take = store.selectedMusicTake {
        HStack {
          Button { if auditionPlaying { store.musicPlayer.pause() } else { store.musicPlayer.play() } }
            label: { Image(systemName: auditionPlaying ? "pause.fill" : "play.fill") }
          Text(take.name).font(.headline); Spacer()
          Text("\(auditionTime, specifier: "%.1f") / \(take.duration, specifier: "%.1f") s").monospacedDigit()
        }
        Slider(value: Binding(get: { min(take.duration, max(0, auditionTime)) }, set: { value in
          store.musicPlayer.seek(to: CMTime(seconds: value, preferredTimescale: 48000)); auditionTime = value
        }), in: 0...max(0.01, take.duration))
        if take.musicGeneration?.truncated == true {
          Label("Token budget reached · check the ending", systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
        HStack {
          Button("Add to Music track") { store.placeMusicTake(take) }
          Button("Restore settings") { if let draft = take.musicGeneration?.draft { store.change { $0.musicDraft = draft } } }
          Button("Reveal files") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: take.path)]) }
        }
        if advanced {
          HStack {
            Button("Resynthesize take") { Task { await store.generateMusic(reusing: take) } }
            Button("Decode saved latents") { Task { await store.generateMusic(reusing: take, decodeOnly: true) } }
          }.disabled(store.operationBusy)
          Text("Resynthesis reuses this take’s composition and music tokens, with the current acoustic steps and seed. Decode reruns only the stereo decoder.").foregroundStyle(.secondary)
        }
        Divider()
        HStack {
          Text("Song in-point"); TextField("Seconds", value: $sourceIn, format: .number).frame(width: 85)
          Text("s").foregroundStyle(.secondary); Spacer()
          Button("Use as audio driver") { Task { await store.useMusicDriver(take, sourceIn: sourceIn, muteClipAudio: muteClipAudio) } }
            .disabled(store.selectedClip == nil || store.operationBusy)
        }
        Toggle("Mute clip audio when using the master Music track", isOn: $muteClipAudio)
        Text(store.selectedClip.map { "Target: \($0.name) · \($0.duration.formatted()) s. Prepares a lossless excerpt from this in-point." }
          ?? "Select a native video clip in the movie to use a music excerpt as its audio driver.").foregroundStyle(.secondary)
      } else { Text("Generated takes appear here for audition.").foregroundStyle(.secondary) }
    }.font(.caption).padding(14).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
  }
  var takes: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 7) {
        ForEach(store.musicTakes.reversed()) { take in
          Button { store.auditionMusic(take) } label: {
            HStack {
              Image(systemName: "waveform"); Text(take.name); Spacer()
              Text("\(take.duration, specifier: "%.1f") s · seed \(take.musicGeneration?.request.seed ?? 0)")
            }.padding(8).background(store.selectedMusicAssetID == take.id ? Theme.mint.opacity(0.15) : Theme.raised,
              in: RoundedRectangle(cornerRadius: 6))
          }.buttonStyle(.plain)
        }
      }
    }.frame(maxHeight: 160)
  }
}

private struct MusicSamplingControls: View {
  var title: String
  @Binding var sampling: MusicSampling
  var body: some View {
    DisclosureGroup(title) {
      VStack(spacing: 8) {
        field("Temperature", $sampling.temperature); field("Top P", $sampling.topP)
        integer("Top K", $sampling.topK); field("Repetition penalty", $sampling.repetitionPenalty)
        integer("Repetition window", $sampling.penaltyWindow)
        integer("Minimum tokens", $sampling.minTokens); integer("Maximum tokens", $sampling.maxTokens)
      }.padding(.top, 8)
    }
  }
  func field(_ label: String, _ value: Binding<Double>) -> some View {
    HStack { Text(label); Spacer(); TextField(label, value: value, format: .number).frame(width: 85) }
  }
  func integer(_ label: String, _ value: Binding<Int>) -> some View {
    HStack { Text(label); Spacer(); TextField(label, value: value, format: .number).frame(width: 85) }
  }
}
