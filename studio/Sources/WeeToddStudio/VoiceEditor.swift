import AVFoundation
import StudioCore
import SwiftUI

struct VoiceEditor: View {
  @EnvironmentObject var store: StudioStore
  var draft: VoiceDraft { store.project.voiceDraft ?? VoiceDraft() }
  func field<T>(_ key: WritableKeyPath<VoiceDraft, T>) -> Binding<T> {
    Binding(get: { draft[keyPath:key] }, set: { value in store.change { $0.voiceDraft?[keyPath:key] = value } })
  }
  func reference<T>(_ key: WritableKeyPath<VoiceReference, T>, _ fallback: T) -> Binding<T> {
    Binding(get: { draft.reference?[keyPath:key] ?? fallback }, set: { value in store.change { $0.voiceDraft?.reference?[keyPath:key] = value } })
  }
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Button("Back to movie") { store.musicPlayer.pause(); store.showVoice = false }
        Text("Voice").font(.title2); Spacer()
        TextField("Take name", text: field(\.name)).frame(width: 240)
      }.padding()
      Divider()
      HSplitView {
        ScrollView {
          VStack(alignment: .leading, spacing: 14) {
            Picker("Family", selection: Binding(get: { draft.engine }, set: { store.selectVoiceFamily($0) })) {
              Text("Fish").tag(VoiceEngine.fishS2Pro)
              Text("Qwen").tag(VoiceEngine.qwen3TTS)
            }.disabled(store.operationBusy)
            Picker("Installed model", selection: Binding<UUID?>(get: { store.selectedVoiceModel?.id }, set: { store.selectVoiceModel($0) })) {
              if store.selectedVoiceModel == nil {
                Text(draft.modelID == nil ? "Choose an installed model" : "Selected model is unavailable").tag(UUID?.none)
              }
              ForEach(store.familyVoiceModels) { model in
                Text(model.variantName).tag(Optional(model.id))
              }
            }.disabled(store.operationBusy || store.familyVoiceModels.isEmpty)
            if store.familyVoiceModels.isEmpty {
              Text("No \(draft.engine.label) models are configured. Add an installed model or download one in Runtime settings.")
                .font(.caption).foregroundStyle(.secondary)
            } else if let model = store.selectedVoiceModel, !store.voiceModelAvailable(model) {
              Text("This model is unavailable. Reconnect its drive or update its location in Runtime settings.")
                .font(.caption).foregroundStyle(.orange)
            }
            Button("Manage speech models in Runtime settings…") { store.showRuntime = true }
            Text(store.selectedVoiceModel?.supportsInstructions == true ? "Your model choice is remembered. Choose a built-in voice and describe its delivery." : "Your model choice is remembered. The audio sample below supplies the voice to copy.")
              .font(.caption).foregroundStyle(.secondary)
            Divider()
            if draft.usesDialogue == true {
              Text("Each character has their own voice. Open Speakers in the dialogue editor to configure them.").font(.callout).foregroundStyle(.secondary)
            } else if store.selectedVoiceModel?.supportsInstructions == true {
              QwenVoiceStyleControls(speaker: field(\.presetVoice), instructions: field(\.instructions))
            } else {
            Picker("Reference", selection: field(\.referenceMode)) {
              Text("Audio + transcript").tag(VoiceReferenceMode.audioAndTranscript)
              if draft.engine == .qwen3TTS { Text("Speaker identity only").tag(VoiceReferenceMode.speakerIdentityOnly) }
              if draft.engine == .fishS2Pro { Text("Synthetic voice").tag(VoiceReferenceMode.synthetic) }
            }
            if draft.referenceMode != .synthetic {
              HStack {
                Button("Choose sample…") { store.chooseVoiceReference() }
                Menu("Use existing audio") {
                  Button("Selected clip soundtrack") { store.referenceFromClip() }.disabled(store.selectedClip?.sourcePath.isEmpty != false)
                  ForEach(store.project.assets.filter { $0.kind == .audio || $0.kind == .video }) { asset in
                    Button(asset.name) { store.referenceFromAsset(asset) }
                  }
                  ForEach(store.project.voicePresets ?? []) { preset in
                    Button("Preset · \(preset.name)") { store.change { $0.voiceDraft?.reference = preset.reference } }
                  }
                }
              }
              if let sample = draft.reference {
                Text(URL(fileURLWithPath: sample.path).lastPathComponent).font(.caption)
                HStack {
                  Text("Start"); TextField("Seconds", value: reference(\.start,0), format: .number)
                  Text("Length"); TextField("Seconds", value: reference(\.duration,5), format: .number)
                }
                Text("The selected range is used exactly as entered. Update the transcript to match that range.").font(.caption).foregroundStyle(.secondary)
                Picker("Sample channel", selection: reference(\.channel,"mix")) {
                  Text("Downmix").tag("mix"); Text("Left").tag("left"); Text("Right").tag("right")
                }
                Button("Audition sample range") {
                  store.musicPlayer.pause()
                  let item = AVPlayerItem(url: URL(fileURLWithPath: sample.path))
                  item.forwardPlaybackEndTime = CMTime(seconds: sample.start + sample.duration, preferredTimescale: 48000)
                  store.musicPlayer.replaceCurrentItem(with: item)
                  store.musicPlayer.seek(to: CMTime(seconds: sample.start, preferredTimescale: 48000)); store.musicPlayer.play()
                }
                if draft.referenceMode == .audioAndTranscript {
                  Text("Words spoken in the sample").font(.headline)
                  TextEditor(text: reference(\.transcript,"")).frame(minHeight: 90).border(Color.secondary.opacity(0.3))
                } else { Text("Identity-only mode can be less faithful than audio with a transcript.").font(.caption).foregroundStyle(.secondary) }
                Button("Save reference preset") { store.saveVoicePreset() }
              }
            }
            }
            DisclosureGroup("Generation settings") {
              VStack(alignment: .leading) {
                if draft.engine == .qwen3TTS {
                  Picker("Language", selection: field(\.language)) {
                    ForEach(["auto","english","chinese","japanese","korean","french","german","italian","portuguese","spanish","russian"], id: \.self) { Text($0.capitalized).tag($0) }
                  }
                }
                TextField("Seed", value: field(\.seed), format: .number)
                TextField("Temperature", value: field(\.sampling.temperature), format: .number)
                TextField("Top P", value: field(\.sampling.topP), format: .number)
                TextField("Top K", value: field(\.sampling.topK), format: .number)
                TextField("Maximum audio tokens", value: field(\.sampling.maxTokens), format: .number)
                Text("The model chooses speaking duration. Placement trims the visible region; the full take is retained.").font(.caption).foregroundStyle(.secondary)
              }
            }
          }.padding().textFieldStyle(.roundedBorder)
        }.frame(minWidth: 360, idealWidth: 430, maxWidth: 520)
        VStack(alignment: .leading, spacing: 14) {
          Picker("Script", selection: Binding(get: { draft.usesDialogue == true }, set: { enabled in
            store.change { if enabled { $0.voiceDraft?.startDialogue() } else { $0.voiceDraft?.usesDialogue = false } }
          })) {
            Text("Single voice").tag(false); Text("Conversation").tag(true)
          }.pickerStyle(.segmented)
          if draft.usesDialogue == true { VoiceDialogueEditor().id(draft.id) }
          else {
            Text("Script to speak").font(.headline)
            VoiceScriptEditor(text: field(\.text), engine: draft.engine, contextID: draft.id.uuidString, customVoice: store.selectedVoiceModel?.supportsInstructions == true).id(draft.id)
          }
          Divider()
          Text("Voice takes").font(.headline)
          ScrollView {
            ForEach(store.voiceTakes.reversed()) { asset in
              HStack {
                VStack(alignment: .leading) {
                  Text(asset.name); Text("\(asset.duration, specifier: "%.2f") seconds\(asset.voiceGeneration?.truncated == true ? " · token limit reached" : "")").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Play") { store.auditionMusic(asset) }
                Button("Add to clip") { store.placeVoiceTake(asset) }.disabled(store.selectedClip == nil)
              }.padding(8).background(Theme.raised, in: RoundedRectangle(cornerRadius: 6))
            }
          }
          Button("Stop audition") { store.musicPlayer.pause() }
        }.padding().frame(minWidth: 420)
      }
      Divider()
      HStack {
        if store.bridge.busy { ProgressView().controlSize(.small); Text(store.bridge.message).font(.caption); Button("Cancel") { store.bridge.cancel() } }
        else { Text("Speech inference runs locally in WeeTodd.").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        Button("Generate voice") { Task { await store.generateVoice() } }.buttonStyle(.borderedProminent).disabled(store.operationBusy || store.selectedVoiceModel.map { !store.voiceModelAvailable($0) } != false)
      }.padding()
    }.background(Theme.background).onDisappear { store.musicPlayer.pause() }
  }
}
