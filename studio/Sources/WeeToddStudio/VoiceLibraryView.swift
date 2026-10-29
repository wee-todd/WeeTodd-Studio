import AVFoundation
import StudioCore
import SwiftUI

struct VoiceLibraryView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) var dismiss
  @State private var name = ""
  @State private var direction = ""
  @State private var editing: VoiceProfileReference?
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Voice Library").font(.title2)
        Spacer()
        Button("Done") { dismiss() }
      }
      Text(
        "Saved voices retain their audio samples and transcripts. Reuse them across movies or assign them to a character."
      ).foregroundStyle(.secondary)
      List(store.voiceLibraryProfiles, id: \.reference) { profile in
        HStack {
          VStack(alignment: .leading) {
            Text(profile.name)
            Text("Version \(profile.reference.version) · \(profile.samples.count) samples").font(
              .caption
            ).foregroundStyle(.secondary)
          }
          if profile.samples.contains(where: { !FileManager.default.fileExists(atPath: $0.path) }) {
            Text("Samples unavailable · restore retained files").foregroundStyle(.orange).font(
              .caption)
          }
          Spacer()
          Button("Audition") {
            if let s = profile.samples.first {
              store.musicPlayer.replaceCurrentItem(
                with: AVPlayerItem(url: URL(fileURLWithPath: s.path)))
              store.musicPlayer.play()
            }
          }
          Button("Edit details") {
            editing = profile.reference
            name = profile.name
            direction = profile.description
          }
          Button("Use voice") {
            store.useSavedVoice(profile)
            dismiss()
          }
        }
      }
      Divider()
      Text(editing == nil ? "Save current cloning samples" : "New version of saved voice").font(
        .headline)
      TextField("Voice name", text: $name)
      TextField("Voice description (optional)", text: $direction)
      Button(
        store.voiceLibraryBusy
          ? "Saving…" : editing == nil ? "Save cloned voice" : "Save new version"
      ) {
        Task {
          if let editing {
            await store.updateSavedVoice(editing, name: name, description: direction)
          } else {
            await store.saveClonedVoice(name: name, description: direction)
          }
        }
      }
      .disabled(
        store.voiceLibraryBusy || store.operationBusy
          || name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      if editing != nil {
        Button("Save a different clone instead") {
          editing = nil
          name = ""
          direction = ""
        }
      }
      Text(
        "The selected sample range and any added samples are copied into managed storage. Your original media can then move without breaking this voice."
      ).font(.caption).foregroundStyle(.secondary)
    }.padding(20).frame(minWidth: 600, minHeight: 440).textFieldStyle(.roundedBorder).task {
      await store.refreshSavedVoices()
    }
  }
}
struct CharacterVoiceControls: View {
  @EnvironmentObject var store: StudioStore
  @Binding var binding: CharacterVoiceBinding?
  var onUseVoice: () -> Void = {}
  @State private var showLibrary = false
  var body: some View {
    GroupBox("Character voice") {
      VStack(alignment: .leading, spacing: 8) {
        Text(binding?.profile.name ?? "No voice assigned").font(.headline)
        if let profile = binding?.profile {
          Text(
            "Saved version \(profile.reference.version) · \(profile.samples.count) retained samples"
          ).font(.caption).foregroundStyle(.secondary)
        }
        HStack {
          Menu("Choose saved voice") {
            ForEach(store.voiceLibraryProfiles, id: \.reference) { profile in
              Button(profile.name) { binding = .init(profile: profile) }
            }
          }
          Button("Voice Library…") { showLibrary = true }
        }
        if let profile = binding?.profile {
          HStack {
            Button("Use voice") {
              Task {
                if await store.retainAndUseCharacterVoice(profile) { onUseVoice() }
              }
            }.disabled(store.operationBusy || store.voiceLibraryBusy)
            Button("Audition") {
              if let sample = profile.samples.first {
                store.musicPlayer.replaceCurrentItem(
                  with: AVPlayerItem(url: URL(fileURLWithPath: sample.path)))
                store.musicPlayer.play()
              }
            }
            Button("Clear assignment") { binding = nil }
          }
        }
        Text(
          "This assignment travels with the character. Visual references and their approvals remain separate."
        ).font(.caption).foregroundStyle(.secondary)
      }.frame(maxWidth: .infinity, alignment: .leading)
    }.task { await store.refreshSavedVoices() }.sheet(isPresented: $showLibrary) {
      VoiceLibraryView().environmentObject(store)
    }
  }
}
