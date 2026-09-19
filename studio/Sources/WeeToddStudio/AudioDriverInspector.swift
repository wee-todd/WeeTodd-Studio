import StudioCore
import SwiftUI

struct AudioDriverInspector: View {
  @EnvironmentObject var store: StudioStore
  @State private var muteSource = true
  var body: some View {
    if let clip = store.selectedClip, [.h3, .ltx23, .ltx25].contains(clip.engine) {
      DisclosureGroup("Timeline audio driver") {
        VStack(alignment: .leading, spacing: 10) {
          Picker("Use", selection: Binding(get: { clip.audioDriverSelection?.mode.rawValue ?? "attachment" }, set: { value in
            store.editClip { $0.audioDriverSelection = AudioDriverMode(rawValue: value).map { AudioDriverSelection(mode: $0) }; $0.audioDriverMixKey = nil }
          })) {
            Text("Existing audio attachment").tag("attachment")
            Text("Voice").tag("voice"); Text("Music").tag("music"); Text("Voice + Music").tag("voiceAndMusic")
          }
          if let selection = clip.audioDriverSelection {
            ForEach([AudioTrackRole.voice, .music], id: \.self) { role in
              if selection.mode == .voiceAndMusic || selection.mode.rawValue == role.rawValue {
                Menu("\(role.rawValue.capitalized) tracks") {
                  Button("All \(role.rawValue) tracks") {
                    store.editClip { if role == .voice { $0.audioDriverSelection?.voiceTrackIDs = [] } else { $0.audioDriverSelection?.musicTrackIDs = [] } }
                  }
                  ForEach(store.project.audioTracks.filter { $0.role == role }) { track in
                    Button(track.name) {
                      store.editClip { if role == .voice { $0.audioDriverSelection?.voiceTrackIDs = [track.id] } else { $0.audioDriverSelection?.musicTrackIDs = [track.id] } }
                    }
                  }
                }
              }
            }
            Toggle("Use timeline soundtrack (mute clip audio)", isOn: $muteSource)
            Text("Track gains, pan, fades and mutes are included. Solo affects preview only.").font(.caption).foregroundStyle(.secondary)
            HStack {
              Button("Prepare driver") { Task { _ = await store.prepareAudioDriver(muteSource: muteSource) } }.disabled(store.operationBusy)
              Button("Audition") { store.auditionAudioDriver() }.disabled(clip.audioDriverMixKey == nil)
            }
            if clip.audioDriverMixKey == nil { Text("Prepare the current mix before video generation.").font(.caption).foregroundStyle(.secondary) }
          }
        }.padding(.vertical, 8)
      }
    }
  }
}
