import AppKit
import StudioCore
import UniformTypeIdentifiers

@MainActor extension StudioStore {
  func finishAudioPlacement(_ region: AudioRegion) {
    if project.audioPresentation == nil { change { $0.audioPresentation = AudioPresentation() } }
    musicPlayer.pause(); showVoice = false; showMusic = false
    selectedAudioID = region.id; selectedTrackID = region.trackID; selectedTitleID = nil
    seek(region.start)
    notice = "Added the full audio region to the timeline. Select it to adjust timing and effects."
  }
  func placeAudioTake(_ asset: MediaAsset, role: AudioTrackRole) {
    do {
      var updated = project
      let region = try updated.placeTimelineAudio(asset, role: role, at: project.playbackDuration > 0 ? playhead : 0)
      if updated.audioPresentation == nil {
        var settings = AudioPresentation(); settings.style = role == .voice ? "waveform" : "spectrum"
        settings.source = role == .voice ? "voice" : "music"; updated.audioPresentation = settings
      }
      change { $0 = updated }; finishAudioPlacement(region)
    } catch { self.error = error.localizedDescription }
  }
  func addSong() {
    let panel = NSOpenPanel(); panel.title = "Add song to timeline"; panel.allowedContentTypes = [.audio]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await addSong(from: url) }
  }
  func addSong(from url: URL) async {
    let session = documentSessionID, id = project.id, time = playhead
    do {
      guard let info = try await NativeAssetInspection.inspectAVMedia(url),
        info["kind"] as? String == "audio" else {
        throw StudioError.invalid("Choose a supported local audio file.")
      }
      guard session == documentSessionID, project.id == id else { return }
      guard let duration = info["duration"] as? Double, duration.isFinite, duration > 0 else { throw StudioError.invalid("Choose a playable audio file") }
      var asset = MediaAsset(name: url.deletingPathExtension().lastPathComponent, kind: .audio, path: url.path)
      asset.duration = duration
      seek(time); placeAudioTake(asset, role: .music)
    } catch { self.error = error.localizedDescription }
  }
  func exportTimelineAudio() {
    let panel = NSSavePanel(); panel.nameFieldStringValue = project.name + ".wav"
    panel.allowedContentTypes = [.wav]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let body = try? payload()
    Task {
      do {
        guard let body else { throw StudioError.invalid("Cannot encode this audio timeline") }
        _ = try await bridge.invoke("audio-export", runtime: runtime, payload: body, output: url)
        notice = "Exported full timeline audio to \(url.lastPathComponent)"
      } catch { self.error = error.localizedDescription }
    }
  }
  func chooseCaptionModel(download: Bool) {
    let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
    panel.canCreateDirectories = download
    panel.title = download ? "Choose a library for local caption models" : "Choose WeeTodd-Analysis or wav2vec2-base-960h folder"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    if !download { runtime.captionModelDirectory = url.path; saveRuntime(reloadProfiles: false); return }
    Task {
      do {
        let result = try await bridge.invoke("music-analysis-setup", runtime: runtime, payload: ["directory": url.path])
        guard let folder = result["model_directory"] as? String else { throw StudioError.invalid("Model setup did not return its folder") }
        runtime.captionModelDirectory = folder; saveRuntime(reloadProfiles: false)
      } catch { self.error = error.localizedDescription }
    }
  }
  func generateCaptions(regionID: UUID, text: String) async {
    guard !operationBusy, let region = project.audio.first(where: { $0.id == regionID }), region.instrumental != true else { return }
    let session = documentSessionID, id = project.id
    do {
      let result = try await bridge.invoke("audio-captions", runtime: runtime,
        payload: ["path": region.path, "text": text.replacingOccurrences(of: "\\[[^\\]]*\\]", with: "", options: .regularExpression)], output: dataDirectory.appendingPathComponent("Captions/\(UUID().uuidString)"))
      guard session == documentSessionID, project.id == id,
        let index = project.audio.firstIndex(where: { $0.id == regionID }), project.audio[index] == region else {
        notice = "The audio changed during caption generation. Generate captions again."; return
      }
      struct Cue: Decodable { var start: Double; var end: Double; var text: String }
      let data = try JSONSerialization.data(withJSONObject: result["cues"] ?? [])
      let cues = try JSONDecoder().decode([Cue].self,from:data).map { AudioCaption(start:$0.start,end:$0.end,text:$0.text) }
      change { p in
        p.audio[index].captions = cues
        if p.audioPresentation == nil { p.audioPresentation = AudioPresentation() }
        if p.audioTracks.first(where: { $0.id == region.trackID })?.role == .voice { p.audioPresentation?.voiceCaptions = true }
        else { p.audioPresentation?.lyrics = true }
      }
      notice = cues.isEmpty ? "No timed words detected. Instrumental audio does not need lyrics." : "Captions generated locally. Review the words and timing, especially sung lyrics."
    } catch { self.error = error.localizedDescription }
  }
}
