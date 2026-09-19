import AppKit
import AVFoundation
import Foundation
import StudioCore
import UniformTypeIdentifiers

@MainActor extension StudioStore {
  var musicTakes: [MediaAsset] { project.assets.filter { $0.musicGeneration != nil } }
  var selectedMusicTake: MediaAsset? { musicTakes.first { $0.id == selectedMusicAssetID } }

  func openMusic() {
    if project.musicDraft == nil { change { $0.musicDraft = MusicDraft() } }
    showMusic = true
  }
  func chooseMusicFolder(decoder: Bool = false) {
    let panel = NSOpenPanel()
    panel.title = decoder ? "Choose YuE2 VAE folder" : "Choose YuE2 MLX model folder"
    panel.canChooseDirectories = true; panel.canChooseFiles = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    change {
      if decoder { $0.musicDraft?.vaePath = url.path }
      else { $0.musicDraft?.modelPath = url.path }
    }
    musicModelStatus = nil
  }
  func loadMusicScore() {
    let panel = NSOpenPanel(); panel.title = "Import ABC score"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do {
      let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
      guard size <= 1024 * 1024 else { throw StudioError.invalid("Choose an ABC score smaller than 1 MiB.") }
      let score = try String(contentsOf: url, encoding: .utf8)
      change { $0.musicDraft?.abc = score; if $0.musicDraft?.cot == "off" { $0.musicDraft?.cot = "full" } }
    } catch { self.error = error.localizedDescription }
  }
  func downloadMusicModel() {
    guard !operationBusy else { return }
    let panel = NSOpenPanel(); panel.title = "Choose a library for YuE2 (4.53 GB · CC BY-NC 4.0)"
    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let folder = panel.url else { return }
    Task { await downloadMusicModel(to: folder) }
  }
  func downloadMusicModel(to folder: URL) async {
    guard !operationBusy, let draft = project.musicDraft else { return }
    let session = documentSessionID, projectID = project.id
      do {
        let result = try await bridge.invoke("music-download", runtime: runtime, payload: ["directory": folder.path])
        guard let path = result["model_path"] as? String else { throw StudioError.invalid("No music model was prepared.") }
        guard documentSessionID == session, project.id == projectID,
          project.musicDraft?.id == draft.id, project.musicDraft?.modelPath == draft.modelPath,
          project.musicDraft?.precision == draft.precision else {
          notice = "YuE2 model downloaded to \(path)."; return
        }
        change { $0.musicDraft?.modelPath = path; $0.musicDraft?.precision = "auto" }
        musicModelStatus = "Verified 8-bit YuE2 model ready."
      } catch { self.error = error.localizedDescription }
  }
  func inspectMusicModel() async {
    guard !operationBusy, let draft = project.musicDraft else { return }
    do {
      let request = try draft.request()
      let result = try await bridge.invoke("music-inspect", runtime: runtime,
        payload: ["music": try request.object()])
      guard project.musicDraft == draft else { return }
      musicModelStatus = result["description"] as? String ?? "Model files validated. Ready for native YuE2 generation."
    } catch { self.error = error.localizedDescription; musicModelStatus = nil }
  }
  func planMusic() async {
    guard !operationBusy, let draft = project.musicDraft else { return }
    let session = documentSessionID, projectID = project.id
    do {
      var request = try draft.request(); request.abc = nil
      let job = dataDirectory.appendingPathComponent("Music/\(UUID().uuidString)")
      let result = try await bridge.invoke("music-plan", runtime: runtime,
        payload: ["music": try request.object()], output: job)
      guard let abc = result["abc"] as? String, !abc.isEmpty else {
        throw StudioError.invalid("No score was produced. Choose Melody or Melody + chords composition.")
      }
      guard documentSessionID == session, project.id == projectID, project.musicDraft == draft else {
        notice = "Score saved at \(job.path). The music draft changed."; return
      }
      change { $0.musicDraft?.abc = abc }
      notice = "Score composed. Review or edit it, then Generate music to use this score."
    } catch { self.error = error.localizedDescription }
  }
  func generateMusic(reusing take: MediaAsset? = nil, decodeOnly: Bool = false) async {
    guard !operationBusy, let currentDraft = project.musicDraft else { return }
    var draft = take?.musicGeneration?.draft ?? currentDraft
    if take != nil && !decodeOnly { draft.steps = currentDraft.steps; draft.seed = currentDraft.seed }
    let projectID = project.id, session = documentSessionID
    let job = dataDirectory.appendingPathComponent("Music/\(UUID().uuidString)")
    do {
      let request = try draft.request()
      let command = take == nil ? "music-generate" : (decodeOnly ? "music-decode" : "music-resynthesize")
      var body: [String: Any] = ["music": try request.object()]
      if let generation = take?.musicGeneration {
        body["source_artifacts"] = generation.artifacts
        if !decodeOnly { body["steps"] = draft.steps; body["seed"] = draft.seed }
      }
      let result = try await bridge.invoke(command, runtime: runtime, payload: body, output: job)
      guard let audio = result["audio"] as? String, !audio.isEmpty,
        let duration = result["duration"] as? Double, duration.isFinite, duration > 0,
        result["sample_rate"] as? Int == 48000, result["channels"] as? Int == 2,
        let artifacts = result["artifacts"] as? String else {
        throw StudioError.invalid("YuE2 did not return a verified 48 kHz stereo take. See the job log at \(job.path).")
      }
      guard project.id == projectID, documentSessionID == session else {
        notice = "Music take saved at \(audio). The destination movie changed."
        return
      }
      let flags = result["truncated"] as? [String: Bool] ?? [:]
      var asset = MediaAsset(name: draft.name, kind: .audio, path: audio)
      asset.duration = duration
      let effective = try (result["request"] as? [String: Any]).map {
        try JSONDecoder().decode(MusicRequest.self, from: JSONSerialization.data(withJSONObject: $0))
      } ?? request
      asset.musicGeneration = MusicGeneration(draft: draft, request: effective, artifacts: artifacts,
        truncated: flags.values.contains(true), timings: result["timings"] as? [String: Double] ?? [:])
      change { $0.assets.append(asset) }
      if project.musicDraft == currentDraft { selectedMusicAssetID = asset.id; auditionMusic(asset, play: false) }
      notice = flags.values.contains(true)
        ? "Take saved, but a token limit was reached. Audition it or increase the generation budget."
        : "Music take saved. Audition it, then add it to the Music track or use it to drive a video clip."
    } catch { self.error = error.localizedDescription }
  }
  func auditionMusic(_ asset: MediaAsset, play: Bool = true) {
    player.pause(); isPlaying = false
    selectedMusicAssetID = asset.id
    musicPlayer.pause()
    musicPlayer.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: asset.path)))
    if play { musicPlayer.play() }
  }
  func placeMusicTake(_ asset: MediaAsset) {
    do {
      var updated = project
      let region = try updated.placeMusic(asset, at: max(0, playhead), trackID: selectedTrackID)
      change { $0 = updated }
      selectedAudioID = region.id; selectedTitleID = nil
      notice = "Added \(asset.name) to the Music track at \(region.start.formatted()) seconds."
    } catch { self.error = error.localizedDescription }
  }
  func useMusicDriver(_ asset: MediaAsset, sourceIn: Double, muteClipAudio: Bool) async {
    guard !operationBusy, let clip = selectedClip else { return }
    guard [.h3, .ltx23, .ltx25].contains(clip.engine),
      profiles.contains(where: { $0.engine == clip.engine.rawValue && $0.generation?.supportedTasks.contains("a2v") == true }) else {
      error = "Set up an audio-driven video recipe for the selected native model first."
      return
    }
    guard sourceIn.isFinite, sourceIn >= 0, clip.duration.isFinite, clip.duration > 0,
      sourceIn + clip.duration <= asset.duration else {
      error = "Choose a song interval long enough for the selected video clip."
      return
    }
    if clip.engine == .ltx25 {
      guard let index = project.clips.firstIndex(where: { $0.id == clip.id }) else { return }
      change { project in
        if !project.assets.contains(where: { $0.id == asset.id }) { project.assets.append(asset) }
        var driver = Attachment(assetID: asset.id, role: .audioDriver)
        driver.audioSourceStart = sourceIn; driver.audioSourceDuration = clip.duration
        project.clips[index].attachments.removeAll { $0.role == .audioDriver }
        project.clips[index].attachments.append(driver)
        project.clips[index].selectGenerationTask("a2v"); project.clips[index].profileID = "auto"
        if muteClipAudio { project.clips[index].volume = 0 }
      }
      notice = "Attached the original song interval. Consecutive intervals can drive one continuous LTX 2.5 scene."
      return
    }
    let projectID = project.id, session = documentSessionID
    let key = clip.generationFingerprint
    let start = project.clips.firstIndex(where: { $0.id == clip.id }).map { project.start(of: $0) } ?? 0
    do {
      let result = try await bridge.invoke("music-excerpt", runtime: runtime,
        payload: ["path": asset.path, "source_in": sourceIn, "duration": clip.duration,
                  "video_request": try payload()],
        output: dataDirectory.appendingPathComponent("Music/Excerpts"))
      guard let path = result["path"] as? String, let duration = result["duration"] as? Double,
        duration.isFinite, duration > 0 else { throw StudioError.invalid("No valid music excerpt was produced.") }
      guard project.id == projectID, documentSessionID == session,
        let index = project.clips.firstIndex(where: { $0.id == clip.id }),
        project.clips[index].generationFingerprint == key, project.start(of: index) == start else {
        notice = "Music excerpt saved at \(path). The target clip changed; attach it again."
        return
      }
      var excerpt = MediaAsset(name: "\(asset.name) · \(sourceIn.formatted())s", kind: .audio,
        path: path, scope: .clip, owner: clip.id)
      excerpt.duration = duration
      change { project in
        project.assets.append(excerpt)
        project.clips[index].attachments.removeAll { $0.role == .audioDriver }
        project.clips[index].attachments.append(Attachment(assetID: excerpt.id, role: .audioDriver))
        project.clips[index].selectGenerationTask("a2v")
        project.clips[index].profileID = "auto"
        if muteClipAudio { project.clips[index].volume = 0 }
      }
      notice = muteClipAudio
        ? "Song excerpt attached. Clip audio is muted; place the matching song region on the Music track."
        : "Song excerpt attached as the audio driver. Clip audio remains enabled."
    } catch { self.error = error.localizedDescription }
  }
  func exportMusicRequest() {
    guard let draft = project.musicDraft else { return }
    do {
      let request = try draft.request()
      let panel = NSSavePanel(); panel.nameFieldStringValue = "music-job.json"
      panel.allowedContentTypes = [.json]
      guard panel.runModal() == .OK, let url = panel.url else { return }
      try JSONSerialization.data(withJSONObject: ["music": try request.object()], options: [.prettyPrinted, .sortedKeys])
        .write(to: url, options: .atomic)
      notice = "Saved the exact native music request for headless generation."
    } catch { self.error = error.localizedDescription }
  }
}
