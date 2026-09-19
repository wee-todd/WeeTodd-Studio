import AVFoundation
import StudioCore

@MainActor extension StudioStore {
  func prepareAudioDriver(muteSource: Bool? = nil) async -> Bool {
    guard !operationBusy, let clip = selectedClip, let selection = clip.audioDriverSelection else { return false }
    guard [.h3, .ltx23, .ltx25].contains(clip.engine), clip.musicSource == nil,
      !project.isContinuousSceneMember(clip) else {
      error = "Timeline audio drivers require an independent native H3 or LTX clip. Keep a planned song scene's shared source, or separate this clip first."
      return false
    }
    let session = documentSessionID, projectID = project.id, revision = project.audioDriverRevision(for: clip)
    let generation = clip.generationFingerprint
    do {
      var body = try payload(); body["selection"] = try selection.object()
      let result = try await bridge.invoke("audio-driver", runtime: runtime, payload: body, output: dataDirectory.appendingPathComponent("AudioMix"))
      guard let path = result["path"] as? String, let key = result["mix_key"] as? String,
        let duration = result["duration"] as? Double else { throw StudioError.invalid("No audio driver was prepared.") }
      guard documentSessionID == session, project.id == projectID,
        let index = project.clips.firstIndex(where: { $0.id == clip.id }),
        project.clips[index].generationFingerprint == generation,
        project.audioDriverRevision(for: project.clips[index]) == revision else {
        notice = "Audio driver saved at \(path). The clip or its audio changed; prepare it again."
        return false
      }
      var asset = MediaAsset(name: "\(clip.name) · \(selection.mode.rawValue) driver", kind: .audio, path: path, scope: .clip, owner: clip.id)
      asset.duration = duration
      change { p in
        p.assets.append(asset)
        p.clips[index].attachments.removeAll { $0.role == .audioDriver }
        p.clips[index].attachments.append(Attachment(assetID: asset.id, role: .audioDriver))
        p.clips[index].audioDriverMixKey = key
        p.clips[index].selectGenerationTask("a2v"); p.clips[index].profileID = "auto"
        if let muteSource { p.clips[index].volume = muteSource ? 0 : 1 }
      }
      notice = "Exact audio driver prepared. Audition it before generating video."
      return true
    } catch { self.error = error.localizedDescription; return false }
  }
  func auditionAudioDriver() {
    guard let clip = selectedClip, clip.audioDriverMixKey != nil,
      let attachment = clip.attachments.first(where: { $0.role == .audioDriver }),
      let asset = project.assets.first(where: { $0.id == attachment.assetID }) else { return }
    auditionMusic(asset)
  }
}

@MainActor extension StudioStore {
  func crossfadeAudio(_ id: UUID) {
    guard let a = project.audio.first(where: { $0.id == id }) else { return }
    let start = (try? resolvedAudioStart(a, in: project)) ?? a.start
    guard let b = project.resolvedAudio.filter({ $0.id != id && $0.trackID == a.trackID && $0.start > start }).min(by: { $0.start < $1.start }),
      let ai = project.audio.firstIndex(where: { $0.id == id }), let bi = project.audio.firstIndex(where: { $0.id == b.id }) else {
      error = "Place another region on this track, then overlap their edges to crossfade."; return
    }
    let overlap = start + a.duration - b.start
    guard overlap > 0, overlap <= min(a.duration, b.duration) else {
      error = "Overlap the two audio regions first; their shared range becomes the crossfade."; return
    }
    change { p in
      p.audio[ai].envelope = nil; p.audio[bi].envelope = nil
      p.audio[ai].fadeOut = overlap; p.audio[bi].fadeIn = overlap
      p.audio[ai].fadeCurve = "equalPower"; p.audio[bi].fadeCurve = "equalPower"
    }
  }
}
