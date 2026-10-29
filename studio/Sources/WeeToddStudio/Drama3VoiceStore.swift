import Drama3Contracts
import Foundation
import StudioCore

@MainActor extension StudioStore {
  func generateDramaVoice() async {
    guard !operationBusy else { return }
    migrateVoiceModelLocation()
    guard let model = selectedVoiceModel, var draft = project.voiceDraft else {
      error = "Choose an installed Drama 3 model in Runtime settings."
      return
    }
    do {
      if let dialogue = draft.dialogue, draft.usesDialogue == true {
        var mapping = draft.dramaSpeakerMap ?? Drama3SpeakerMap()
        _ = try mapping.assign(dialogue.speakers.map(\.id))
        if mapping != draft.dramaSpeakerMap {
          change { $0.voiceDraft?.dramaSpeakerMap = mapping }
          draft = project.voiceDraft!
        }
      }
      let session = documentSessionID
      let projectID = project.id
      let job = dataDirectory.appendingPathComponent("Voice/\(UUID().uuidString)")
      var resolved = draft
      resolved.modelPath = model.path
      resolved.precision = "auto"
      var native = try resolved.drama3Request()
      native.outputDirectory = job.path
      native.cacheDirectory =
        dataDirectory.appendingPathComponent("Voice Cache/ReferenceCodes").path
      if native.mode == .direct || native.mode == .infill, let parentID = draft.dramaParentTakeID {
        guard let parent = voiceTakes.first(where: { $0.id == parentID }),
          let generation = parent.voiceGeneration, generation.request.drama3 != nil
        else { throw StudioError.invalid("Choose a native Drama 3 take for this operation.") }
        let root = URL(fileURLWithPath: generation.artifacts)
        let original = try Drama3TakeBundle.request(at: root)
        let receipt = try await Task.detached { try Drama3TakeBundle.verify(at: root) }.value
        guard let codec = receipt.codecFingerprint else {
          throw StudioError.invalid(
            "This take predates codec identity tracking. Import its audio for a new edit.")
        }
        native.contextCodecFingerprint = codec
        let codes = Drama3Codes(
          path: root.appendingPathComponent(receipt.codes).path, frames: receipt.codeFrames,
          sha256: receipt.codesSHA256)
        if draft.dramaMode == "direct" {
          guard
            !(draft.dramaDirection ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          else { throw StudioError.invalid("Enter a director note for the selected take.") }
          var priorText = original.text
          if !original.direction.isEmpty {
            priorText = "Direction: \(original.direction)\nLine: \(priorText)"
          }
          native.history = original.history + [.init(text: priorText, codes: codes)]
        } else if draft.dramaMode == "infill" {
          native.sourceAudio = nil
          native.sourceCodes = codes
        }
      }
      var request = try resolved.request(resolvedDrama3: native)
      let bytes = try JSONEncoder().encode(native)
      let payload = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
      _ = try await bridge.invoke(
        native.mode == .streamingText ? "drama3-stream" : "drama3-render", runtime: runtime,
        payload: payload, output: job)
      let receipt = try await Task.detached { try Drama3TakeBundle.verify(at: job) }.value
      guard receipt.id == native.id else {
        throw StudioError.invalid("Speech output belongs to another request.")
      }
      let frozen = try Drama3TakeBundle.request(at: job)
      request.drama3 = frozen
      request.text = frozen.text
      let audio = job.appendingPathComponent(receipt.audio).path
      guard project.id == projectID, documentSessionID == session else {
        notice = "Voice take saved at \(audio). The destination movie changed."
        return
      }
      var asset = MediaAsset(name: draft.name, kind: .audio, path: audio)
      asset.duration = receipt.duration
      asset.voiceGeneration = VoiceGeneration(
        draft: draft, request: request, artifacts: job.path, sampleRate: receipt.sampleRate,
        frames: receipt.frames, truncated: receipt.truncated)
      change { $0.assets.append(asset) }
      if project.voiceDraft == draft { selectedVoiceAssetID = asset.id }
      notice =
        receipt.truncated
        ? "Saved a voice take requiring review: "
          + (receipt.warnings?.joined(separator: " ")
            ?? "Generation ended before the script completed.")
        : "Native Drama 3 take saved. Audition it, revise it, or add it to the timeline."
    } catch { self.error = error.localizedDescription }
  }
  func prepareDramaRevision(_ asset: MediaAsset, infill: Bool) {
    guard !operationBusy, let generation = asset.voiceGeneration,
      generation.request.engine == .fishDrama3
    else { return }
    var draft = generation.draft
    if draft.usesDialogue != true, let performed = generation.request.drama3?.text {
      draft.text = performed
    }
    draft.id = UUID()
    draft.name = asset.name + (infill ? " · edit" : " · directed")
    draft.dramaParentTakeID = asset.id
    draft.dramaMode = infill ? "infill" : "direct"
    draft.dramaDirection = ""
    draft.dramaSourceAudio = asset.path
    draft.dramaEditStart = 0
    draft.dramaEditEnd = min(1, asset.duration)
    change { $0.voiceDraft = draft }
    showVoice = true
  }
}
