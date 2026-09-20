import AppKit
import Foundation
import StudioCore
import UniformTypeIdentifiers

extension StudioStore {
  var rippleClip: Clip? { project.clips.first { $0.id == rippleClipID } }

  func openRipple() {
    guard let clip = selectedClip, !clip.sourcePath.isEmpty else {
      error = "Select an imported or generated video clip first."; return
    }
    rippleClipID = clip.id; rippleSelectedTakeID = nil
    rippleInspection = nil; rippleInspectionKey = nil
    if clip.rippleDraft == nil {
      updateRipple { $0 = RippleDraft(clip: clip, frameRate: clip.settings(in: project).fps) }
    }
  }

  func updateRipple(_ edit: (inout RippleDraft) -> Void) {
    guard let index = project.clips.firstIndex(where: { $0.id == rippleClipID }) else { return }
    var draft = project.clips[index].rippleDraft
      ?? RippleDraft(clip: project.clips[index], frameRate: project.clips[index].settings(in: project).fps)
    edit(&draft)
    change { $0.clips[index].rippleDraft = draft }
  }

  func restartRippleFromCurrentClip() {
    guard let clip = rippleClip else { return }
    updateRipple { $0 = RippleDraft(clip: clip, frameRate: clip.settings(in: project).fps) }
    rippleInspection = nil; rippleInspectionKey = nil; rippleSelectedTakeID = nil
  }

  @discardableResult func addRippleReference(frame: Int) throws -> UUID {
    guard let draft = rippleClip?.rippleDraft, frame >= 0, frame < draft.frameCount else {
      throw StudioError.invalid("Choose a frame inside the captured source interval.")
    }
    if let existing = draft.references.first(where: { $0.frame == frame }) { return existing.id }
    guard draft.references.count < 9 else { throw StudioError.invalid("Ripple supports up to nine distinct source frames.") }
    let reference = RippleReference(frame: frame)
    updateRipple { $0.references.append(reference) }
    return reference.id
  }

  func setRippleReferenceFrame(_ referenceID: UUID, frame: Int) throws {
    guard let draft = rippleClip?.rippleDraft,
      let reference = draft.references.first(where: { $0.id == referenceID }) else { return }
    guard frame != reference.frame else { return }
    guard reference.frame != 0, frame > 0, frame < draft.frameCount,
      !draft.references.contains(where: { $0.id != referenceID && $0.frame == frame }) else {
      throw StudioError.invalid("Keep the required first frame at 0 and assign every other reference to a distinct later frame.")
    }
    updateRipple { draft in
      if let index = draft.references.firstIndex(where: { $0.id == referenceID }) {
        draft.references[index].frame = frame
        draft.references[index].originalPath = ""
        draft.references[index].path = ""
      }
    }
  }

  func removeRippleReference(_ referenceID: UUID) {
    updateRipple { draft in
      draft.references.removeAll { $0.id == referenceID && $0.frame != 0 }
    }
  }

  func importRippleImage(referenceID: UUID) {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    updateRipple { draft in
      guard let index = draft.references.firstIndex(where: { $0.id == referenceID }) else { return }
      draft.references[index].path = url.path
    }
  }

  func inspectRipple() async {
    guard !operationBusy, let clip = rippleClip, let draft = clip.rippleDraft else { return }
    let session = documentSessionID
    do {
      let result = try await bridge.invoke("ripple-inspect", runtime: runtime,
        payload: ["ripple": try draft.bridgeObject(requireReferences: false)])
      guard session == documentSessionID, rippleClipID == clip.id,
        rippleClip?.rippleDraft == draft else { return }
      var inspected = draft
      // Once images are assigned, their frame grid is immutable. A new empty draft
      // adopts source cadence before any frame is extracted or edited.
      if draft.sourceInspected != true, draft.references.allSatisfy({ $0.path.isEmpty && $0.originalPath.isEmpty }),
        let fps = result["source_frame_rate"] as? Double, fps.isFinite, (1...60).contains(fps) {
        inspected.sourceInspected = true
        inspected.frameRate = fps
        let source = result["source"] as? [String: Any] ?? result
        if let width = source["width"] as? Int, let height = source["height"] as? Int,
          width > 0, height > 0 {
          let scale = min(1, 1920 / Double(max(width, height)))
          inspected.width = max(32, Int((Double(width) * scale / 32).rounded()) * 32)
          inspected.height = max(32, Int((Double(height) * scale / 32).rounded()) * 32)
        }
        updateRipple { $0 = inspected }
      }
      if let sourceFPS = result["source_frame_rate"] as? Double,
        abs(inspected.frameRate - sourceFPS) > max(1, sourceFPS) * 0.00001 {
        throw StudioError.invalid("The saved Ripple frame rate differs from the source video. Start a new draft from the current clip so frame references remain exact.")
      }
      if let start = result["source_preview_start"] as? Double, start.isFinite, start >= 0 {
        inspected.sourcePreviewStart = start
        updateRipple { $0.sourcePreviewStart = start }
      }
      rippleInspection = result; rippleInspectionKey = inspected
      notice = (result["has_audio"] as? Bool == true)
        ? "Ripple source checked. Original clip audio will be preserved by default."
        : "Ripple source checked. This silent clip is valid."
    } catch { if session == documentSessionID { self.error = error.localizedDescription } }
  }

  func extractRippleFrame(referenceID: UUID) async {
    guard !operationBusy, let clip = rippleClip, let draft = clip.rippleDraft,
      let reference = draft.references.first(where: { $0.id == referenceID }) else { return }
    let session = documentSessionID
    do {
      var request = try draft.bridgeObject(requireReferences: false)
      guard reference.frame >= 0, reference.frame < draft.frameCount else {
        throw StudioError.invalid("Choose a frame inside the source interval.")
      }
      request["frame"] = reference.frame
      let directory = dataDirectory.appendingPathComponent("Ripple/Frames/\(UUID().uuidString)")
      let result = try await bridge.invoke("ripple-frame", runtime: runtime,
        payload: ["ripple": request], output: directory)
      guard let image = result["image_path"] as? String ?? result["path"] as? String, !image.isEmpty else {
        throw StudioError.invalid("Ripple did not return the extracted frame.")
      }
      guard session == documentSessionID, rippleClipID == clip.id, rippleClip?.rippleDraft == draft else {
        notice = "Extracted source frame saved at \(image). The Ripple inputs changed."; return
      }
      updateRipple { draft in
        if let index = draft.references.firstIndex(where: { $0.id == referenceID }) {
          draft.references[index].originalPath = image
        }
      }
    } catch { if session == documentSessionID { self.error = error.localizedDescription } }
  }

  func generateRipple(clipID: UUID? = nil) async {
    let target = clipID.flatMap { id in project.clips.first { $0.id == id } } ?? rippleClip
    guard !operationBusy, let clip = target, let draft = clip.rippleDraft else { return }
    let session = documentSessionID
    let job = dataDirectory.appendingPathComponent("Ripple/Takes/\(UUID().uuidString)")
    do {
      let body = try draft.bridgeObject()
      let result = try await bridge.invoke("ripple-generate", runtime: runtime,
        payload: ["ripple": body], output: job)
      guard let path = result["video_path"] as? String ?? result["path"] as? String,
        !path.isEmpty, path != draft.sourcePath,
        let duration = result["duration"] as? Double, duration.isFinite,
        abs(duration - draft.duration) <= 1 / draft.frameRate + 0.001,
        let returnedFPS = result["frame_rate"] as? Double, returnedFPS.isFinite,
        abs(returnedFPS - draft.frameRate) <= max(1, draft.frameRate) * 0.00001,
        result["frames"] as? Int == draft.frameCount,
        result["width"] as? Int == draft.width, result["height"] as? Int == draft.height,
        let hasAudio = result["has_audio"] as? Bool,
        draft.audioPolicy != .silent || !hasAudio,
        let receipt = result["receipt_path"] as? String, !receipt.isEmpty,
        let artifacts = result["artifacts_directory"] as? String, !artifacts.isEmpty else {
        throw StudioError.invalid("Ripple did not return a verified new take covering the clip interval. Artifacts: \(job.path)")
      }
      guard let frozen = result["frozen_references"] as? [[String: Any]], frozen.count == draft.references.count,
        let sourceHash = result["source_sha256"] as? String, sourceHash.count == 64,
        sourceHash.allSatisfy({ $0.isHexDigit }) else {
        throw StudioError.invalid("Ripple did not return frozen replay references and a source identity. Artifacts: \(job.path)")
      }
      var replayDraft = draft
      replayDraft.sourceSHA256 = sourceHash
      var frozenFrames = Set<Int>()
      for reference in frozen {
        guard let frame = reference["frame"] as? Int, frozenFrames.insert(frame).inserted,
          let imagePath = reference["path"] as? String, !imagePath.isEmpty,
          let strength = reference["strength"] as? Double,
          let index = replayDraft.references.firstIndex(where: { $0.frame == frame }),
          strength == replayDraft.references[index].strength else {
          throw StudioError.invalid("Ripple returned inconsistent frozen reference identities. Artifacts: \(job.path)")
        }
        replayDraft.references[index].path = imagePath
      }
      let take = RippleTake(draft: replayDraft, path: path, receiptPath: receipt,
        artifactsDirectory: artifacts, hasAudio: hasAudio, submittedDraftFingerprint: draft.inputFingerprint)
      guard documentSessionID == session,
        let index = project.clips.firstIndex(where: { $0.id == clip.id }) else {
        notice = "Ripple take saved at \(path). The destination movie changed."; return
      }
      var asset = MediaAsset(name: clip.name + " · Ripple", kind: .video, path: path, scope: .clip, owner: clip.id)
      asset.duration = duration; asset.width = draft.width; asset.height = draft.height; asset.fps = draft.frameRate
      change { project in
        if project.clips[index].rippleTakes == nil { project.clips[index].rippleTakes = [] }
        project.clips[index].rippleTakes!.append(take)
        project.assets.append(asset)
      }
      if rippleClipID == clip.id && project.clips[index].rippleDraft == draft {
        rippleSelectedTakeID = take.id
      }
      notice = "Ripple take saved. Review it, then Apply take to replace the timeline source."
    } catch { if documentSessionID == session { self.error = error.localizedDescription } }
  }

  func canApplyRipple(_ take: RippleTake, to clip: Clip) -> Bool {
    guard let draft = clip.rippleDraft else { return false }
    let sameInputs = draft == take.draft || draft.inputFingerprint == take.submittedDraftFingerprint
    return sameInputs && take.draft.sourceMatches(clip)
      && clip.rippleTakes?.contains(where: { $0.id == take.id }) == true
  }

  func applyRipple(_ take: RippleTake) {
    guard let index = project.clips.firstIndex(where: { $0.id == rippleClipID }),
      canApplyRipple(take, to: project.clips[index]) else {
      error = "The source clip or Ripple inputs changed. Restore this take's inputs before applying it."; return
    }
    change { project in
      // Retain the source interval as its own take before explicitly adopting the output.
      let source = project.clips[index]
      if !source.versions.contains(where: { $0.path == source.sourcePath }) {
        project.clips[index].versions.append(RenderVersion(path: source.sourcePath,
          seed: source.seed, prompt: source.prompt, recipePath: "", usableSourceIn: source.sourceIn,
          usableDuration: source.duration))
      }
      if !source.versions.contains(where: { $0.path == take.path }) {
        project.clips[index].versions.append(RenderVersion(path: take.path, seed: take.draft.seed,
          prompt: take.draft.prompt, recipePath: take.receiptPath, usableSourceIn: 0,
          usableDuration: take.draft.duration))
      }
      _ = project.separateContinuousSceneMember(clipID: source.id)
      project.clips[index].sourcePath = take.path
      project.clips[index].sourceIn = 0
      project.clips[index].duration = take.draft.duration
      project.clips[index].renderedSignature = ""
      project.clips[index].motionResult = nil
      project.clips[index].reviewedTakeFingerprint = project.clips[index].reuseFingerprint
    }
    refreshPreview()
    notice = "Ripple take applied. The original source and trim are retained in the Ripple draft and take history."
  }

  func restoreRippleTakeInputs(_ take: RippleTake) {
    guard let index = project.clips.firstIndex(where: { $0.id == rippleClipID }),
      project.clips[index].rippleTakes?.contains(where: { $0.id == take.id }) == true else { return }
    change {
      $0.clips[index].rippleDraft = take.draft
      $0.clips[index].sourcePath = take.draft.sourcePath
      $0.clips[index].sourceIn = take.draft.sourceIn
      $0.clips[index].duration = take.draft.duration
      $0.clips[index].renderedSignature = ""
      $0.clips[index].motionResult = nil
      $0.clips[index].reviewedTakeFingerprint = $0.clips[index].reuseFingerprint
    }
    rippleInspection = nil; rippleInspectionKey = nil
    refreshPreview()
    notice = "Restored the original source interval and this take’s Ripple inputs."
  }
}
