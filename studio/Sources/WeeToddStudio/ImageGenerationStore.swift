import AppKit
import Foundation
import StudioCore

@MainActor extension StudioStore {
  var imagePreflightIssues: [String] {
    if let messages = imageEstimate?["issues"] as? [String] { return messages }
    return (imageEstimate?["issues"] as? [[String: Any]] ?? []).map {
      $0["message"] as? String ?? "Generation settings need attention."
    }
  }

  func prepareNativeImageModel() async {
    guard !operationBusy, !preparingImageRequest else { return }
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
    panel.prompt = "Prepare here"; panel.message = "Choose model storage. First-time setup needs about 60 GiB; existing verified files are reused. Qwen’s research license applies."
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let session = documentSessionID, destination = imageDraft?.storageKey
    do {
      let result = try await bridge.invoke("image-model-prepare", runtime: runtime, payload: ["directory": url.path])
      guard let path = result["manifestPath"] as? String else { throw StudioError.invalid("Model preparation returned no manifest.") }
      guard documentSessionID == session, imageDraft?.storageKey == destination else {
        notice = "Qwen model prepared at \(path)."; return
      }
      if imageDraft?.nativeImage == nil { imageDraft?.nativeImage = NativeImageSettings() }
      imageDraft?.nativeImage?.manifestPath = path; imageEstimate = nil
      notice = "Qwen is ready for local image generation."
    } catch { self.error = error.localizedDescription }
  }
  func imagePayload(_ draft: DrawThingsImageDraft, connection: DrawThingsConnection?) async throws -> [String: Any] {
    // Hash captured files away from MainActor. Return encoded data across the concurrency boundary.
    if draft.executionProvider == .nativeMLX {
      let data = try await Task.detached(priority: .userInitiated) {
        try JSONSerialization.data(withJSONObject: draft.nativeRequest(id: UUID().uuidString))
      }.value
      return ["nativeImageRequest": try JSONSerialization.jsonObject(with: data)]
    }
    guard let connection else { throw StudioError.invalid("Choose a Draw Things connection.") }
    let data = try await Task.detached(priority: .userInitiated) {
      try JSONSerialization.data(withJSONObject: draft.request(id: UUID().uuidString))
    }.value
    return ["connection": try connection.object(), "drawThingsRequest": try JSONSerialization.jsonObject(with: data)]
  }
  func prepareImageGeneration() async {
    guard let draft = imageDraft, !operationBusy, !preparingImageRequest else { return }
    preparingImageRequest = true
    defer { preparingImageRequest = false }
    let connection = drawThingsConnections.first(where: { $0.id == draft.profileID })
    let referenceKey = beginImageAttempt(draft)
    do {
      let session = documentSessionID
      let payload = try await imagePayload(draft, connection: connection)
      guard documentSessionID == session, imageDraft == draft else { return }
      let result = try await bridge.invoke(draft.executionProvider == .nativeMLX ? "image-preflight" : "dt-estimate",
        runtime: runtime, payload: payload)
      guard documentSessionID == session, imageDraft == draft else { return }
      imageEstimate = result
    } catch { recordImageFailure(error.localizedDescription, referenceKey: referenceKey) }
  }
  func generateImageAsset() async {
    guard let draft = imageDraft, !operationBusy, !preparingImageRequest else { return }
    preparingImageRequest = true
    defer { preparingImageRequest = false }
    let connection = drawThingsConnections.first(where: { $0.id == draft.profileID })
    let documentSession = documentSessionID
    let inputPaths = Set(([draft.canvas].compactMap { $0 }.filter { $0.enabled }
      + draft.moodboard.filter { $0.enabled && $0.strength > 0 }).map { $0.path })
    let inputAssetIDs = allAssets.filter { inputPaths.contains($0.path) }.map { $0.id.uuidString }
    let referenceKey = beginImageAttempt(draft)
    do {
      let output = Self.supportDirectory.appendingPathComponent("Generated Images/\(UUID().uuidString)")
      try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
      var payload = try await imagePayload(draft, connection: connection)
      guard documentSessionID == documentSession, imageDraft == draft else { return }
      if draft.executionProvider == .nativeMLX { payload["expectedFingerprint"] = imageEstimate?["fingerprint"] }
      payload["name"] = draft.name; payload["scope"] = draft.destination.scope.rawValue
      payload["project"] = try project.object()
      if let owner = draft.destination.owner { payload["owner"] = owner.uuidString }
      let result = try await bridge.invoke(draft.executionProvider == .nativeMLX ? "image-generate" : "dt-generate-image", runtime: runtime, payload: payload, output: output)
      guard let value = result["asset"] as? [String: Any], let path = value["path"] as? String else {
        throw StudioError.invalid("The image engine did not return a saved image.")
      }
      guard documentSession == documentSessionID else {
        notice = "Generated image saved at \(path). The destination movie changed."; return
      }
      var asset = try draft.destination.asset(name: draft.name, path: path, in: project)
      asset.width = value["width"] as? Int ?? draft.width
      asset.height = value["height"] as? Int ?? draft.height
      var provenance = ImageGeneration(provider: draft.executionProvider.rawValue, requestFingerprint: result["fingerprint"] as? String ?? "",
        modelID: draft.modelID, prompt: draft.prompt)
      provenance.profileID = draft.executionProvider == .drawThings ? draft.profileID : nil
      provenance.negativePrompt = draft.negativePrompt
      if draft.executionProvider == .nativeMLX, let normalized = result["normalizedRequest"] {
        provenance.nativeRequest = try JSONDecoder().decode([String: JSONValue].self,
          from: JSONSerialization.data(withJSONObject: normalized))
      }
      provenance.configuration = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject:
          (result["normalizedRequest"] as? [String: Any])?["configuration"] ?? draft.configuration))
      provenance.inputIDs = inputAssetIDs
      provenance.referenceSheet = draft.referenceSheet
      provenance.rippleReference = draft.rippleReference
      provenance.generatedAt = Date(); asset.generation = provenance
      if asset.scope == .global { globalAssets.append(asset); saveGlobals() }
      else { change { $0.assets.append(asset) } }
      selectedAssetID = asset.id
      if imageDraft == draft { imagePreviewPath = path }
      notice = "Generated \(draft.name) in \(asset.scope.rawValue) assets."
    } catch { recordImageFailure(error.localizedDescription, referenceKey: referenceKey) }
  }
}
