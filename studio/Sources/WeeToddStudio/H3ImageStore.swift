import Foundation
import StudioCore

struct PreparedH3Image {
  let draft: DrawThingsImageDraft
  let sessionID: UUID
  let requestID: UUID
  let seed: UInt32
  let recipe: Data
  let directory: URL
  let fingerprint: String
}

@MainActor extension StudioStore {
  func prepareH3ImageGeneration() async {
    guard let draft = imageDraft, draft.executionProvider == .nativeH3,
      !operationBusy, !preparingImageRequest else { return }
    preparingImageRequest = true
    defer { preparingImageRequest = false }
    let referenceKey = beginImageAttempt(draft), session = documentSessionID, requestID = UUID()
    do {
      guard draft.seed == -1 || (0...Int(UInt32.max)).contains(draft.seed) else {
        throw StudioError.invalid("Use a UInt32 seed or −1 for a random image seed.")
      }
      let seed = draft.seed == -1 ? UInt32.random(in: .min ... .max) : UInt32(draft.seed)
      let directory = dataDirectory.appendingPathComponent("Cache/H3 Images/" + requestID.uuidString)
      let recipe = try await Task.detached(priority: .userInitiated) {
        let bytes = try NativeH3ImagePreparation.recipe(draft: draft, requestID: requestID, seed: seed)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: directory.appendingPathComponent("recipe.json"), options: .withoutOverwriting)
        return bytes
      }.value
      guard documentSessionID == session, imageDraft == draft else { return }
      let result = try await bridge.invoke("h3-native-preflight", runtime: runtime,
        payload: ["recipePath": directory.appendingPathComponent("recipe.json").path],
        output: directory.appendingPathComponent("preflight"))
      guard documentSessionID == session, imageDraft == draft else { return }
      try NativeH3ImageResult.validateIdentity(result: result, fingerprint: nil,
        requestID: requestID, seed: seed, width: draft.width, height: draft.height)
      guard result["eligibility"] as? String == "allowed", let fingerprint = result["fingerprint"] as? String else {
        throw StudioError.invalid("H3 image settings were not admitted.")
      }
      preparedH3Image = PreparedH3Image(draft: draft, sessionID: session, requestID: requestID,
        seed: seed, recipe: recipe, directory: directory, fingerprint: fingerprint)
      imageEstimate = result
    } catch { preparedH3Image = nil; imageEstimate = nil; recordImageFailure(error.localizedDescription, referenceKey: referenceKey) }
  }

  func generateH3ImageAsset() async {
    guard let draft = imageDraft, draft.executionProvider == .nativeH3,
      !operationBusy, !preparingImageRequest else { return }
    // A finished random-seed attempt is checked again on the next click, so
    // each generation gets a new seed while the authored draft retains −1.
    if preparedH3Image?.draft != draft || preparedH3Image?.sessionID != documentSessionID {
      await prepareH3ImageGeneration()
    }
    guard let prepared = preparedH3Image, prepared.draft == draft,
      prepared.sessionID == documentSessionID, imageEstimate?["eligibility"] as? String == "allowed" else { return }
    preparingImageRequest = true
    defer { preparingImageRequest = false }
    let referenceKey = beginImageAttempt(draft)
    do {
      var root = try JSONSerialization.jsonObject(with: prepared.recipe) as! [String: Any]
      var fields = root["still_image"] as! [String: Any]
      fields["expected_fingerprint"] = prepared.fingerprint; root["still_image"] = fields
      let request = prepared.directory.appendingPathComponent("render.json")
      try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys]).write(to: request, options: .atomic)
      let output = dataDirectory.appendingPathComponent("Generated Images/" + UUID().uuidString)
      try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
      let result = try await bridge.invoke("h3-native-render", runtime: runtime,
        payload: ["recipePath": request.path], output: output)
      let bytes = try JSONSerialization.data(withJSONObject: result)
      let image = try await Task.detached(priority: .userInitiated) {
        let value = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
        return try NativeH3ImageResult.image(result: value, output: output,
          fingerprint: prepared.fingerprint, requestID: prepared.requestID, seed: prepared.seed,
          width: draft.width, height: draft.height)
      }.value
      preparedH3Image = nil
      guard documentSessionID == prepared.sessionID else {
        notice = "H3 image saved at \(image.path). The destination movie changed."; return
      }
      var asset = try draft.destination.asset(name: draft.name, path: image.path, in: project)
      asset.width = draft.width; asset.height = draft.height
      var provenance = ImageGeneration(provider: ImageExecutionProvider.nativeH3.rawValue,
        requestFingerprint: prepared.fingerprint, modelID: draft.modelID, prompt: draft.prompt)
      provenance.nativeRequest = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject: result["normalizedRequest"]!))
      provenance.configuration = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject: (result["normalizedRequest"] as! [String: Any])["configuration"]!))
      let paths = Set(draft.activeImageInputs.map { $0.input.path })
      provenance.inputIDs = allAssets.filter { paths.contains($0.path) }.map { $0.id.uuidString }
      provenance.referenceSheet = draft.referenceSheet; provenance.rippleReference = draft.rippleReference
      provenance.generatedAt = Date(); asset.generation = provenance
      if asset.scope == .global { globalAssets.append(asset); saveGlobals() }
      else { change { $0.assets.append(asset) } }
      selectedAssetID = asset.id
      if imageDraft == draft { imagePreviewPath = image.path }
      notice = "Generated \(draft.name) with H3 · seed \(prepared.seed)."
    } catch {
      preparedH3Image = nil; imageEstimate = nil
      recordImageFailure(error.localizedDescription, referenceKey: referenceKey)
    }
  }
}
