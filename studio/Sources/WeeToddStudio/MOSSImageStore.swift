import Foundation
import ImageLayout
import StudioCore

struct PreparedMOSSImage {
  let draft: DrawThingsImageDraft
  let sessionID, requestID: UUID
  let seed: UInt32
  let request: Data
  let directory: URL
  let fingerprint: String
}

@MainActor extension StudioStore {
  func reopenMOSSLayout(_ asset: MediaAsset) {
    guard !operationBusy, let generation = asset.generation, generation.provider == ImageExecutionProvider.nativeMOSS.rawValue,
      let layout = generation.mossLayout else { return }
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: project.id))
    draft.selectProvider(.nativeMOSS); draft.name = asset.name + " · layout edit"
    draft.prompt = generation.prompt; draft.mossLayout = layout; draft.width = layout.width; draft.height = layout.height
    draft.modelID = generation.modelID
    if let request = generation.nativeRequest,
      let data = try? JSONEncoder().encode(request), let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let configuration = root["configuration"] as? [String: Any] {
      draft.mossImage?.transformerDirectory = root["transformerDirectory"] as? String ?? ""
      draft.mossImage?.sharedDirectory = root["sharedDirectory"] as? String ?? ""
      draft.mossImage?.variant = root["modelVariant"] as? String ?? "sft"
      draft.mossImage?.sampler = configuration["sampler"] as? String ?? "euler"
      draft.mossImage?.schedule = configuration["schedule"] as? String ?? "release"
      draft.mossImage?.skipLayers = configuration["skipLayers"] as? [Int] ?? [22]
      draft.mossImage?.skipStart = configuration["skipStart"] as? Double ?? 0
      draft.mossImage?.skipStop = configuration["skipStop"] as? Double ?? 1
      draft.mossImage?.weightCacheGB = configuration["weightCacheGB"] as? Int ?? 0
      draft.mossImage?.quantizationBits = configuration["quantizationBits"] as? Int
      draft.mossImage?.regionalLayout = configuration["regionalLayout"] as? Bool
      draft.steps = configuration["steps"] as? Int ?? 50; draft.seed = configuration["seed"] as? Int ?? -1
      draft.guidance = configuration["guidance"] as? Double ?? 3; draft.shift = configuration["shift"] as? Double ?? 5
      draft.negativePrompt = configuration["negativePrompt"] as? String ?? ""
      if let descriptors = root["loras"] as? [[String: Any]] {
        draft.mossImage?.loras = descriptors.compactMap { value in
          guard let path = value["path"] as? String, let strength = value["strength"] as? Double, let enabled = value["enabled"] as? Bool else { return nil }
          var adapter = MediaAsset(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent, kind: .lora, path: path, scope: .global)
          adapter.loraModel = .mossImage
          if value["profile"] as? String == "sft_4step_v2" { adapter.loraProfile = "moss_sft_4step_v2" }
          return LoRAMember(asset: adapter, strength: strength, enabled: enabled)
        }
        if draft.mossImage?.loras?.contains(where: { $0.isEnabled && $0.strength != 0 && $0.asset.loraProfile == "moss_sft_4step_v2" }) == true {
          draft.mossImage?.stockRecipe = generation.mossStockRecipe ?? MOSSRecipeSnapshot()
        }
      }
    }
    imageDraft = draft; imagePreviewPath = nil; imageEstimate = nil; preparedMOSSImage = nil
  }
  func prepareMOSSImageGeneration() async {
    guard let draft = imageDraft, draft.executionProvider == .nativeMOSS,
      !operationBusy, !preparingImageRequest else { return }
    preparingImageRequest = true; defer { preparingImageRequest = false }
    let session = documentSessionID, requestID = UUID(), referenceKey = beginImageAttempt(draft)
    do {
      guard draft.seed == -1 || (0...Int(UInt32.max)).contains(draft.seed) else { throw StudioError.invalid("Use a UInt32 seed or −1 for a random seed.") }
      let seed = draft.seed == -1 ? UInt32.random(in: .min ... .max) : UInt32(draft.seed)
      let directory = dataDirectory.appendingPathComponent("Cache/MOSS Images/" + requestID.uuidString)
      let request = try await Task.detached(priority: .userInitiated) {
        let bytes = try NativeMOSSImagePreparation.request(draft: draft, requestID: requestID, seed: seed)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: directory.appendingPathComponent("request.json"), options: .withoutOverwriting)
        return bytes
      }.value
      guard documentSessionID == session, imageDraft == draft else { return }
      let result = try await bridge.invoke("moss-native-preflight", runtime: runtime,
        payload: ["recipePath": directory.appendingPathComponent("request.json").path], output: directory.appendingPathComponent("preflight"))
      guard documentSessionID == session, imageDraft == draft else { return }
      try NativeMOSSImageResult.validateIdentity(result, requestID: requestID, fingerprint: nil, seed: seed, width: draft.width, height: draft.height)
      try NativeMOSSImageResult.validateRegionalControl(result, required: draft.mossImage?.regionalLayout == true)
      guard let fingerprint = result["fingerprint"] as? String else { throw StudioError.invalid("MOSS settings were not admitted.") }
      preparedMOSSImage = .init(draft: draft, sessionID: session, requestID: requestID, seed: seed,
        request: request, directory: directory, fingerprint: fingerprint)
      imageEstimate = result
    } catch is CancellationError {
      preparedMOSSImage = nil; imageEstimate = nil; notice = "MOSS settings check cancelled."
    } catch { preparedMOSSImage = nil; imageEstimate = nil; recordImageFailure(error.localizedDescription, referenceKey: referenceKey) }
  }

  func generateMOSSImageAsset() async {
    guard let draft = imageDraft, draft.executionProvider == .nativeMOSS, !operationBusy, !preparingImageRequest else { return }
    if preparedMOSSImage?.draft != draft || preparedMOSSImage?.sessionID != documentSessionID { await prepareMOSSImageGeneration() }
    guard let prepared = preparedMOSSImage, prepared.draft == draft, prepared.sessionID == documentSessionID,
      imageEstimate?["eligibility"] as? String == "allowed" else { return }
    preparingImageRequest = true; defer { preparingImageRequest = false }
    let referenceKey = beginImageAttempt(draft)
    do {
      guard var root = try JSONSerialization.jsonObject(with: prepared.request) as? [String: Any] else { throw StudioError.invalid("Invalid prepared MOSS request.") }
      root["expectedFingerprint"] = prepared.fingerprint
      let file = prepared.directory.appendingPathComponent("render.json")
      try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]).write(to: file, options: .atomic)
      let output = dataDirectory.appendingPathComponent("Generated Images/" + UUID().uuidString)
      let result = try await bridge.invoke("moss-native-render", runtime: runtime, payload: ["recipePath": file.path], output: output)
      let bytes = try JSONSerialization.data(withJSONObject: result)
      let image = try await Task.detached(priority: .userInitiated) {
        guard let value = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else { throw StudioError.invalid("Invalid MOSS result.") }
        return try NativeMOSSImageResult.image(value, output: output, requestID: prepared.requestID,
          fingerprint: prepared.fingerprint, seed: prepared.seed, width: draft.width, height: draft.height)
      }.value
      preparedMOSSImage = nil
      guard documentSessionID == prepared.sessionID else { notice = "MOSS image saved at \(image.path). The destination movie changed."; return }
      var asset = try draft.destination.asset(name: draft.name, path: image.path, in: project)
      asset.width = draft.width; asset.height = draft.height
      var provenance = ImageGeneration(provider: ImageExecutionProvider.nativeMOSS.rawValue,
        requestFingerprint: prepared.fingerprint, modelID: draft.modelID, prompt: draft.prompt)
      provenance.nativeRequest = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject: result["normalizedRequest"] ?? [:]))
      provenance.configuration = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject: (result["normalizedRequest"] as? [String: Any])?["configuration"] ?? [:]))
      if let sentLayout = (result["normalizedRequest"] as? [String: Any])?["layout"] {
        let admitted = try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONSerialization.data(withJSONObject: sentLayout))
        let sourceObject = (result["normalizedRequest"] as? [String: Any])?["sourceImage"]
        var source: MOSSLayoutSource?
        if let sourceObject {
          var object = sourceObject as? [String: Any] ?? [:]; object["enabled"] = true
          source = try JSONDecoder().decode(MOSSLayoutSource.self, from: JSONSerialization.data(withJSONObject: object))
        }
        provenance.mossLayout = try NativeMOSSImagePreparation.layoutForSavedTake(admitted: admitted, draft: draft, admittedSource: source)
      }
      provenance.effectivePrompt = result["effectivePrompt"] as? String
      provenance.mossStockRecipe = draft.mossImage?.stockRecipe
      provenance.referenceSheet = draft.referenceSheet; provenance.rippleReference = draft.rippleReference
      provenance.generatedAt = Date(); asset.generation = provenance
      if asset.scope == .global { globalAssets.append(asset); saveGlobals() } else { change { $0.assets.append(asset) } }
      selectedAssetID = asset.id
      if imageDraft == draft { imagePreviewPath = image.path }
      notice = "Generated \(draft.name) with MOSS · seed \(prepared.seed)."
    } catch is CancellationError {
      preparedMOSSImage = nil; imageEstimate = nil; notice = "MOSS image generation cancelled."
    } catch { preparedMOSSImage = nil; imageEstimate = nil; recordImageFailure(error.localizedDescription, referenceKey: referenceKey) }
  }
}
