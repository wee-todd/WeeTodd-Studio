import AppKit
import Foundation
import StudioCore

@MainActor extension CharacterSheetSessionController {
  func validateRecipeDependencies() async throws {
    guard let connection, connection.route == "grpc" else { throw StudioError.invalid("Choose a Draw Things Local connection.") }
    if let failure = catalogFailure, failure.connectionID == connection.id {
      throw StudioError.invalid(failure.message)
    }
    let initialName = models.first { $0.id == document.draft.modelID }?.name.lowercased().filter { $0.isLetter || $0.isNumber } ?? ""
    guard initialName.contains("krea2"), initialName.contains("turbo") else { throw StudioError.invalid("Choose the installed Krea 2 Turbo model for the initial sheet.") }
    guard let sheetLoRA = document.draft.characterSheetLoRAID,
      loras(model: document.draft.modelID).contains(where: { $0.id == sheetLoRA }) else {
      throw StudioError.invalid("Choose the installed compatible four-panel character LoRA.")
    }
    let refineName = models.first { $0.id == document.refinement.modelID }?.name.lowercased().filter { $0.isLetter || $0.isNumber } ?? ""
    guard refineName.contains("klein"), refineName.contains("9b"), !refineName.contains("base") else {
      throw StudioError.invalid("Choose FLUX.2 klein 9B for panel refinement.")
    }
    func adapterKey(_ item: (id: String, name: String)) -> String { (item.id + item.name).lowercased().filter { $0.isLetter || $0.isNumber } }
    let available = loras(model: document.refinement.modelID)
    guard available.contains(where: { $0.id == document.refinement.detailLoRAID && Self.isCharacterDetailLoRA($0) }) else {
      throw StudioError.invalid("Choose the compatible HighResolution9B detail LoRA.")
    }
    guard (0...(Int(UInt32.max) - 1003)).contains(document.refinement.seed) else {
      throw StudioError.invalid("Use a panel seed between 0 and 4294966292.")
    }
    if document.refinement.replaceFaces {
      guard available.contains(where: { $0.id == document.refinement.headLoRAID && adapterKey($0).contains("bfsheadv1fluxklein9bstep3750rank64") }), let head = document.headReference else {
        throw StudioError.invalid("Prepare the current reference head and select its compatible BFS rank-64 LoRA.")
      }
      let selectedFace = document.sources["face"] ?? ""
      let valid = try await Task.detached {
        try CharacterArtifactHash.file(selectedFace) == head.sourceSHA256
          && CharacterArtifactHash.file(head.sourcePath) == head.sourceSHA256
          && CharacterArtifactHash.file(head.whiteMattePath) == head.artifactSHA256["whiteMatte"]
          && CharacterArtifactHash.file(head.rgbaCutoutPath) == head.artifactSHA256["rgbaCutout"]
      }.value
      guard valid, head.matteRGB == [255, 255, 255] else {
        throw StudioError.invalid("The prepared reference head changed. Remove its background again before rendering.")
      }
    }
  }
  func generateInitial() async throws {
    try checkCancellation(); try await validateRecipeDependencies()
    guard compiled.canGenerate else { throw StudioError.invalid("Resolve the required fields and compiler diagnostics before generating.") }
    guard !document.pipeline.stages.contains(where: { $0.key.hasPrefix("initial-") && [.submitted, .uncertain].contains($0.state) }) else {
      throw StudioError.invalid("The initial sheet submission is uncertain. Inspect its saved output and explicitly allow a retry in Generation.")
    }
    var draft = document.draft
    var context = ReferenceSheetContext(subjectKey: document.subjectKey ?? "character:" + id.uuidString,
      name: document.title, kind: .character, description: document.originalDescription)
    context.characterDefinition = document.definition
    context.apply(to: &draft)
    draft.characterPanel = nil
    if let adapter = draft.characterSheetLoRAID, !draft.loras.contains(where: { $0.modelID == adapter }) { draft.loras.append(DrawThingsLoRA(modelID: adapter)) }
    if draft.seed < 0 { draft.seed = Int.random(in: 0...Int(UInt32.max)) }
    status = "Generating initial sheet · 1920 × 1088"
    let asset = try await generate(draft, stageKey: "initial-\(UUID().uuidString)")
    document.initialSheetPath = asset.path; document.panels = nil; document.cropsApproved = false
    previewPath = asset.path; save()
    try await detectPanels()
  }
  func generate(_ draft: DrawThingsImageDraft, stageKey: String) async throws -> MediaAsset {
    guard let connection else { throw StudioError.invalid("Choose a Draw Things Local connection.") }
    try checkCancellation()
    let encoded = try await Task.detached { try JSONSerialization.data(withJSONObject: draft.request(id: UUID().uuidString)) }.value
    let request = try JSONSerialization.jsonObject(with: encoded)
    let inputHashes = ((request as? [String: Any])?["inputs"] as? [[String: Any]] ?? []).compactMap { $0["sha256"] as? String }
    var payload: [String: Any] = ["connection": try connection.object(), "drawThingsRequest": request,
      "name": draft.name, "scope": "global", "project": [:]]
    let output = storage.directory(id: id).appendingPathComponent("Generations/\(UUID().uuidString)")
    let digest = try await Task.detached { try draft.characterExecutionDigest() }.value
    if document.pipeline.canReuse(key: stageKey, inputDigest: digest), let path = document.pipeline.stages.first(where: { $0.key == stageKey })?.outputPath,
      let existing = document.candidates.first(where: { $0.path == path }) { return existing }
    if let stage = document.pipeline.stages.first(where: { $0.key == stageKey }), [.submitted, .uncertain].contains(stage.state) {
      throw StudioError.invalid("This panel may already have been submitted. Inspect the saved job/output before explicitly retrying.")
    }
    let estimate = try await bridge.invoke("dt-estimate", runtime: store.runtime, payload: payload)
    guard estimate["eligibility"] as? String == "allowed" else {
      throw StudioError.invalid((estimate["issues"] as? [String])?.joined(separator: "\n") ?? "Draw Things preflight did not allow this image request.")
    }
    try checkCancellation()
    document.pipeline.record(key: stageKey, inputDigest: digest, state: .submitted, recoveryDirectory: output.path, draft: draft); save()
    payload["name"] = draft.name
    do {
      let result = try await bridge.invoke("dt-generate-image", runtime: store.runtime, payload: payload, output: output)
      guard let value = result["asset"] as? [String: Any], let path = value["path"] as? String else { throw StudioError.invalid("Draw Things returned no saved image.") }
      let hash = try await Task.detached { try CharacterArtifactHash.file(path) }.value
      var asset = MediaAsset(name: draft.name, kind: .image, path: path, scope: .global)
      asset.width = value["width"] as? Int ?? draft.width; asset.height = value["height"] as? Int ?? draft.height
      var provenance = ImageGeneration(provider: "drawThings", requestFingerprint: result["fingerprint"] as? String ?? digest,
        modelID: draft.modelID, prompt: draft.prompt)
      provenance.profileID = draft.profileID; provenance.referenceSheet = draft.referenceSheet
      provenance.characterPanel = draft.characterPanel
      provenance.configuration = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject: (result["normalizedRequest"] as? [String: Any])?["configuration"] ?? draft.configuration))
      provenance.generatedAt = Date(); asset.generation = provenance
      document.candidates.append(asset)
      document.pipeline.record(key: stageKey, inputDigest: digest, state: .completed, outputPath: path, outputHash: hash, inputHashes: inputHashes, draft: draft)
      previewPath = path; save(); try checkCancellation(); return asset
    } catch {
      if document.pipeline.stages.first(where: { $0.key == stageKey })?.state != .completed {
        document.pipeline.record(key: stageKey, inputDigest: digest, state: .uncertain, recoveryDirectory: output.path,
          message: "Inspect saved job state before retrying: " + error.localizedDescription, draft: draft); save()
      }
      throw error
    }
  }
  func detectPanels() async throws {
    guard let path = document.initialSheetPath else { throw StudioError.invalid("Generate or select an initial character sheet first.") }
    status = "Detecting actual panel boundaries…"
    let detection = try await CharacterPanelDetector().detect(source: URL(fileURLWithPath: path))
    try checkCancellation()
    guard document.initialSheetPath == path else { throw StudioError.invalid("The initial sheet changed during detection.") }
    document.panels = detection; document.cropsApproved = false; previewPath = path; save()
    status = "Review detected panel crops"
  }
  func approveCrops() {
    guard let detection = document.panels, detection.candidates.count == 4,
      Set(detection.candidates.map(\.role)).count == 4,
      let path = document.initialSheetPath, (try? CharacterArtifactHash.file(path)) == detection.sourceSHA256 else {
      error = "Review exactly four distinct views from the current initial sheet."; return
    }
    let values = detection.candidates.map(\.sourcePixelRect).sorted { $0.x < $1.x }
    guard values.allSatisfy({ $0.isValid(inWidth: 1920, height: 1088) }),
      zip(values, values.dropFirst()).allSatisfy({ $0.maxX <= $1.x }) else {
      error = "Panel crops must stay inside the 1920 × 1088 sheet and must not overlap."; return
    }
    document.cropsApproved = true; save()
  }
  func refinePanels() async throws {
    try await validateRecipeDependencies(); try checkCancellation()
    guard document.cropsApproved, let detection = document.panels, detection.candidates.count == 4,
      let source = document.initialSheetPath else { throw StudioError.invalid("Detect and approve four panel crops first.") }
    let currentHash = try await Task.detached { try CharacterArtifactHash.file(source) }.value
    guard currentHash == detection.sourceSHA256 else { throw StudioError.invalid("The initial sheet changed. Detect and review its panels again.") }
    guard Set(detection.candidates.map(\.role)).count == 4,
      detection.candidates.allSatisfy({ $0.sourcePixelRect.isValid(inWidth: 1920, height: 1088) }) else {
      throw StudioError.invalid("Review four distinct panel roles with crop bounds inside the initial sheet.")
    }
    var assembly: [[String: Any]] = []
    var usedStageKeys: [String] = []
    for (index, panel) in detection.candidates.enumerated() {
      try checkCancellation()
      let rect = panel.sourcePixelRect
      var settings = document.refinement
      let separated = settings.twoPass && settings.replaceFaces
      status = separated ? "Preparing native-size \(panel.role.label.lowercased()) · \(index + 1)/4"
        : "Upscaling \(panel.role.label.lowercased()) · \(index + 1)/4"
      let prepared = try await bridge.invoke("character-panel-prepare", runtime: store.runtime,
        payload: ["source": source, "rect": ["x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height],
          "scale": separated ? 1 : 2],
        output: storage.directory(id: id).appendingPathComponent("Panels/\(panel.id.uuidString)"))
      guard let input = prepared["padded_path"] as? String, let dimensions = prepared["padded_dimensions"] as? [Int], dimensions.count == 2 else { throw StudioError.invalid("Panel preparation did not return valid dimensions.") }
      let key = "panel-\(panel.id.uuidString)"
      var draft = try CharacterPanelRecipe.makeDraft(role: panel.role, definition: document.definition, settings: settings,
        profileID: document.draft.profileID, panelPath: input, headPath: document.headReference?.whiteMattePath,
        width: dimensions[0], height: dimensions[1], documentID: id, seed: settings.seed + index)
      // Source/crop/ref/mode fingerprints distinguish same-path derivative replacements.
      let identity = try CharacterArtifactHash.value(["source": currentHash, "rect": try CharacterArtifactHash.value(rect),
        "head": settings.replaceFaces ? (try CharacterArtifactHash.file(document.headReference!.whiteMattePath)) : "",
        "mode": separated ? "native-head-then-2x-detail-v2" : "one", "input": prepared["padded_sha256"] as? String ?? ""])
      status = "Refining \(panel.role.label.lowercased()) · \(index + 1)/4"
      if settings.twoPass && settings.replaceFaces {
        // Head-first pass excludes the detail adapter, then runs the ordinary detail recipe.
        draft = try CharacterPanelRecipe.headOnlyDraft(from: draft)
        status = "Replacing head \(panel.role.label.lowercased()) · pass 1/2 · \(index + 1)/4"
      }
      usedStageKeys.append(key + ":" + identity)
      var asset = try await generate(draft, stageKey: key + ":" + identity)
      if settings.twoPass && settings.replaceFaces {
        settings.replaceFaces = false
        status = "Upscaling swapped \(panel.role.label.lowercased()) · \(index + 1)/4"
        // Remove native transport padding before the one and only 2× enlargement.
        let detailInput = try await bridge.invoke("character-panel-prepare", runtime: store.runtime,
          payload: ["source": asset.path, "rect": ["x": 0, "y": 0, "width": rect.width, "height": rect.height], "scale": 2],
          output: storage.directory(id: id).appendingPathComponent("Panels/\(panel.id.uuidString)/Detail Input"))
        try checkCancellation()
        guard let detailPath = detailInput["padded_path"] as? String,
          let detailDimensions = detailInput["padded_dimensions"] as? [Int], detailDimensions.count == 2 else {
          throw StudioError.invalid("Detail preparation did not return valid 2× dimensions.")
        }
        draft = try CharacterPanelRecipe.makeDraft(role: panel.role, definition: document.definition, settings: settings,
          profileID: document.draft.profileID, panelPath: detailPath, headPath: nil,
          width: detailDimensions[0], height: detailDimensions[1], documentID: id, seed: settings.seed + index + 1000,
          preservesInputHead: true)
        status = "Detailing \(panel.role.label.lowercased()) · pass 2/2 · \(index + 1)/4"
        usedStageKeys.append(key + ":detail:" + identity)
        asset = try await generate(draft, stageKey: key + ":detail:" + identity)
      }
      assembly.append(["path": asset.path, "x": rect.x * 2, "y": rect.y * 2, "width": rect.width * 2, "height": rect.height * 2])
    }
    status = "Assembling refined sheet…"
    let assembled = try await bridge.invoke("character-reassemble", runtime: store.runtime,
      payload: ["width": 3840, "height": 2176, "panels": assembly], output: storage.directory(id: id).appendingPathComponent("Assemblies/\(UUID().uuidString)"))
    if let path = assembled["path"] as? String {
      var asset = MediaAsset(name: document.title + " · refined character sheet", kind: .image, path: path, scope: .global)
      asset.width = 3840; asset.height = 2176
      var context = ReferenceSheetContext(subjectKey: document.subjectKey ?? "character:" + id.uuidString, name: document.title,
        kind: .character, description: document.originalDescription)
      context.characterDefinition = document.definition
      var generation = ImageGeneration(provider: "drawThings", requestFingerprint: try CharacterArtifactHash.value(document.pipeline),
        modelID: document.refinement.modelID, prompt: compiled.prompt)
      generation.referenceSheet = context; generation.generatedAt = Date()
      generation.characterAssembly = CharacterAssemblyProvenance(detection: detection,
        head: document.refinement.replaceFaces ? document.headReference : nil,
        stages: usedStageKeys.compactMap { key in document.pipeline.stages.first { $0.key == key } })
      asset.generation = generation
      document.candidates.append(asset); previewPath = path; save()
    }
  }
  func allowRetry(stageKey: String) {
    guard !running, let index = document.pipeline.stages.firstIndex(where: { $0.key == stageKey }),
      [.submitted, .uncertain].contains(document.pipeline.stages[index].state) else { return }
    let alert = NSAlert()
    alert.messageText = "Allow this generation to be submitted again?"
    alert.informativeText = "Inspect the saved output and Draw Things first. An interrupted request may still finish; retrying can create a duplicate. Completed panels will be reused."
    alert.addButton(withTitle: "Allow Retry"); alert.addButton(withTitle: "Keep Uncertain")
    guard alert.runModal() == .alertFirstButtonReturn else { return }
    document.pipeline.stages[index].state = .failed
    document.pipeline.stages[index].message = "Retry explicitly enabled after reviewing the interrupted submission."
    save()
  }
  func exportDocument() {
    let panel = NSSavePanel(); panel.nameFieldStringValue = document.title + ".characterdirector"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do { try storage.export(document, to: url) } catch { self.error = error.localizedDescription }
  }
  func importDocument(_ url: URL) async throws {
    let imported = try storage.importDocument(from: url)
    store.characterDirector.open(documentID: imported.id)
  }
}
