import AppKit
import Foundation
import StudioCore

struct PreparedDrawThingsClip {
  var projectID: UUID
  var documentSessionID: UUID
  var clipID: UUID
  var signature: String
  var connection: DrawThingsConnection
}

extension StudioStore {
  struct DiscoveredDrawThingsLoRA {
    var id: String; var name: String; var family: String; var compatibleModelIDs: [String]
  }
  var canGenerateSelected: Bool {
    guard let clip = selectedClip else { return false }
    guard clip.engine == .drawThings else { return preparedRecipe != nil && preparedFingerprint == signature(for: clip) }
    guard let prepared = preparedDrawThingsClip else { return false }
    return prepared.projectID == project.id && prepared.documentSessionID == documentSessionID
      && prepared.clipID == clip.id
      && prepared.signature == signature(for: clip)
      && drawThingsConnections.contains(prepared.connection)
  }
  func drawThingsModelFamily(profileID: String, modelID: String) -> String {
    (drawThingsCatalogs[profileID]?["models"] as? [[String: Any]])?
      .first(where: { $0["id"] as? String == modelID })?["family"] as? String ?? ""
  }
  func drawThingsModelModifier(profileID: String, modelID: String) -> String? {
    (drawThingsCatalogs[profileID]?["models"] as? [[String: Any]])?
      .first(where: { $0["id"] as? String == modelID })?["modifier"] as? String
  }
  func prepareDrawThingsClip() async {
    guard !operationBusy else { return }
    guard let clip = selectedClip,
      let connection = drawThingsConnections.first(where: { $0.id == clip.drawThings?.profileID }) else {
      error = "Choose a Draw Things connection in the clip inspector."
      return
    }
    preparedDrawThingsClip = nil
    preparingDrawThings = true
    defer { preparingDrawThings = false }
    let projectID = project.id
    let session = documentSessionID
    let key = generationRequestKey(for: clip)
    let settings = runtime
    func isCurrent() -> Bool {
      documentSessionID == session && project.id == projectID && selectedClipID == clip.id
        && selectedClip.map { generationRequestKey(for: $0) == key } == true
        && drawThingsConnections.contains(connection)
    }
    do {
      let inputIssues = clip.drawThingsConditioningIssues(assets: allAssets)
      if !inputIssues.isEmpty {
        throw StudioError.invalid(inputIssues.joined(separator: "\n"))
      }
      var body = try payload()
      body["connection"] = try connection.object()
      try await attachmentDigests.resolve(drawThingsAttachmentPaths(clip))
      guard isCurrent() else { throw StudioError.invalid("Clip or connection changed during preflight. Prepare it again.") }
      let snapshot = signature(for: clip)
      var result = try await bridge.invoke("dt-prepare-clip", runtime: settings, payload: body)
      guard isCurrent(),
        selectedClip.map({ signature(for: $0) }) == snapshot,
        drawThingsConnections.contains(connection) else {
        throw StudioError.invalid("Clip or connection changed during preflight. Prepare it again.")
      }
      result["studioSignature"] = snapshot
      drawThingsClipEstimates[clip.id] = result
      preparedPrompt = clip.prompt
      preparedReport = String(decoding: try JSONSerialization.data(withJSONObject: result,
        options: [.prettyPrinted, .sortedKeys]), as: UTF8.self)
      if result["eligibility"] as? String == "allowed" {
        preparedDrawThingsClip = PreparedDrawThingsClip(projectID: projectID, documentSessionID: session, clipID: clip.id,
          signature: snapshot, connection: connection)
        if let i = project.clips.firstIndex(where: { $0.id == clip.id }) {
          project.clips[i].validatedSignature = snapshot
        }
        validationErrors[clip.id] = nil
        notice = "Draw Things preflight passed. Review the prompt and CU, then generate."
      } else {
        notice = "Draw Things needs attention. Review the preflight details."
      }
    } catch {
      guard documentSessionID == session else { return }
      self.error = error.localizedDescription
      if isCurrent() { validationErrors[clip.id] = error.localizedDescription }
    }
  }
  func renderDrawThingsClip() async {
    guard !operationBusy else { return }
    guard let clip = selectedClip, let prepared = preparedDrawThingsClip,
      canGenerateSelected else { error = "Prepare the current clip before generating."; return }
    preparingDrawThings = true
    defer { preparingDrawThings = false }
    let settings = runtime
    let key = generationRequestKey(for: clip)
    do {
      var body = try payload()
      body["connection"] = try prepared.connection.object()
      try await attachmentDigests.resolve(drawThingsAttachmentPaths(clip), force: true)
      guard documentSessionID == prepared.documentSessionID, selectedClipID == clip.id,
        selectedClip.map({ generationRequestKey(for: $0) == key }) == true,
        canGenerateSelected else {
        throw StudioError.invalid("Clip, connection, or attachment contents changed. Prepare the clip again.")
      }
      let output = dataDirectory.appendingPathComponent("Jobs/\(UUID().uuidString)/render")
      let result = try await bridge.invoke("dt-generate-clip", runtime: settings, payload: body, output: output)
      guard let video = result["video"] as? String, FileManager.default.fileExists(atPath: video),
        let manifest = result["manifestPath"] as? String else {
        throw StudioError.invalid("Draw Things did not return a completed video with audio.")
      }
      guard documentSessionID == prepared.documentSessionID, project.id == prepared.projectID,
        project.clips.contains(where: { $0.id == clip.id }) else {
        throw StudioError.invalid("The destination project or clip changed. The completed video is saved at \(video)")
      }
      let stillCurrent = project.clips.first(where: { $0.id == clip.id })
        .map { $0 == clip && signature(for: $0) == prepared.signature } ?? false
      change { p in
        guard let i = p.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        p.clips[i].versions.append(RenderVersion(path: video, seed: clip.seed, prompt: clip.prompt,
          recipePath: manifest, usableSourceIn: 0,
          usableDuration: result["endpointDuration"] as? Double ?? clip.duration))
        if stillCurrent {
          p.clips[i].sourcePath = video
          p.clips[i].sourceIn = 0
          p.clips[i].applyDrawThingsEndpointDuration(result["endpointDuration"] as? Double)
          p.clips[i].renderedSignature = signature(for: p.clips[i])
        }
        var asset = MediaAsset(name: clip.name + " render", kind: .video, path: video,
          scope: .clip, owner: clip.id)
        asset.duration = result["endpointDuration"] as? Double ?? clip.duration
        asset.width = clip.generationWidth; asset.height = clip.generationHeight
        p.assets.append(asset)
      }
      preparedDrawThingsClip = nil
      if stillCurrent && selectedClipID == clip.id {
        showPrompt = false
        refreshPreview()
      }
      notice = stillCurrent ? "Draw Things video and audio saved to clip versions and Clip Assets."
        : "Render saved as a version. The clip changed during generation; prepare its new settings."
    } catch { self.error = error.localizedDescription }
  }
  private func drawThingsAttachmentPaths(_ clip: Clip) -> [String] {
    clip.attachments.compactMap { attachment in allAssets.first { $0.id == attachment.assetID }?.path }
  }
  func loadDrawThingsConnections() {
    let url = Self.supportDirectory.appendingPathComponent("drawthings-connections.json")
    if let data = try? Data(contentsOf: url), let values = try? JSONDecoder().decode([DrawThingsConnection].self, from: data) {
      drawThingsConnections = values
    }
  }
  func saveDrawThingsConnections() {
    do {
      try JSONEncoder().encode(drawThingsConnections).write(
        to: Self.supportDirectory.appendingPathComponent("drawthings-connections.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func testDrawThings(_ connection: DrawThingsConnection) async {
    await discoverDrawThings(connection, force: true)
    guard drawThingsConnections.contains(connection) else { return }
    if let failure = drawThingsDiscovery.errors[connection.id] { error = failure }
    else { notice = "Connected to \(connection.name)." }
  }
  func discoverDrawThings(_ connection: DrawThingsConnection, force: Bool = false) async {
    guard drawThingsConnections.contains(connection) else { return }
    let settings = runtime
    await drawThingsDiscovery.load(connection, force: force) {
      let discoveryBridge = Bridge()
      return try await discoveryBridge.invoke("dt-discover", runtime: settings,
        payload: ["connection": try connection.object()])
    }
    if drawThingsConnections.contains(connection), imageDraft?.profileID == connection.id {
      configureCharacterSheetAdapter()
    }
  }
  func drawThingsModels(_ profileID: String, operation: String, task: String? = nil) -> [(id: String, name: String)] {
    guard let catalog = drawThingsCatalogs[profileID], let rules = catalog["capabilities"] as? [String: Any] else { return [] }
    let names = (catalog["models"] as? [[String: Any]] ?? []).reduce(into: [String: String]()) { result, item in
      if let id = item["id"] as? String, let name = item["name"] as? String { result[id] = name }
    }
    let taskModels = task.map { DrawThingsTaskFilter.modelIDs(in: rules, task: $0) }
    return rules.keys.filter { id in
      ((rules[id] as? [String: Any])?["operations"] as? [String: Any])?[operation] != nil
        && (taskModels?.contains(id) ?? true)
    }.map { (id: $0, name: names[$0] ?? $0) }.sorted {
      let order = $0.name.localizedCaseInsensitiveCompare($1.name)
      return order == .orderedSame ? $0.id < $1.id : order == .orderedAscending
    }
  }
  func drawThingsConnections(for task: String) -> [DrawThingsConnection] {
    drawThingsConnections.filter {
      drawThingsCatalogs[$0.id] == nil || !drawThingsModels($0.id, operation: "video", task: task).isEmpty
    }
  }
  func drawThingsLoRAs(profileID: String, modelID: String) -> [DiscoveredDrawThingsLoRA] {
    (drawThingsCatalogs[profileID]?["loras"] as? [[String: Any]] ?? []).compactMap { item in
      guard let id = item["id"] as? String, let name = item["name"] as? String,
        let family = item["family"] as? String,
        let compatible = item["compatibleModelIDs"] as? [String], compatible.contains(modelID)
      else { return nil }
      return DiscoveredDrawThingsLoRA(id: id, name: name, family: family, compatibleModelIDs: compatible)
    }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }
  func loadDrawThingsLoRAGroups() {
    let url = dataDirectory.appendingPathComponent("drawthings-lora-groups.json")
    if let data = try? Data(contentsOf: url),
      let groups = try? JSONDecoder().decode([DrawThingsLoRAGroup].self, from: data) {
      drawThingsLoRAGroups = groups
    }
  }
  func saveDrawThingsLoRAGroups() {
    do {
      try JSONEncoder().encode(drawThingsLoRAGroups).write(
        to: dataDirectory.appendingPathComponent("drawthings-lora-groups.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func beginImageGeneration(scope: AssetScope) {
    guard scope != .clip || selectedClipID != nil else { return }
    let destination = ImageAssetDestination(scope: scope, projectID: project.id,
      owner: scope == .clip ? selectedClipID : nil)
    if let session = imageWorkspaceLibrary.sessions[destination.storageKey] {
      restoringImageWorkspace = true
      imageDraft = session.draft; imagePreviewPath = session.previewPath; imageEstimate = nil
      restoringImageWorkspace = false; persistImageWorkspace()
      return
    }
    var draft = DrawThingsImageDraft(destination: destination)
    draft.profileID = drawThingsConnections.first?.id ?? ""
    imageDraft = draft; imageEstimate = nil; imagePreviewPath = nil
  }
}
