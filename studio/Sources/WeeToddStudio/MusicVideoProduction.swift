import AppKit
import CryptoKit
import StudioCore
import SwiftUI

@MainActor extension StudioStore {
  func productionExecutionFingerprint() throws -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    let pieces = [try encoder.encode(runtime), try encoder.encode(globalAssets),
                  try encoder.encode(loraGroups), try encoder.encode(drawThingsConnections),
                  try encoder.encode(profiles.sorted { $0.id < $1.id })]
    var data = Data()
    for piece in pieces { data.append(piece); data.append(0) }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private func decodeProduction(_ result: [String: Any]) throws -> MusicVideoProductionStatus {
    try JSONDecoder().decode(MusicVideoProductionStatus.self, from: JSONSerialization.data(withJSONObject: result))
  }
  func createProduction(maxRetries: Int, allowRemote: Bool) async {
    guard !operationBusy, !productionRunning, !project.clips.isEmpty else { return }
    productionRunning = true
    defer { productionRunning = false }
    let session = documentSessionID, snapshot = project
    do {
      let fingerprint = try snapshot.productionInputFingerprint()
      let execution = try productionExecutionFingerprint()
      try await revalidateExistingNativeTakes()
      guard documentSessionID == session, project.id == snapshot.id,
        try project.productionInputFingerprint() == fingerprint,
        try productionExecutionFingerprint() == execution else {
        notice = "The movie changed while revalidating existing takes. Prepare it again."
        return
      }
      var body = try payload()
      body["generateIDs"] = project.clips.filter {
        $0.engine != .movie && ((!$0.hasReviewedReusedTake && $0.renderedSignature != signature(for: $0))
          || !FileManager.default.fileExists(atPath: $0.sourcePath))
      }.map { $0.id.uuidString }
      body["drawThingsConnections"] = try drawThingsConnections.map { try $0.object() }
      body["maxRetries"] = maxRetries; body["allowRemote"] = allowRemote
      let directory = dataDirectory.appendingPathComponent("Productions/\(UUID().uuidString)")
      let result = try await bridge.invoke("production-create", runtime: runtime, payload: body, output: directory)
      guard documentSessionID == session, project.id == snapshot.id,
        try project.productionInputFingerprint() == fingerprint,
        try productionExecutionFingerprint() == execution else {
        notice = "Production snapshot saved at \(directory.path). The movie changed while preparing it."
        return
      }
      let status = try decodeProduction(result)
      guard status.projectID == project.id else { throw StudioError.invalid("Production belongs to another movie.") }
      change { $0.production = MusicVideoProduction(jobDirectory: status.jobDirectory,
        inputFingerprint: fingerprint, executionFingerprint: execution) }
      productionStatus = status
      notice = "Production ready. Start renders queued shots and assembles the movie with its audio tracks."
    } catch {
      if session == documentSessionID, project.id == snapshot.id { self.error = error.localizedDescription }
    }
  }
  func refreshProduction() async {
    guard !operationBusy, !productionRunning, let saved = project.production else { return }
    let session = documentSessionID
    do {
      let result = try await bridge.invoke("production-status", runtime: runtime, payload: ["jobDirectory": saved.jobDirectory])
      guard session == documentSessionID, project.production == saved else { return }
      productionStatus = try decodeProduction(result)
    } catch {
      if session == documentSessionID, project.production == saved { self.error = error.localizedDescription }
    }
  }
  func runProduction() async {
    guard !operationBusy, !productionRunning, let saved = project.production else { return }
    let session = documentSessionID, id = project.id
    guard (try? project.productionInputFingerprint()) == saved.inputFingerprint,
      (try? productionExecutionFingerprint()) == saved.executionFingerprint, !saved.applied else {
      error = "The edit changed. Create a new production snapshot; previous outputs are preserved."
      return
    }
    productionRunning = true
    // Rendering occupies the main bridge for minutes. Read status independently so
    // the sheet can show completed/running shots without cancelling that process.
    let statusBridge = bridge.independent(), settings = runtime
    let polling = Task { @MainActor in
      while !Task.isCancelled {
        guard session == documentSessionID, project.id == id, project.production == saved,
          (try? productionExecutionFingerprint()) == saved.executionFingerprint else { return }
        do {
          let result = try await statusBridge.invoke("production-status", runtime: settings,
            payload: ["jobDirectory": saved.jobDirectory])
          guard !Task.isCancelled, session == documentSessionID, project.id == id,
            project.production == saved,
            (try? productionExecutionFingerprint()) == saved.executionFingerprint else { return }
          let status = try decodeProduction(result)
          guard status.projectID == id, status.jobDirectory == saved.jobDirectory else { return }
          productionStatus = status
        } catch {
          // A transient status read must not fail the independent render. Its final
          // response remains authoritative and reports any production failure.
        }
        do { try await Task.sleep(for: .seconds(2)) } catch { return }
      }
    }
    defer {
      polling.cancel()
      statusBridge.cancel()
      productionRunning = false
    }
    do {
      let result = try await bridge.invoke("production-run", runtime: runtime, payload: ["jobDirectory": saved.jobDirectory])
      guard session == documentSessionID, project.id == id, project.production == saved,
        (try? productionExecutionFingerprint()) == saved.executionFingerprint else { return }
      productionStatus = try decodeProduction(result)
      notice = "Movie assembled. Preview it, then apply the generated takes to your timeline."
    } catch {
      guard session == documentSessionID, project.id == id, project.production == saved else { return }
      self.error = error.localizedDescription
      if let result = try? await bridge.invoke("production-status", runtime: runtime, payload: ["jobDirectory": saved.jobDirectory]),
        session == documentSessionID, project.id == id, project.production == saved {
        productionStatus = try? decodeProduction(result)
      }
    }
  }
  func applyProductionTakes() async {
    guard !operationBusy, !productionRunning, let saved = project.production,
      let status = productionStatus, status.status == "completed", status.projectID == project.id,
      status.jobDirectory == saved.jobDirectory, let path = status.resolvedProjectPath else { return }
    let session = documentSessionID
    productionRunning = true
    defer { productionRunning = false }
    do {
      guard (try productionExecutionFingerprint()) == saved.executionFingerprint else {
        throw StudioError.invalid("Runtime settings or global references changed. Create a new production for these inputs.")
      }
      _ = try await bridge.invoke("production-verify", runtime: runtime, payload: ["jobDirectory": saved.jobDirectory])
      guard session == documentSessionID, project.production == saved,
        (try productionExecutionFingerprint()) == saved.executionFingerprint else { return }
      let resolved = try await Task.detached(priority: .userInitiated) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count <= 64_000_000 else { throw StudioError.invalid("Production project exceeds its size limit.") }
        let result = try JSONDecoder().decode(StudioProject.self, from: data)
        for clip in result.clips where !FileManager.default.fileExists(atPath: clip.sourcePath) {
          throw StudioError.invalid("A generated take is missing. Open its production folder to inspect it.")
        }
        return result
      }.value
      guard session == documentSessionID, project.production == saved,
        (try productionExecutionFingerprint()) == saved.executionFingerprint else { return }
      var updated = project
      try updated.applyProduction(resolved, expectedFingerprint: saved.inputFingerprint)
      var descriptions: [UUID: [String: Any]] = [:]
      for clip in updated.clips where clip.engine != .movie && clip.engine != .drawThings {
        var body = try payload(); body["project"] = try updated.object(); body["clipID"] = clip.id.uuidString
        descriptions[clip.id] = try await descriptionBridge.invoke("describe-generation", runtime: runtime, payload: body)
        guard session == documentSessionID, project.production == saved,
          (try project.productionInputFingerprint()) == saved.inputFingerprint,
          (try productionExecutionFingerprint()) == saved.executionFingerprint else { return }
      }
      change { $0 = updated }
      // Scene versions stay a shared take and use Studio's canonical input identity.
      for index in project.clips.indices {
        let clip = project.clips[index]
        let sceneKey = continuousSceneDependencyKey(for: clip)
        if let last = project.clips[index].versions.indices.last,
          project.clips[index].versions[last].sceneTakeID != nil {
          project.clips[index].versions[last].sceneInputFingerprint = sceneKey
        }
        if var description = descriptions[clip.id] {
          description["studioInput"] = generationRequestKey(for: project.clips[index])
          description["studioEngine"] = clip.engine.rawValue
          description["studioTask"] = clip.inferredTask
          description["studioProfile"] = clip.profileID
          generationDescriptions[clip.id] = description
        }
        project.clips[index].renderedSignature = signature(for: project.clips[index])
      }
      changed(); refreshPreview()
      notice = "Generated takes applied. The original song, production objects and shot timing are preserved."
    } catch {
      if session == documentSessionID, project.production == saved { self.error = error.localizedDescription }
    }
  }
}

struct MusicVideoProductionView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @State private var retries = 1
  @State private var allowRemote = false
  private var current: MusicVideoProductionStatus? {
    guard let value = store.productionStatus, value.projectID == store.project.id,
      value.jobDirectory == store.project.production?.jobDirectory else { return nil }
    return value
  }
  private var busy: Bool { store.productionRunning || store.operationBusy }
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        Text("Produce movie").font(.title2)
        Spacer()
        Button("Done") { dismiss() }.disabled(busy)
      }
      Text("Render the reviewed timeline in order, then assemble its titles and audio tracks. Continuous scenes render together. Completed takes are retained when you pause or retry.")
      HStack {
        Picker("Local retries per shot", selection: $retries) {
          ForEach(0...3, id: \.self) { Text(String($0)).tag($0) }
        }.frame(width: 220)
        Toggle("Include self-hosted Draw Things", isOn: $allowRemote)
      }.disabled(busy)
      Text("Cloud API shots require their normal individual cost confirmation. Generate them first. Self-hosted submissions are never retried automatically.")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        Button(store.project.production == nil ? "Prepare production" : "New production snapshot") {
          Task { await store.createProduction(maxRetries: retries, allowRemote: allowRemote) }
        }.disabled(busy || store.project.clips.isEmpty)
        Button("Refresh status") { Task { await store.refreshProduction() } }
          .disabled(busy || store.project.production == nil)
      }
      if let current {
        Text(current.status.capitalized).font(.headline)
        List(current.units) { unit in
          HStack {
            VStack(alignment: .leading) {
              Text(unit.name)
              if let error = unit.error { Text(error).font(.caption).foregroundStyle(.orange) }
            }
            Spacer()
            Text("\(unit.status) · \(unit.attempts) attempts").font(.caption)
          }
        }.frame(minHeight: 140, maxHeight: 330)
        if let error = current.error { Text(error).font(.caption).foregroundStyle(.orange) }
        if let output = current.outputPath {
          HStack {
            Button("Preview movie") { NSWorkspace.shared.open(URL(fileURLWithPath: output)) }
            Button("Show output") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: output)]) }
            Button("Apply generated takes") { Task { await store.applyProductionTakes() } }
              .disabled(busy || store.project.production?.applied == true)
          }
        }
      }
      if store.productionRunning {
        HStack {
          ProgressView().controlSize(.small)
          Text(store.bridge.message).font(.caption)
          Spacer()
          Button("Pause") { store.bridge.cancel() }
        }
      } else if store.project.production?.applied != true {
        Button(current?.status == "ready" ? "Start production" : "Resume production") { Task { await store.runProduction() } }
          .disabled(busy || store.project.production == nil || current?.status == "completed")
      }
    }.padding(24).frame(width: 750)
      .interactiveDismissDisabled(busy)
      .task { await store.refreshProduction() }
  }
}
