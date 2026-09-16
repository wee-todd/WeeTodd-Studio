import AppKit
import StudioCore
import SwiftUI

/// Independent setup job; closing cancels only this sheet's operation.
struct AssistantModelSetupView: View {
  @ObservedObject var store: StudioStore
  let currentModelPath: String
  let onSelect: (String) -> Void
  @Environment(\.dismiss) private var dismiss
  @StateObject private var bridge: Bridge
  @State private var catalog: AssistantModelCatalog?
  @State private var inspection: AssistantModelInspection?
  @State private var candidates: [URL] = []
  @State private var selectedPath = ""
  @AppStorage("weetodd.assistantModelDestination") private var destination = LocalPromptModels.studioDirectory.path
  @State private var error = ""
  @State private var healthDetails = ""

  init(store: StudioStore, currentModelPath: String = "", onSelect: @escaping (String) -> Void) {
    self.store = store; self.currentModelPath = currentModelPath; self.onSelect = onSelect
    _bridge = StateObject(wrappedValue: store.bridge.independent())
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack { Text("Set Up Director Model").font(.title2); Spacer(); Button("Done") { dismiss() } }
      Text("Use an installed Qwen3.5 checkpoint or download Qwen3.5 4B for local text and image understanding. The Draw Things app is not required.")
      GroupBox("Use installed") {
        VStack(alignment: .leading, spacing: 10) {
          if candidates.isEmpty { Text("No supported checkpoint found in the standard model folders.").foregroundStyle(.secondary) }
          else {
            Picker("Checkpoint", selection: $selectedPath) {
              Text("Choose installed model").tag("")
              ForEach(candidates, id: \.path) { item in Text(LocalPromptModels.label(for: item.path) + " · " + item.deletingLastPathComponent().lastPathComponent).tag(item.path) }
            }
          }
          HStack {
            Button("Locate…") { locate() }
            Button("Rescan") { candidates = LocalPromptModels.discover(including: selectedPath) }
            Button("Verify installed") { Task { await perform("assistant-model-inspect", payload: ["path": selectedPath]) } }.disabled(selectedPath.isEmpty)
          }
          if !selectedPath.isEmpty { Text(selectedPath).font(.caption).textSelection(.enabled) }
        }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
      }.disabled(bridge.busy)
      GroupBox("Download Qwen3.5 4B") {
        VStack(alignment: .leading, spacing: 10) {
          if let catalog {
            Text("Download: \(ByteCountFormatter.string(fromByteCount: catalog.downloadBytes, countStyle: .file)) · Required free space: \(ByteCountFormatter.string(fromByteCount: catalog.requiredDiskBytes, countStyle: .file))")
            HStack {
              if let url = URL(string: catalog.sourceURL) { Link("Model source", destination: url) }
              if let url = URL(string: catalog.licenseURL) { Link("Source license", destination: url) }
            }
            Text(catalog.notice).font(.caption).foregroundStyle(.secondary)
          }
          HStack { Text(destination).font(.caption).textSelection(.enabled); Spacer(); Button("Choose destination…") { chooseDestination() } }
          Button("Download / Resume") { Task { await perform("assistant-model-download", payload: ["destination": destination]) } }.disabled(catalog == nil)
          Text("Existing models are preserved. Interrupted downloads resume in the same destination; the checksum is verified before use.").font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
      }.disabled(bridge.busy)
      if bridge.busy {
        HStack { ProgressView(); Text(bridge.message); Spacer(); Button("Cancel") { bridge.cancel() } }
      }
      if let inspection {
        Text(inspection.message).font(.callout)
        HStack {
          Button(inspection.vision ? "Check text + image inference" : "Check text inference") {
            Task { await perform("assistant-model-health", payload: ["path": inspection.path]) }
          }.disabled(store.operationBusy)
          Button("Use this model") { onSelect(inspection.path); dismiss() }.disabled(!inspection.selectable)
        }.disabled(bridge.busy)
        Text("The health check makes local model calls using a small synthetic test image. It does not generate images or certify creative quality.").font(.caption).foregroundStyle(.secondary)
        if store.operationBusy { Text("Wait for the current Studio job before checking model inference.").font(.caption).foregroundStyle(.secondary) }
      }
      if !healthDetails.isEmpty { DisclosureGroup("Health check responses") { ScrollView { Text(healthDetails).font(.caption.monospaced()).textSelection(.enabled) }.frame(maxHeight: 100) } }
      if !error.isEmpty { Text(error).foregroundStyle(.red).textSelection(.enabled) }
    }.padding(22).frame(width: 700)
      .task {
        candidates = LocalPromptModels.discover(including: currentModelPath)
        if candidates.contains(where: { $0.path == currentModelPath }) { selectedPath = currentModelPath }
        await loadCatalog()
      }
      .onChange(of: selectedPath) { _, value in if inspection?.path != value { inspection = nil; healthDetails = "" } }
      .onDisappear { if bridge.busy { bridge.cancel() } }
  }

  private func locate() {
    let panel = NSOpenPanel(); panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    if !candidates.contains(url) { candidates.append(url) }; selectedPath = url.path
    Task { await perform("assistant-model-inspect", payload: ["path": url.path]) }
  }
  private func chooseDestination() {
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let url = panel.url else { return }; destination = url.path
  }
  private func loadCatalog() async {
    do {
      let value = try await bridge.invoke("assistant-model-catalog", runtime: store.runtime, payload: [:])
      catalog = try JSONDecoder().decode(AssistantModelCatalog.self, from: JSONSerialization.data(withJSONObject: value))
    } catch { self.error = "Set up or update the managed renderer in Studio Settings. " + error.localizedDescription }
  }
  private func perform(_ command: String, payload: [String: Any]) async {
    guard command != "assistant-model-health" || !store.operationBusy else {
      error = "Wait for the current Studio job before checking model inference."; return
    }
    error = ""; inspection = nil; healthDetails = ""
    do {
      let value = try await bridge.invoke(command, runtime: store.runtime, payload: payload)
      let result = try JSONDecoder().decode(AssistantModelInspection.self, from: JSONSerialization.data(withJSONObject: value))
      let url = URL(fileURLWithPath: result.path)
      if !candidates.contains(url) { candidates.append(url) }
      selectedPath = result.path; inspection = result
      if let checks = value["checks"] { healthDetails = String(data: try JSONSerialization.data(withJSONObject: checks, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8) ?? "" }
    } catch { self.error = error.localizedDescription }
  }
}
