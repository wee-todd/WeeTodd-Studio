import AppKit
import Drama3Contracts
import Foundation
import StudioCore

@MainActor extension StudioStore {
  var nativeDramaCatalogURL: URL {
    let bundled = Bundle.main.resourceURL?.appendingPathComponent("Drama3ModelCatalog.json")
    return bundled.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
      ?? URL(fileURLWithPath: runtime.root).appendingPathComponent("studio/Resources/drama3_model_catalog.json")
  }

  func nativeDramaReuseRoots(_ model: URL) -> [String] {
    var roots = installedVoiceModels.map(\.path) + [model.path, model.deletingLastPathComponent().path]
    var ancestor = model.standardizedFileURL.resolvingSymlinksInPath()
    while ancestor.path != "/" {
      if ancestor.lastPathComponent == "ai-models" { roots.append(ancestor.path); break }
      ancestor.deleteLastPathComponent()
    }
    return Array(Set(roots))
  }

  func addNativeDramaModel(at root: URL) async {
    guard !operationBusy else { return }
    nativeVoiceSetupMessage = "Checking native speech checkpoint…"
    defer { nativeVoiceSetupMessage = nil }
    do {
      let admission = try await Task.detached { try Drama3ModelLoader.inspect(at: root) }.value
      guard admission.ready else {
        throw StudioError.invalid("Complete the native speech checkpoint: \(admission.missing.joined(separator:", ")).")
      }
      try registerVoiceModel(["model": ["path": root.path, "engine": "fishDrama3",
        "name": "Fish Drama 3 · " + root.lastPathComponent, "kind": "drama3"]])
      notice = admission.modes.contains(.scene)
        ? "Native Drama 3 model checked, including scene controls."
        : "Native Drama 3 model added. Use Complete scene controls in Speech models to enable scene tasks."
    } catch { self.error = error.localizedDescription }
  }

  func cancelNativeDramaSetup() {
    nativeVoiceSetupTask?.cancel()
    if bridge.busy { bridge.cancel() }
  }

  func installNativeDramaSceneControls(model: URL, source: URL) async throws {
    nativeVoiceSetupMessage = "Extracting native scene controls…"
    _ = try await bridge.invoke("drama3-supplement-head", runtime: runtime,
      payload: ["modelPath": model.path, "source": source.path],
      output: dataDirectory.appendingPathComponent("VoiceSetup/\(UUID().uuidString)"))
    try Task.checkCancellation()
    let admission = try await Task.detached { try Drama3ModelLoader.inspect(at: model) }.value
    guard admission.ready, admission.modes.contains(.scene) else {
      throw StudioError.invalid("Scene controls did not pass native model inspection. Retry setup.")
    }
    try FileManager.default.removeItem(at: source)
  }

  func completeNativeDramaSceneControls(_ model: InstalledVoiceModel) {
    guard !operationBusy, model.engine == .fishDrama3 else { return }
    let root = URL(fileURLWithPath: model.path), catalog = nativeDramaCatalogURL
    let roots = nativeDramaReuseRoots(root)
    let attempt = UUID(); nativeVoiceSetupAttempt = attempt
    nativeVoiceSetupMessage = "Checking native scene controls…"
    nativeVoiceSetupTask = Task {
      defer { if nativeVoiceSetupAttempt == attempt { nativeVoiceSetupMessage = nil; nativeVoiceSetupTask = nil } }
      do {
        let admission = try await Task.detached { try Drama3ModelLoader.inspect(at: root) }.value
        guard admission.ready else { throw StudioError.invalid("Restore the installed model's missing files before completing scene controls.") }
        if admission.modes.contains(.scene) { notice = "Native scene controls are already ready."; return }
        let source = try await NativeModelDownloads.prepareDramaSceneSource(catalog: catalog,
          model: root, existingRoots: roots, progress: { [weak self] message, _ in
            Task { @MainActor in
              guard self?.nativeVoiceSetupAttempt == attempt, self?.nativeVoiceSetupTask != nil else { return }
              self?.nativeVoiceSetupMessage = message
            }
          })
        try Task.checkCancellation()
        try await installNativeDramaSceneControls(model: root, source: source)
        notice = "Native Drama 3 scene controls are ready."
      } catch is CancellationError {
        notice = "Scene-control setup cancelled. Retained source files can be reused when you retry."
      } catch { self.error = error.localizedDescription }
    }
  }

  func downloadNativeDramaModel(_ id: String) {
    guard !operationBusy else { return }
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
    panel.title = "Choose the shared native voice model library"
    guard panel.runModal() == .OK, let directory = panel.url else { return }
    let catalog = nativeDramaCatalogURL, roots = nativeDramaReuseRoots(directory)
    let attempt = UUID(); nativeVoiceSetupAttempt = attempt
    nativeVoiceSetupMessage = "Preparing native voice download…"
    nativeVoiceSetupTask = Task {
      defer { if nativeVoiceSetupAttempt == attempt { nativeVoiceSetupMessage = nil; nativeVoiceSetupTask = nil } }
      do {
        let root = try await NativeModelDownloads.prepare(id: id, catalog: catalog,
          destination: directory, existingRoots: roots) { [weak self] message, _ in
            Task { @MainActor in
              guard self?.nativeVoiceSetupAttempt == attempt, self?.nativeVoiceSetupTask != nil else { return }
              self?.nativeVoiceSetupMessage = message
            }
          }
        // Register the usable speech checkpoint even if scene extraction fails or is cancelled.
        // The explicit completion action can then retry against this same model folder.
        try registerVoiceModel(["model": ["path": root.path, "engine": "fishDrama3",
          "name": "Fish Drama 3 · " + root.lastPathComponent, "kind": "drama3"]])
        try Task.checkCancellation()
        try await installNativeDramaSceneControls(model: root,
          source: root.appendingPathComponent(".original-head-source.safetensors"))
        notice = "Native Drama 3 model and scene controls are ready."
      } catch is CancellationError {
        notice = "Speech model setup cancelled. Verified files remain reusable; use Complete scene controls for an installed model."
      } catch { self.error = error.localizedDescription }
    }
  }
}
