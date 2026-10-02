import Foundation
import StudioCore

@MainActor final class ModelSetupState: ObservableObject {
  @Published var presets: [ModelSetupPreset] = []
  @Published var downloads: [ModelSetupDownload] = []
  @Published var loadingCatalog = false
  @Published var scanning = false
  @Published var catalogError = ""
  @Published var selectedPreset: ModelSetupPreset?
  @Published var roots: [String] = []
  @Published var selection = ModelSetupSelection()
  @Published var memoryMode = ModelSetupMemoryMode.automatic
  @Published var warnings: [String] = []
  @Published var error = ""
  @Published var resultPath = ""
  @Published var selectedDownloadID = ""
  @Published var downloadDestination = ""
  @Published var downloadMessage = ""
  @Published var setupLog = ""
  @Published var downloading = false
  @Published var downloadFraction = 0.0
  @Published var downloadStatus = ""
  var nativeDownloadCatalogURL: URL?
  var readNativeDownloadToken: () async throws -> String? = {
    let saved = try await BackgroundCredential.read { try ModelDownloadToken.read() }
    let environment = try ModelDownloadToken.environment(ProcessInfo.processInfo.environment, savedToken: saved)
    return environment["HF_TOKEN"]
  }
  private var nativeDownloadTask: Task<URL, Error>?
  private var nativeDownloadCancelled = false
  private var downloadAttempt = UUID()
  private let catalogBridge = Bridge()

  static func rendererAvailable(_ runtime: RuntimeSettings) -> Bool {
    nativeH3Available(runtime) || nativeLTXAvailable(runtime) || pythonAvailable(runtime)
  }

  private static func nativeH3Available(_ runtime: RuntimeSettings) -> Bool {
    runtime.usesNativeH3 && FileManager.default.isExecutableFile(atPath: runtime.h3WorkerPath ?? "")
  }

  private static func nativeLTXAvailable(_ runtime: RuntimeSettings) -> Bool {
    runtime.usesNativeLTX25 && FileManager.default.isExecutableFile(atPath: runtime.ltx25WorkerPath ?? "")
  }

  private static func pythonAvailable(_ runtime: RuntimeSettings) -> Bool {
    FileManager.default.isExecutableFile(atPath: runtime.pythonPath)
      && FileManager.default.fileExists(atPath: runtime.root + "/scripts/studio_bridge.py")
  }

  func loadCatalog(runtime: RuntimeSettings) async {
    guard !loadingCatalog else { return }
    loadingCatalog = true
    catalogError = ""
    defer { loadingCatalog = false }
    do {
      presets = NativeModelSetup.catalog().filter {
        ($0.engine == "h3" && Self.nativeH3Available(runtime)) ||
          ($0.engine == "ltx25" && Self.nativeLTXAvailable(runtime))
      }
      let bundled = nativeDownloadCatalogURL ?? NativeModelDownloads.bundledCatalog
      let local = URL(fileURLWithPath: runtime.root).appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
      let source = [bundled, local].compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
      nativeDownloadCatalogURL = source
      downloads = try source.map { try NativeModelDownloads.catalog(at: $0).map(\.descriptor) } ?? []
      downloads = downloads.filter {
        ($0.supports(engine: "h3") && Self.nativeH3Available(runtime)) ||
          ($0.supports(engine: "ltx25") && Self.nativeLTXAvailable(runtime))
      }
      if Self.pythonAvailable(runtime) {
        let catalog = try await catalogBridge.invoke("setup-catalog", runtime: runtime, payload: [:])
        let legacy = try decode([ModelSetupPreset].self, from: catalog["presets"] ?? [])
        presets += legacy.filter {
          !($0.engine == "h3" && Self.nativeH3Available(runtime)) &&
            !($0.engine == "ltx25" && Self.nativeLTXAvailable(runtime))
        }
        let downloadCatalog = try await catalogBridge.invoke(
          "setup-downloads", runtime: runtime, payload: [:])
        let legacyDownloads = try decode([ModelSetupDownload].self, from: downloadCatalog["downloads"] ?? [])
        downloads += legacyDownloads.filter {
          !($0.supports(engine: "h3") && Self.nativeH3Available(runtime)) &&
            !($0.supports(engine: "ltx25") && Self.nativeLTXAvailable(runtime))
        }
      }
    } catch {
      catalogError = "Optional legacy model setup is unavailable: " + error.localizedDescription
    }
  }

  func begin(_ preset: ModelSetupPreset) {
    selectedPreset = preset
    roots = []
    selection = ModelSetupSelection()
    memoryMode = .automatic
    warnings = []
    error = ""
    resultPath = ""
    downloadMessage = ""
    selectedDownloadID = ""
    setupLog = ""
  }

  func scan(store: StudioStore) async {
    guard let preset = selectedPreset, !roots.isEmpty, !scanning else { return }
    scanning = true
    error = ""
    resultPath = ""
    defer { scanning = false; captureLog(store.bridge) }
    do {
      if preset.id.hasPrefix("swift-") {
        let presetID = preset.id, selectedRoots = roots
        let result = try await Task.detached(priority: .userInitiated) {
          try NativeModelSetup.scan(presetID: presetID, roots: selectedRoots)
        }.value
        guard selectedPreset?.id == presetID, roots == selectedRoots else { return }
        selection.applyScan(result.candidates)
        warnings = result.warnings
        return
      }
      let response = try await store.bridge.invoke(
        "setup-scan", runtime: store.runtime, payload: ["presetID": preset.id, "roots": roots])
      selection.applyScan(response["candidates"] as? [String: [String]] ?? [:])
      warnings = response["warnings"] as? [String] ?? []
    } catch { self.error = error.localizedDescription }
  }

  func createRecipe(store: StudioStore) async {
    guard let preset = selectedPreset, selection.missingComponents(for: preset).isEmpty else {
      return
    }
    error = ""
    do {
      if preset.id.hasPrefix("swift-") {
        let recipe = try NativeModelSetup.recipe(preset: preset,
          selected: selection.components, memoryMode: memoryMode)
        let staged = try NativeModelSetup.stage(recipe, directory: store.runtime.profilesDirectory)
        if preset.task == "t2v" {
          let probe = StudioStore.supportDirectory.appendingPathComponent("SetupPreflight/\(UUID().uuidString)")
          defer { try? FileManager.default.removeItem(at: probe) }
          do {
            _ = try await store.bridge.invoke(preset.engine == "h3" ? "h3-native-preflight" : "ltx-native-preflight",
              runtime: store.runtime, payload: ["recipePath": staged], output: probe)
          } catch {
            try? FileManager.default.removeItem(atPath: staged)
            throw error
          }
        }
        resultPath = staged
        warnings = preset.task == "t2v"
          ? ["Swift worker preflight passed. Clip settings are validated again when preparing it."]
          : ["Component paths are linked. Import clip media and prepare the clip to run Swift worker preflight."]
        captureLog(store.bridge)
        await store.reloadProfiles()
        store.notice = preset.task == "t2v" ? "Native model recipe created and preflighted."
          : "Native model recipe created. Prepare a clip with media to validate it."
        return
      }
      let response = try await store.bridge.invoke(
        "setup-create", runtime: store.runtime,
        payload: [
          "presetID": preset.id, "components": selection.components,
          "memoryMode": memoryMode.rawValue,
          "memoryGB": Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824,
        ])
      guard let recipePath = response["recipePath"] as? String, !recipePath.isEmpty else {
        throw StudioError.invalid("Setup did not return a recipe. Open the log for details.")
      }
      resultPath = recipePath
      warnings = response["warnings"] as? [String] ?? []
      captureLog(store.bridge)
      await store.reloadProfiles()
      store.notice =
        "Model recipe created. Select it for a compatible clip, then prepare the clip to validate its media and settings."
    } catch {
      captureLog(store.bridge)
      self.error = error.localizedDescription
    }
  }

  func useRecipeForSelectedClip(store: StudioStore) {
    guard let preset = selectedPreset, let clip = store.selectedClip,
      preset.supports(clip), !resultPath.isEmpty, !store.bridge.busy
    else { return }
    guard FileManager.default.fileExists(atPath: resultPath) else {
      error = "The created recipe is missing. Create the recipe again before selecting it."
      resultPath = ""
      return
    }
    let recipePath = resultPath
    store.editClip { $0.profileID = recipePath }
    error = ""
    store.notice =
      "Recipe selected for \(clip.name). Prepare the clip to validate its media and settings."
  }

  func download(store: StudioStore) async {
    guard !selectedDownloadID.isEmpty, !downloadDestination.isEmpty, !downloading else { return }
    error = ""
    downloadMessage = ""
    defer { captureLog(store.bridge) }
    do {
      if selectedPreset?.id.hasPrefix("swift-") == true {
        guard let catalog = nativeDownloadCatalogURL else { throw StudioError.invalid("The native download catalog is missing from this app.") }
        downloading = true; downloadFraction = 0; downloadStatus = "Checking model download access…"
        nativeDownloadCancelled = false; downloadAttempt = UUID()
        let attempt = downloadAttempt
        defer { downloading = false; nativeDownloadTask = nil }
        let token = try await readNativeDownloadToken()
        guard !nativeDownloadCancelled else { throw CancellationError() }
        let id = selectedDownloadID, destination = URL(fileURLWithPath: downloadDestination), selectedRoots = roots
        let task = Task.detached(priority: .utility) {
          try await NativeModelDownloads.prepare(id: id, catalog: catalog, destination: destination,
            existingRoots: selectedRoots, token: token, progress: { [weak self] message, fraction in
              Task { @MainActor in
                guard self?.downloading == true, self?.downloadAttempt == attempt else { return }
                self?.downloadFraction = fraction; self?.downloadStatus = message
              }
            })
        }
        nativeDownloadTask = task
        let path = try await task.value.path
        if !roots.contains(path) { roots.append(path) }
        downloadMessage = "Pinned model files verified and installed by Swift. Scan this folder to choose its components."
        setupLog = String((setupLog + "\n" + downloadMessage + "\n" + path).suffix(60000))
        return
      }
      let response = try await store.bridge.invoke(
        "setup-download", runtime: store.runtime,
        payload: [
          "downloadID": selectedDownloadID, "destination": downloadDestination,
          "existingRoots": roots,
        ])
      guard let path = response["path"] as? String, !path.isEmpty else {
        throw StudioError.invalid("Download did not return a model path. Open the log for details.")
      }
      if !roots.contains(path) { roots.append(path) }
      downloadMessage =
        response["message"] as? String ?? "Model prepared. Scan the selected folders to use it."
    } catch is CancellationError {
      downloadMessage = "Download cancelled. Retry to resume the retained partial files."
    } catch { self.error = error.localizedDescription }
  }

  func cancelDownload() { nativeDownloadCancelled = true; nativeDownloadTask?.cancel() }

  private func captureLog(_ bridge: Bridge) {
    setupLog = String((setupLog + "\n" + bridge.log).suffix(60000))
  }

  private func decode<T: Decodable>(_ type: T.Type, from object: Any) throws -> T {
    try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: object))
  }
}
