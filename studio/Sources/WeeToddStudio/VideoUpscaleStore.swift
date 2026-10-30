import AppKit
import Foundation
import StudioCore
import UniformTypeIdentifiers

struct VideoUpscalePreset: Codable, Equatable, Identifiable {
  var id = UUID()
  var name: String
  var settings: FlashVSRSettings
}
private struct VideoUpscalePreferences: Codable {
  var settings: FlashVSRSettings
  var runtime: FlashVSRRuntime
  var presets: [VideoUpscalePreset]
  var tool: NativeUpscalerTool?
  var h3Settings: H3RegenerateSettings?
  var modelLibrary: String?
}
@MainActor extension StudioStore {
  func loadVideoUpscalePreferences() {
    let file = dataDirectory.appendingPathComponent("video-upscale.json")
    if let data = try? Data(contentsOf: file), let value = try? JSONDecoder().decode(VideoUpscalePreferences.self, from: data) {
      videoUpscaleSettings = value.settings; videoUpscaleRuntime = value.runtime; videoUpscalePresets = value.presets
      videoUpscaleTool = value.tool ?? .flashVSR
      h3RegenerateSettings = value.h3Settings ?? .init()
      nativeUpscalerModelLibrary = value.modelLibrary ?? h3RegenerateSettings.modelLibraryRoot
    }
    let sibling = Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddCLI").path ?? ""
    if FileManager.default.isExecutableFile(atPath: sibling) { videoUpscaleRuntime.supervisorPath = sibling }
    let nativeWorker = Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddFlashVSRWorker").path ?? ""
    if videoUpscaleRuntime.swiftWorkerPath == nil, FileManager.default.isExecutableFile(atPath: nativeWorker) {
      videoUpscaleRuntime.swiftWorkerPath = nativeWorker
    }
    for (key, current, configured, name) in [
      (0, videoUpscaleRuntime.ffmpegPath, runtime.ffmpegPath, "ffmpeg"),
      (1, videoUpscaleRuntime.ffprobePath, runtime.ffprobePath, "ffprobe")
    ] where current.isEmpty {
      let candidates = [configured, "/opt/homebrew/bin/" + name, "/usr/local/bin/" + name]
      if let found = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
        if key == 0 { videoUpscaleRuntime.ffmpegPath = found } else { videoUpscaleRuntime.ffprobePath = found }
      }
    }
    if videoUpscaleSettings.backend == .vpipe && videoUpscaleRuntime.vpipePath.isEmpty {
      let candidates = ["/Applications/Vpipe Manager.app/Contents/Helpers/vpipe", FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Vpipe Manager.app/Contents/Helpers/vpipe").path]
      videoUpscaleRuntime.vpipePath = candidates.first { FileManager.default.isExecutableFile(atPath: $0) } ?? ""
    }
  }
  func saveVideoUpscalePreferences() {
    do {
      try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
      let value = VideoUpscalePreferences(settings: videoUpscaleSettings, runtime: videoUpscaleRuntime, presets: videoUpscalePresets,
        tool: videoUpscaleTool, h3Settings: h3RegenerateSettings, modelLibrary: nativeUpscalerModelLibrary)
      try JSONEncoder().encode(value).write(to: dataDirectory.appendingPathComponent("video-upscale.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func saveVideoUpscalePreset() {
    let name = videoUpscalePresetName.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else { return }
    if let index = videoUpscalePresets.firstIndex(where: { $0.name == name }) { videoUpscalePresets[index].settings = videoUpscaleSettings }
    else { videoUpscalePresets.append(.init(name: name, settings: videoUpscaleSettings)) }
    saveVideoUpscalePreferences()
  }
  func openVideoUpscale() {
    guard !videoUpscaleBusy else { showVideoUpscale = true; return }
    loadVideoUpscalePreferences(); pausePlayback()
    if let asset = selectedAsset, asset.kind == .video {
      if videoUpscaleSourcePath != asset.path {
        videoUpscalePlan = nil; videoUpscaleResult = nil; videoUpscaleStatus = ""
        h3RegeneratePlan = nil; h3RegenerateResult = nil
      }
      videoUpscaleSourcePath = asset.path; videoUpscaleSourceAssetID = asset.id
      if let saved = asset.upscaleReceipt, let data = try? Data(contentsOf: URL(fileURLWithPath: saved)),
        let receipt = try? JSONDecoder().decode(H3RegenerateReceipt.self, from: data),
        URL(fileURLWithPath: receipt.video.path).resolvingSymlinksInPath().path == URL(fileURLWithPath: asset.path).resolvingSymlinksInPath().path,
        FileManager.default.isReadableFile(atPath: receipt.source.path), FileManager.default.isReadableFile(atPath: receipt.video.path) {
        h3RegenerateResult = receipt; h3RegenerateSettings = receipt.settings; videoUpscaleTool = .h3Regenerate
        videoUpscaleSourcePath = receipt.source.path; videoUpscaleSourceAssetID = nil
      }
      if let saved = asset.upscaleReceipt, let data = try? Data(contentsOf: URL(fileURLWithPath: saved)),
        let receipt = try? JSONDecoder().decode(FlashVSRReceipt.self, from: data) {
        let recorded = URL(fileURLWithPath: receipt.video.path).resolvingSymlinksInPath().path
        if recorded == URL(fileURLWithPath: asset.path).resolvingSymlinksInPath().path,
          FileManager.default.isReadableFile(atPath: receipt.plan.source.path),
          FileManager.default.isReadableFile(atPath: receipt.video.path),
          receipt.sourcePreview.map({ FileManager.default.isReadableFile(atPath: $0.path) }) ?? true {
          videoUpscaleResult = receipt; videoUpscaleSettings = receipt.plan.request.settings
          videoUpscaleTool = .flashVSR
          videoUpscaleSourcePath = receipt.plan.source.path; videoUpscaleSourceAssetID = nil
        } else {
          videoUpscalePlan = nil; videoUpscaleResult = nil
          videoUpscaleStatus = "Selected local result. The recorded comparison media is unavailable; use the original take in Versions for recipe retry."
        }
      }
    } else if let clip = selectedClip, !clip.sourcePath.isEmpty {
      if videoUpscaleSourcePath != clip.sourcePath {
        videoUpscalePlan = nil; videoUpscaleResult = nil; videoUpscaleStatus = ""
        h3RegeneratePlan = nil; h3RegenerateResult = nil
      }
      videoUpscaleSourcePath = clip.sourcePath; videoUpscaleSourceAssetID = nil
    }
    showVideoUpscale = true
  }
  func chooseVideoUpscaleSource() {
    let panel = NSOpenPanel(); panel.title = "Choose source movie"; panel.canChooseDirectories = false
    panel.allowedContentTypes = [.movie]; panel.allowsOtherFileTypes = true
    guard panel.runModal() == .OK, let url = panel.url else { return }
    videoUpscaleSourcePath = url.path; videoUpscaleSourceAssetID = nil; videoUpscalePlan = nil; videoUpscaleResult = nil
    h3RegeneratePlan = nil; h3RegenerateResult = nil
  }
  func connectVideoUpscale(_ kind: String) {
    let panel = NSOpenPanel(); panel.title = kind == "models" ? "Connect FlashVSR+ model folder" : "Connect " + kind
    panel.canChooseDirectories = kind == "models"; panel.canChooseFiles = kind != "models"
    panel.treatsFilePackagesAsDirectories = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    switch kind {
    case "models": videoUpscaleRuntime.modelRoot = url.path
    case "Swift worker": videoUpscaleRuntime.swiftWorkerPath = url.path
    case "vpipe": videoUpscaleRuntime.vpipePath = url.pathExtension == "app" ? url.appendingPathComponent("Contents/Helpers/vpipe").path : url.path
    case "FFmpeg": videoUpscaleRuntime.ffmpegPath = url.path
    case "FFprobe": videoUpscaleRuntime.ffprobePath = url.path
    default: return
    }
    videoUpscalePlan = nil; saveVideoUpscalePreferences()
  }
  func downloadVideoUpscaleModels() {
    guard !videoUpscaleBusy else { return }
    if videoUpscaleSettings.backend == .swiftMLX && nativeUpscalerModelLibrary.isEmpty { chooseNativeUpscalerLibrary() }
    guard videoUpscaleSettings.backend == .swiftMLX ? !nativeUpscalerModelLibrary.isEmpty : !videoUpscaleRuntime.modelRoot.isEmpty else { return }
    saveVideoUpscalePreferences(); videoUpscaleBusy = true
    let root = URL(fileURLWithPath: videoUpscaleRuntime.modelRoot), settings = videoUpscaleSettings
    let library = nativeUpscalerModelLibrary
    let logs = dataDirectory.appendingPathComponent("Downloads/FlashVSR/" + UUID().uuidString)
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        if settings.backend == .swiftMLX {
          let catalog = try self.nativeUpscalerCatalogURL()
          let token = try await self.nativeUpscalerDownloadToken()
          let package = "flashvsr-native-" + settings.variant.rawValue + "-" + (settings.decoder == .tiny ? "tiny" : "full")
          let result = try await NativeUpscalerSetup.download(packageID: package, catalog: catalog,
            modelLibrary: URL(fileURLWithPath: library), reuseRoots: root.path == "/" ? [] : [root.path], token: token, progress: { [weak self] message, fraction in
              Task { @MainActor in self?.videoUpscaleStatus = message + " · " + String(Int(fraction * 100)) + "%" }
            })
          if self.videoUpscaleSettings == settings, self.nativeUpscalerModelLibrary == library {
            self.videoUpscaleRuntime.modelRoot = result.path; self.saveVideoUpscalePreferences()
          }
        } else {
          try await FlashVSRModelDownloader.download(root: root, decoder: settings.decoder, logDirectory: logs, emit: { [weak self] message in
            Task { @MainActor in self?.videoUpscaleStatus = message }
          })
        }
        self.videoUpscaleStatus = "Selected FlashVSR+ variant and decoder verified. Native archive preparation uses Swift."
      } catch is CancellationError { self.videoUpscaleStatus = "Download cancelled. Partial files can resume in the selected model folder." }
      catch { self.videoUpscaleStatus = error.localizedDescription }
    }
  }
  func runVideoUpscale(preflightOnly: Bool = false) {
    guard !videoUpscaleBusy, !videoUpscaleSourcePath.isEmpty else { return }
    saveVideoUpscalePreferences()
    let request = VideoUpscaleRequest(sourcePath: videoUpscaleSourcePath, settings: videoUpscaleSettings, runtime: videoUpscaleRuntime)
    let session = documentSessionID, projectID = project.id, sourceAssetID = videoUpscaleSourceAssetID
    let directory = dataDirectory.appendingPathComponent("Upscales/" + UUID().uuidString)
    videoUpscaleBusy = true; videoUpscaleStatus = "Checking the source, selected model and engine…"
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let plan = try await FlashVSRService.preflight(request: request)
        try Task.checkCancellation()
        self.videoUpscalePlan = plan
        if preflightOnly { self.videoUpscaleStatus = "Ready · \(plan.frames) frames · \(plan.media.fps.argument) FPS"; return }
        let receipt = try await FlashVSRService.run(plan: plan, directory: directory, emit: { [weak self] event in
          Task { @MainActor in self?.videoUpscaleStatus = event.message }
        })
        self.videoUpscaleResult = receipt
        if self.videoUpscaleSourcePath == request.sourcePath {
          _ = self.acceptVideoUpscale(receipt, directory: directory, session: session, projectID: projectID, sourceAssetID: sourceAssetID)
        }
        self.videoUpscaleStatus = "Verified · \(receipt.plan.frames) frames · \(Int(receipt.elapsedSeconds.rounded())) s"
      } catch is CancellationError { self.videoUpscaleStatus = "Cancelled. The source movie is unchanged." }
      catch { self.videoUpscaleStatus = error.localizedDescription }
    }
  }
  @discardableResult func acceptVideoUpscale(_ receipt: FlashVSRReceipt, directory: URL, session: UUID,
    projectID: UUID, sourceAssetID: UUID?) -> Bool {
    guard documentSessionID == session, project.id == projectID else {
      notice = "Upscale saved to " + directory.path + ". The active document changed."; return false
    }
    if let sourceAssetID {
      guard allAssets.contains(where: { $0.id == sourceAssetID && $0.path == receipt.plan.request.sourcePath }) else { return false }
    }
    var asset = MediaAsset(name: "FlashVSR+ " + receipt.plan.request.settings.variant.rawValue.capitalized + " · " + receipt.plan.request.settings.decoder.label + " · " + URL(fileURLWithPath: receipt.plan.source.path).deletingPathExtension().lastPathComponent,
      kind: .video, path: receipt.video.path, scope: .project)
    asset.width = receipt.plan.request.settings.width; asset.height = receipt.plan.request.settings.height
    asset.fps = receipt.plan.media.fps.value; asset.duration = Double(receipt.plan.frames) / asset.fps
    asset.upscaleReceipt = directory.appendingPathComponent("receipt.json").path
    change { $0.assets.append(asset) }; selectedAssetID = asset.id
    return true
  }
  func nativeUpscalerCatalogURL() throws -> URL {
    let local = URL(fileURLWithPath: runtime.root).appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    guard let url = [NativeModelDownloads.bundledCatalog, local].compactMap({ $0 }).first(where: { FileManager.default.isReadableFile(atPath: $0.path) }) else {
      throw StudioError.invalid("The native model catalog is missing. Use a complete Studio app build.")
    }
    return url
  }
  func nativeUpscalerDownloadToken() async throws -> String? {
    let environment = ProcessInfo.processInfo.environment
    return try await BackgroundCredential.read { try ModelDownloadToken.resolveExisting(environment: environment) }
  }
  func chooseNativeUpscalerLibrary() {
    let panel = NSOpenPanel(); panel.title = "Choose the shared AI model library"
    panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let url = panel.url else { return }
    nativeUpscalerModelLibrary = url.path; h3RegenerateSettings.modelLibraryRoot = url.path
    videoUpscalePlan = nil; h3RegeneratePlan = nil; saveVideoUpscalePreferences()
  }
}
