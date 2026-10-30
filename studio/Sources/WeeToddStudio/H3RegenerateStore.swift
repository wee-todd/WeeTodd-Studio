import AppKit
import Foundation
import StudioCore

enum NativeUpscalerTool: String, Codable, CaseIterable { case flashVSR, h3Regenerate }

@MainActor extension StudioStore {
  var h3RegenerateRuntime: H3RegenerateRuntime {
    .init(ffmpegPath: videoUpscaleRuntime.ffmpegPath, ffprobePath: videoUpscaleRuntime.ffprobePath,
      workerPath: runtime.h3WorkerPath ?? "")
  }
  func editH3Regenerate(_ edit: (inout H3RegenerateSettings) -> Void) {
    edit(&h3RegenerateSettings); h3RegeneratePlan = nil
  }
  func prepareH3RegenerateModels() {
    guard !videoUpscaleBusy else { return }
    if nativeUpscalerModelLibrary.isEmpty { chooseNativeUpscalerLibrary() }
    guard !nativeUpscalerModelLibrary.isEmpty else { return }
    let library = nativeUpscalerModelLibrary, profiles = runtime.profilesDirectory
    let worker = runtime.h3WorkerPath ?? "", includeFast = h3RegenerateSettings.decoder == .fastH3
    videoUpscaleBusy = true; videoUpscaleStatus = "Checking installed H3 models before acquiring missing components…"
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let token = try await self.nativeUpscalerDownloadToken()
        let result = try await NativeUpscalerSetup.prepareH3Restoration(catalog: self.nativeUpscalerCatalogURL(),
          modelLibrary: URL(fileURLWithPath: library), profileDirectory: URL(fileURLWithPath: profiles),
          workerPath: worker, includeFastDecoder: includeFast, token: token, progress: { [weak self] message, fraction in
            Task { @MainActor in self?.videoUpscaleStatus = message + " · " + String(Int(fraction * 100)) + "%" }
          })
        try Task.checkCancellation()
        guard self.nativeUpscalerModelLibrary == library else { throw StudioError.invalid("The selected model library changed during setup. Verified files remain in the original library.") }
        self.editH3Regenerate {
          $0.baseProfilePath = result.profileURL.path; $0.modelLibraryRoot = library
          $0.publisherEncoderPath = result.publisherEncoderURL.path; $0.fastDecoderPath = result.fastDecoderURL?.path ?? ""
        }
        self.saveVideoUpscalePreferences(); await self.reloadProfiles()
        self.videoUpscaleStatus = "H3 model setup complete. Source dimensions, timing and memory are checked before each restoration."
      } catch is CancellationError { self.videoUpscaleStatus = "H3 setup cancelled. Verified components remain reusable in the selected model library." }
      catch { self.videoUpscaleStatus = error.localizedDescription }
    }
  }
  func chooseH3RegenerateProfile() {
    let panel = NSOpenPanel(); panel.title = "Choose ordinary H3 Ref2VA components"; panel.allowedContentTypes = [.json]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    editH3Regenerate { $0.baseProfilePath = url.path }; saveVideoUpscalePreferences()
  }
  func runH3Regenerate(preflightOnly: Bool = false) {
    guard !videoUpscaleBusy, !videoUpscaleSourcePath.isEmpty else { return }
    saveVideoUpscalePreferences()
    let request = H3RegenerateRequest(sourcePath: videoUpscaleSourcePath, settings: h3RegenerateSettings, runtime: h3RegenerateRuntime)
    let session = documentSessionID, projectID = project.id, assetID = videoUpscaleSourceAssetID
    let directory = dataDirectory.appendingPathComponent("H3Regenerate/" + UUID().uuidString)
    videoUpscaleBusy = true; videoUpscaleStatus = "Checking H3 source cadence, components and restoration profile…"
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let service = H3RegenerateService(), plan = try await service.preflight(request: request)
        try Task.checkCancellation(); self.h3RegeneratePlan = plan
        if preflightOnly { self.videoUpscaleStatus = "Ready · \(plan.frames) source frames · 24 FPS · three refinement steps"; return }
        let receipt = try await service.run(plan: plan, directory: directory, emit: { [weak self] event in
          Task { @MainActor in self?.videoUpscaleStatus = event.message }
        })
        try Task.checkCancellation(); self.h3RegenerateResult = receipt
        if self.videoUpscaleSourcePath == request.sourcePath {
          _ = self.acceptH3Regenerate(receipt, directory: directory, session: session, projectID: projectID, sourceAssetID: assetID)
        }
        self.videoUpscaleStatus = "H3 verified · \(receipt.frames) frames · \(Int(receipt.elapsedSeconds.rounded())) s"
      } catch is CancellationError { self.videoUpscaleStatus = "Restoration cancelled. Original movie retained." }
      catch { self.videoUpscaleStatus = error.localizedDescription }
    }
  }
  @discardableResult func acceptH3Regenerate(_ receipt: H3RegenerateReceipt, directory: URL, session: UUID,
    projectID: UUID, sourceAssetID: UUID?) -> Bool {
    guard documentSessionID == session, project.id == projectID,
      URL(fileURLWithPath: videoUpscaleSourcePath).resolvingSymlinksInPath().path == receipt.source.path else { return false }
    if let sourceAssetID {
      guard allAssets.contains(where: { $0.id == sourceAssetID && URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path == receipt.source.path }) else { return false }
    }
    var asset = MediaAsset(name: "H3 Regenerate · " + URL(fileURLWithPath: receipt.source.path).deletingPathExtension().lastPathComponent,
      kind: .video, path: receipt.video.path, scope: .project)
    asset.width = receipt.settings.targetWidth; asset.height = receipt.settings.targetHeight
    asset.fps = receipt.fps.value; asset.duration = Double(receipt.frames) / asset.fps
    asset.upscaleReceipt = directory.appendingPathComponent("receipt.json").path
    change { $0.assets.append(asset) }; selectedAssetID = asset.id; return true
  }
  func editH3VideoProcessing(_ edit: (inout H3RegenerateSettings) -> Void) {
    guard let clip = selectedClip, var recipe = clip.videoProcessing,
      let index = recipe.steps.firstIndex(where: \.isH3Regenerate), var settings = recipe.steps[index].h3Regenerate else { return }
    edit(&settings); recipe.steps[index].h3Regenerate = settings
    let ids = Set(videoProcessingClips(for: clip).map(\.id))
    change { project in for i in project.clips.indices where ids.contains(project.clips[i].id) {
      project.clips[i].videoProcessing = recipe
    } }
  }
}
