import Foundation
import StudioCore

@MainActor extension StudioStore {
  func runRelayVSRUpscale(preflightOnly: Bool = false) {
    guard !operationBusy, !videoUpscaleSourcePath.isEmpty else { return }
    saveVideoUpscalePreferences()
    let source = videoUpscaleSourcePath, sourceAssetID = videoUpscaleSourceAssetID
    let session = documentSessionID, projectID = project.id, settings = runtime
    let scale = relayVSRScale
    let directory = dataDirectory.appendingPathComponent("Upscales/" + UUID().uuidString)
    videoUpscaleBusy = true
    videoUpscaleStatus = "Checking the RelayVSR source and checkpoints…"
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let prepared = preflightOnly ? FileManager.default.temporaryDirectory
          .appendingPathComponent(UUID().uuidString) : directory.appendingPathComponent("prepared")
        defer { if preflightOnly { try? FileManager.default.removeItem(at: prepared) } }
        let recipe = try await NativeRelayVSRPreparation.prepareMovie(source: source,
          worker: settings.relayVSRWorkerPath, models: settings.relayVSRModelDirectory,
          scale: scale, destination: prepared)
        try Task.checkCancellation()
        if preflightOnly {
          self.videoUpscaleStatus = "Source movie and both RelayVSR checkpoint files are readable."
          return
        }
        let result = try await withTaskCancellationHandler {
          try await self.bridge.invoke("relay-native-render", runtime: settings,
            payload: ["recipePath": recipe], output: directory.appendingPathComponent("render"))
        } onCancel: { [weak self] in
          Task { @MainActor in self?.bridge.cancel() }
        }
        try Task.checkCancellation()
        guard let output = result["video"] as? String else {
          throw StudioError.invalid("The RelayVSR worker did not return a movie.")
        }
        let info = try await Self.inspectNativeMovie(output, requiresAudio: false)
        try Task.checkCancellation()
        guard self.documentSessionID == session, self.project.id == projectID,
          self.videoUpscaleSourcePath == source, self.relayVSRScale == scale,
          sourceAssetID.map({ id in self.allAssets.contains(where: { $0.id == id && $0.path == source }) }) ?? true
        else {
          self.videoUpscaleStatus = "RelayVSR saved its result at \(output). The source or document changed."
          return
        }
        self.relayVSRResult = (source, output)
        var asset = MediaAsset(name: "RelayVSR \(scale.label) · " + URL(fileURLWithPath: source).deletingPathExtension().lastPathComponent,
          kind: .video, path: output, scope: .project)
        asset.width = info["width"] as? Int ?? 0
        asset.height = info["height"] as? Int ?? 0
        asset.fps = info["fps"] as? Double ?? 0
        asset.duration = info["duration"] as? Double ?? 0
        self.change { $0.assets.append(asset) }
        self.selectedAssetID = asset.id
        self.videoUpscaleStatus = "RelayVSR \(scale.label) complete. Added to Project Assets."
      } catch {
        self.videoUpscaleStatus = Task.isCancelled ? "RelayVSR cancelled." : error.localizedDescription
      }
    }
  }
}
