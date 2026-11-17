import InferenceContracts
import Foundation
import StudioCore

@MainActor extension StudioStore {
  func invokeReCaVSRWorker(_ command: String, runtime: RuntimeSettings, recipe: String, output: URL) async throws -> [String: Any] {
    try await withTaskCancellationHandler {
      let result = try await bridge.invoke(command, runtime: runtime, payload: ["recipePath": recipe], output: output)
      try Task.checkCancellation()
      return result
    } onCancel: { [weak self] in Task { @MainActor in self?.bridge.cancel() } }
  }

  static func verifyReCaVSRResult(output: String, recipe: String) async throws -> String {
    let verification = Task.detached {
      let expected = try NativeReCaVSRRecipe.prepare(Data(contentsOf: URL(fileURLWithPath: recipe)))
      let file = output + ".json"
      let receipt = try JSONDecoder().decode(NativeReCaVSRReceipt.self, from: Data(contentsOf: URL(fileURLWithPath: file)))
      try receipt.validate(expected: expected, output: output)
      try receipt.video.verify()
      return file
    }
    return try await withTaskCancellationHandler(operation: { try await verification.value }, onCancel: { verification.cancel() })
  }
  func runReCaVSRUpscale(preflightOnly: Bool = false) {
    guard !operationBusy, !videoUpscaleSourcePath.isEmpty else { return }
    saveVideoUpscalePreferences()
    let source = videoUpscaleSourcePath, sourceAssetID = videoUpscaleSourceAssetID
    let session = documentSessionID, projectID = project.id, settings = runtime
    let options = reCaVSRSettings
    let directory = dataDirectory.appendingPathComponent("Upscales/" + UUID().uuidString)
    videoUpscaleBusy = true
    videoUpscaleStatus = "Checking the ReCaVSR source and checkpoints…"
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleBusy = false; self.videoUpscaleTask = nil }
      do {
        let prepared = preflightOnly ? FileManager.default.temporaryDirectory
          .appendingPathComponent(UUID().uuidString) : directory.appendingPathComponent("prepared")
        defer { if preflightOnly { try? FileManager.default.removeItem(at: prepared) } }
        let job = Task.detached {
          try await NativeReCaVSRPreparation.prepareMovie(source: source,
            worker: settings.reCaVSRWorkerPath, models: settings.reCaVSRModelDirectory,
            settings: options, destination: prepared)
        }
        let recipe = try await withTaskCancellationHandler(operation: { try await job.value }, onCancel: { job.cancel() })
        try Task.checkCancellation()
        if preflightOnly {
          _ = try await self.invokeReCaVSRWorker("recavsr-native-preflight", runtime: settings,
            recipe: recipe, output: prepared.appendingPathComponent("check"))
          self.videoUpscaleStatus = "ReCaVSR assets, source and memory admission verified. No inference ran."
          return
        }
        let result = try await self.invokeReCaVSRWorker("recavsr-native-render", runtime: settings,
          recipe: recipe, output: directory.appendingPathComponent("render"))
        try Task.checkCancellation()
        guard let output = result["video"] as? String else {
          throw StudioError.invalid("The ReCaVSR worker did not return a movie.")
        }
        let info = try await Self.inspectNativeMovie(output, requiresAudio: false)
        let receiptPath = try await Self.verifyReCaVSRResult(output: output, recipe: recipe)
        try Task.checkCancellation()
        guard self.documentSessionID == session, self.project.id == projectID,
          self.videoUpscaleSourcePath == source,
          self.reCaVSRSettings == options,
          sourceAssetID.map({ id in self.allAssets.contains(where: { $0.id == id && $0.path == source }) }) ?? true
        else {
          self.videoUpscaleStatus = "ReCaVSR saved its result at \(output). The source or document changed."
          return
        }
        self.reCaVSRResult = (source, output)
        var asset = MediaAsset(name: "ReCaVSR · " + URL(fileURLWithPath: source).deletingPathExtension().lastPathComponent,
          kind: .video, path: output, scope: .project)
        asset.upscaleReceipt = receiptPath
        asset.width = info["width"] as? Int ?? 0
        asset.height = info["height"] as? Int ?? 0
        asset.fps = info["fps"] as? Double ?? 0
        asset.duration = info["duration"] as? Double ?? 0
        self.change { $0.assets.append(asset) }
        self.selectedAssetID = asset.id
        self.videoUpscaleStatus = "ReCaVSR complete. Added to Project Assets."
      } catch {
        self.videoUpscaleStatus = Task.isCancelled ? "ReCaVSR cancelled." : error.localizedDescription
      }
    }
  }
}
