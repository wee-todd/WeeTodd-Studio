import Foundation
import StudioCore

@MainActor extension StudioStore {
  /// Reuses weight-free native request admission. No media extraction, worker call or generation.
  func admitClipPresetApplications(_ applications: [ClipPresetApplication], project input: StudioProject) async throws {
    var proposed = input
    for application in applications {
      if let index = proposed.clips.firstIndex(where: { $0.id == application.clip.id }) {
        proposed.clips[index] = application.clip
        // Describe validates composed settings. Creative readiness belongs to generation;
        // this temporary text never enters the applied clip, saved baseline, or a queued request.
        if proposed.clips[index].prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { proposed.clips[index].prompt = "Preset settings admission." }
      }
    }
    let settings = runtime, project = proposed.generationInputProject
    for application in applications {
      let clip = application.clip
      guard clip.engine == .h3 && settings.usesNativeH3 || clip.engine == .ltx25 && settings.usesNativeLTX25,
        !clip.usesNativeMovieUpscaler else { continue }
      if clip.profileID != "auto" {
        guard let profile = try? GenerationLibraryModelImport.json(URL(fileURLWithPath: clip.profileID)), profile["format"] as? String == "weetodd-headless-v2" else { continue }
      }
      let result = try await Task.detached(priority: .userInitiated) {
        var request: [String: Any] = ["project": try project.object(), "globalAssets": [], "clipID": clip.id.uuidString]
        request["runtime"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(settings))
        if clip.engine == .ltx25, clip.inferredTask != "video_upscale" {
          var originalRequest = request; originalRequest["project"] = try input.generationInputProject.object()
          let original = input.clips.first { $0.id == clip.id }
          let originalFPS = original?.settings(in: input).fps ?? project.settings.fps
          let previous = original?.engine == .ltx25 ? (try? NativeLTXPreparation.presetClock(request: originalRequest)) ?? [originalFPS, originalFPS] : [originalFPS, originalFPS]
          guard try NativeLTXPreparation.presetClock(request: request) == previous else {
            throw StudioError.invalid("This model preset would change the effective video frame rate. Choose a model with the current source clock.")
          }
        }
        return clip.engine == .h3 ? try NativeH3Preparation.describe(request: request) : try NativeLTXPreparation.describe(request: request)
      }.value
      try Task.checkCancellation()
      let errors = result["readinessErrors"] as? [String] ?? []
      guard errors.isEmpty else { throw StudioError.invalid(errors.joined(separator: "\n")) }
    }
  }
}
