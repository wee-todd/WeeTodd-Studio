import Foundation
import StudioCore

@MainActor extension StudioStore {
  /// Restore process-local validation before deciding which saved takes need inference.
  /// A fresh description still detects changed models, profiles and conditioning.
  func revalidateExistingNativeTakes() async throws {
    let session = documentSessionID, projectID = project.id
    let candidates = project.clips.filter {
      $0.engine != .movie && $0.engine != .drawThings && !$0.hasReviewedReusedTake
        && !$0.renderedSignature.isEmpty && FileManager.default.fileExists(atPath: $0.sourcePath)
    }
    for clip in candidates {
      let key = generationRequestKey(for: clip)
      if generationDescriptions[clip.id]?["studioInput"] as? String == key { continue }
      var body = try payload()
      body["clipID"] = clip.id.uuidString
      var description = try await descriptionBridge.independent().invoke(
        "describe-generation", runtime: runtime, payload: body)
      guard documentSessionID == session, project.id == projectID,
        let current = project.clips.first(where: { $0.id == clip.id }),
        generationRequestKey(for: current) == key else {
        throw StudioError.invalid("The movie changed while revalidating existing takes. Prepare it again.")
      }
      let readiness = description["readinessErrors"] as? [String] ?? []
      guard readiness.isEmpty else {
        validationErrors[clip.id] = readiness.joined(separator: "\n")
        throw StudioError.invalid("Cannot revalidate \(clip.name): \(readiness.joined(separator: "; "))")
      }
      description["studioInput"] = key
      description["studioEngine"] = clip.engine.rawValue
      description["studioTask"] = clip.inferredTask
      description["studioProfile"] = clip.profileID
      generationDescriptions[clip.id] = description
      validationErrors[clip.id] = nil
    }
  }
}
