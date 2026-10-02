import Foundation
import StudioCore

@MainActor extension StudioStore {
  var nativeHeadlessEligible: Bool {
    !project.clips.isEmpty && project.clips.allSatisfy { clip in
      clip.engine == .movie || clip.engine == .h3 && runtime.usesNativeH3
        || clip.engine == .ltx25 && runtime.usesNativeLTX25
    }
  }
  /// Export reuses the native preparation/compiler; it never recomposes through Python.
  func exportNativeHeadlessJob(body: [String: Any], to url: URL, clipOnly: Bool) async throws {
    let snapshot = project, settings = runtime, session = documentSessionID
    let execution = try productionExecutionFingerprint()
    var frozenProject = snapshot
    if clipOnly {
      guard let selected = selectedClip else { throw StudioError.invalid("Select a clip first.") }
      // The native finishing admission runs before any preparation. Added timeline media
      // remains explicit rather than silently disappearing from a clip-only export.
      guard snapshot.audio.isEmpty, snapshot.titles.isEmpty,
        try snapshot.continuousSceneMembers(for: selected).isEmpty else {
        throw StudioError.invalid("Export the complete scene or use the Python job exporter for timeline titles/audio.")
      }
      var clip = selected; clip.transition = "cut"
      frozenProject.clips = [clip]; frozenProject.name = clip.name
    }
    var generated = Set(body["generateIDs"] as? [String] ?? [])
    var sceneOwners: [UUID: UUID] = [:]
    for clip in frozenProject.clips where generated.contains(clip.id.uuidString) {
      let members = try snapshot.continuousSceneMembers(for: clip)
      for member in members {
        generated.insert(member.id.uuidString); sceneOwners[member.id] = members[0].id
      }
    }
    guard !frozenProject.clips.contains(where: {
      generated.contains($0.id.uuidString) && $0.rippleDraft != nil
    }) else {
      throw StudioError.invalid("Native headless export cannot generate a pending Ripple edit. Generate and apply the Ripple take in Studio before exporting; an ordinary LTX recipe cannot replace its frozen edit request.")
    }
    var workerPaths: [String: String] = [:]
    if let worker = settings.h3WorkerPath { workerPaths["h3"] = worker }
    if let worker = settings.ltx25WorkerPath { workerPaths["ltx25"] = worker }
    try NativeHeadlessJob.validateFinishing(frozenProject)
    let inputs = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".inputs-" + UUID().uuidString)
    var published = false
    defer { if !published { try? FileManager.default.removeItem(at: inputs) } }
    var recipes: [String: NativeHeadlessJob.Recipe] = [:]
    for clip in frozenProject.clips where generated.contains(clip.id.uuidString)
      && (sceneOwners[clip.id] == nil || sceneOwners[clip.id] == clip.id) {
      guard project == snapshot, documentSessionID == session,
        try productionExecutionFingerprint() == execution else {
        throw StudioError.invalid("The movie changed during job export. Export the current edit again.")
      }
      let target = inputs.appendingPathComponent(clip.id.uuidString)
      var request = body; request["clipID"] = clip.id.uuidString
      let h3 = clip.engine == .h3
      let result = try await bridge.invoke(h3 ? "h3-native-prepare" : "ltx-native-prepare",
        runtime: settings, payload: request, output: target)
      guard let recipePath = result["recipePath"] as? String else { throw StudioError.invalid("Native job preparation returned no recipe.") }
      _ = try await bridge.invoke(h3 ? "h3-native-preflight" : "ltx-native-preflight",
        runtime: settings, payload: ["recipePath": recipePath], output: target.appendingPathComponent("preflight"))
      recipes[clip.id.uuidString] = NativeHeadlessJob.Recipe(engine: clip.engine.rawValue,
        bytes: try Data(contentsOf: URL(fileURLWithPath: recipePath)), signature: signature(for: clip),
        report: try JSONSerialization.data(withJSONObject: result["report"] ?? [:]))
    }
    guard project == snapshot, documentSessionID == session,
      try productionExecutionFingerprint() == execution else {
      throw StudioError.invalid("The movie changed during job export. Export the current edit again.")
    }
    let job = try NativeHeadlessJob(project: frozenProject, recipes: recipes, workers: workerPaths, ffmpeg: settings.ffmpegPath)
    try job.write(to: url); published = true
  }
}
